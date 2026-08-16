"""
generate_queries.py — use an LLM (Gemini) to propose fresh search queries for
the AIM Composites crawler, then optionally launch pdf_crawler.py with them.

Reruns of pdf_crawler with the same queries find nothing new (the crawler is
incremental and dedups against state.json). To actually grow the corpus you
need *different* search intents. This script asks Gemini for a batch of new,
non-overlapping queries about thermoplastic composites — spanning matrices
(PEEK/PEKK/PPS/PEI/PA/PP/PC/LCP…), fiber types, and property kinds — that
avoid the ones already built in to pdf_crawler.

It reuses the same Gemini REST setup as the extraction pipeline (extraction.py:
model, URL template, retry/backoff, structured-output config) so behavior
matches the rest of the project, and it pulls the existing query list straight
from pdf_crawler so the "avoid these" set never drifts out of sync.

Usage:
    set GEMINI_API_KEY=...                       (Windows)   /  export … (bash)

    python generate_queries.py --n 12            # just print 12 new queries
    python generate_queries.py --n 12 --out queries.txt
    python generate_queries.py --n 12 --run --out-dir ./crawl_out
    python generate_queries.py --n 8 --focus "wear and fatigue of PPS composites"

With --run it hands the generated queries to pdf_crawler.main(...) directly
(academic sources only by default — the datasheet seed list is fixed, so new
queries don't change it).

Author: Mathias Heider, AIM Composites project, June 2026.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests

# Reuse the project's existing Gemini config and query list — single source of
# truth, so this script tracks any changes made there. (These constants live in
# extraction.py since the hardening phase; importing them from batch_ingest
# raised ImportError and made this script unrunnable.)
from extraction import GEMINI_MODEL, GEMINI_URL_TEMPLATE, REQUEST_TIMEOUT_S, gemini_request
from pdf_crawler import DEFAULT_QUERIES

# Structured-output schema: force a clean JSON list of query strings.
QUERY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "queries": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["queries"],
}

PROMPT_TEMPLATE = (
    "You are helping build a materials-property database focused on "
    "THERMOPLASTIC COMPOSITES and their constituents.\n\n"
    "Propose {n} web-search queries that would find open-access research papers "
    "AND manufacturer datasheets containing real mechanical/thermal property "
    "data.\n\n"
    "Rules for each query:\n"
    "- 3 to 8 words, the way someone would actually type into a search engine "
    "(no boolean operators, no quotes).\n"
    "- Be specific: name a matrix polymer and/or a fiber, plus a property or "
    "the word 'datasheet'.\n"
    "- Spread the {n} queries ACROSS:\n"
    "    matrices: PEEK, PEKK, PPS, PEI/Ultem, PA6, PA66, PA12, polypropylene, "
    "polycarbonate, PET, PETG, ABS, LCP, PPSU;\n"
    "    fibers: carbon, glass, aramid; unidirectional, woven, and short-fiber "
    "forms;\n"
    "    properties: tensile, flexural, compressive, interlaminar shear (ILSS), "
    "impact, fatigue, creep, CTE, glass transition, crystallinity, fiber "
    "volume fraction, thermal conductivity.\n"
    "- Do NOT duplicate or closely paraphrase any of these EXISTING queries:\n"
    "{existing}\n"
    "{focus}"
    "Return ONLY JSON of the form {{\"queries\": [\"...\", \"...\"]}}."
)


def generate_queries(n: int, focus: str, api_key: str, model: str) -> list[str]:
    existing = "\n".join(f"  - {q}" for q in DEFAULT_QUERIES)
    focus_line = (f"- Bias the set toward this theme: {focus}.\n" if focus else "")
    prompt = PROMPT_TEMPLATE.format(n=n, existing=existing, focus=focus_line)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.9,            # we want variety, not determinism
            "responseMimeType": "application/json",
            "responseSchema": QUERY_SCHEMA,
        },
    }
    url = GEMINI_URL_TEMPLATE.format(model=model, key=api_key)
    resp = gemini_request(url, payload, timeout=REQUEST_TIMEOUT_S)  # retry/backoff
    if resp is None:
        return []
    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        return []
    text = ""
    for part in candidates[0].get("content", {}).get("parts", []):
        t = (part.get("text") or "").strip()
        if t.startswith("{"):
            text = t
            break
    if not text:
        return []
    obj = json.loads(text)
    return [str(q).strip() for q in obj.get("queries", []) if str(q).strip()]


def dedup_against_existing(queries: list[str]) -> list[str]:
    """Drop exact/case-insensitive repeats of built-in or earlier queries."""
    seen = {q.lower() for q in DEFAULT_QUERIES}
    out: list[str] = []
    for q in queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            out.append(q)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=12, help="how many queries to ask for")
    ap.add_argument("--focus", default="", help="optional theme to bias toward")
    ap.add_argument("--model", default=GEMINI_MODEL, help="Gemini model id")
    ap.add_argument("--out", type=Path, default=None,
                    help="also write the queries to this file (one per line)")
    ap.add_argument("--run", action="store_true",
                    help="immediately crawl with the generated queries")
    ap.add_argument("--out-dir", default="./crawl_out",
                    help="pdf_crawler output dir when --run is used")
    ap.add_argument("--max-per-query", type=int, default=8,
                    help="passed through to pdf_crawler when --run is used")
    ap.add_argument("--include-datasheets", action="store_true",
                    help="with --run, also crawl datasheet seeds (default: skip, "
                         "since the seed list does not depend on the queries)")
    args = ap.parse_args(argv)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment.",
              file=sys.stderr)
        return 2

    try:
        raw = generate_queries(args.n, args.focus, api_key, args.model)
    except requests.RequestException as e:
        print(f"ERROR: Gemini request failed: {e}", file=sys.stderr)
        return 1
    except (ValueError, KeyError) as e:
        print(f"ERROR: could not parse Gemini response: {e}", file=sys.stderr)
        return 1

    queries = dedup_against_existing(raw)
    if not queries:
        print("No new queries produced (all duplicated existing ones).",
              file=sys.stderr)
        return 1

    print(f"# {len(queries)} new queries:", file=sys.stderr)
    for q in queries:
        print(q)

    if args.out:
        args.out.write_text("\n".join(queries) + "\n", encoding="utf-8")
        print(f"# written to {args.out}", file=sys.stderr)

    if args.run:
        import pdf_crawler
        crawl_argv = ["--out", args.out_dir, "--queries", *queries,
                      "--max-per-query", str(args.max_per_query)]
        if not args.include_datasheets:
            crawl_argv.append("--skip-datasheets")
        print(f"# launching crawler with {len(queries)} queries...", file=sys.stderr)
        return pdf_crawler.main(crawl_argv)

    return 0


if __name__ == "__main__":
    sys.exit(main())
