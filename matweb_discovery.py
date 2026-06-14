"""
matweb_discovery.py — MatWeb-as-index discovery for the AIM Composites
Materials Database crawler.

MatWeb (matweb.com) sits behind Cloudflare bot protection, and its license
restricts reproducing datasheet *content*. So this module deliberately does
NOT scrape property data from MatWeb. Instead it uses MatWeb as what it is
best at — an index of who makes what:

  Stage 1 (browser)  Search MatWeb for material keywords via Playwright
                     driving a real Chromium window. Harvest only the
                     material *names* from the result list
                     (e.g. "Ensinger Tecapeek IM CF30 black ...").
                     -> matweb_index.csv
  Stage 2 (ddgs)     For each material name, web-search for the official
                     manufacturer datasheet PDF (excluding matweb.com).
                     -> matweb_candidates.csv
  Stage 3 (optional) Hand candidates to pdf_crawler's download machinery
                     (dedup, %PDF check, provenance) -> <out>/pdfs/ +
                     sources.csv, ready for batch_ingest.py.

If Cloudflare shows an interactive challenge, solve it yourself in the
browser window the script opens — the script just waits. It will not try
to bypass bot checks, and it rate-limits itself on MatWeb's pages.

Usage:
    pip install playwright ddgs requests
    playwright install chromium

    python matweb_discovery.py --out ./crawl_out
    python matweb_discovery.py --out ./crawl_out --queries "PPS glass fiber" "PEKK"
    python matweb_discovery.py --out ./crawl_out --no-download   # stop at candidates CSV

Stage-1 result parsing verified against live MatWeb markup (June 2026):
result rows are <a href="/search/DataSheet.aspx?MatGUID=<32 hex>">name</a>.

Author: Mathias Heider, AIM Composites project, June 2026.
Discovery/yield ideas adapted from InDesDatabaseTool `agentic` branch.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import time
import urllib.parse
from pathlib import Path

log = logging.getLogger("matweb_discovery")

MATWEB_SEARCH = "https://www.matweb.com/search/QuickText.aspx?SearchText={q}"
PAGE_DELAY_S = 4.0            # politeness between MatWeb page loads
CHALLENGE_WAIT_S = 90         # max seconds to wait for Cloudflare/human
MAX_NAMES_PER_QUERY = 60
DDG_RESULTS_PER_NAME = 6

DEFAULT_QUERIES = [
    "PEEK carbon fiber",
    "PEEK glass fiber",
    "PEKK carbon fiber",
    "PPS carbon fiber",
    "PPS glass fiber",
    "PEI carbon fiber",
    "polyamide 66 glass fiber",
    "polypropylene glass fiber composite",
]

# Words stripped from material names before web-searching (colors, forms)
NAME_NOISE = re.compile(
    r"\b(black|natural|uncolored|grey|gray|blue|white|dry|conditioned|"
    r"extruded|compression molded|rods?|plates?|tubes?|stock shapes?)\b",
    re.IGNORECASE)


# ---------------------------------------------------------------------------
# Stage 1: harvest material names from MatWeb search results (real browser)
# ---------------------------------------------------------------------------

def harvest_matweb_index(queries: list[str], headless: bool) -> list[dict]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.error("Playwright not installed: pip install playwright && "
                  "playwright install chromium")
        sys.exit(1)

    rows: list[dict] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        page = browser.new_page()
        for q in queries:
            url = MATWEB_SEARCH.format(q=urllib.parse.quote(q))
            log.info("[matweb] %s", q)
            page.goto(url, wait_until="domcontentloaded")
            # Wait for result links; if Cloudflare interjects, give the human
            # time to clear it. We do not attempt to bypass the check.
            try:
                page.wait_for_selector('a[href*="MatGUID"]',
                                       timeout=CHALLENGE_WAIT_S * 1000)
            except Exception:
                title = page.title()
                if "moment" in title.lower():
                    log.warning("Cloudflare challenge not cleared for %r — "
                                "rerun without --headless and solve it in the "
                                "window.", q)
                else:
                    log.info("No results for %r", q)
                continue
            links = page.eval_on_selector_all(
                'a[href*="DataSheet.aspx?MatGUID="]',
                "els => els.map(e => e.textContent.trim())")
            n = 0
            for name in links:
                name = re.sub(r"\s+", " ", name).strip()
                if not name or name.lower().startswith("overview of materials"):
                    continue          # generic aggregate entries, not a grade
                rows.append({"query": q, "material_name": name})
                n += 1
                if n >= MAX_NAMES_PER_QUERY:
                    break
            log.info("[matweb] %d names for %r", n, q)
            time.sleep(PAGE_DELAY_S)
        browser.close()

    # de-dup, keep first occurrence
    seen, out = set(), []
    for r in rows:
        if r["material_name"] not in seen:
            seen.add(r["material_name"])
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Stage 2: find official manufacturer datasheet PDFs for each name
# ---------------------------------------------------------------------------

def find_datasheet_candidates(index_rows: list[dict]):
    """Yield datasheet-PDF candidates for each harvested material name.

    Implemented as a generator so the caller can persist rows incrementally:
    a Cloudflare hiccup or DDG rate-limit partway through a long run then keeps
    everything found so far instead of losing the batch.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        log.error("ddgs not installed: pip install ddgs")
        sys.exit(1)

    seen_urls: set[str] = set()
    with DDGS() as ddgs:
        for r in index_rows:
            name = NAME_NOISE.sub("", r["material_name"])
            name = re.sub(r"\s+", " ", name).strip()
            short = " ".join(name.split()[:7])       # manufacturer + grade
            query = f'"{short}" datasheet filetype:pdf'
            try:
                results = list(ddgs.text(query, max_results=DDG_RESULTS_PER_NAME))
            except Exception as e:
                log.warning("search failed for %r: %s", short, e)
                time.sleep(5)
                continue
            for res in results:
                url = res.get("href", "")
                low = url.lower()
                if not url or url in seen_urls:
                    continue
                if "matweb.com" in low:
                    continue                          # index only — no content
                if not (low.endswith(".pdf") or "/pdf" in low):
                    continue
                seen_urls.add(url)
                yield {
                    "title": res.get("title") or short,
                    "pdf_url": url,
                    "source": "matweb_index",
                    "query": r["material_name"],
                }
            time.sleep(1.0)                           # be polite to DDG too


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    global MAX_NAMES_PER_QUERY, DDG_RESULTS_PER_NAME
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="./crawl_out")
    ap.add_argument("--queries", nargs="*", default=None)
    ap.add_argument("--headless", action="store_true",
                    help="run browser headless (Cloudflare challenges can't "
                         "be solved manually in this mode)")
    ap.add_argument("--no-download", action="store_true",
                    help="stop after writing matweb_candidates.csv")
    ap.add_argument("--skip-harvest", action="store_true",
                    help="reuse existing matweb_index.csv, skip the browser")
    ap.add_argument("--max-names", type=int, default=MAX_NAMES_PER_QUERY,
                    metavar="N", help="max material names kept per MatWeb query")
    ap.add_argument("--ddg-results", type=int, default=DDG_RESULTS_PER_NAME,
                    metavar="N", help="web-search results examined per name")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")

    MAX_NAMES_PER_QUERY = args.max_names
    DDG_RESULTS_PER_NAME = args.ddg_results
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index_csv = out / "matweb_index.csv"
    cand_csv = out / "matweb_candidates.csv"

    # Stage 1
    if args.skip_harvest and index_csv.exists():
        index_rows = list(csv.DictReader(index_csv.open(encoding="utf-8")))
        log.info("Reusing %d names from %s", len(index_rows), index_csv)
    else:
        index_rows = harvest_matweb_index(args.queries or DEFAULT_QUERIES,
                                          headless=args.headless)
        with index_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["query", "material_name"])
            w.writeheader()
            w.writerows(index_rows)
        log.info("Stage 1: %d material names -> %s", len(index_rows), index_csv)
    if not index_rows:
        log.error("No material names harvested; nothing to do.")
        return 1

    # Stage 2 — write each candidate as it is found so an interrupted run
    # keeps its partial results.
    n_cands = 0
    with cand_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["title", "pdf_url", "source", "query"])
        w.writeheader()
        for cand in find_datasheet_candidates(index_rows):
            w.writerow(cand)
            f.flush()
            n_cands += 1
    log.info("Stage 2: %d datasheet candidates -> %s", n_cands, cand_csv)

    # Stage 3
    if args.no_download:
        log.info("Run: python pdf_crawler.py --out %s --from-csv %s "
                 "--skip-academic --skip-datasheets", out, cand_csv)
        return 0
    import pdf_crawler
    return pdf_crawler.main(["--out", str(out), "--from-csv", str(cand_csv),
                             "--skip-academic", "--skip-datasheets"])


if __name__ == "__main__":
    sys.exit(main())
