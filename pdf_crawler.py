"""
pdf_crawler.py — source-discovery crawler for the AIM Composites Materials
Database (component 1 of the autonomous-ingestion architecture, Section 6
of the project report).

Finds and downloads PDFs about thermoplastic composites from:

  1. OpenAlex          — open-access papers (no API key)
  2. Semantic Scholar  — open-access papers (no API key, free tier)
  3. arXiv             — preprints (no API key)
  4. Unpaywall         — OA-PDF fallback for DOIs OpenAlex has no PDF for
  5. Datasheets        — best-effort depth-1 crawl of manufacturer seed pages,
                         harvesting links to .pdf files (respects robots.txt)
  6. --from-csv FILE   — candidate URLs produced elsewhere, e.g. by
                         matweb_discovery.py (MatWeb-as-index workflow)

Crawl-strategy ideas (HEAD pre-checks, junk-URL filters, per-source yield
tracking) adapted from InDesDatabaseTool `agentic` branch (Frontier.py,
Downloader.py, Discovery.py).

Output:
  <out>/pdfs/*.pdf       downloaded, verified (%PDF magic), sha256-deduped
  <out>/sources.csv      provenance: filename, title, doi, url, year, source, sha256, query
  <out>/state.json       seen URL/hash sets — reruns resume without re-downloading

The pdfs/ folder feeds directly into batch_ingest.py --input.

Usage:
    python pdf_crawler.py --out ./crawl_out --max-per-query 10
    python pdf_crawler.py --out ./crawl_out --queries "PEEK carbon fiber tensile" "PPS GF30 datasheet"
    python pdf_crawler.py --out ./crawl_out --skip-datasheets
    python pdf_crawler.py --out ./crawl_out --max-total 50
    python pdf_crawler.py --out ./crawl_out --seed-urls https://www.ensingerplastics.com/en/shapes/products

Requires: requests  (pip install requests)

Author: Mathias Heider, AIM Composites project, June 2026.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sys
import time
import urllib.parse
import urllib.robotparser
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Optional

import requests

log = logging.getLogger("pdf_crawler")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONTACT_EMAIL = "mathias.heider@gmail.com"  # used for polite API identification
USER_AGENT = f"AIMCompositesCrawler/0.1 (research; mailto:{CONTACT_EMAIL})"

# Default search intents. Edit freely or override with --queries.
DEFAULT_QUERIES = [
    "thermoplastic composite mechanical properties",
    "carbon fiber PEEK composite tensile properties",
    "glass fiber polypropylene composite properties",
    "carbon fiber PPS laminate mechanical",
    "PEKK composite thermoforming properties",
    "PEI carbon fiber composite processing",
    "PA66 glass fiber composite mechanical properties",
    "thermoplastic prepreg consolidation properties",
]

# Keywords used to score relevance of titles/abstracts. A candidate must
# reach MIN_SCORE to be downloaded.
MATERIAL_KEYWORDS = {
    "thermoplastic": 3, "composite": 2, "laminate": 2, "prepreg": 2,
    "peek": 3, "pekk": 3, "pps": 2, "pei": 2, "paek": 3, "ultem": 2,
    "polypropylene": 2, "polyamide": 2, "pa6": 2, "pa66": 2, "pa12": 2,
    "nylon": 1, "petg": 1, "pet": 1, "abs": 1, "pc": 1,
    "carbon fiber": 2, "carbon fibre": 2, "glass fiber": 2, "glass fibre": 2,
    "cf/": 2, "gf/": 2, "fiber reinforced": 2, "fibre reinforced": 2,
    "tensile": 1, "modulus": 1, "flexural": 1, "strength": 1,
    "datasheet": 2, "mechanical propert": 2, "thermoforming": 1,
    "consolidation": 1, "crystallinity": 1, "matrix": 1,
}
MIN_SCORE = 4

# Best-effort manufacturer/datasheet seed pages (depth-1: fetch page, harvest
# links to PDFs and to same-domain pages that look like datasheet listings).
DEFAULT_SEED_URLS = [
    "https://www.victrex.com/en/datasheets",
    "https://www.ensingerplastics.com/en/shapes/plastic-material-selection",
    "https://www.toraytac.com/products",
    "https://www.solvay.com/en/chemical-categories/composite-materials",
    "https://www.avient.com/products",
]

MAX_PDF_BYTES = 40 * 1024 * 1024     # skip anything larger than 40 MB
MIN_PDF_BYTES = 10 * 1024            # smaller is probably junk, not a paper/datasheet
REQUEST_TIMEOUT = 30                  # seconds
PER_DOMAIN_DELAY = 1.5                # politeness delay between hits to a domain
DATASHEET_LINK_LIMIT = 40             # max PDF links harvested per seed page

# Transient-failure retry policy. OpenAlex and (especially) Semantic Scholar's
# free tier return 429s under load, so a request that fails once is often fine
# moments later. Retry connection errors and 429/5xx, honoring Retry-After.
MAX_RETRIES = 3
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
BACKOFF_BASE = 2.0                    # seconds; exponential per attempt

# URL pre-filter for *academic* candidates (idea from InDesDatabaseTool
# agentic branch, Downloader.is_likely_paper_pdf): skip obvious non-papers
# before spending a download on them.
JUNK_URL_PATTERNS = (
    "syllabus", "application", "template", "newsletter",
    "handbook", "slide", "presentation", "poster", "cv.pdf", "resume",
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """A discovered PDF candidate, pre-download."""
    title: str
    pdf_url: str
    source: str                       # openalex | arxiv | unpaywall | datasheet
    query: str = ""
    doi: str = ""
    year: str = ""
    abstract: str = ""


class CrawlerState:
    """Persisted seen-set so reruns are incremental."""

    def __init__(self, path: Path):
        self.path = path
        self.seen_urls: set[str] = set()
        self.seen_hashes: set[str] = set()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self.seen_urls = set(data.get("seen_urls", []))
                self.seen_hashes = set(data.get("seen_hashes", []))
            except Exception:
                log.warning("Could not parse state file; starting fresh.")

    def save(self) -> None:
        self.path.write_text(json.dumps({
            "seen_urls": sorted(self.seen_urls),
            "seen_hashes": sorted(self.seen_hashes),
        }, indent=1))


class _DomainThrottle:
    def __init__(self, delay: float):
        self.delay = delay
        self._last: dict[str, float] = {}

    def wait(self, url: str) -> None:
        dom = urllib.parse.urlparse(url).netloc
        now = time.monotonic()
        last = self._last.get(dom, 0.0)
        if now - last < self.delay:
            time.sleep(self.delay - (now - last))
        self._last[dom] = time.monotonic()


THROTTLE = _DomainThrottle(PER_DOMAIN_DELAY)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


def _retry_after(r: requests.Response) -> Optional[float]:
    """Seconds from a Retry-After header (delta-seconds form), else None.

    The HTTP-date form is not parsed; callers fall back to exponential
    backoff when this returns None.
    """
    val = r.headers.get("Retry-After")
    if not val:
        return None
    try:
        return max(0.0, float(val))
    except ValueError:
        return None


def http_get(url: str, **kw) -> Optional[requests.Response]:
    """Polite GET with throttling and bounded retry/backoff.

    Retries connection errors and 429/5xx responses up to MAX_RETRIES times,
    honoring a Retry-After header when present. Returns None on a
    non-retryable status or once retries are exhausted.
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            THROTTLE.wait(url)
            r = SESSION.get(url, timeout=REQUEST_TIMEOUT, **kw)
            if r.status_code == 200:
                return r
            if r.status_code in RETRY_STATUS and attempt < MAX_RETRIES:
                delay = _retry_after(r) or BACKOFF_BASE * (2 ** attempt)
                log.debug("GET %s -> %s; retry %d/%d in %.1fs",
                          url, r.status_code, attempt + 1, MAX_RETRIES, delay)
                time.sleep(delay)
                continue
            log.debug("GET %s -> %s", url, r.status_code)
            return None
        except requests.RequestException as e:
            if attempt < MAX_RETRIES:
                delay = BACKOFF_BASE * (2 ** attempt)
                log.debug("GET %s failed: %s; retry %d/%d in %.1fs",
                          url, e, attempt + 1, MAX_RETRIES, delay)
                time.sleep(delay)
                continue
            log.debug("GET %s failed: %s", url, e)
            return None
    return None


def _json(r: Optional[requests.Response]) -> dict:
    """Parse a JSON body defensively; returns {} on None or non-JSON.

    Some endpoints answer 200 with an HTML error/challenge page, which would
    otherwise raise inside .json() and abort the whole query.
    """
    if r is None:
        return {}
    try:
        return r.json() or {}
    except ValueError:
        log.debug("Non-JSON 200 body from %s", r.url)
        return {}


def relevance_score(text: str) -> int:
    t = text.lower()
    return sum(w for k, w in MATERIAL_KEYWORDS.items() if k in t)


def slugify(title: str, maxlen: int = 70) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
    return s[:maxlen] or "untitled"


# ---------------------------------------------------------------------------
# Source 1: OpenAlex
# ---------------------------------------------------------------------------

def search_openalex(query: str, limit: int) -> Iterable[Candidate]:
    url = (
        "https://api.openalex.org/works"
        f"?search={urllib.parse.quote(query)}"
        "&filter=open_access.is_oa:true"
        f"&per-page={min(limit * 3, 50)}"
        f"&mailto={CONTACT_EMAIL}"
    )
    r = http_get(url)
    if r is None:
        log.warning("OpenAlex unreachable for query %r", query)
        return
    n = 0
    for work in _json(r).get("results", []):
        if n >= limit:
            break
        title = work.get("display_name") or ""
        loc = work.get("best_oa_location") or {}
        pdf_url = loc.get("pdf_url")
        doi = (work.get("doi") or "").replace("https://doi.org/", "")
        # Reconstruct abstract from OpenAlex's inverted index for scoring.
        abstract = ""
        inv = work.get("abstract_inverted_index")
        if inv:
            pos: dict[int, str] = {}
            for word, idxs in inv.items():
                for i in idxs:
                    pos[i] = word
            abstract = " ".join(pos[i] for i in sorted(pos))[:2000]
        if not pdf_url and doi:
            pdf_url = unpaywall_pdf_url(doi)
            src = "unpaywall"
        else:
            src = "openalex"
        if not pdf_url:
            continue
        yield Candidate(
            title=title, pdf_url=pdf_url, source=src, query=query,
            doi=doi, year=str(work.get("publication_year") or ""),
            abstract=abstract,
        )
        n += 1


def unpaywall_pdf_url(doi: str) -> Optional[str]:
    r = http_get(f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}"
                 f"?email={CONTACT_EMAIL}")
    loc = _json(r).get("best_oa_location") or {}
    return loc.get("url_for_pdf")


# ---------------------------------------------------------------------------
# Source 2: Semantic Scholar (free tier, no key; ~100 req / 5 min)
# ---------------------------------------------------------------------------

S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,abstract,year,externalIds,openAccessPdf"
# The unauthenticated S2 endpoint shares one tiny global rate bucket and 429s
# almost constantly. A free key (https://www.semanticscholar.org/product/api)
# gives a dedicated quota; set S2_API_KEY in the env or pass --s2-api-key.
S2_API_KEY = os.environ.get("S2_API_KEY") or os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or ""

def search_semantic_scholar(query: str, limit: int) -> Iterable[Candidate]:
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else None
    r = http_get(S2_API, params={"query": query, "limit": min(limit * 3, 50),
                                 "fields": S2_FIELDS}, headers=headers)
    if r is None:
        log.warning("Semantic Scholar unreachable for query %r%s", query,
                    "" if S2_API_KEY else " (no API key — set S2_API_KEY to avoid 429s)")
        return
    n = 0
    for p in (_json(r).get("data") or []):
        if n >= limit:
            break
        ext = p.get("externalIds") or {}
        if "ArXiv" in ext:
            continue                       # arXiv source already covers these
        oa = p.get("openAccessPdf") or {}
        pdf_url = oa.get("url")
        if not pdf_url:
            continue
        yield Candidate(
            title=p.get("title") or "", pdf_url=pdf_url,
            source="semanticscholar", query=query,
            doi=str(ext.get("DOI") or ""), year=str(p.get("year") or ""),
            abstract=(p.get("abstract") or "")[:2000],
        )
        n += 1


# ---------------------------------------------------------------------------
# Source 3: arXiv
# ---------------------------------------------------------------------------

ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}

def search_arxiv(query: str, limit: int) -> Iterable[Candidate]:
    q = urllib.parse.quote(f'all:"{query}"' if " " in query else f"all:{query}")
    url = (f"https://export.arxiv.org/api/query?search_query={q}"
           f"&max_results={limit}&sortBy=relevance")
    r = http_get(url)
    if r is None:
        log.warning("arXiv unreachable for query %r", query)
        return
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return
    for entry in root.findall("a:entry", ARXIV_NS):
        title = re.sub(r"\s+", " ", entry.findtext("a:title", "", ARXIV_NS)).strip()
        abstract = re.sub(r"\s+", " ", entry.findtext("a:summary", "", ARXIV_NS)).strip()
        year = (entry.findtext("a:published", "", ARXIV_NS) or "")[:4]
        pdf_url = ""
        for link in entry.findall("a:link", ARXIV_NS):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href", "")
        if pdf_url:
            yield Candidate(title=title, pdf_url=pdf_url, source="arxiv",
                            query=query, year=year, abstract=abstract)


# ---------------------------------------------------------------------------
# Source 3: manufacturer datasheet pages (best effort)
# ---------------------------------------------------------------------------

class _LinkExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[tuple[str, str]] = []   # (href, anchor text)
        self._href: Optional[str] = None
        self._text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._text = []

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._href:
            self.links.append((self._href, " ".join(self._text).strip()))
            self._href = None


_ROBOTS_CACHE: dict[str, urllib.robotparser.RobotFileParser] = {}

def robots_allows(url: str) -> bool:
    dom = urllib.parse.urlparse(url)
    base = f"{dom.scheme}://{dom.netloc}"
    rp = _ROBOTS_CACHE.get(base)
    if rp is None:
        rp = urllib.robotparser.RobotFileParser()
        try:
            r = http_get(base + "/robots.txt")
            rp.parse(r.text.splitlines() if r is not None else [])
        except Exception:
            rp.parse([])
        _ROBOTS_CACHE[base] = rp
    try:
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        return True


def crawl_datasheet_seed(seed_url: str) -> Iterable[Candidate]:
    """Depth-1 crawl: harvest PDF links from the seed page and from
    same-domain subpages whose anchor text looks datasheet-related."""
    if not robots_allows(seed_url):
        log.info("robots.txt disallows %s — skipping", seed_url)
        return
    r = http_get(seed_url)
    if r is None:
        log.warning("Seed unreachable: %s", seed_url)
        return

    def extract(page_url: str, html: str) -> tuple[list[tuple[str, str]], list[str]]:
        p = _LinkExtractor()
        try:
            p.feed(html)
        except Exception:
            pass
        pdfs, subpages = [], []
        dom = urllib.parse.urlparse(page_url).netloc
        for href, text in p.links:
            full = urllib.parse.urljoin(page_url, href)
            u = urllib.parse.urlparse(full)
            if u.scheme not in ("http", "https"):
                continue
            if u.path.lower().endswith(".pdf"):
                pdfs.append((full, text))
            elif (u.netloc == dom and
                  re.search(r"datasheet|data-sheet|tds|product|material|grade",
                            (href + " " + text).lower())):
                subpages.append(full)
        return pdfs, subpages

    pdfs, subpages = extract(seed_url, r.text)
    for sub in subpages[:8]:                      # depth 1, bounded
        if not robots_allows(sub):
            continue
        sr = http_get(sub)
        if sr is not None:
            more, _ = extract(sub, sr.text)
            pdfs.extend(more)

    seen = set()
    for full, text in pdfs[:DATASHEET_LINK_LIMIT]:
        if full in seen:
            continue
        seen.add(full)
        title = text or Path(urllib.parse.urlparse(full).path).stem
        yield Candidate(title=title, pdf_url=full, source="datasheet",
                        query=seed_url)


# ---------------------------------------------------------------------------
# Download + provenance
# ---------------------------------------------------------------------------

def looks_like_junk(url: str) -> bool:
    u = url.lower()
    return any(p in u for p in JUNK_URL_PATTERNS)


def head_precheck(url: str) -> bool:
    """HEAD request before download (agentic-branch Downloader idea): reject
    blocked / wrong-type / wrong-sized URLs without spending bandwidth on a
    GET. Permissive on errors — many servers mishandle HEAD."""
    try:
        THROTTLE.wait(url)
        h = SESSION.head(url, timeout=10, allow_redirects=True)
        if h.status_code in (401, 403, 404, 410):
            # Paywalled or bot-blocked (MDPI/SAGE/Wiley/Hindawi do this); the
            # GET would return the same block page, so don't bother.
            log.debug("HEAD %s blocked/missing: %s", h.status_code, url)
            return False
        ctype = (h.headers.get("Content-Type") or "").lower()
        if ctype.startswith("text/html"):
            log.debug("HEAD content-type %s, not a PDF: %s", ctype, url)
            return False
        size = int(h.headers.get("Content-Length") or 0)
        if size and not (MIN_PDF_BYTES <= size <= MAX_PDF_BYTES):
            log.debug("HEAD size %d out of range: %s", size, url)
            return False
    except (requests.RequestException, ValueError):
        pass
    return True


def download_pdf(cand: Candidate, pdf_dir: Path, state: CrawlerState) -> Optional[dict]:
    if cand.pdf_url in state.seen_urls:
        return None
    state.seen_urls.add(cand.pdf_url)

    if not head_precheck(cand.pdf_url):
        return None
    r = http_get(cand.pdf_url, stream=True, allow_redirects=True)
    if r is None:
        return None
    chunks, size = [], 0
    try:
        for chunk in r.iter_content(chunk_size=65536):
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_PDF_BYTES:
                log.info("Too large, skipping: %s", cand.pdf_url)
                return None
    except requests.RequestException:
        return None
    data = b"".join(chunks)
    if len(data) < MIN_PDF_BYTES:
        log.debug("Too small (%d B), skipping: %s", len(data), cand.pdf_url)
        return None
    if not data.startswith(b"%PDF"):
        log.debug("Not a PDF (no %%PDF magic): %s", cand.pdf_url)
        return None

    sha = hashlib.sha256(data).hexdigest()
    if sha in state.seen_hashes:
        log.info("Duplicate content, skipping: %s", cand.title[:60])
        return None
    state.seen_hashes.add(sha)

    fname = f"{cand.source}_{slugify(cand.title)}_{sha[:8]}.pdf"
    (pdf_dir / fname).write_bytes(data)
    log.info("Saved [%s] %s  (%.1f kB)", cand.source, fname, size / 1024)
    return {
        "filename": fname, "title": cand.title, "doi": cand.doi,
        "url": cand.pdf_url, "year": cand.year, "source": cand.source,
        "sha256": sha, "query": cand.query, "bytes": size,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

class _LimitReached(Exception):
    """Raised internally once --max-total downloads have been saved."""


def main(argv: Optional[list[str]] = None) -> int:
    global CONTACT_EMAIL, USER_AGENT, S2_API_KEY
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="./crawl_out", help="output directory")
    ap.add_argument("--queries", nargs="*", default=None,
                    help="search queries (default: built-in thermoplastic-composite set)")
    ap.add_argument("--max-per-query", type=int, default=8,
                    help="max PDFs to download per query per source")
    ap.add_argument("--max-total", type=int, default=None, metavar="N",
                    help="stop the whole run after N new PDFs (default: no cap)")
    ap.add_argument("--email", default=None, metavar="ADDR",
                    help="contact email for polite API identification "
                         f"(default: {CONTACT_EMAIL})")
    ap.add_argument("--s2-api-key", default=None, metavar="KEY",
                    help="Semantic Scholar API key (else uses $S2_API_KEY); "
                         "without one S2 is heavily rate-limited (429)")
    ap.add_argument("--skip-datasheets", action="store_true",
                    help="skip the manufacturer datasheet crawl")
    ap.add_argument("--skip-academic", action="store_true",
                    help="skip OpenAlex/arXiv")
    ap.add_argument("--seed-urls", nargs="*", default=None,
                    help="datasheet seed pages (default: built-in list)")
    ap.add_argument("--from-csv", default=None, metavar="FILE",
                    help="also download candidates from a CSV with columns "
                         "title,pdf_url[,source,query,doi,year] "
                         "(e.g. produced by matweb_discovery.py)")
    ap.add_argument("--min-score", type=int, default=MIN_SCORE,
                    help="minimum relevance score for academic results")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s")

    if args.email:
        CONTACT_EMAIL = args.email
        USER_AGENT = f"AIMCompositesCrawler/0.1 (research; mailto:{CONTACT_EMAIL})"
        SESSION.headers.update({"User-Agent": USER_AGENT})
    if args.s2_api_key:
        S2_API_KEY = args.s2_api_key

    out = Path(args.out)
    pdf_dir = out / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output dir: %s  (PDFs -> %s)", out.resolve(), pdf_dir.resolve())
    state = CrawlerState(out / "state.json")
    csv_path = out / "sources.csv"

    fields = ["filename", "title", "doi", "url", "year", "source",
              "sha256", "query", "bytes"]
    new_csv = not csv_path.exists()
    csv_file = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    if new_csv:
        writer.writeheader()

    queries = args.queries or DEFAULT_QUERIES
    seeds = args.seed_urls or DEFAULT_SEED_URLS
    downloaded = 0
    yield_by_source: dict[str, int] = {}    # agentic-branch idea: track which
                                            # sources actually produce PDFs

    def save_row(row: Optional[dict]) -> None:
        nonlocal downloaded
        if row:
            writer.writerow(row)
            csv_file.flush()
            downloaded += 1
            yield_by_source[row["source"]] = yield_by_source.get(row["source"], 0) + 1
            if args.max_total and downloaded >= args.max_total:
                raise _LimitReached

    try:
        if args.from_csv:
            log.info("=== candidates from %s ===", args.from_csv)
            with open(args.from_csv, newline="", encoding="utf-8") as f:
                for rec in csv.DictReader(f):
                    if not rec.get("pdf_url"):
                        continue
                    c = Candidate(
                        title=rec.get("title", ""), pdf_url=rec["pdf_url"],
                        source=rec.get("source", "csv"), query=rec.get("query", ""),
                        doi=rec.get("doi", ""), year=rec.get("year", ""))
                    save_row(download_pdf(c, pdf_dir, state))
            state.save()

        if not args.skip_academic:
            for q in queries:
                log.info("=== query: %s ===", q)
                cands = list(search_openalex(q, args.max_per_query))
                cands += list(search_semantic_scholar(q, args.max_per_query))
                cands += list(search_arxiv(q, args.max_per_query))
                for c in cands:
                    if looks_like_junk(c.pdf_url):
                        log.debug("Junk URL filter: %s", c.pdf_url)
                        continue
                    score = relevance_score(c.title + " " + c.abstract)
                    if score < args.min_score:
                        log.debug("Low relevance (%d): %s", score, c.title[:70])
                        continue
                    save_row(download_pdf(c, pdf_dir, state))
                state.save()

        if not args.skip_datasheets:
            for seed in seeds:
                log.info("=== datasheet seed: %s ===", seed)
                for c in crawl_datasheet_seed(seed):
                    # Datasheet anchors are short; score title only, lower bar.
                    if relevance_score(c.title) < 1:
                        continue
                    save_row(download_pdf(c, pdf_dir, state))
                state.save()
    except _LimitReached:
        log.info("Reached --max-total=%d; stopping.", args.max_total)
    finally:
        state.save()
        csv_file.close()

    if yield_by_source:
        log.info("Yield by source: %s",
                 ", ".join(f"{k}={v}" for k, v in sorted(yield_by_source.items())))
    log.info("Done. %d new PDFs in %s — feed this to batch_ingest.py --input %s",
             downloaded, pdf_dir.resolve(), pdf_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())

