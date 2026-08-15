r"""
webscraper_demo.py — a tiny demo of the datasheet web-scraper in pdf_crawler.py.

This isolates the "walk into a page and grab the PDFs" step (`crawl_datasheet_seed`
+ `_LinkExtractor`) so you can see it work without running the whole crawler.

Two modes:

  # 1. OFFLINE (default) — no network. Parses a bundled sample datasheet page
  #    with the REAL _LinkExtractor + the same PDF/subpage filter the scraper
  #    uses, and prints what it would harvest.
  python webscraper_demo.py

  # 2. LIVE — runs the actual crawl_datasheet_seed() against a real vendor page
  #    (robots-respecting, GET only). Lists the PDF candidates; does NOT download.
  python webscraper_demo.py --url https://www.toraytac.com/products
  python webscraper_demo.py --url https://www.victrex.com/en/datasheets --download 2
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.parse
from pathlib import Path

# Import the real scraper pieces from the crawler.
from pdf_crawler import _LinkExtractor, crawl_datasheet_seed

# A stand-in vendor "datasheets" listing page. Mix of: direct PDF links, a
# datasheet-ish subpage link, an off-domain PDF, and junk — so you can watch the
# filter keep the right ones.
SAMPLE_HTML = """
<html><body>
  <h1>Acme Composites — Product Datasheets</h1>
  <ul>
    <li><a href="/downloads/acme_peek_cf_tape_tds.pdf">PEEK / Carbon UD Tape — TDS</a></li>
    <li><a href="/downloads/acme_pps_gf_laminate.pdf">PPS / Glass Laminate datasheet</a></li>
    <li><a href="files/Acme_PEI_grade_overview.PDF">PEI grade overview</a></li>
    <li><a href="/products/thermoplastics/">All thermoplastic grades</a></li>
    <li><a href="/about/careers/">Careers</a></li>
    <li><a href="https://cdn.othersite.com/random_brochure.pdf">Off-domain brochure</a></li>
    <li><a href="/media/acme-presentation-slides.pdf">Company presentation</a></li>
  </ul>
</body></html>
"""
SAMPLE_BASE = "https://www.acme-composites.example/datasheets"

# Same regex the scraper uses to decide a subpage is worth following (pdf_crawler.py:462).
_SUBPAGE_RE = re.compile(r"datasheet|data-sheet|tds|product|material|grade")


def _extract(page_url: str, html: str):
    """Mirror of the scraper's inner extract(): split links into PDFs vs subpages."""
    p = _LinkExtractor()
    p.feed(html)
    dom = urllib.parse.urlparse(page_url).netloc
    pdfs, subpages = [], []
    for href, text in p.links:
        full = urllib.parse.urljoin(page_url, href)
        u = urllib.parse.urlparse(full)
        if u.scheme not in ("http", "https"):
            continue
        if u.path.lower().endswith(".pdf"):
            pdfs.append((full, text, u.netloc == dom))
        elif u.netloc == dom and _SUBPAGE_RE.search((href + " " + text).lower()):
            subpages.append(full)
    return pdfs, subpages


def offline_demo() -> None:
    print("=" * 72)
    print("OFFLINE demo — parsing a bundled sample page with the real _LinkExtractor")
    print("=" * 72)
    print(f"seed page: {SAMPLE_BASE}\n")

    all_links = _LinkExtractor()
    all_links.feed(SAMPLE_HTML)
    print(f"Step 1 — found {len(all_links.links)} <a> links on the page:")
    for href, text in all_links.links:
        print(f"    {href:<45} {text!r}")

    pdfs, subpages = _extract(SAMPLE_BASE, SAMPLE_HTML)
    print(f"\nStep 2 — filtered to PDF links (same-domain harvested, off-domain noted):")
    for full, text, same in pdfs:
        tag = "harvest" if same else "off-domain"
        print(f"    [{tag:<10}] {text or '(no title)':<32} {full}")

    print(f"\nStep 3 — datasheet-ish subpages the scraper would follow (depth-1, max 8):")
    for s in subpages:
        print(f"    -> {s}")

    kept = [(f, t) for f, t, same in pdfs if same]
    print(f"\nResult: {len(kept)} PDF candidate(s) it would queue for download "
          f"(off-domain + non-datasheet are dropped).")
    print("Each becomes a Candidate(title, pdf_url, source='datasheet') -> download_pdf()")
    print("which then enforces the %PDF- magic-byte + PyMuPDF validation before saving.")


def live_demo(url: str, download: int) -> None:
    print("=" * 72)
    print("LIVE demo — running the real crawl_datasheet_seed() (robots-respecting)")
    print("=" * 72)
    print(f"seed: {url}\n")
    cands = list(crawl_datasheet_seed(url))
    if not cands:
        print("No PDF candidates found (page unreachable, robots-disallowed, or no PDFs).")
        return
    print(f"Harvested {len(cands)} PDF candidate(s):\n")
    for i, c in enumerate(cands, 1):
        print(f"  {i:>2}. {c.title[:60]:<60} {c.pdf_url}")

    if download:
        import logging
        logging.basicConfig(level=logging.INFO)
        from pdf_crawler import CrawlerState, download_pdf
        out = Path("webscraper_demo_out")
        out.mkdir(exist_ok=True)
        state = CrawlerState(out / "state.json")
        print(f"\nDownloading first {download} (with %PDF- validation) into {out}/ ...")
        for c in cands[:download]:
            rec = download_pdf(c, out, state)
            print(f"    {'saved ' + rec['filename'] if rec else 'rejected/failed'} <- {c.pdf_url[:60]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", help="run live against this seed page instead of the offline sample")
    ap.add_argument("--download", type=int, default=0,
                    help="(live only) actually download the first N harvested PDFs")
    args = ap.parse_args()
    try:  # show unicode (em-dashes, etc.) cleanly even on a cp1252 console
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if args.url:
        live_demo(args.url, args.download)
    else:
        offline_demo()
    return 0


if __name__ == "__main__":
    sys.exit(main())
