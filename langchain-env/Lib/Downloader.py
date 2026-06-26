"""
downloader.py
Downloads PDFs for relevant papers.
Pre-filters before downloading to avoid junk files.
Saves to ./downloads/ (relative path — portable across machines).
"""
import re
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional

# Relative path — works on any machine regardless of OS
DOWNLOAD_DIR = Path(__file__).parent / "downloads"
DOWNLOAD_DIR.mkdir(exist_ok=True)

MIN_PDF_BYTES = 50_000     # 50 KB  — smaller = probably not a real paper
MAX_PDF_BYTES = 30_000_000 # 30 MB  — larger = probably a book or dataset dump
RATE_LIMIT    = 1.5        # seconds between downloads (politeness)


def is_likely_paper_pdf(url: str) -> bool:
    """
    Pre-filter: check URL structure before downloading.
    Avoids fetching lab manuals, admin forms, slide decks.
    """
    url_lower = url.lower()

    # Must end in .pdf or contain /pdf/ in path
    if not (".pdf" in url_lower or "/pdf/" in url_lower):
        return False

    # Skip obvious non-papers
    skip_patterns = [
        "syllabus", "form", "application", "template",
        "brochure", "newsletter", "manual", "handbook",
        "slide", "presentation", "poster"
    ]
    if any(p in url_lower for p in skip_patterns):
        return False

    # Bonus: URL contains year (e.g. /2023/, /2024/) — good sign
    return True


def download_pdf(paper: Dict) -> Optional[str]:
    """
    Download a paper's PDF. Returns local file path string, or None on failure.
    Pre-checks URL structure, file size, and content-type.
    """
    pdf_url = paper.get("pdf_url", "")
    if not pdf_url:
        return None

    if not is_likely_paper_pdf(pdf_url):
        print(f"  [Download] Skipped (URL filter): {pdf_url[:60]}")
        return None

    # Build a clean filename from paper ID
    safe_id = re.sub(r"[^\w\-]", "_", paper["id"])
    filepath = DOWNLOAD_DIR / f"{safe_id}.pdf"

    # Skip if already downloaded
    if filepath.exists():
        return str(filepath)

    try:
        # HEAD request first — check size and content-type without downloading
        head = requests.head(pdf_url, timeout=10, allow_redirects=True)
        content_type = head.headers.get("Content-Type", "")
        content_length = int(head.headers.get("Content-Length", 0))

        if "pdf" not in content_type.lower() and content_length > 0:
            print(f"  [Download] Skipped (not PDF content-type): {pdf_url[:60]}")
            return None

        if content_length > MAX_PDF_BYTES:
            print(f"  [Download] Skipped (too large {content_length//1e6:.1f}MB): {pdf_url[:60]}")
            return None

        # Now actually download
        resp = requests.get(pdf_url, timeout=60, stream=True)
        resp.raise_for_status()

        data = b""
        for chunk in resp.iter_content(chunk_size=8192):
            data += chunk
            if len(data) > MAX_PDF_BYTES:
                print(f"  [Download] Aborted (exceeded size limit): {pdf_url[:60]}")
                return None

        if len(data) < MIN_PDF_BYTES:
            print(f"  [Download] Skipped (too small {len(data)} bytes): {pdf_url[:60]}")
            return None

        filepath.write_bytes(data)
        print(f"  [Download] Saved: {filepath.name} ({len(data)//1024} KB)")
        time.sleep(RATE_LIMIT)
        return str(filepath)

    except requests.RequestException as e:
        print(f"  [Download] Failed: {e}")
        return None


def download_batch(papers: List[Dict]) -> List[Dict]:
    """
    Download PDFs for a list of papers. Updates pdf_path on each.
    
    Note: All input papers should already be marked as relevant.
    Papers that fail to download are still returned (without pdf_path).
    """
    downloaded = 0
    failed = 0
    
    for paper in papers:
        path = download_pdf(paper)
        if path:
            paper["pdf_path"] = path
            downloaded += 1
        else:
            failed += 1

    print(f"  [Download] {downloaded}/{len(papers)} PDFs saved | {failed} failed to download")
    return papers