"""
sources/semantic_scholar.py
Fetches papers from Semantic Scholar API.
Free tier: 100 req / 5 min. We stay well within that.
Skips papers that have an ArXiv ID (already fetched by arxiv.py).
"""
import time
import requests
from datetime import datetime, timedelta
from typing import List, Dict

S2_API     = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS     = "paperId,title,abstract,authors,year,publicationDate,externalIds,openAccessPdf"
RATE_LIMIT = 3  # seconds

QUERIES = [
    "composite materials mechanical properties",
    "fiber reinforced polymer composite",
    "carbon fiber composite tensile",
    "glass fiber epoxy mechanical",
    "polymer nanocomposite properties",
    "hybrid composite flexural strength",
]


def fetch(days_back: int = 1, max_per_query: int = 50) -> List[Dict]:
    since_year = (datetime.now() - timedelta(days=days_back)).year
    papers, seen = [], set()
    retry_count = 0
    max_retries = 3

    for query in QUERIES:
        for attempt in range(max_retries):
            try:
                resp = requests.get(
                    S2_API,
                    params={"query": query, "limit": max_per_query,
                            "fields": FIELDS, "year": f"{since_year}-"},
                    headers={"User-Agent": "MaterialsDBCrawler/1.0"},
                    timeout=30,
                )
                if resp.status_code == 429:
                    # Rate limited — exponential backoff
                    wait_time = 10 * (2 ** attempt)  # 10s, 20s, 40s
                    print(f"  [S2] Rate limited (429). Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                
                resp.raise_for_status()
                data = resp.json()
                break  # Success, exit retry loop
                
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"  [S2] Attempt {attempt+1} failed for '{query}': {e}. Retrying...")
                    time.sleep(5 * (attempt + 1))
                else:
                    print(f"  [S2] Request failed for '{query}' after {max_retries} attempts: {e}")
                continue

        else:
            # All retries exhausted
            print(f"  [S2] Skipping '{query}' — max retries exceeded")
            continue

        for paper in _parse(data.get("data", []), days_back):
            if paper["id"] not in seen:
                seen.add(paper["id"])
                papers.append(paper)

        time.sleep(RATE_LIMIT)

    print(f"  [S2] {len(papers)} papers found")
    return papers


def _parse(raw: list, days_back: int) -> List[Dict]:
    cutoff = (datetime.now() - timedelta(days=days_back)).date()
    papers = []

    for p in raw:
        try:
            s2_id = p.get("paperId", "")
            if not s2_id:
                continue

            # Skip if also on ArXiv — already handled
            if p.get("externalIds", {}).get("ArXiv"):
                continue

            pub_str = p.get("publicationDate", "")
            if pub_str:
                from datetime import date
                pub_date = date.fromisoformat(pub_str)
                if pub_date < cutoff:
                    continue

            pdf_url = (p.get("openAccessPdf") or {}).get("url", "")
            authors = [a.get("name", "") for a in p.get("authors", [])]

            papers.append({
                "id": f"s2_{s2_id}",
                "source": "semantic_scholar",
                "title": (p.get("title") or "").strip(),
                "abstract": (p.get("abstract") or "").strip(),
                "authors": authors,
                "url": f"https://www.semanticscholar.org/paper/{s2_id}",
                "pdf_url": pdf_url,
                "published": pub_str,
                "is_relevant": 0,
            })
            
            if not pdf_url:
                print(f"  [S2] No open-access PDF for: {(p.get('title') or '')[:50]}")
        except Exception as e:
            print(f"  [S2] Parse error: {e}")
            continue

    return papers