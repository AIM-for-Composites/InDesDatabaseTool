"""
sources/arxiv.py
Fetches papers from ArXiv API. No BeautifulSoup needed —
the API returns structured XML including direct PDF URLs.
"""
import time
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Dict

ARXIV_API  = "http://export.arxiv.org/api/query"
NS         = {"atom": "http://www.w3.org/2005/Atom"}
RATE_LIMIT = 3  # seconds between requests (ArXiv asks for this)

QUERIES = [
    "composite materials mechanical properties",
    "carbon fiber reinforced polymer",
    "fiber reinforced composite",
    "polymer matrix composite",
    "glass fiber epoxy mechanical",
    "nanocomposite polymer properties",
    "hybrid composite flexural strength",
]


def fetch(days_back: int = 1, max_per_query: int = 3) -> List[Dict]:
    since = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
    papers, seen = [], set()
    for query in QUERIES:
        params = {
            "search_query": f"(cat:cond-mat.mtrl-sci OR cat:physics.app-ph) AND all:{query}",
            "start": 0,
            "max_results": max_per_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        try:
            resp = requests.get(ARXIV_API, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  [ArXiv] Request failed for '{query}': {e}")
            time.sleep(2)
            continue

        for paper in _parse(resp.text):
            if paper["id"] in seen:
                continue
            seen.add(paper["id"])
            #if paper.get("published", "") >= since:
            if len(papers) < max_per_query:
                papers.append(paper)

        time.sleep(RATE_LIMIT)

    print(f"  [ArXiv] {len(papers)} papers found")
    return papers


def _parse(xml_text: str) -> List[Dict]:
    papers = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return papers

    for entry in root.findall("atom:entry", NS):
        try:
            arxiv_id = entry.find("atom:id", NS).text.strip().split("/abs/")[-1]
            title    = entry.find("atom:title", NS).text.strip().replace("\n", " ")
            abstract = entry.find("atom:summary", NS).text.strip().replace("\n", " ")
            published = entry.find("atom:published", NS).text.strip()[:10]
            authors  = [a.find("atom:name", NS).text.strip()
                        for a in entry.findall("atom:author", NS)]
            pdf_url  = next(
                (l.attrib["href"] for l in entry.findall("atom:link", NS)
                 if l.attrib.get("type") == "application/pdf"), ""
            )
            papers.append({
                "id": f"arxiv_{arxiv_id}",
                "source": "arxiv",
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": pdf_url,
                "published": published,
                "is_relevant": 0,
            })
        except (AttributeError, StopIteration):
            continue
    return papers