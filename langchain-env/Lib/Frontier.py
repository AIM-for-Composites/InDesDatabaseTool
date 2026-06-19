"""
frontier.py
Priority queue over sources.json.
Best-First Search — always processes the highest-scored URL next.
Higher priority number = crawled first.
"""
import json
import heapq
from pathlib import Path
from typing import List, Dict

SOURCES_PATH = Path(__file__).parent / "sources.json"

# Keywords that boost a URL's priority score
RELEVANCE_KEYWORDS = [
    "composite", "fiber", "fibre", "polymer", "epoxy",
    "tensile", "flexural", "nanocomposite", "laminate",
    "carbon fiber", "glass fiber", "mechanical properties"
]


def load_frontier() -> List[Dict]:
    """Load sources.json and return as list of source dicts."""
    if not SOURCES_PATH.exists():
        return []
    with open(SOURCES_PATH) as f:
        return json.load(f)


def save_frontier(sources: List[Dict]):
    """Write sources list back to sources.json."""
    with open(SOURCES_PATH, "w") as f:
        json.dump(sources, f, indent=2)


def score_url(url: str, domain: str = "") -> int:
    """
    Score a URL for crawl priority.
    Higher = more important.
    Used by Best-First Search to decide what to crawl next.
    """
    score = 5  # base score

    text = (url + " " + domain).lower()

    # Boost for keyword relevance in URL
    for kw in RELEVANCE_KEYWORDS:
        if kw in text:
            score += 2

    # Boost for academic domains
    academic_tlds = [".edu", ".ac.uk", ".ac.in", ".edu.au", "researchgate",
                     "springer", "elsevier", "wiley", "mdpi", "arxiv",
                     "semanticscholar", "sciencedirect"]
    for tld in academic_tlds:
        if tld in text:
            score += 3
            break

    # Penalty for deep paths (likely nav pages, not content)
    depth = url.count("/") - 2  # subtract protocol slashes
    score -= max(0, depth - 3)

    return max(1, score)  # never below 1


def get_next_sources(n: int = 10) -> List[Dict]:
    """
    Return the top-n sources to crawl next, ordered by priority.
    Implements Best-First Search via heapq.
    Uses negative priority because heapq is a min-heap.
    """
    sources = load_frontier()
    if not sources:
        return []

    # Build a max-heap using negative priority
    # Use index as tiebreaker to avoid comparing dict objects
    heap = []
    for idx, source in enumerate(sources):
        priority = source.get("priority", score_url(source["url"]))
        heapq.heappush(heap, (-priority, idx, source))

    result = []
    for _ in range(min(n, len(heap))):
        _, _, source = heapq.heappop(heap)
        result.append(source)

    return result


def append_source(url: str, discovered_by: str = "web_search") -> bool:
    """
    Add a new source URL to sources.json if not already present.
    Returns True if added, False if duplicate.
    """
    sources = load_frontier()
    existing_urls = {s["url"] for s in sources}

    if url in existing_urls:
        return False

    from urllib.parse import urlparse
    domain = urlparse(url).netloc

    new_source = {
        "url": url,
        "type": "scrape",
        "domain": domain,
        "added": __import__("datetime").datetime.utcnow().strftime("%Y-%m-%d"),
        "discovered_by": discovered_by,
        "priority": score_url(url, domain),
        "yield_score": None
    }
    sources.append(new_source)
    save_frontier(sources)
    print(f"[Frontier] Added new source: {url} (priority={new_source['priority']})")
    return True


def update_yield_score(url: str, papers_saved: int):
    """
    After a crawl run, update the yield_score for a source.
    This is what your RL scheduler will read later to prioritise sources.
    """
    sources = load_frontier()
    for source in sources:
        if source["url"] == url:
            # Running average — if no previous score, use current
            prev = source.get("yield_score") or papers_saved
            source["yield_score"] = round((prev + papers_saved) / 2, 2)
            break
    save_frontier(sources)