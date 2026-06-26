"""
discovery.py
Weekly job — finds new academic sources via DuckDuckGo search.
Runs separately from the daily crawler.
New sources get appended to sources.json for the next daily run.
"""
from ddgs import DDGS
from Frontier import append_source, score_url
from urllib.parse import urlparse

SEARCH_QUERIES = [
    "composites fiber polymer mechanical properties open access journal",
    "fiber reinforced composite research papers free pdf",
    "polymer matrix composite materials science journal",
    "carbon fiber composite tensile strength open access",
    "nanocomposite materials properties research",
]

# Minimum score a discovered URL must have to be added
MIN_SCORE = 6

# Domains to always skip
BLOCKLIST = [
    "youtube.com", "twitter.com", "linkedin.com", "facebook.com",
    "reddit.com", "quora.com", "wikipedia.org", "amazon.com",
    "google.com", "bing.com"
]


def run_discovery(max_results_per_query: int = 10):
    """Search DuckDuckGo and add high-scoring academic sources to frontier."""
    print("\n[Discovery] Starting weekly source discovery...")
    added = 0

    with DDGS() as ddgs:
        for query in SEARCH_QUERIES:
            print(f"  Searching: '{query}'")
            try:
                results = list(ddgs.text(query, max_results=max_results_per_query))
            except Exception as e:
                print(f"  [Discovery] Search failed: {e}")
                continue

            for r in results:
                url = r.get("href", "")
                if not url:
                    continue

                domain = urlparse(url).netloc

                # Skip blocklisted domains
                if any(b in domain for b in BLOCKLIST):
                    continue

                # Score it
                score = score_url(url, domain)
                if score < MIN_SCORE:
                    continue

                # Try to add to frontier
                if append_source(url, discovered_by="duckduckgo_weekly"):
                    added += 1

    print(f"[Discovery] Done — {added} new sources added to sources.json\n")


if __name__ == "__main__":
    run_discovery()