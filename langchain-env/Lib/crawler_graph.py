"""
crawler_graph.py
LangGraph pipeline for the materials crawler agent.

Nodes (in order):
  1. load_frontier   — reads sources.json, builds priority queue
  2. fetch_papers    — calls ArXiv + S2 APIs
  3. filter_relevant — keyword fast-path + Claude fallback
  4. download_pdfs   — pre-filter + download to ./downloads/
  5. save_to_db      — dedup + insert new papers
  6. extract_pdfs    — extract properties from PDFs
  7. log_run         — write crawl log entry

State flows through all nodes. Each node returns only what it changes.
"""

import time
from datetime import datetime
from typing import List, Dict, TypedDict, Annotated
import operator
import pandas as pd

from langgraph.graph import StateGraph, END

from Frontier    import get_next_sources, update_yield_score
from Arxiv            import fetch as fetch_arxiv
from Semantic_Scholar import fetch as fetch_s2
from Relevance   import filter_papers
from Downloader  import download_batch
from Database    import init_db, is_seen_paper, save_paper, save_crawl_log
from Extraction  import extract_all_downloaded_pdfs


# ── Shared state flowing through all nodes ───────────────────

class CrawlerState(TypedDict):
    run_date:        str
    days_back:       int
    sources:         List[Dict]                          # from frontier
    papers_fetched:  Annotated[List[Dict], operator.add] # raw from APIs
    papers_relevant: Annotated[List[Dict], operator.add] # passed relevance
    papers_saved:    Annotated[List[Dict], operator.add] # new + in DB
    extracted_df:    object                              # pandas DataFrame from extraction
    errors:          Annotated[List[str],  operator.add]
    run_start:       float


# ── Node 1: Load frontier ────────────────────────────────────

def load_frontier(state: CrawlerState) -> dict:
    """
    Read sources.json and get top sources ordered by priority.
    Best-First Search — highest priority URLs processed first.
    """
    print("\n[Node 1] Loading frontier...")
    sources = get_next_sources(n=20)  # top 20 by priority score
    print(f"  {len(sources)} sources queued (Best-First order)")
    return {"sources": sources}


# ── Node 2: Fetch papers ─────────────────────────────────────

def fetch_papers(state: CrawlerState) -> dict:
    """
    Fetch from ArXiv and Semantic Scholar APIs.
    Each source in the frontier is processed; API sources get full fetch.
    """
    print(f"\n[Node 2] Fetching papers (days_back={state['days_back']})...")
    all_papers = []
    errors = []

    # ArXiv
    try:
        papers = fetch_arxiv(days_back=state["days_back"])
        all_papers.extend(papers)
    except Exception as e:
        errors.append(f"ArXiv: {e}")
        print(f"  [ERROR] ArXiv: {e}")

    # Semantic Scholar
    # try:
    #     papers = fetch_s2(days_back=state["days_back"])
    #     all_papers.extend(papers)
    # except Exception as e:
    #     errors.append(f"S2: {e}")
    #     print(f"  [ERROR] S2: {e}")

    print(f"  Total fetched: {len(all_papers)}")
    return {"papers_fetched": all_papers, "errors": errors}


# ── Node 3: Filter relevance ─────────────────────────────────

def filter_relevance(state: CrawlerState) -> dict:
    """
    Two-stage relevance filter.
    Stage 1: keywords (free). Stage 2: Claude Haiku (cheap, uncertain only).
    """
    papers = state["papers_fetched"]
    print(f"\n[Node 3] Filtering relevance ({len(papers)} papers)...")

    if not papers:
        return {"papers_relevant": []}

    checked = filter_papers(papers)
    relevant = [p for p in checked if p.get("is_relevant") == 1]
    
    return {"papers_relevant": relevant}


# ── Node 4: Download PDFs ────────────────────────────────────

def download_pdfs(state: CrawlerState) -> dict:
    """
    Download PDFs for relevant papers.
    Pre-filters by URL structure + HEAD request before full download.
    Respects 1.5s rate limit between downloads.
    Depth limit: only follows direct PDF URLs (depth=1), not crawling further.
    """
    papers = state["papers_relevant"]
    print(f"\n[Node 4] Downloading PDFs ({len(papers)} relevant papers)...")

    if not papers:
        return {"papers_relevant": []}

    updated = download_batch(papers)
    return {"papers_relevant": updated}


# ── Node 5: Save to DB ───────────────────────────────────────

def save_to_db(state: CrawlerState) -> dict:
    """
    Dedup + save relevant papers to SQLite.
    Skips any paper ID already in the DB.
    """
    papers = state["papers_relevant"]
    print(f"\n[Node 5] Saving to DB ({len(papers)} papers)...")

    saved = []
    dupes = 0
    errors = []

    for paper in papers:
        try:
            if is_seen_paper(paper["id"]):
                dupes += 1
                continue
            if save_paper(paper):
                saved.append(paper)
        except Exception as e:
            errors.append(f"Save error: {e}")

    print(f"  Saved: {len(saved)} | Dupes skipped: {dupes}")
    return {"papers_saved": saved, "errors": errors}


# ── Node 6: Extract PDFs ─────────────────────────────────────

def extract_pdfs(state: CrawlerState) -> dict:
    """
    Extract material properties from downloaded PDFs using Gemini RAG.
    One-by-one with rate limiting to respect API limits.
    """
    print(f"\n[Node 6] Extracting properties from PDFs...")
    
    try:
        extracted_df = extract_all_downloaded_pdfs()
        print(f"  Extracted {len(extracted_df)} properties")
        return {"extracted_df": extracted_df}
    except Exception as e:
        print(f"  [ERROR] Extraction failed: {e}")
        return {"extracted_df": pd.DataFrame(), "errors": [f"Extraction: {e}"]}


# ── Node 7: Log the run ──────────────────────────────────────

def log_run(state: CrawlerState) -> dict:
    """
    Write summary to crawl_log table.
    Update yield_score for each source (feeds RL scheduler later).
    """
    duration = time.time() - state.get("run_start", time.time())
    run_date = state["run_date"]

    # Log per-source breakdown
    for source_name in ["arxiv", "semantic_scholar"]:
        found    = sum(1 for p in state["papers_fetched"] if p.get("source") == source_name)
        relevant = sum(1 for p in state["papers_relevant"] if p.get("source") == source_name)
        saved    = sum(1 for p in state["papers_saved"] if p.get("source") == source_name)
        pdfs     = sum(1 for p in state["papers_saved"]
                       if p.get("source") == source_name and p.get("pdf_path"))

        save_crawl_log({
            "run_date": run_date, "source": source_name,
            "found": found, "relevant": relevant,
            "saved": saved, "pdfs_downloaded": pdfs,
            "errors": "; ".join(state.get("errors", [])),
            "duration_s": round(duration, 2)
        })

        # Update yield_score in sources.json (RL scheduler reads this later)
        # update_yield_score(
        #     "https://arxiv.org" if source_name == "arxiv"
        #     else "https://api.semanticscholar.org",
        #     saved
        # )

    total = len(state["papers_saved"])
    pdfs  = sum(1 for p in state["papers_saved"] if p.get("pdf_path"))

    print(f"\n{'='*45}")
    print(f"  Run date : {run_date}")
    print(f"  Fetched  : {len(state['papers_fetched'])}")
    print(f"  Relevant : {len(state['papers_relevant'])}")
    print(f"  Saved    : {total}")
    print(f"  PDFs     : {pdfs}")
    print(f"  Duration : {duration:.1f}s")
    if state.get("errors"):
        print(f"  Errors   : {state['errors']}")
    print(f"{'='*45}\n")

    return {}


# ── Build + run ──────────────────────────────────────────────

def build_graph():
    graph = StateGraph(CrawlerState)

    graph.add_node("load_frontier",   load_frontier)
    graph.add_node("fetch_papers",    fetch_papers)
    graph.add_node("filter_relevance",filter_relevance)
    graph.add_node("download_pdfs",   download_pdfs)
    graph.add_node("save_to_db",      save_to_db)
    graph.add_node("extract_pdfs",    extract_pdfs)
    graph.add_node("log_run",         log_run)

    graph.set_entry_point("load_frontier")
    graph.add_edge("load_frontier",    "fetch_papers")
    graph.add_edge("fetch_papers",     "filter_relevance")
    graph.add_edge("filter_relevance", "download_pdfs")
    graph.add_edge("download_pdfs",    "save_to_db")
    graph.add_edge("save_to_db",       "extract_pdfs")
    graph.add_edge("extract_pdfs",     "log_run")
    graph.add_edge("log_run",          END)

    return graph.compile()

run_count = 0
MAX_RUNS = 5

def run_crawler(days_back: int = 1):
    global run_count
    run_count += 1
    if run_count >= MAX_RUNS:
        import Scheduler
        Scheduler.scheduler.shutdown(wait=False)
        return
    init_db()
    graph = build_graph()
    result = graph.invoke({
        "run_date":        datetime.now().strftime("%Y-%m-%d"),
        "days_back":       days_back,
        "sources":         [],
        "papers_fetched":  [],
        "papers_relevant": [],
        "papers_saved":    [],
        "extracted_df":    pd.DataFrame(),
        "errors":          [],
        "run_start":       time.time(),
    })
    
    # Display final extracted data
    if not result["extracted_df"].empty:
        print(f"\n{'='*80}")
        print("  FINAL EXTRACTED PROPERTIES")
        print(f"{'='*80}")
        print(result["extracted_df"].to_string())
        print(f"{'='*80}\n")


if __name__ == "__main__":
    run_crawler(days_back=7)  # Search last 7 days for initial test