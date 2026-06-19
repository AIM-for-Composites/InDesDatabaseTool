"""
database.py
SQLite storage for crawled papers + URL dedup + crawl logs.
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "materials.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        -- Papers we've found and saved
        CREATE TABLE IF NOT EXISTS papers (
            id          TEXT PRIMARY KEY,
            source      TEXT NOT NULL,
            title       TEXT NOT NULL,
            abstract    TEXT,
            authors     TEXT,
            url         TEXT,
            pdf_url     TEXT,
            pdf_path    TEXT,
            published   TEXT,
            is_relevant INTEGER DEFAULT 0,
            crawled_at  TEXT NOT NULL
        );

        -- Every URL we've ever visited or queued (dedup table)
        CREATE TABLE IF NOT EXISTS seen_urls (
            url_hash    TEXT PRIMARY KEY,
            url         TEXT NOT NULL,
            seen_at     TEXT NOT NULL
        );

        -- Log of each crawler run
        CREATE TABLE IF NOT EXISTS crawl_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date    TEXT NOT NULL,
            source      TEXT,
            found       INTEGER DEFAULT 0,
            relevant    INTEGER DEFAULT 0,
            saved       INTEGER DEFAULT 0,
            pdfs_downloaded INTEGER DEFAULT 0,
            errors      TEXT,
            duration_s  REAL
        );
    """)
    conn.commit()
    conn.close()
    print(f"[DB] Ready at {DB_PATH}")


# ── URL dedup ────────────────────────────────────────────────

def url_hash(url: str) -> str:
    import hashlib
    return hashlib.md5(url.strip().lower().encode()).hexdigest()


def is_seen_url(url: str) -> bool:
    conn = get_conn()
    row = conn.execute(
        "SELECT 1 FROM seen_urls WHERE url_hash = ?", (url_hash(url),)
    ).fetchone()
    conn.close()
    return row is not None


def mark_url_seen(url: str):
    conn = get_conn()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO seen_urls (url_hash, url, seen_at) VALUES (?,?,?)",
            (url_hash(url), url, datetime.utcnow().isoformat())
        )
        conn.commit()
    finally:
        conn.close()


# ── Papers ───────────────────────────────────────────────────

def is_seen_paper(paper_id: str) -> bool:
    conn = get_conn()
    row = conn.execute(
        "SELECT 1 FROM papers WHERE id = ?", (paper_id,)
    ).fetchone()
    conn.close()
    return row is not None


def save_paper(paper: dict) -> bool:
    """Returns True if inserted (new), False if duplicate."""
    if is_seen_paper(paper["id"]):
        return False
    conn = get_conn()
    try:
        conn.execute("""
            INSERT INTO papers
              (id, source, title, abstract, authors, url, pdf_url,
               pdf_path, published, is_relevant, crawled_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            paper["id"],
            paper["source"],
            paper["title"],
            paper.get("abstract", ""),
            json.dumps(paper.get("authors", [])),
            paper.get("url", ""),
            paper.get("pdf_url", ""),
            paper.get("pdf_path", ""),
            paper.get("published", ""),
            int(paper.get("is_relevant", 0)),
            datetime.utcnow().isoformat()
        ))
        conn.commit()
        return True
    finally:
        conn.close()


# ── Crawl log ────────────────────────────────────────────────

def save_crawl_log(log: dict):
    conn = get_conn()
    try:
        conn.execute("""
            INSERT INTO crawl_log
              (run_date, source, found, relevant, saved,
               pdfs_downloaded, errors, duration_s)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            log["run_date"], log.get("source", ""),
            log.get("found", 0), log.get("relevant", 0),
            log.get("saved", 0), log.get("pdfs_downloaded", 0),
            log.get("errors", ""), log.get("duration_s", 0.0)
        ))
        conn.commit()
    finally:
        conn.close()