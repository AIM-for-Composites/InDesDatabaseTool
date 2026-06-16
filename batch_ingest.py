r"""
batch_ingest.py — preliminary autonomous batch-ingestion prototype for the
AIM Composites Materials Database.

This script takes a folder of PDFs as input and exercises the extraction +
validation + insertion stages of the target autonomous-ingestion architecture
described in Section 6 of the project report:

  PDFs  ->  Gemini extraction  ->  grounding+validation  ->  dedup  ->  SQLite
                                                         \-> review_queue.csv (a view)

It intentionally does NOT do source discovery (component 1) and does NOT run
the plot-extraction / image-mapping pipeline. It is a minimal closed loop
sufficient to characterize the cost, throughput, and failure modes of the
extraction-plus-validation stage on a fixed corpus.

As of the extraction-hardening phase, all extraction logic (prompt, schema,
grounding, unit normalization, classification, dedup keys) lives in
``extraction.py`` — this file is just the driver + SQLite mirror. Nothing is
silently dropped: flagged rows are inserted with a ``status`` and
``flag_reason``, and ``review_queue.csv`` is a ``SELECT ... WHERE status != 'ok'``
export rather than a discard pile.

Usage:
    export GEMINI_API_KEY=...
    python batch_ingest.py --input ./pdfs --db ./materials_mirror.sqlite \
        --review review_queue.csv --report run_report.json

    # one-off, non-destructive column migration of an existing DB:
    python batch_ingest.py --migrate --db ./materials_mirror.sqlite

Author: Mathias Heider, ME8930 course project, May 2026.
Extraction prompt and schema now centralized in extraction.py (was adapted from
the live Streamlit app's page_files/categorized/Backend/upload_backend.py,
co-developed with Abhijit on the AIM Composites HF Space).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import sqlite3
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

import requests

import extraction
from extraction import Extraction, PropertyRow, extract_from_pdf, to_rows, verify_against_text
from migrate import EXTRA_COLUMNS, ensure_columns


# ---------------------------------------------------------------------------
# SQLite mirror of the Postgres schema
# ---------------------------------------------------------------------------

# Base (legacy) columns, kept so the CSV export and page1.py keep working. The
# hardening-phase columns are added on top by migrate.ensure_columns().
SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS Polymers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    material_name TEXT, material_abbreviation TEXT, section TEXT,
    property_name TEXT, value TEXT, unit TEXT, english TEXT,
    test_condition TEXT, comments TEXT
);
CREATE TABLE IF NOT EXISTS Fibers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    material_name TEXT, material_abbreviation TEXT, section TEXT,
    property_name TEXT, value TEXT, unit TEXT, english TEXT,
    test_condition TEXT, comments TEXT
);
CREATE TABLE IF NOT EXISTS Composites_materials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    material_name TEXT, material_abbreviation TEXT, section TEXT,
    property_name TEXT, value TEXT, unit TEXT, english TEXT,
    test_condition TEXT, comments TEXT
);
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pdf_filename TEXT UNIQUE,
    pdf_sha1 TEXT,
    ingested_at TEXT,
    material_class TEXT,
    material_abbreviation TEXT
);
"""

TABLE_FOR_CLASS = {
    "Polymer": "Polymers",
    "Fiber": "Fibers",
    "Composite": "Composites_materials",
}
ALL_TABLES = tuple(TABLE_FOR_CLASS.values())


# ---------------------------------------------------------------------------
# Run bookkeeping
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PdfResult:
    pdf: str
    elapsed_s: float
    materials: int
    extracted: int
    inserted: int
    flagged: int          # inserted with status != 'ok'
    duplicates: int
    material_classes: list[str]
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------


def init_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA_DDL)
    # Add the hardening-phase columns if they aren't there yet (idempotent).
    for table in ALL_TABLES:
        ensure_columns(conn, table)
    conn.commit()
    return conn


# Column order used for inserts: legacy columns first (positional compat), then
# the hardening-phase columns. Mirrors migrate.EXTRA_COLUMNS.
_LEGACY_COLS = [
    "material_name", "material_abbreviation", "section", "property_name",
    "value", "unit", "english", "test_condition", "comments",
]
_INSERT_COLS = _LEGACY_COLS + [name for name, _type in EXTRA_COLUMNS]


def _row_values(row: PropertyRow) -> tuple:
    from datetime import datetime, timezone

    mapping = {
        "material_name": row.material_name,
        "material_abbreviation": row.material_abbreviation,
        "section": row.section,
        "property_name": row.property_name,
        "value": row.value,
        "unit": row.unit,
        "english": row.english,
        "test_condition": row.test_condition,
        "comments": row.comments,
        "material_key": row.material_key,
        "material_class": row.material_class,
        "trade_grade": row.trade_grade,
        "manufacturer": row.manufacturer,
        "matrix": row.matrix,
        "fiber": row.fiber,
        "fiber_volume_fraction": row.fiber_volume_fraction,
        "value_raw": row.value_raw,
        "value_num": row.value_num,
        "value_min": row.value_min,
        "value_max": row.value_max,
        "qualifier": row.qualifier,
        "unit_canonical": row.unit_canonical,
        "value_si": row.value_si,
        "source_pdf": row.source_pdf,
        "source_sha1": row.source_sha1,
        "page": row.page,
        "source_quote": row.source_quote,
        "status": row.status,
        "flag_reason": row.flag_reason,
        "model": row.model,
        "prompt_version": row.prompt_version,
        "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return tuple(mapping[c] for c in _INSERT_COLS)


def already_inserted(conn: sqlite3.Connection, table: str, row: PropertyRow) -> bool:
    """Source-aware dedup (Task 6).

    Grain = (source_sha1, material_key, section, property_name, test_condition,
    value_raw). Skip only a true re-ingest of the *same measurement from the
    same PDF*; the same property from a different PDF (independent repeat) is
    kept.
    """
    cur = conn.execute(
        f"SELECT 1 FROM {table} "
        f"WHERE IFNULL(source_sha1,'') = IFNULL(?, '') "
        f"  AND IFNULL(material_key,'') = IFNULL(?, '') "
        f"  AND IFNULL(section,'') = IFNULL(?, '') "
        f"  AND IFNULL(property_name,'') = IFNULL(?, '') "
        f"  AND IFNULL(test_condition,'') = IFNULL(?, '') "
        f"  AND IFNULL(value_raw,'') = IFNULL(?, '') "
        f"LIMIT 1",
        (row.source_sha1, row.material_key, row.section,
         row.property_name, row.test_condition, row.value_raw),
    )
    return cur.fetchone() is not None


def insert_row(conn: sqlite3.Connection, table: str, row: PropertyRow) -> None:
    placeholders = ", ".join("?" for _ in _INSERT_COLS)
    cols = ", ".join(_INSERT_COLS)
    conn.execute(
        f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
        _row_values(row),
    )


def record_source(
    conn: sqlite3.Connection,
    pdf_path: Path,
    sha1: str,
    material_class: Optional[str],
    abbr: Optional[str],
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO sources "
        "(pdf_filename, pdf_sha1, ingested_at, material_class, material_abbreviation) "
        "VALUES (?, ?, datetime('now'), ?, ?)",
        (pdf_path.name, sha1, material_class, abbr),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _empty_result(pdf_path: Path, started: float, error: str) -> PdfResult:
    return PdfResult(
        pdf=pdf_path.name,
        elapsed_s=time.time() - started,
        materials=0,
        extracted=0,
        inserted=0,
        flagged=0,
        duplicates=0,
        material_classes=[],
        error=error,
    )


def process_pdf(
    pdf_path: Path,
    conn: sqlite3.Connection,
    api_key: str,
) -> PdfResult:
    started = time.time()
    pdf_bytes = pdf_path.read_bytes()
    sha1 = hashlib.sha1(pdf_bytes).hexdigest()

    # Skip if we already ingested this exact file.
    already = conn.execute(
        "SELECT 1 FROM sources WHERE pdf_sha1 = ? LIMIT 1", (sha1,)
    ).fetchone()
    if already:
        return _empty_result(pdf_path, started, "skipped_seen_sha1")

    try:
        extracted = extract_from_pdf(pdf_bytes, pdf_path.name, api_key)
    except requests.RequestException as exc:
        return _empty_result(pdf_path, started, f"gemini_error:{exc}")

    if extracted.doc_status == "scanned_no_text":
        # Don't fabricate rows from an image-only PDF (Task 10).
        record_source(conn, pdf_path, sha1, "scanned_no_text", None)
        conn.commit()
        return _empty_result(pdf_path, started, "scanned_no_text")
    if extracted.doc_status != "ok" or not extracted.materials:
        return _empty_result(pdf_path, started, "empty_extraction")

    # Ground every value against the PDF text (Task 1).
    page_texts = extraction.pdf_page_texts(pdf_bytes)
    verify_against_text(extracted, page_texts)

    rows = to_rows(extracted, pdf_path.name, sha1)

    classes = [extraction.classify_material(m) for m in extracted.materials]
    primary_class = classes[0] if classes else None
    primary_abbr = rows[0].material_abbreviation if rows else None
    record_source(conn, pdf_path, sha1, primary_class, primary_abbr)

    inserted = flagged = duplicates = 0
    for row in rows:
        table = TABLE_FOR_CLASS.get(row.material_class, "Polymers")
        if already_inserted(conn, table, row):
            duplicates += 1
            continue
        insert_row(conn, table, row)
        inserted += 1
        if row.status != "ok":
            flagged += 1
    conn.commit()

    return PdfResult(
        pdf=pdf_path.name,
        elapsed_s=time.time() - started,
        materials=len(extracted.materials),
        extracted=len(rows),
        inserted=inserted,
        flagged=flagged,
        duplicates=duplicates,
        material_classes=sorted(set(classes)),
    )


# ---------------------------------------------------------------------------
# review_queue.csv as a view over flagged rows (Task 4)
# ---------------------------------------------------------------------------

_REVIEW_COLUMNS = [
    "table_name", "source_pdf", "page", "status", "flag_reason",
    "material_name", "material_key", "material_class", "section",
    "property_name", "value_raw", "value_num", "unit", "unit_canonical",
    "value_si", "test_condition", "source_quote", "comments",
]


def export_review_queue(conn: sqlite3.Connection, path: Path) -> int:
    """Write review_queue.csv as SELECT ... WHERE status != 'ok' across tables."""
    import csv

    select_cols = [c for c in _REVIEW_COLUMNS if c != "table_name"]
    rows: list[list[Any]] = []
    for table in ALL_TABLES:
        cur = conn.execute(
            f"SELECT {', '.join(select_cols)} FROM {table} "
            f"WHERE IFNULL(status,'ok') != 'ok'"
        )
        for r in cur.fetchall():
            rows.append([table, *r])

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_REVIEW_COLUMNS)
        writer.writerows(rows)
    return len(rows)


def promote_review_queue(conn: sqlite3.Connection, path: Path) -> int:
    """Re-admit corrected rows from a review CSV as status='ok' (Task 4, optional).

    Matches on (table_name, source_pdf, material_key, property_name,
    test_condition, value_raw) and sets status='ok', flag_reason='promoted'.
    """
    import csv

    promoted = 0
    with path.open("r", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            table = r.get("table_name")
            if table not in ALL_TABLES:
                continue
            cur = conn.execute(
                f"UPDATE {table} SET status='ok', flag_reason='promoted' "
                f"WHERE IFNULL(source_pdf,'')=? AND IFNULL(material_key,'')=? "
                f"  AND IFNULL(property_name,'')=? AND IFNULL(test_condition,'')=? "
                f"  AND IFNULL(value_raw,'')=?",
                (r.get("source_pdf", ""), r.get("material_key", ""),
                 r.get("property_name", ""), r.get("test_condition", ""),
                 r.get("value_raw", "")),
            )
            promoted += cur.rowcount
    conn.commit()
    return promoted


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def summarize(results: list[PdfResult]) -> dict[str, Any]:
    total_pdfs = len(results)
    successes = [r for r in results if not r.error]
    extracted = sum(r.extracted for r in successes)
    inserted = sum(r.inserted for r in successes)
    flagged = sum(r.flagged for r in successes)
    duplicates = sum(r.duplicates for r in successes)
    materials = sum(r.materials for r in successes)
    total_elapsed = sum(r.elapsed_s for r in results)
    avg_elapsed = total_elapsed / total_pdfs if total_pdfs else 0.0
    error_breakdown: dict[str, int] = {}
    for r in results:
        if r.error:
            key = r.error.split(":", 1)[0]
            error_breakdown[key] = error_breakdown.get(key, 0) + 1
    return {
        "pdfs_seen": total_pdfs,
        "pdfs_ok": len(successes),
        "errors_by_kind": error_breakdown,
        "materials_extracted": materials,
        "rows_extracted": extracted,
        "rows_inserted": inserted,
        "rows_flagged_in_db": flagged,
        "rows_duplicate_skipped": duplicates,
        "insert_rate": inserted / extracted if extracted else 0.0,
        "flag_rate": flagged / inserted if inserted else 0.0,
        "duplicate_rate": duplicates / extracted if extracted else 0.0,
        "avg_seconds_per_pdf": round(avg_elapsed, 2),
        "total_seconds": round(total_elapsed, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        help="Folder of PDFs to ingest")
    parser.add_argument("--db", default=Path("materials_mirror.sqlite"),
                        type=Path, help="SQLite mirror path")
    parser.add_argument("--review", default=Path("review_queue.csv"),
                        type=Path, help="CSV view of flagged (status != 'ok') rows")
    parser.add_argument("--report", default=Path("run_report.json"),
                        type=Path, help="JSON run summary")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N PDFs (for testing)")
    parser.add_argument("--migrate", action="store_true",
                        help="Add hardening-phase columns to an existing DB and exit")
    parser.add_argument("--promote", type=Path, default=None,
                        help="Re-admit corrected rows from a review CSV as status='ok'")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger("batch_ingest")

    if args.migrate:
        conn = init_db(args.db)
        conn.close()
        log.info("Migration complete: %s", args.db)
        return 0

    if args.promote:
        conn = init_db(args.db)
        n = promote_review_queue(conn, args.promote)
        export_review_queue(conn, args.review)
        conn.close()
        log.info("Promoted %d rows to status='ok' from %s", n, args.promote)
        return 0

    if not args.input:
        log.error("--input is required (folder of PDFs).")
        return 2

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        log.error("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")
        return 2

    pdfs = sorted(p for p in args.input.rglob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        log.error("No PDFs found under %s", args.input)
        return 2

    log.info("Ingesting %d PDFs into %s", len(pdfs), args.db)
    conn = init_db(args.db)
    results: list[PdfResult] = []
    for i, pdf in enumerate(pdfs, start=1):
        log.info("[%d/%d] %s", i, len(pdfs), pdf.name)
        result = process_pdf(pdf, conn, api_key)
        results.append(result)
        log.info(
            "  -> materials=%d classes=%s extracted=%d inserted=%d "
            "flagged=%d duplicates=%d elapsed=%.1fs error=%s",
            result.materials,
            ",".join(result.material_classes) or "-",
            result.extracted,
            result.inserted,
            result.flagged,
            result.duplicates,
            result.elapsed_s,
            result.error,
        )

    n_flagged = export_review_queue(conn, args.review)
    summary = summarize(results)
    args.report.write_text(json.dumps(summary, indent=2))
    log.info("Run summary written to %s", args.report)
    log.info("Review queue (%d flagged rows) written to %s", n_flagged, args.review)

    conn.close()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
