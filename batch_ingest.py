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

    # write to the shared Postgres (the DB the HF Space reads) instead of
    # SQLite — env: DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD or DATABASE_URL.
    # Requires a one-time `python pg_migrate.py --apply` first (see pg_mirror.py):
    python batch_ingest.py --pg --input ./pdfs

    # figure & graph mining (opt-in; SQLite only; see FIGURES.md):
    python batch_ingest.py --figures --input ./pdfs --db ./materials_mirror.sqlite
    python batch_ingest.py --figures --no-figure-mining --input ./pdfs   # harvest+classify only

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
from migrate import (
    EXTRA_COLUMNS,
    backfill_material_key_grade,
    ensure_columns,
    ensure_figures_table,
    ensure_sources_sha1_unique,
)


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
    pdf_filename TEXT,
    pdf_sha1 TEXT UNIQUE,
    ingested_at TEXT,
    material_class TEXT,
    material_abbreviation TEXT
);
"""
# `sources` identity is the content hash, not the basename: two different PDFs
# that share a filename (vendorA/datasheet.pdf vs vendorB/datasheet.pdf — likely,
# since --input is rglob'd) used to collide on a `pdf_filename UNIQUE` +
# INSERT OR IGNORE, so the second was never recorded and got re-sent to Gemini
# on every run. migrate.ensure_sources_sha1_unique() rebuilds legacy tables.

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
    # figure-mining phase (all zero when --figures is off)
    figures_found: int = 0        # harvested PNGs (after junk filters + cap)
    figures_mined: int = 0        # plot/table figures that returned a readout
    figure_rows: int = 0          # origin='figure' rows inserted
    figure_duplicates: int = 0    # figure rows skipped by the dedup grain
    vision_calls: int = 0         # classify + mining Gemini calls
    figure_error: Optional[str] = None   # non-fatal: text rows still inserted
    figure_filters: dict[str, int] = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------


def init_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA_DDL)
    # Add the hardening-phase columns if they aren't there yet (idempotent).
    for table in ALL_TABLES:
        ensure_columns(conn, table)
        # Rows written before trade_grade joined material_key get re-keyed
        # once, so a re-ingest dedups against them instead of doubling them.
        backfill_material_key_grade(conn, table)
    # Legacy DBs keyed `sources` on pdf_filename; rebuild to pdf_sha1 (idempotent).
    ensure_sources_sha1_unique(conn)
    # Figure provenance table (figure-mining phase; idempotent).
    ensure_figures_table(conn)
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

    # Export guard (figure-mining phase). Every consumer that shows rows to
    # the app filters on status='ok', so status is the publish gate. A figure
    # row must therefore never be INSERTED as 'ok' — only --promote may set
    # that, after a human looked at the PNG. Enforced here, the single point
    # both the SQLite and Postgres insert paths go through.
    status, flag_reason = row.status, row.flag_reason
    if (row.origin or "text") == "figure" and status == "ok":
        status = "figure_estimate"
        flag_reason = ("figure row inserted with status=ok; downgraded — only "
                       "--promote may publish a figure reading"
                       + (f"; {row.flag_reason}" if row.flag_reason else ""))

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
        "status": status,
        "flag_reason": flag_reason,
        "model": row.model,
        "prompt_version": row.prompt_version,
        "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "origin": row.origin or "text",
        "figure_id": row.figure_id or None,
    }
    return tuple(mapping[c] for c in _INSERT_COLS)


def seen_sha1(conn: sqlite3.Connection, sha1: str) -> bool:
    """True if this exact PDF (by sha1) was already ingested."""
    cur = conn.execute("SELECT 1 FROM sources WHERE pdf_sha1 = ? LIMIT 1", (sha1,))
    return cur.fetchone() is not None


def already_inserted(conn: sqlite3.Connection, table: str, row: PropertyRow) -> bool:
    """Source-aware dedup (Task 6).

    Grain = (source_sha1, material_key, section, property_name, test_condition,
    value_raw, origin). Skip only a true re-ingest of the *same measurement
    from the same PDF*; the same property from a different PDF (independent
    repeat) is kept. `origin` is part of the grain on purpose: a text row and
    a figure row reporting the same number both survive — that agreement is
    signal, not duplication (figure-mining phase).
    """
    cur = conn.execute(
        f"SELECT 1 FROM {table} "
        f"WHERE IFNULL(source_sha1,'') = IFNULL(?, '') "
        f"  AND IFNULL(material_key,'') = IFNULL(?, '') "
        f"  AND IFNULL(section,'') = IFNULL(?, '') "
        f"  AND IFNULL(property_name,'') = IFNULL(?, '') "
        f"  AND IFNULL(test_condition,'') = IFNULL(?, '') "
        f"  AND IFNULL(value_raw,'') = IFNULL(?, '') "
        f"  AND IFNULL(origin,'text') = IFNULL(?, 'text') "
        f"LIMIT 1",
        (row.source_sha1, row.material_key, row.section,
         row.property_name, row.test_condition, row.value_raw,
         row.origin or "text"),
    )
    return cur.fetchone() is not None


_FIGURE_COLS = [
    "figure_id", "source_pdf", "source_sha1", "page", "bbox", "caption",
    "figure_kind", "material_key", "image_path", "image_sha256", "width_px",
    "height_px", "route", "mining_status", "n_values", "model",
    "figure_prompt_version", "extracted_at",
]


def upsert_figure(conn: sqlite3.Connection, fig: Any) -> None:
    """Insert or refresh one harvested figure's provenance row (keyed on
    figure_id = sha of the PNG, so a re-harvest is idempotent while a later
    classify/mine pass can update kind/status)."""
    from datetime import datetime, timezone

    vals = {
        "figure_id": fig.figure_id, "source_pdf": fig.source_pdf,
        "source_sha1": fig.source_sha1, "page": fig.page,
        "bbox": json.dumps(list(fig.bbox)), "caption": fig.caption,
        "figure_kind": fig.figure_kind, "material_key": fig.material_key,
        "image_path": fig.image_path, "image_sha256": fig.image_sha256,
        "width_px": fig.width_px, "height_px": fig.height_px, "route": fig.route,
        "mining_status": fig.mining_status, "n_values": fig.n_values,
        "model": fig.model, "figure_prompt_version": fig.figure_prompt_version,
        "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    cols = ", ".join(_FIGURE_COLS)
    ph = ", ".join("?" for _ in _FIGURE_COLS)
    upd = ", ".join(f"{c}=excluded.{c}" for c in _FIGURE_COLS if c != "figure_id")
    conn.execute(
        f"INSERT INTO figures ({cols}) VALUES ({ph}) "
        f"ON CONFLICT(figure_id) DO UPDATE SET {upd}",
        tuple(vals[c] for c in _FIGURE_COLS),
    )


def figures_recorded_for(conn: sqlite3.Connection, sha1: str) -> int:
    """How many figures the `figures` table already holds for this PDF."""
    cur = conn.execute("SELECT count(*) FROM figures WHERE source_sha1 = ?", (sha1,))
    return int(cur.fetchone()[0])


def materials_for_source(conn: sqlite3.Connection, sha1: str) -> list["extraction.Material"]:
    """Rebuild the text-pass material list of an already-ingested PDF from its
    rows, so the figure stage can run on a PDF whose text pass happened in an
    earlier run (backfill mode)."""
    seen: dict[str, extraction.Material] = {}
    for table in ALL_TABLES:
        cur = conn.execute(
            f"SELECT DISTINCT material_name, material_abbreviation, material_class, "
            f"trade_grade, manufacturer, matrix, fiber, fiber_volume_fraction "
            f"FROM {table} WHERE source_sha1 = ? AND IFNULL(origin,'text') = 'text'",
            (sha1,),
        )
        for r in cur.fetchall():
            m = extraction.Material(
                material_name=r[0] or "", material_abbreviation=r[1] or "",
                material_class=r[2] or "", trade_grade=r[3] or "",
                manufacturer=r[4] or "", matrix=r[5] or "", fiber=r[6] or "",
                fiber_volume_fraction=r[7] or "",
            )
            seen.setdefault(extraction.material_key(m), m)
    return list(seen.values())


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


@dataclasses.dataclass
class FigureOptions:
    """--figures settings handed to process_pdf (None = figure stage off)."""
    out_dir: Path = Path("crawl_out/figures")
    max_figures: int = 12
    mine: bool = True          # False = harvest + classify only (cheap mode)


def _run_figure_stage(
    pdf_path: Path, pdf_bytes: bytes, sha1: str, text_materials: list,
    api_key: str, conn: Any, db: Any, opts: FigureOptions, result: PdfResult,
) -> None:
    """harvest -> classify -> mine -> insert figure rows. NEVER raises: any
    failure lands in result.figure_error and is counted; the PDF's text rows
    are already committed by the time this runs."""
    try:
        import figures as F
        stage = F.run_figure_stage(
            pdf_bytes, pdf_path.name, sha1, text_materials, api_key,
            out_dir=opts.out_dir, max_figures=opts.max_figures, mine=opts.mine,
        )
        result.figures_found = len(stage.figures)
        result.figures_mined = stage.mined_figures
        result.vision_calls = stage.vision.total
        result.figure_filters = {
            k: v for k, v in dataclasses.asdict(stage.harvest).items() if v
        }
        if stage.error:
            result.figure_error = stage.error
        for fig in stage.figures:
            db.upsert_figure(conn, fig)
        frows = fdups = 0
        for row in stage.rows:
            table = TABLE_FOR_CLASS.get(row.material_class, "Polymers")
            if db.already_inserted(conn, table, row):
                fdups += 1
                continue
            db.insert_row(conn, table, row)
            frows += 1
        result.figure_rows = frows
        result.figure_duplicates = fdups
        # figure rows are never 'ok' -> they all count as flagged
        result.flagged += frows
        result.inserted += frows
        conn.commit()
        if stage.vision.failed_calls and not result.figure_error:
            result.figure_error = f"vision_calls_failed:{stage.vision.failed_calls}"
    except Exception as exc:  # pragma: no cover - defensive; run_figure_stage already guards
        log = logging.getLogger("batch_ingest")
        log.exception("figure stage crashed for %s", pdf_path.name)
        result.figure_error = f"figure_stage_error:{type(exc).__name__}:{exc}"
        try:
            conn.rollback()
        except Exception:
            pass


def process_pdf(
    pdf_path: Path,
    conn: Any,
    api_key: str,
    db: Any = None,
    figure_opts: Optional[FigureOptions] = None,
) -> PdfResult:
    # `db` = backend module providing seen_sha1 / record_source /
    # already_inserted / insert_row. Defaults to this module (SQLite);
    # main() passes pg_mirror for --pg. Same logic either way.
    db = db or sys.modules[__name__]
    started = time.time()
    pdf_bytes = pdf_path.read_bytes()
    sha1 = hashlib.sha1(pdf_bytes).hexdigest()

    # Skip if we already ingested this exact file — unless figures are on and
    # this PDF has none recorded yet, in which case run ONLY the figure stage
    # (backfill for PDFs whose text pass predates --figures).
    if db.seen_sha1(conn, sha1):
        result = _empty_result(pdf_path, started, "skipped_seen_sha1")
        if figure_opts is not None and hasattr(db, "figures_recorded_for") \
                and db.figures_recorded_for(conn, sha1) == 0:
            mats = db.materials_for_source(conn, sha1)
            _run_figure_stage(pdf_path, pdf_bytes, sha1, mats, api_key, conn, db,
                              figure_opts, result)
            result.elapsed_s = time.time() - started
        return result

    try:
        extracted = extract_from_pdf(pdf_bytes, pdf_path.name, api_key)
    except requests.RequestException as exc:
        return _empty_result(pdf_path, started, f"gemini_error:{exc}")

    if extracted.doc_status == "scanned_no_text":
        # Don't fabricate rows from an image-only PDF (Task 10). Whole-page
        # scans are not figures either — the figure stage is skipped too.
        db.record_source(conn, pdf_path, sha1, "scanned_no_text", None)
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
    db.record_source(conn, pdf_path, sha1, primary_class, primary_abbr)

    inserted = flagged = duplicates = 0
    for row in rows:
        table = TABLE_FOR_CLASS.get(row.material_class, "Polymers")
        if db.already_inserted(conn, table, row):
            duplicates += 1
            continue
        db.insert_row(conn, table, row)
        inserted += 1
        if row.status != "ok":
            flagged += 1
    conn.commit()   # text rows are safe on disk before the figure stage runs

    result = PdfResult(
        pdf=pdf_path.name,
        elapsed_s=time.time() - started,
        materials=len(extracted.materials),
        extracted=len(rows),
        inserted=inserted,
        flagged=flagged,
        duplicates=duplicates,
        material_classes=sorted(set(classes)),
    )

    if figure_opts is not None:
        _run_figure_stage(pdf_path, pdf_bytes, sha1, extracted.materials, api_key,
                          conn, db, figure_opts, result)
        result.elapsed_s = time.time() - started
    return result


# ---------------------------------------------------------------------------
# review_queue.csv as a view over flagged rows (Task 4)
# ---------------------------------------------------------------------------

_REVIEW_COLUMNS = [
    "table_name", "source_pdf", "page", "status", "flag_reason",
    "material_name", "material_key", "material_class", "section",
    "property_name", "value_raw", "value_num", "unit", "unit_canonical",
    "value_si", "test_condition", "source_quote", "comments",
    # figure-mining phase: figure rows land here automatically (status is
    # never 'ok'); a reviewer opens the PNG behind figure_id and --promote is
    # how one gets blessed.
    "origin", "figure_id",
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

    Matches on the full dedup grain — (table_name, source_pdf, material_key,
    section, property_name, test_condition, value_raw, origin) — and only
    touches rows whose status is not already 'ok', so promoting one flagged
    row cannot rewrite the flag_reason of an already-ok sibling in another
    section, and promoting a text row cannot silently bless the figure row
    that reports the same number (or vice versa). A CSV without an `origin`
    column (pre-figure-phase export) matches text rows only.
    Sets status='ok', flag_reason='promoted'.
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
                f"  AND IFNULL(section,'')=? "
                f"  AND IFNULL(property_name,'')=? AND IFNULL(test_condition,'')=? "
                f"  AND IFNULL(value_raw,'')=? "
                f"  AND IFNULL(origin,'text')=? "
                f"  AND IFNULL(status,'ok') != 'ok'",
                (r.get("source_pdf") or "", r.get("material_key") or "",
                 r.get("section") or "",
                 r.get("property_name") or "", r.get("test_condition") or "",
                 r.get("value_raw") or "",
                 (r.get("origin") or "text").strip() or "text"),
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
    # figure-mining phase (all PdfResults, incl. backfill on seen PDFs)
    fig_filters: dict[str, int] = {}
    fig_errors: dict[str, int] = {}
    for r in results:
        for k, v in (r.figure_filters or {}).items():
            fig_filters[k] = fig_filters.get(k, 0) + v
        if r.figure_error:
            key = r.figure_error.split(":", 1)[0]
            fig_errors[key] = fig_errors.get(key, 0) + 1
    figure_rows = sum(r.figure_rows for r in results)
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
        "figures": {
            "figures_found": sum(r.figures_found for r in results),
            "figures_mined": sum(r.figures_mined for r in results),
            "figure_rows": figure_rows,
            "figure_rows_duplicate_skipped": sum(r.figure_duplicates for r in results),
            "vision_calls": sum(r.vision_calls for r in results),
            "figure_errors_by_kind": fig_errors,
            "harvest_filters": fig_filters,
        },
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
    parser.add_argument("--pg", action="store_true",
                        help="Write to the shared Postgres (env DB_HOST/... or "
                             "DATABASE_URL) instead of the local SQLite mirror")
    # --- figure-mining phase (opt-in) ---
    parser.add_argument("--figures", action="store_true",
                        help="Also harvest figures from each PDF, classify them with one "
                             "vision call per PDF, and mine plots/table-images for "
                             "property values (origin='figure', status='figure_estimate')")
    parser.add_argument("--figures-dir", type=Path, default=Path("crawl_out/figures"),
                        help="Where harvested figure PNGs go (<dir>/<sha1>/p<page>_<n>.png)")
    parser.add_argument("--max-figures-per-pdf", type=int, default=12,
                        help="Hard cap on harvested figures per PDF (bounds vision calls)")
    parser.add_argument("--no-figure-mining", action="store_true",
                        help="With --figures: harvest + classify only, skip the "
                             "per-figure mining calls (cheap mode)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger("batch_ingest")

    # Select the storage backend: this module (SQLite, default) or pg_mirror.
    if args.pg:
        import pg_mirror as db
        log.info("Postgres mode: %s", db.config_summary())
    else:
        db = sys.modules[__name__]

    figure_opts: Optional[FigureOptions] = None
    if args.figures:
        if args.pg:
            # Not half-supported silently: the Postgres mirror has no figures
            # table and its dedup index does not include `origin` yet, so a
            # figure row equal to a text row on the old grain would violate
            # it. See FIGURES.md / FOLLOWUPS.md.
            log.error("--figures is not yet supported with --pg (SQLite mirror only). "
                      "Run without --pg, or see FOLLOWUPS.md for the Postgres port.")
            return 2
        figure_opts = FigureOptions(out_dir=args.figures_dir,
                                    max_figures=args.max_figures_per_pdf,
                                    mine=not args.no_figure_mining)

    if args.migrate:
        if args.pg:
            log.error("Schema changes to the shared Postgres are deliberately "
                      "kept in one place: run `python pg_migrate.py` (dry-run) "
                      "then `python pg_migrate.py --apply`.")
            return 2
        conn = init_db(args.db)
        conn.close()
        log.info("Migration complete: %s", args.db)
        return 0

    if args.promote:
        conn = db.connect_from_env() if args.pg else init_db(args.db)
        if args.pg:
            db.check_schema(conn)
        n = db.promote_review_queue(conn, args.promote)
        db.export_review_queue(conn, args.review)
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

    if args.pg:
        log.info("Ingesting %d PDFs into Postgres (%s)", len(pdfs), db.config_summary())
        conn = db.connect_from_env()
        db.check_schema(conn)  # refuse to run against an unmigrated schema
    else:
        log.info("Ingesting %d PDFs into %s", len(pdfs), args.db)
        conn = init_db(args.db)
    results: list[PdfResult] = []
    for i, pdf in enumerate(pdfs, start=1):
        log.info("[%d/%d] %s", i, len(pdfs), pdf.name)
        result = process_pdf(pdf, conn, api_key, db=db, figure_opts=figure_opts)
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
        if figure_opts is not None:
            log.info(
                "  -> figures: found=%d mined=%d rows=%d dup=%d vision_calls=%d error=%s",
                result.figures_found, result.figures_mined, result.figure_rows,
                result.figure_duplicates, result.vision_calls, result.figure_error,
            )

    n_flagged = db.export_review_queue(conn, args.review)
    summary = summarize(results)
    args.report.write_text(json.dumps(summary, indent=2))
    log.info("Run summary written to %s", args.report)
    log.info("Review queue (%d flagged rows) written to %s", n_flagged, args.review)

    conn.close()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
