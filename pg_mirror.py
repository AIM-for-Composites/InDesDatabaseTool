r"""
pg_mirror.py — Postgres backend for batch_ingest.py (the pipeline → HF Space bridge).

Writes pipeline rows into the SAME Postgres database the Hugging Face Space
(aim4composites/MaterialsDatabase) reads, so the site shows pipeline output
with zero Space code changes.

Behavior mirrors the SQLite path in batch_ingest.py exactly:

  * legacy columns (`value`, `unit`, `english`, ...) keep being populated —
    the Space's data_loader.py SELECTs exactly those
  * hardening columns (status, provenance, value_si, ...) are added ONCE by
    ``python pg_migrate.py --apply`` (with CSV backup); this module never
    alters the schema — ``check_schema()`` refuses to ingest until migrated
  * dedup grain = (source_sha1, material_key, section, property_name,
    test_condition, value_raw): SELECT-then-INSERT like SQLite, plus a partial
    unique index as a safety net against double-writers
  * review_queue.csv export and --promote work identically

Config — same env names the Space already uses, read from the environment only:

    DATABASE_URL                     postgresql://user:pass@host:5432/db  (wins if set)
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    DB_SSLMODE                       default "require" (use "prefer" for local test PG)

Usage:
    python pg_migrate.py                 # dry-run: inspect, print planned changes
    python pg_migrate.py --apply         # backup CSVs, add columns/indexes/sources
    python batch_ingest.py --pg --input crawl_out/pdfs
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import psycopg
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "psycopg is required for the Postgres mirror: pip install 'psycopg[binary]'"
    ) from exc

import batch_ingest as _bi  # single source of truth: columns, row mapping, tables
from extraction import PropertyRow
from migrate import EXTRA_COLUMNS

TARGET_TABLES: tuple[str, ...] = tuple(_bi.ALL_TABLES)  # Polymers, Fibers, Composites_materials

_LEGACY_COLS = list(_bi._LEGACY_COLS)
_INSERT_COLS = list(_bi._INSERT_COLS)
_DEDUP_COLS = ("source_sha1", "material_key", "section", "property_name",
               "test_condition", "value_raw")


# ---------------------------------------------------------------------------
# Connection / config
# ---------------------------------------------------------------------------


def config_summary() -> str:
    """Human-readable, password-free description of the target DB."""
    url = os.environ.get("DATABASE_URL")
    if url:
        tail = url.split("@")[-1] if "@" in url else url
        return f"DATABASE_URL → {tail}"
    return (f"{os.environ.get('DB_USER', '?')}@{os.environ.get('DB_HOST', '?')}:"
            f"{os.environ.get('DB_PORT', '5432')}/{os.environ.get('DB_NAME', '?')}")


def connect_from_env() -> "psycopg.Connection":
    """Connect using DATABASE_URL or the Space's DB_* env vars. SSL on by default."""
    sslmode = os.environ.get("DB_SSLMODE", "require")
    url = os.environ.get("DATABASE_URL")
    if url:
        return psycopg.connect(url, sslmode=sslmode, connect_timeout=15)

    missing = [k for k in ("DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD")
               if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            "Postgres config incomplete — set DATABASE_URL or the missing env "
            f"vars: {', '.join(missing)}"
        )
    return psycopg.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ.get("DB_PORT", "5432")),
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        sslmode=sslmode,
        connect_timeout=15,
    )


def _q(table: str) -> str:
    """Quote a known table name (defensive: only whitelisted tables)."""
    if table not in TARGET_TABLES and table != "sources":
        raise ValueError(f"unexpected table: {table!r}")
    return f'"{table}"'


# ---------------------------------------------------------------------------
# Schema inspection / migration primitives (invoked by pg_migrate.py)
# ---------------------------------------------------------------------------


def _pg_type(sqlite_type: str) -> str:
    """Map migrate.py's SQLite column types to Postgres types."""
    t = sqlite_type.strip()
    upper = t.upper()
    if upper.startswith("TEXT"):
        return "text" + t[4:]          # keeps e.g. " DEFAULT 'ok'"
    if upper.startswith("REAL"):
        return "double precision" + t[4:]
    if upper.startswith("INTEGER"):
        return "integer" + t[7:]
    raise ValueError(f"unmapped SQLite type: {sqlite_type!r}")


def existing_columns(conn, table: str) -> dict[str, str]:
    """{column_name: data_type} for a table (empty dict if table missing)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name=%s "
            "ORDER BY ordinal_position",
            (table,),
        )
        return {r[0]: r[1] for r in cur.fetchall()}


def missing_columns(conn, table: str) -> list[tuple[str, str]]:
    """(name, pg_type) for each hardening column the table doesn't have yet."""
    have = existing_columns(conn, table)
    return [(name, _pg_type(t)) for name, t in EXTRA_COLUMNS if name not in have]


def table_count(conn, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {_q(table)}")
        return cur.fetchone()[0]


def sources_table_exists(conn) -> bool:
    return bool(existing_columns(conn, "sources"))


def dedup_index_name(table: str) -> str:
    return f"ix_{table.lower()}_pipeline_dedup"


def dedup_index_exists(conn, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_indexes WHERE schemaname='public' AND indexname=%s",
                    (dedup_index_name(table),))
        return cur.fetchone() is not None


def ensure_schema(conn) -> dict[str, list[str]]:
    """Apply the additive migration. Returns {table: [columns added]}.

    Only pg_migrate.py should call this (explicit --apply); batch_ingest --pg
    never alters the schema.
    """
    added: dict[str, list[str]] = {}
    with conn.cursor() as cur:
        for table in TARGET_TABLES:
            cols = missing_columns(conn, table)
            for name, pg_type in cols:
                cur.execute(f"ALTER TABLE {_q(table)} ADD COLUMN IF NOT EXISTS {name} {pg_type}")
            added[table] = [name for name, _ in cols]

            # Partial unique index = dedup safety net. Ignores legacy rows
            # (source_sha1 IS NULL), so it can't conflict with InDeS data.
            exprs = ", ".join(f"(COALESCE({c}, ''))" for c in _DEDUP_COLS)
            cur.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS {dedup_index_name(table)} "
                f"ON {_q(table)} ({exprs}) WHERE source_sha1 IS NOT NULL"
            )

        # Doc-level bookkeeping table (mirrors the SQLite `sources` table).
        cur.execute(
            "CREATE TABLE IF NOT EXISTS sources ("
            " id bigint GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,"
            " pdf_filename text UNIQUE,"
            " pdf_sha1 text,"
            " ingested_at text,"
            " material_class text,"
            " material_abbreviation text)"
        )
    conn.commit()
    return added


def check_schema(conn) -> None:
    """Raise (with instructions) unless every table has all hardening columns."""
    problems: list[str] = []
    for table in TARGET_TABLES:
        have = existing_columns(conn, table)
        if not have:
            problems.append(f"table {table} is missing")
            continue
        missing = [name for name, _ in EXTRA_COLUMNS if name not in have]
        if missing:
            problems.append(f"{table}: missing {len(missing)} columns ({', '.join(missing[:4])}…)")
    if not sources_table_exists(conn):
        problems.append("sources table is missing")
    if problems:
        raise RuntimeError(
            "Postgres schema is not migrated yet — run `python pg_migrate.py` "
            "(dry-run) then `python pg_migrate.py --apply`. Problems: "
            + "; ".join(problems)
        )


def backup_tables_csv(conn, out_dir: Path) -> dict[str, int]:
    """Dump the three material tables + row counts to CSVs. Cheap insurance."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for table in TARGET_TABLES:
        cols = list(existing_columns(conn, table))
        path = out_dir / f"{table}.csv"
        with conn.cursor() as cur, path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(cols)
            cur.execute(f"SELECT {', '.join(cols)} FROM {_q(table)}")
            n = 0
            while True:
                batch = cur.fetchmany(5000)
                if not batch:
                    break
                writer.writerows(batch)
                n += len(batch)
        counts[table] = n
    return counts


# ---------------------------------------------------------------------------
# Ingest operations — signature-compatible with batch_ingest's SQLite versions
# ---------------------------------------------------------------------------


def seen_sha1(conn, sha1: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM sources WHERE pdf_sha1 = %s LIMIT 1", (sha1,))
        return cur.fetchone() is not None


def record_source(conn, pdf_path: Path, sha1: str,
                  material_class: Optional[str], abbr: Optional[str]) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO sources "
            "(pdf_filename, pdf_sha1, ingested_at, material_class, material_abbreviation) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (pdf_filename) DO NOTHING",
            (pdf_path.name, sha1, now, material_class, abbr),
        )


def already_inserted(conn, table: str, row: PropertyRow) -> bool:
    """Same dedup grain and NULL semantics as the SQLite version."""
    conds = " AND ".join(f"COALESCE({c}, '') = COALESCE(%s, '')" for c in _DEDUP_COLS)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT 1 FROM {_q(table)} WHERE {conds} LIMIT 1",
            (row.source_sha1, row.material_key, row.section,
             row.property_name, row.test_condition, row.value_raw),
        )
        return cur.fetchone() is not None


def insert_row(conn, table: str, row: PropertyRow) -> None:
    values = [None if v == "" else v for v in _bi._row_values(row)]
    cols = ", ".join(_INSERT_COLS)
    placeholders = ", ".join("%s" for _ in _INSERT_COLS)
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {_q(table)} ({cols}) VALUES ({placeholders})", values
        )


def export_review_queue(conn, path: Path) -> int:
    """review_queue.csv = SELECT ... WHERE status != 'ok' across the 3 tables."""
    select_cols = [c for c in _bi._REVIEW_COLUMNS if c != "table_name"]
    rows: list[list[Any]] = []
    with conn.cursor() as cur:
        for table in TARGET_TABLES:
            cur.execute(
                f"SELECT {', '.join(select_cols)} FROM {_q(table)} "
                f"WHERE COALESCE(status, 'ok') <> 'ok'"
            )
            for r in cur.fetchall():
                rows.append([table, *r])

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_bi._REVIEW_COLUMNS)
        writer.writerows(rows)
    return len(rows)


def promote_review_queue(conn, path: Path) -> int:
    """Re-admit corrected rows from a review CSV as status='ok'."""
    promoted = 0
    with path.open("r", newline="", encoding="utf-8") as fh, conn.cursor() as cur:
        for r in csv.DictReader(fh):
            table = r.get("table_name")
            if table not in TARGET_TABLES:
                continue
            cur.execute(
                f"UPDATE {_q(table)} SET status='ok', flag_reason='promoted' "
                f"WHERE COALESCE(source_pdf,'')=%s AND COALESCE(material_key,'')=%s "
                f"  AND COALESCE(property_name,'')=%s AND COALESCE(test_condition,'')=%s "
                f"  AND COALESCE(value_raw,'')=%s",
                (r.get("source_pdf") or "", r.get("material_key") or "",
                 r.get("property_name") or "", r.get("test_condition") or "",
                 r.get("value_raw") or ""),
            )
            promoted += cur.rowcount
    conn.commit()
    return promoted
