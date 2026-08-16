r"""
migrate.py — non-destructive column migration for the materials mirror.

Adds the extraction-hardening columns to the existing ``Polymers`` / ``Fibers``
/ ``Composites_materials`` tables via ``ALTER TABLE ... ADD COLUMN``. The
original columns (``value``, ``unit``, ``english`` ...) are left untouched and
keep being populated, so the CSV export and the Streamlit ``page1.py`` continue
to work.

SQLite ``ADD COLUMN`` is cheap and never rewrites existing rows; existing rows
get NULL for the new columns. New rows written by ``batch_ingest.py`` fill them.

Usage:
    python migrate.py --db ./materials_mirror.sqlite          # back up + migrate
    python migrate.py --db ./materials_mirror.sqlite --no-backup
    # equivalently:  python batch_ingest.py --migrate --db ...
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

# (column_name, sqlite_type) — appended after the legacy columns. Order is the
# canonical insert order used by batch_ingest._INSERT_COLS, so keep it stable.
EXTRA_COLUMNS: list[tuple[str, str]] = [
    ("material_key", "TEXT"),
    ("material_class", "TEXT"),
    ("trade_grade", "TEXT"),
    ("manufacturer", "TEXT"),
    ("matrix", "TEXT"),
    ("fiber", "TEXT"),
    ("fiber_volume_fraction", "TEXT"),
    ("value_raw", "TEXT"),
    ("value_num", "REAL"),
    ("value_min", "REAL"),
    ("value_max", "REAL"),
    ("qualifier", "TEXT"),
    ("unit_canonical", "TEXT"),
    ("value_si", "REAL"),
    ("source_pdf", "TEXT"),
    ("source_sha1", "TEXT"),
    ("page", "INTEGER"),
    ("source_quote", "TEXT"),
    ("status", "TEXT DEFAULT 'ok'"),
    ("flag_reason", "TEXT"),
    ("model", "TEXT"),
    ("prompt_version", "TEXT"),
    # Populated explicitly by batch_ingest at insert time. SQLite forbids a
    # non-constant DEFAULT (datetime('now')) on ALTER TABLE ADD COLUMN, so this
    # stays a plain column rather than carrying a SQL default.
    ("extracted_at", "TEXT"),
]

TARGET_TABLES = ("Polymers", "Fibers", "Composites_materials")


def _existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def ensure_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    """Add any missing EXTRA_COLUMNS to `table`. Returns the columns added.

    Idempotent: safe to call on every run. Only touches a table that exists.
    """
    info = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if not info:
        return []
    have = _existing_columns(conn, table)
    added: list[str] = []
    for name, coltype in EXTRA_COLUMNS:
        if name not in have:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {coltype}")
            added.append(name)
    return added


def _sources_unique_column(conn: sqlite3.Connection) -> str | None:
    """Which column carries the UNIQUE constraint on `sources` (None if no table)."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='sources'"
    ).fetchone()
    if not row or not row[0]:
        return None
    ddl = row[0].lower()
    if "pdf_sha1 text unique" in ddl:
        return "pdf_sha1"
    if "pdf_filename text unique" in ddl:
        return "pdf_filename"
    return ""


def ensure_sources_sha1_unique(conn: sqlite3.Connection) -> bool:
    """Rebuild `sources` so its UNIQUE key is `pdf_sha1`, not `pdf_filename`.

    Legacy DBs keyed the doc logbook on the basename. Two different PDFs that
    share a filename (likely — --input is rglob'd across vendor folders) then
    collided: `INSERT OR IGNORE` dropped the second, `seen_sha1()` never saw
    it, and it was re-sent to Gemini on every run. Content hash is the real
    identity. SQLite cannot alter a UNIQUE constraint in place, so this is a
    copy-rebuild; it keeps the earliest row per sha1 if duplicates exist.
    Idempotent — returns True only when a rebuild happened.
    """
    key = _sources_unique_column(conn)
    if key in (None, "pdf_sha1"):
        return False
    # One transaction: Python's sqlite3 executescript() autocommits each
    # statement otherwise, and a crash between the CREATE and the RENAME left
    # every later init_db() failing on "sources__new already exists" (or, after
    # the DROP, silently orphaned the whole logbook). DDL is transactional in
    # SQLite, so BEGIN/COMMIT makes the rebuild all-or-nothing; the leading
    # DROP IF EXISTS makes a retry after a crash self-healing.
    conn.executescript(
        """
        BEGIN;
        DROP TABLE IF EXISTS sources__new;
        CREATE TABLE sources__new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_filename TEXT,
            pdf_sha1 TEXT UNIQUE,
            ingested_at TEXT,
            material_class TEXT,
            material_abbreviation TEXT
        );
        INSERT INTO sources__new
            (id, pdf_filename, pdf_sha1, ingested_at, material_class, material_abbreviation)
        SELECT id, pdf_filename, pdf_sha1, ingested_at, material_class, material_abbreviation
        FROM sources
        WHERE id IN (SELECT MIN(id) FROM sources GROUP BY IFNULL(pdf_sha1, '__null__' || id));
        DROP TABLE sources;
        ALTER TABLE sources__new RENAME TO sources;
        COMMIT;
        """
    )
    return True


def backfill_material_key_grade(conn: sqlite3.Connection, table: str) -> int:
    """Re-key pipeline rows written before trade_grade became part of material_key.

    Rows ingested before 2026-08 carry ``material_key = <name>`` while their
    ``trade_grade`` column is populated; the current rule is
    ``<name>|<grade>`` (see extraction.material_key). Without this backfill a
    re-ingest of the same PDF would not dedup against the old rows and would
    insert every graded row a second time. Only touches pipeline rows
    (``source_sha1`` set) whose key has no ``|`` yet. Idempotent. Returns the
    number of rows updated. Computed in Python so it is byte-identical to
    what extraction.material_key() produces.
    """
    import re as _re
    info = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if not info:
        return 0
    have = _existing_columns(conn, table)
    if not {"material_key", "trade_grade", "source_sha1"} <= have:
        return 0
    rows = conn.execute(
        f"SELECT id, material_key, trade_grade FROM {table} "
        f"WHERE source_sha1 IS NOT NULL AND IFNULL(trade_grade,'') <> '' "
        f"  AND IFNULL(material_key,'') <> '' AND material_key NOT LIKE '%|%'"
    ).fetchall()
    n = 0
    for rid, key, grade in rows:
        g = _re.sub(r"\s+", " ", (grade or "").strip().lower())
        if g and g != key and g not in key:
            conn.execute(f"UPDATE {table} SET material_key=? WHERE id=?", (f"{key}|{g}", rid))
            n += 1
    return n


def migrate(db_path: Path, backup: bool = True) -> dict[str, list[str]]:
    if backup and db_path.exists():
        bak = db_path.with_suffix(db_path.suffix + ".bak")
        shutil.copy2(db_path, bak)
        print(f"Backed up {db_path} -> {bak}", file=sys.stderr)

    conn = sqlite3.connect(str(db_path))
    try:
        result: dict[str, list[str]] = {}
        for table in TARGET_TABLES:
            added = ensure_columns(conn, table)
            n = backfill_material_key_grade(conn, table)
            if n:
                added = added + [f"(re-keyed {n} rows: material_key += '|trade_grade')"]
            result[table] = added
        if ensure_sources_sha1_unique(conn):
            result["sources"] = ["UNIQUE(pdf_sha1) (rebuilt from UNIQUE(pdf_filename))"]
        else:
            result["sources"] = []
        conn.commit()
    finally:
        conn.close()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path, help="SQLite mirror path")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip the .bak backup before altering")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"DB {args.db} does not exist; nothing to migrate "
              f"(a fresh DB is created with all columns by batch_ingest).",
              file=sys.stderr)
        return 1

    result = migrate(args.db, backup=not args.no_backup)
    for table, added in result.items():
        if not added:
            print(f"{table}: already up to date")
        elif table == "sources":
            print(f"{table}: rebuilt -> {added[0]}")
        else:
            print(f"{table}: added {len(added)} columns: {', '.join(added)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
