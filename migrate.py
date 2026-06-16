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
            result[table] = added
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
        if added:
            print(f"{table}: added {len(added)} columns: {', '.join(added)}")
        else:
            print(f"{table}: already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
