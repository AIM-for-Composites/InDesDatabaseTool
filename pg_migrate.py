r"""
pg_migrate.py — one-time, additive migration of the shared Postgres database
(the one the HF Space reads) so batch_ingest.py --pg can write to it.

What --apply does (all additive, nothing dropped or rewritten):
  1. CSV backup of Polymers / Fibers / Composites_materials (default on)
  2. ALTER TABLE ADD COLUMN IF NOT EXISTS for each hardening column
     (migrate.EXTRA_COLUMNS: provenance, structured values, status, ...)
  3. Partial unique dedup index per table (ignores legacy rows)
  4. CREATE TABLE IF NOT EXISTS sources (doc-level bookkeeping), keyed on
     pdf_sha1; a legacy sources table keyed on pdf_filename is re-keyed in
     place (drop the filename UNIQUE, add a unique index on pdf_sha1)

Legacy note: `status text DEFAULT 'ok'` backfills existing rows with 'ok'
(instant in Postgres ≥ 11) — so the ~30k pre-existing InDeS rows read
status='ok' and would survive a future `WHERE status='ok'` display filter.

DRY-RUN IS THE DEFAULT. Without --apply this script only reads
information_schema and prints what it would do.

Usage:
    export DB_HOST=... DB_PORT=5432 DB_NAME=... DB_USER=... DB_PASSWORD=...
    python pg_migrate.py                     # inspect + plan (no writes)
    python pg_migrate.py --apply             # backup, then migrate
    python pg_migrate.py --apply --no-backup
    python pg_migrate.py --apply --backup-dir ./pg_backup_20260716
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pg_mirror
from migrate import EXTRA_COLUMNS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Actually run the migration (default is dry-run)")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip the CSV backup before altering (not recommended)")
    parser.add_argument("--backup-dir", type=Path, default=None,
                        help="Backup folder (default pg_backup_<UTC timestamp>/)")
    args = parser.parse_args()

    print(f"Target: {pg_mirror.config_summary()}", file=sys.stderr)
    conn = pg_mirror.connect_from_env()
    try:
        # ---------- inspect ----------
        print("\n=== Current state ===")
        for table in pg_mirror.TARGET_TABLES:
            have = pg_mirror.existing_columns(conn, table)
            if not have:
                print(f"  {table}: TABLE MISSING")
                continue
            n = pg_mirror.table_count(conn, table)
            missing = pg_mirror.missing_columns(conn, table)
            idx = pg_mirror.dedup_index_exists(conn, table)
            print(f"  {table}: {n} rows, {len(have)} columns "
                  f"({len(missing)} hardening columns missing, "
                  f"dedup index {'present' if idx else 'missing'})")
        if pg_mirror.sources_table_exists(conn):
            keyed = pg_mirror.sources_sha1_index_exists(conn)
            print(f"  sources table: present (keyed on "
                  f"{'pdf_sha1' if keyed else 'pdf_filename — legacy'})")
        else:
            print("  sources table: missing")

        # ---------- plan ----------
        print("\n=== Plan ===")
        any_change = False
        for table in pg_mirror.TARGET_TABLES:
            if not pg_mirror.existing_columns(conn, table):
                print(f"  !! {table} missing — will NOT create material tables "
                      f"(they belong to the app); aborting would-be changes for it")
                continue
            missing = pg_mirror.missing_columns(conn, table)
            if missing:
                any_change = True
                print(f"  {table}: ADD {len(missing)} columns: "
                      + ", ".join(name for name, _ in missing))
            if not pg_mirror.dedup_index_exists(conn, table):
                any_change = True
                print(f"  {table}: CREATE UNIQUE INDEX {pg_mirror.dedup_index_name(table)} "
                      f"(partial, legacy rows unaffected)")
        if not pg_mirror.sources_table_exists(conn):
            any_change = True
            print("  CREATE TABLE sources (keyed on pdf_sha1)")
        elif not pg_mirror.sources_sha1_index_exists(conn):
            any_change = True
            print("  sources: DROP UNIQUE(pdf_filename), CREATE UNIQUE INDEX "
                  f"{pg_mirror.SOURCES_SHA1_INDEX} ON (pdf_sha1) — content hash is "
                  "the doc identity; same-basename PDFs no longer collide "
                  "(batch_ingest --pg refuses to run until this is applied)")
        if not any_change:
            print("  Nothing to do — schema already migrated.")

        if not args.apply:
            print("\nDry-run only. Re-run with --apply to execute.")
            return 0
        if not any_change:
            return 0

        # ---------- backup ----------
        if not args.no_backup:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out = args.backup_dir or Path(f"pg_backup_{stamp}")
            print(f"\nBacking up tables to {out}/ ...")
            counts = pg_mirror.backup_tables_csv(conn, out)
            for table, n in counts.items():
                print(f"  {table}.csv: {n} rows")

        # ---------- apply ----------
        print("\nApplying migration ...")
        added = pg_mirror.ensure_schema(conn)
        for table, cols in added.items():
            print(f"  {table}: added {len(cols)} columns"
                  + (f" ({', '.join(cols)})" if cols else " (already up to date)"))
        pg_mirror.check_schema(conn)
        print("Migration complete and verified. batch_ingest.py --pg is now allowed.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
