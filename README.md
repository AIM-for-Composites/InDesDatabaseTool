# AIM Composites Materials Database

Tooling for an autonomous materials-database pipeline: discover materials
literature/datasheets on the web, extract structured property data from the
PDFs with an LLM, validate and ground it, and mirror it into a queryable
database. Built for the ME8930 course project.

## Pipeline

```
 pdf_crawler.py / matweb_discovery.py     1. discover + download source PDFs
            │
            ▼
 extraction.py  (Gemini structured output)  2. PDF → typed materials + properties
            │   • grounds every value in the PDF text
            │   • unit-normalizes (pint) → value_si / unit_canonical
            ▼
 batch_ingest.py                            3. validate → dedup → SQLite mirror
            │                                  flagged rows kept with a status
            ▼
 review_queue.csv  (SELECT WHERE status != 'ok')   +   run_report.json
```

See **[EXTRACTION.md](EXTRACTION.md)** for the extraction schema, per-row
statuses, unit handling, and DB columns.

## Components

| File | Purpose |
|------|---------|
| `pdf_crawler.py` | Crawl OpenAlex / arXiv / Unpaywall / vendor datasheets; polite retry/backoff; download PDFs to `crawl_out/` |
| `matweb_discovery.py` | Browser + DuckDuckGo source discovery |
| `generate_queries.py` | Build the search-query set |
| `crawler_ui.py` | Streamlit UI for the crawler |
| `extraction.py` | **Single source of truth** for the Gemini prompt/schema + all post-extraction processing (grounding, unit normalization, classification, dedup) |
| `batch_ingest.py` | Batch driver: PDFs → extraction → validation → SQLite mirror + review queue |
| `migrate.py` | Non-destructive DB migration (also `batch_ingest.py --migrate`): hardening columns + `sources` re-key on `pdf_sha1` |
| `pg_mirror.py` / `pg_migrate.py` | Postgres backend for `--pg` (the DB the HF Space reads) + its dry-run-default migration |
| `eval/` | Eval harness scoring extraction against hand-labeled gold PDFs (`python -m eval`; `--selfcheck` and `--gold-check` run offline) |
| `tests/` | pytest regression suite over the pure logic (value parsing, unit families, grounding, dedup/migration, transport, crawler state) — `python -m pytest tests/` |

## Setup

```bash
pip install -r requirements.txt
playwright install chromium          # only needed for matweb_discovery.py
export GEMINI_API_KEY=...            # required for extraction/ingest/eval
```

Python 3.11+. The API key is read from the environment only — never hardcode it.

## Usage

```bash
# 1. crawl PDFs
python pdf_crawler.py                 # writes crawl_out/pdfs/

# 2. ingest into the SQLite mirror
python batch_ingest.py --input crawl_out/pdfs --db materials_mirror.sqlite \
    --review review_queue.csv --report run_report.json

# migrate an existing DB to the latest columns (backs up .sqlite first)
python batch_ingest.py --migrate --db materials_mirror.sqlite

# 3. score extraction quality against the gold set
python -m eval --report eval_report.json
python -m eval --selfcheck            # offline check of the scorer, no API key needed
python -m eval --gold-check           # offline: every gold value/alias is in its PDF text

# 4. regression tests (offline, ~2 s)
python -m pytest tests/
```

### Postgres mode (feed the HF Space's database)

`batch_ingest.py --pg` writes the same rows into the shared Postgres that the
[HF Space](https://huggingface.co/spaces/aim4composites/MaterialsDatabase)
reads (legacy columns stay populated, so the Space needs no changes). Config
comes from the environment — the same names the Space uses: `DB_HOST`,
`DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` (or a single `DATABASE_URL`);
`DB_SSLMODE` defaults to `require`.

```bash
pip install "psycopg[binary]"
python pg_migrate.py                  # one-time, DRY-RUN: shows planned changes
python pg_migrate.py --apply          # CSV-backs-up the tables, then adds columns/indexes
python batch_ingest.py --pg --input crawl_out/pdfs   # ingest straight to Postgres
```

`pg_migrate.py` is additive only (new columns, a partial dedup index, a
`sources` table); `batch_ingest.py --pg` refuses to run until the migration has
been applied. See `pg_mirror.py` for details.

## Notes

- `crawl_out/` (downloaded PDFs, crawl state) is gitignored — it's regenerable.
- Open follow-ups (incl. a key to rotate in the companion app repo and a prompt
  version to sync) are tracked in **[FOLLOWUPS.md](FOLLOWUPS.md)**.
