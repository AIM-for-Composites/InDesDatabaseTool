# AIM Composites Materials Database

> **About this branch.** This is the AIM extraction/ingestion pipeline. It has
> its own history and shares no commits with this repository's other branches —
> it is a separate codebase kept here so the team has one place to look. The
> autonomous agent that runs this pipeline on a schedule lives in the Hugging
> Face Space `aim4composites/AutonomousAgent`; the materials site lives in
> `aim4composites/MaterialsDatabase`.
>
> Two things are deliberately absent: the `eval/gold/*.pdf` source documents
> (third-party papers and datasheets we cannot redistribute — the hand-written
> `.json` labels are here) and the `paper/` manuscript drafts.

Tooling for an autonomous materials-database pipeline: discover materials
literature/datasheets on the web, extract structured property data from the
PDFs with an LLM, validate and ground it, and mirror it into a queryable
database. Built for the ME8930 course project.

## Pipeline

```
 pdf_crawler.py / matweb_discovery.py     1. discover + download source PDFs
            │
            ▼
 extraction.py  (Gemini structured output)  2. PDF text → typed materials + properties
            │   • grounds every value in the PDF text
            │   • unit-normalizes (pint) → value_si / unit_canonical
            ├──────────────────────────────┐
            │                              ▼
            │              figures.py  (--figures, opt-in)   2b. PDF figures → estimates
            │                • harvest: rasters + vector plots → PNGs (+ caption, page, bbox)
            │                • ONE vision call/PDF classifies; one call/plot mines salient values
            │                • origin='figure', status='figure_estimate' — never 'ok' unpromoted
            ▼                              │
 batch_ingest.py  ◄────────────────────────┘ 3. validate → dedup (grain + origin) → SQLite mirror
            │                                  flagged rows kept with a status; figures table
            ▼
 review_queue.csv  (SELECT WHERE status != 'ok', incl. every figure row)   +   run_report.json
            │
            └─ --promote  (the only path from figure_estimate to ok)
```

See **[EXTRACTION.md](EXTRACTION.md)** for the extraction schema, per-row
statuses, unit handling, and DB columns, and **[FIGURES.md](FIGURES.md)** for
figure harvesting, classification, mining, the `figures` table and the cost
model.

## Components

| File | Purpose |
|------|---------|
| `pdf_crawler.py` | Crawl OpenAlex / arXiv / Unpaywall / vendor datasheets; polite retry/backoff; download PDFs to `crawl_out/` |
| `matweb_discovery.py` | Browser + DuckDuckGo source discovery |
| `generate_queries.py` | Build the search-query set |
| `crawler_ui.py` | Streamlit UI for the crawler |
| `extraction.py` | **Single source of truth** for the Gemini prompt/schema + all post-extraction processing (grounding, unit normalization, classification, dedup) |
| `figures.py` | **Single source of truth for figures**: PyMuPDF harvest (raster + vector, captions, junk filters), one-call-per-PDF vision classification, per-plot mining → `origin='figure'` estimate rows |
| `batch_ingest.py` | Batch driver: PDFs → extraction (→ `--figures` stage) → validation → SQLite mirror + review queue |
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

# 2b. also mine figures (plots / table images) — estimates, quarantined until promoted
python batch_ingest.py --figures --input crawl_out/pdfs --db materials_mirror.sqlite
python batch_ingest.py --figures --no-figure-mining --input crawl_out/pdfs   # harvest+classify only

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
been applied — re-run it after pulling a change that extends
`migrate.EXTRA_COLUMNS` (the figure-mining phase added `origin`/`figure_id`).
See `pg_mirror.py` for details.

## Notes

- `crawl_out/` (downloaded PDFs, crawl state) is gitignored — it's regenerable.
- Open follow-ups (incl. a key to rotate in the companion app repo and a prompt
  version to sync) are tracked in **[FOLLOWUPS.md](FOLLOWUPS.md)**.
