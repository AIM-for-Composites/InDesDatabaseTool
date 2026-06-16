# Claude Code prompt — connect the web scraper to the PDF scraper

**How to use:** open a terminal in your AIM Composites **project folder** (the one
with `pdf_crawler.py`, `batch_ingest.py`, etc.), run `claude`, and paste everything
below the line into it. It's written so Claude Code verifies the facts against the
real files before changing anything.

---

## Goal

Wire my existing scripts into **one standalone command-line pipeline** that runs end
to end: discover → download PDFs → extract with Gemini → load into the database the
live Streamlit app actually reads. Today the crawler and the ingester exist but
nothing runs them together, and ingested data lands in a SQLite file the app never
opens. Close that gap. Keep it standalone in this project folder — the only thing you
touch in the app repo is its `data/*.csv` files.

Before writing code, read the files named below and confirm my description matches.
If anything differs, tell me before proceeding.

## What already exists (don't reinvent it)

All in this project folder unless noted:

- **`generate_queries.py`** — asks Gemini for fresh search queries; with `--run` it
  already does `import pdf_crawler; pdf_crawler.main(crawl_argv)`. Reads `GEMINI_API_KEY`.
- **`matweb_discovery.py`** — Playwright + DuckDuckGo discovery → `matweb_candidates.csv`
  (columns include `title,pdf_url`). Optional; needs `playwright install chromium`.
- **`pdf_crawler.py`** — multi-source PDF discovery/download (OpenAlex, Semantic
  Scholar, arXiv, Unpaywall, datasheet seeds, and `--from-csv`). Entry point is
  `main(argv: list[str] | None) -> int`. Output:
  - `<out>/pdfs/*.pdf`  (verified `%PDF`, sha256-deduped)
  - `<out>/sources.csv` (provenance: filename,title,doi,url,year,source,sha256,query,bytes)
  - `<out>/state.json`  (seen URLs/hashes — reruns are incremental)
- **`batch_ingest.py`** — folder of PDFs → Gemini extract → validate → dedup →
  classify (Polymer/Fiber/Composite) → SQLite. CLI: `--input --db --review --report
  --limit`. Reads `GEMINI_API_KEY` (or `GOOGLE_API_KEY`). Writes tables
  `Polymers`, `Fibers`, `Composites_materials` (cols: material_name,
  material_abbreviation, section, property_name, value, unit, english,
  test_condition, comments) and a `sources` table (pdf_filename, pdf_sha1,
  ingested_at, material_class, material_abbreviation). Also `review_queue.csv` +
  `run_report.json`. NOTE: it only exposes `main()` via argparse — see refactor below.
- **`crawler_ui.py`** — a stdlib local web UI that shells out to the crawler. Leave
  it as-is (it's the future "button" path; we're doing CLI now).
- **The app repo** (`github.com/AIM-for-Composites/InDesDatabaseTool`, branch
  `agentic`) — Streamlit app. The search page `page_files/categorized/page1.py`
  reads CSVs from `data/` and shows them. The upload page (`page6.py`) extracts a
  single PDF with the *same* Gemini schema but only stores it in `st.session_state`
  — it is never written to disk, so it vanishes when the session ends.

**The existing seam:** `pdf_crawler.py`'s `<out>/pdfs/` folder is literally designed
to be `batch_ingest.py --input`. That half works. The missing half is getting
`batch_ingest`'s SQLite into the app's `data/*.csv`, plus a single runner.

## The app's data contract (read `page1.py` to confirm)

`page1.py` loads one CSV per material class and concatenates session uploads:

```
"Composites": "data/Composites_material_data.csv"
"Polymers":   "data/polymers_material_data.csv"   # <-- lowercase 'p'
"Fibers":     "data/Fibers_material_data.csv"
```

The three CSVs do **not** share a column order — match each file's existing header
exactly when appending:

- `Composites_material_data.csv`: `source_file, material_name, material_abbreviation,
  section, property_name, value, unit, english, test_condition, comments,
  source_file_excel`
- `Polymers_material_data.csv`: `material_abbreviation, material_name, section,
  property_name, value, english, comments, unit, test_condition, source_file`
- `Fibers_material_data.csv`: `material_name, material_abbreviation, section,
  property_name, value, unit, english, test_condition, comments, source_file`

## What to build

A new **`pipeline.py`** in this project folder, runnable as `python pipeline.py ...`,
that chains the stages and adds the missing CSV-export stage:

1. **(optional) generate queries** — `--generate N` reuses `generate_queries.py`.
   Default off; otherwise use `--queries "..." "..."` or the crawler's built-in set.
2. **(optional) MatWeb discovery** — `--matweb` runs `matweb_discovery.py` first and
   feeds its `matweb_candidates.csv` to the crawler via `--from-csv`. Default off.
3. **crawl** — call `pdf_crawler.main([...])` into a run directory (default
   `./crawl_out`). Pass through `--max-per-query`, `--max-total`, `--skip-datasheets`,
   `--skip-academic`.
4. **ingest** — run `batch_ingest` over `<out>/pdfs` → `materials_mirror.sqlite`.
5. **export → CSV (NEW, this is the connection)** — read the three SQLite tables and
   **append only new rows** into the matching `data/*.csv` in the app repo, preserving
   each file's exact header order (above). Fill `source_file` from the `sources`
   table (join on material_abbreviation / sha); leave `source_file_excel` blank for
   crawled rows. De-dup at the CSV level on
   `(material_abbreviation, section, property_name, value, test_condition)` so reruns
   add nothing. Write UTF-8.

Default `python pipeline.py` runs stages 3→5. Flags add 1 and 2.

Key flags: `--out ./crawl_out`, `--db <path off OneDrive>`,
`--csv-out <path to InDesDatabaseTool/data>` (required for stage 5),
`--max-total N`, `--dry-run` (do everything except write CSVs; print what would change).

## Refactor allowed (keep it minimal)

- In `batch_ingest.py`, extract the body of `main()` into a callable like
  `run_ingest(input_dir, db_path, review_path, report_path, limit=None) -> dict`
  and have `main()` just parse args and call it, so `pipeline.py` can import it
  instead of shelling out. Don't change the extraction/validation logic.
- Factor the Gemini call + schema + prompt that are duplicated between
  `batch_ingest.py` and the app's `page6.py` / `Backend/Pdf_DataExtraction.py` into a
  shared `extraction.py` **only if** it's clean to do here; otherwise leave a TODO.

## Hard constraints

- **No hardcoded API keys.** Read `GEMINI_API_KEY` from the environment everywhere.
  Heads-up: a real Gemini key is currently committed in the repo at
  `page_files/categorized/Backend/Pdf_DataExtraction.py` (`GEMINI_KEY = "AIza..."`).
  Flag it; I'll rotate it and move it to env.
- **Idempotent.** Rely on the crawler's `state.json`, `batch_ingest`'s `sources`/sha
  dedup, and the new CSV-level dedup so re-running never duplicates work or rows.
- **Don't bloat the app repo.** The only write into it is `data/*.csv`. Everything
  else stays here.
- Put the SQLite DB somewhere **not** synced by OneDrive (OneDrive locking breaks
  SQLite writes) — make `--db` default to a local temp path and document it.

## Also fix while you're here

- `page1.py` asks for `data/polymers_material_data.csv` (lowercase) but the file is
  `Polymers_material_data.csv`. On Windows it works, but it'll break the Polymers tab
  on Linux/HF Spaces. Make the reference match the real filename.

## Acceptance test (show me this works)

1. `python pipeline.py --max-total 3 --skip-datasheets --csv-out <repo>/data`
   downloads ~3 PDFs, ingests them, and adds rows to the correct CSV by class.
2. Run it again → **0** new rows added (dedup proven).
3. `--dry-run` prints intended CSV changes and writes nothing.
4. In the repo, `streamlit run app.py` → Categorized Search shows at least one of the
   newly crawled materials.
5. `review_queue.csv` and `run_report.json` are produced and summarize counts.

## Deliverables

`pipeline.py`, the minimal `batch_ingest.py` refactor, the `page1.py` filename fix, a
short `PIPELINE.md` documenting the one command and its flags, and a one-line summary
of the committed-key issue. Don't commit/push anything — leave changes staged for me
to review.
