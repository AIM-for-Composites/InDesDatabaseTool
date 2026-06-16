# AIM Composites Materials Database — Extraction Pipeline

An end-to-end pipeline that builds a structured materials-property database from
the open literature and vendor datasheets. It **discovers** candidate PDFs on
the web, **extracts** structured material/property data from them with an LLM,
**grounds and validates** every value against the source text, and **mirrors**
the result into a queryable SQLite database with full provenance.

Built for the ME8930 course project. The headline design goal of the current
("hardening") phase: **never silently corrupt or drop data** — every value is
traceable to the sentence it came from, every number is unit-normalized before
it's judged, and every flagged row is kept with a reason rather than discarded.

---

## Table of contents

1. [Architecture at a glance](#architecture-at-a-glance)
2. [Data flow, step by step](#data-flow-step-by-step)
3. [The extraction module (`extraction.py`)](#the-extraction-module-extractionpy)
   - [Model JSON schema](#model-json-schema)
   - [Text grounding](#text-grounding)
   - [Unit normalization](#unit-normalization)
   - [Per-row status](#per-row-status)
   - [Classification](#classification)
   - [Provenance & dedup](#provenance--dedup)
   - [Large & scanned PDFs](#large--scanned-pdfs)
   - [Reliability](#reliability)
4. [Database schema](#database-schema)
5. [Setup](#setup)
6. [Usage](#usage)
7. [The eval harness](#the-eval-harness)
8. [What changed in the hardening phase](#what-changed-in-the-hardening-phase)
9. [Design decisions & rationale](#design-decisions--rationale)
10. [Limitations & follow-ups](#limitations--follow-ups)
11. [File map](#file-map)

---

## Architecture at a glance

```
 ┌─────────────────────────┐
 │  Source discovery        │   pdf_crawler.py · matweb_discovery.py
 │  (OpenAlex / arXiv /      │   generate_queries.py · crawler_ui.py
 │   Unpaywall / datasheets) │
 └────────────┬─────────────┘
              │  crawl_out/pdfs/*.pdf   (gitignored — regenerable)
              ▼
 ┌─────────────────────────┐
 │  Extraction              │   extraction.py  (Gemini structured output)
 │  PDF → typed materials   │   • value parsing (value_raw → num/min/max/qualifier)
 │  + properties            │   • text grounding (source_quote + page)
 │                          │   • unit normalization (pint → value_si / unit_canonical)
 └────────────┬─────────────┘
              │  Extraction object
              ▼
 ┌─────────────────────────┐
 │  Validate → dedup →      │   batch_ingest.py
 │  SQLite mirror           │   • insert ALL rows, each with a status
 │                          │   • source-aware dedup
 └────────────┬─────────────┘
              │
       ┌──────┴───────────────────────────┐
       ▼                                    ▼
 materials_mirror.sqlite            review_queue.csv
 (Polymers / Fibers /               (a VIEW: SELECT … WHERE status != 'ok')
  Composites_materials + sources)    run_report.json
```

Three big ideas distinguish this from a naive "LLM → database" dump:

- **Grounding.** A number is only trusted (`status='ok'`) if it can be located
  in the PDF's extracted text. Anything else is kept but marked `unverified`.
- **Unit awareness.** Values are parsed into structured numerics and converted
  to canonical/SI units *before* any plausibility check, so `1.2 GPa` and
  `1200 MPa` are understood to be the same thing.
- **No data loss.** Failing validation flags a row; it does not delete it. The
  review queue is a *query over the database*, not a separate discard pile.

---

## Data flow, step by step

1. **Discover** — `generate_queries.py` builds a search-query set;
   `pdf_crawler.py` queries OpenAlex / arXiv / Unpaywall and scrapes vendor
   datasheet pages, applying a URL pre-filter, polite per-domain throttling, and
   bounded retry/backoff. PDFs land in `crawl_out/pdfs/` with a persisted
   seen-set so reruns are incremental.
2. **Extract** — for each PDF, `extraction.extract_from_pdf()` sends it to
   Gemini with a structured-output schema (`temperature=0`) and returns a typed
   `Extraction` (a list of `Material`, each with a list of `Property`).
3. **Ground** — `extraction.verify_against_text()` pulls the PDF's text with
   PyMuPDF and checks each value (and a chunk of its `source_quote`) against the
   cited page, falling back to any page. It also folds in unit and plausibility
   checks, assigning a `status` per property.
4. **Flatten** — `extraction.to_rows()` turns the nested `Extraction` into flat
   `PropertyRow`s (materials × properties), attaching provenance
   (`source_pdf`, `source_sha1`, `page`, `source_quote`) and the resolved
   `material_class`.
5. **Persist** — `batch_ingest.py` routes each row to `Polymers` / `Fibers` /
   `Composites_materials` by class, skips true re-ingests via a source-aware
   dedup key, and inserts everything else — flagged rows included, carrying
   their `status`/`flag_reason`.
6. **Report** — `review_queue.csv` is exported as a view over all
   `status != 'ok'` rows; `run_report.json` summarizes throughput, insert/flag
   rates, and error kinds.

---

## The extraction module (`extraction.py`)

`extraction.py` is the **single source of truth** for the Gemini prompt/schema
and all post-extraction processing. (Historically this logic was copy-pasted and
drifted across `batch_ingest.py` and the companion Streamlit app repo; it is now
centralized here, with `PROMPT_VERSION = "2.0"` stamped on every row.)

### Model JSON schema

The model returns a **list of materials**, each with structured, unit-aware
properties plus a verbatim source quote and page number:

```jsonc
{
  "materials": [
    {
      "material_name": "string",
      "material_abbreviation": "string",
      "material_class": "Polymer | Fiber | Composite",   // constrained enum
      "trade_grade": "string",            // '' if absent
      "manufacturer": "string",           // '' if absent
      "matrix": "string",                 // composites; '' otherwise
      "fiber": "string",                  // composites
      "fiber_volume_fraction": "string",  // composites, e.g. '55%'
      "properties": [
        {
          "section": "<enum>",            // Mechanical, Thermal, Electrical, …
          "property_name": "string",
          "value_raw": "string",          // exactly as printed: "≤ -18.0", "100–120"
          "value_num": 0.0,               // single value, else null
          "value_min": 0.0,               // range low, else null
          "value_max": 0.0,               // range high, else null
          "qualifier": "",                // '', '<', '<=', '>', '>=', '~', '±'
          "unit": "string",
          "test_condition": "string",
          "comments": "string",
          "source_quote": "string",       // verbatim sentence/cell from the PDF
          "page": 0                       // 1-based page number
        }
      ]
    }
  ]
}
```

`section` and `material_class` are **enums** in the response schema, so the model
can't drift into "Mechanical Properties" vs "Mechanical". Free-text variants are
still normalized on ingest as a safety net.

If the model omits the structured numerics, `extraction.py` backfills
`value_num`/`value_min`/`value_max`/`qualifier` from `value_raw` with its own
parser (handles ranges `100–120` / `100 to 120`, qualifiers `≤ -18`, plus-minus
`5.0 ± 0.2`, thousands separators, and scientific notation). The old
single-material shape (`material_name` + flat `mechanical_properties`) is still
accepted for backward compatibility.

### Text grounding

`verify_against_text()` normalizes whitespace/unicode and checks that each
`value_raw` (or a chunk of its `source_quote`) appears in the text of the cited
page, falling back to any page. Found → eligible for `ok`; not found →
`unverified`. This is the core anti-hallucination guard: a number the model
invented that isn't anywhere in the document can never be marked `ok`.

### Unit normalization

Each property is matched to a **family** by keyword (e.g. "tensile modulus",
"glass transition", "density"). The value is converted with [pint] to a canonical
unit and to SI base units, stored as `unit_canonical` and `value_si`.
Plausibility ranges are checked on the **converted** value:

| Family | Canonical unit | Plausibility range |
|--------|----------------|--------------------|
| Tensile / flexural / shear / storage modulus | GPa | 0.001 – 1000 |
| Tensile / flexural / compressive / shear strength | MPa | 0.5 – 10000 |
| Glass transition / melting / crystallization / decomposition / HDT | °C | per-property (e.g. Tg −150…600) |
| CTE | ppm/°C | −50 – 500 |
| Density | g/cm³ | 0.1 – 12 |
| Elongation | % | 0.001 – 2000 |

If a unit can't be parsed or is dimensionally wrong for the family (e.g. a
modulus reported in °C), the row gets `status='unit_review'` instead of a guessed
number. Properties outside these families are stored as-is with no range check.

### Per-row status

Every inserted row carries a `status` and, when not `ok`, a `flag_reason`.
Precedence when several issues apply:
`empty_value` → `unverified` → `unit_review` → `out_of_range` → `ok`.

| status | meaning |
|--------|---------|
| `ok` | grounded in the PDF, unit sane, value in range |
| `unverified` | value not locatable in the PDF text — never silently trusted |
| `unit_review` | unit missing or dimensionally wrong for the property family |
| `out_of_range` | value implausible *after* unit conversion |
| `empty_value` | value is blank / `n/a` / `-` / placeholder |

Document-level outcomes that produce **no** rows: `scanned_no_text` (image-only
PDF), `empty_extraction`, `skipped_seen_sha1`.

### Classification

`classify_material()` uses the model's `material_class` enum as the primary
signal (it already read the paper). A deterministic keyword fallback runs only
when that field is missing/invalid: a non-empty `fiber_volume_fraction` ⇒
Composite; else composite keywords ⇒ Composite; else fiber keywords ⇒ Fiber;
else Polymer. This replaced a buggy classifier that repeated `material_name` N
times and never actually read the section names.

### Provenance & dedup

Every property row carries `source_pdf`, `source_sha1`, `page`, and
`source_quote`. The dedup grain is:

```
(source_sha1, material_key, section, property_name, test_condition, value_raw)
```

`material_key` is the normalized lowercased material name (fallback:
abbreviation). Re-ingesting the same PDF adds 0 rows; the same property measured
in a *different* PDF is kept as an independent repeat (valuable for
cross-checking).

### Large & scanned PDFs

- PDFs over ~15 MB or ~80 pages are uploaded via the **Gemini File API** and
  referenced by URI, instead of base64-inlining them (which would exceed the
  request-size limit).
- A PDF whose average extractable text per page is below a threshold is treated
  as scanned/image-only → `scanned_no_text`, and skipped, so it can't produce
  confident-looking garbage. (OCR via `ocrmypdf`/`pytesseract` is a possible
  future route.)

### Reliability

`gemini_request()` retries `429`/`5xx` and connection errors with bounded
exponential backoff, honoring `Retry-After` (mirrored from the crawler's
`http_get()`), so a transient blip doesn't lose a whole PDF. `temperature=0`
throughout for reproducibility.

---

## Database schema

Three property tables (`Polymers`, `Fibers`, `Composites_materials`) share one
column layout, plus a `sources` table tracking ingested PDFs. The legacy columns
(`value`, `unit`, `english`, `section`, …) remain populated for backward
compatibility (`value` mirrors `value_raw`); the hardening phase adds these
columns **additively**:

```
material_key, material_class, trade_grade, manufacturer, matrix, fiber,
fiber_volume_fraction, value_raw, value_num, value_min, value_max, qualifier,
unit_canonical, value_si, source_pdf, source_sha1, page, source_quote,
status, flag_reason, model, prompt_version, extracted_at
```

`migrate.py` adds these to an existing database with `ALTER TABLE ADD COLUMN`
(it backs up the `.sqlite` first and is idempotent). A fresh database created by
`batch_ingest.py` gets all columns up front. See **EXTRACTION.md** for the full
reference.

---

## Setup

```bash
pip install -r requirements.txt
playwright install chromium          # only needed for matweb_discovery.py
export GEMINI_API_KEY=...            # required for extraction / ingest / eval
```

- **Python 3.11+.**
- The API key is read from the environment only (`GEMINI_API_KEY` or
  `GOOGLE_API_KEY`) — never hardcode it.
- Key dependencies: `requests`, `PyMuPDF` (text grounding + scanned detection),
  `pint` (unit normalization); `playwright` + `ddgs` for discovery.

---

## Usage

```bash
# 1. Crawl PDFs into crawl_out/pdfs/
python pdf_crawler.py

# 2. Ingest a folder of PDFs into the SQLite mirror
python batch_ingest.py --input crawl_out/pdfs --db materials_mirror.sqlite \
    --review review_queue.csv --report run_report.json
python batch_ingest.py --input crawl_out/pdfs --db materials_mirror.sqlite --limit 5   # test run

# Migrate an existing DB to the latest columns (backs up the .sqlite first)
python batch_ingest.py --migrate --db materials_mirror.sqlite
#   or: python migrate.py --db materials_mirror.sqlite

# Re-admit corrected rows from a review CSV as status='ok'
python batch_ingest.py --promote review_queue.csv --db materials_mirror.sqlite

# 3. Score extraction quality against the gold set
python -m eval --report eval_report.json
python -m eval --selfcheck            # offline harness check, no API key
```

---

## The eval harness

`eval/` scores extraction against hand-labeled gold PDFs. For each gold case it
runs `extract_from_pdf` + `verify_against_text` and reports:

- **material presence** — precision/recall of expected materials found
- **(material, property) presence** — precision/recall of expected properties
- **value-within-tolerance** — fraction of matched properties whose value is
  within `tolerance_pct` of gold, **compared in SI** so a ksi/MPa or GPa/MPa unit
  difference is not counted as a value miss

Two gold cases ship:

| case | exercises |
|------|-----------|
| `tc920_pc_abs` | three materials in one datasheet (fiberglass UD tape, carbon UD tape, neat resin) — multi-material separation, ksi/Msi vs MPa/GPa, a Tg range |
| `tc910_pa6` | a single carbon/PA6 composite — ksi/Msi → MPa/GPa normalization |

`python -m eval --selfcheck` validates the scoring logic offline (no API) by
feeding it a near-perfect prediction with some values deliberately expressed in
different units, proving the SI-based value comparison. Run `python -m eval`
after each change and diff against a baseline with `--baseline prev.json`.
See `eval/gold/README.md` for the gold JSON schema and how to add cases.

---

## What changed in the hardening phase

| Before | After |
|--------|-------|
| One `material_name` + one flat property list → multi-material papers collapsed | Schema is a `materials[]` list; each material keeps only its own properties |
| `value` was free text; the validator grabbed the first number and ignored the unit (`1.2 GPa` ≡ `1.2 MPa`; `100–120` → `100`) | Structured `value_num`/`min`/`max`/`qualifier`; pint → `unit_canonical` + `value_si`; range checks run *after* conversion |
| Nothing verified that extracted numbers were in the PDF | Every value grounded against the PDF text (`source_quote` + `page`); not found → `unverified`, never `ok` |
| Flagged rows were dropped — written to a CSV and never inserted | All rows inserted with a `status` + `flag_reason`; `review_queue.csv` is a `SELECT … WHERE status != 'ok'` view; `--promote` re-admits fixes |
| Buggy classifier (looped `material_name`, never read `section`); one miss misrouted a whole material | Model `material_class` enum first; deterministic keyword fallback (Vf ⇒ Composite) |
| Dedup ignored value & source → a second paper's repeat was dropped; rows had no link to their PDF | Source-aware dedup grain; every row carries `source_pdf`/`source_sha1`/`page`/`source_quote` |
| Bare `requests.post`, no retry → transient errors lost whole PDFs | `gemini_request()` with retry/backoff + `Retry-After` |
| Free-text `section` drifted | `section` + `material_class` are enums, normalized on ingest |
| Large PDFs base64-inlined (size-limit risk); scanned PDFs produced garbage | >15 MB / 80 pages → File API; image-only → `scanned_no_text`, skipped |
| `matrix`/`fiber`/`Vf` buried in `comments` | Promoted to real columns |

---

## Design decisions & rationale

- **Grounding over trust.** LLM extraction is fast but confabulates. Rather than
  post-hoc spot-checking, every value is checked against the source text at
  ingest. The cost is a `source_quote` + `page` per property; the benefit is that
  "is this number real?" becomes a column (`status`) instead of a manual audit.
- **Normalize before judging.** Plausibility checks on raw numbers are unit-blind
  and produce false flags (a modulus in MPa looks 1000× too small against a GPa
  range). Converting to canonical/SI first makes the checks meaningful and lets
  data from mixed-unit datasheets coexist.
- **Flag, don't delete.** A flagged value is still evidence. Keeping it with a
  reason (and making the review queue a query) means nothing is lost and
  corrections are a status update, not a re-extraction.
- **One schema, one module.** Three drifting copies of the prompt/schema meant
  three different behaviors. Centralizing in `extraction.py` with a
  `PROMPT_VERSION` makes drift visible and changes atomic.
- **Deterministic where possible.** `temperature=0`, enum-constrained fields, and
  a deterministic classification fallback keep runs reproducible and debuggable.

---

## Limitations & follow-ups

See **FOLLOWUPS.md** for the full list. Highlights:

- **A live eval baseline** (`python -m eval`) requires `GEMINI_API_KEY`. All the
  deterministic logic (parsing, units, grounding, classification, dedup,
  migration) is unit-tested and passing offline; end-to-end numbers need the key.
- **Companion app repo** still holds older, drifted copies of the prompt/schema
  (`page6.py`, `Backend/Pdf_DataExtraction.py`) — they need syncing to
  `PROMPT_VERSION 2.0`, and a real API key committed there must be rotated.
- **Scanned PDFs** are detected and skipped, not OCR'd — wiring `ocrmypdf` is a
  natural extension.
- **Gold set** has two cases; 3–5 spanning polymer/fiber/composite would give
  more robust precision/recall.

---

## File map

| Path | Purpose |
|------|---------|
| `pdf_crawler.py` | Crawl OpenAlex / arXiv / Unpaywall / vendor datasheets; retry/backoff; download to `crawl_out/` |
| `matweb_discovery.py` | Browser + DuckDuckGo source discovery |
| `generate_queries.py` | Build the search-query set |
| `crawler_ui.py` | Streamlit UI for the crawler |
| `extraction.py` | Single source of truth: Gemini prompt/schema + grounding, unit normalization, classification, dedup |
| `batch_ingest.py` | Batch driver: PDFs → extraction → validation → SQLite mirror + review queue |
| `migrate.py` | Non-destructive DB column migration |
| `eval/` | Eval harness (`scoring.py`, `__main__.py`) + hand-labeled gold PDFs |
| `EXTRACTION.md` | Detailed schema / column / status reference |
| `FOLLOWUPS.md` | Manual follow-ups (key rotation, prompt sync, eval baseline) |
| `requirements.txt` | Python dependencies |
| `extraction_evaluation_charts.html`, `gemini_vs_gpt4o_charts.html` | Extraction-evaluation charts from earlier benchmarking |
| `claude_code_*_prompt.md` | Phase specs (crawl→CSV connection; extraction hardening) |

[pint]: https://pint.readthedocs.io/
