# Claude Code prompt — Extraction Hardening Phase

**How to use:** open a terminal in the AIM Composites **project folder** (the one with
`batch_ingest.py`), run `claude`, and paste everything below the line. This phase
reworks the *extraction pipeline* for correctness. It's independent of the
crawl→CSV connection phase, but designed to stay compatible with it (all DB/CSV
changes are additive). Verify every claim against the real files before editing; if
my description is wrong, stop and tell me.

---

## Context

The extraction pipeline today is `batch_ingest.py`: PDF → `call_gemini()` →
`flatten_extraction()` → `classify_material()` → `validate()` → SQLite
(`Polymers`/`Fibers`/`Composites_materials` + `sources`). The same Gemini
prompt/schema is **duplicated and drifted** across `batch_ingest.py`, the app repo's
`page_files/categorized/page6.py`, and `Backend/Pdf_DataExtraction.py`.

A review found the pipeline maximizes throughput while silently corrupting and
discarding data. This phase fixes that. **Priority order matters — do P0 before P1,
and run the eval harness (Task 9) after each P0 change.** Work in small commits;
do not push.

Read first: `batch_ingest.py` (whole file), `page6.py` and
`Backend/Pdf_DataExtraction.py` in the repo (for the duplicated schema/prompt), and
`pdf_crawler.py`'s `http_get()` (reuse its retry logic).

---

## Task 0 — Unify extraction into one module

Create **`extraction.py`** in the project folder as the single source of truth:

- `EXTRACTION_SCHEMA`, `EXTRACTION_PROMPT`, `PROMPT_VERSION = "2.0"`, `GEMINI_MODEL`.
- `extract_from_pdf(pdf_bytes, filename, api_key) -> Extraction` (raw model JSON →
  typed object).
- `verify_against_text(extraction, pdf_text) -> extraction` (Task 1).
- `to_rows(extraction, source_pdf, source_sha1) -> list[PropertyRow]`.
- `gemini_request(...)` with retry/backoff (Task 7).

Refactor `batch_ingest.py` to import from `extraction.py` instead of defining its own
copy. Leave a `# TODO: sync with extraction.py PROMPT_VERSION` note in the repo's
`page6.py` / `Pdf_DataExtraction.py` (different repo — don't edit cross-repo here, just
flag drift). Keep `temperature=0`.

---

## P0 — correctness (do these first)

### Task 1 — Ground every value in the source (highest priority)

Nothing today checks that extracted numbers actually appear in the PDF. Add per-property
grounding:

- Extend the schema so each property also returns `source_quote` (verbatim sentence or
  table cell as printed in the PDF) and `page` (int).
- After extraction, pull the PDF text with PyMuPDF (`fitz`, already used by `page6.py`):
  `page_text = [p.get_text() for p in fitz.open(stream=pdf_bytes, filetype="pdf")]`.
- For each property, normalize whitespace/unicode and check that `value_raw` (and ideally
  a chunk of `source_quote`) occurs in the text of the cited page (fall back to any page).
- If found → `status="ok"`. If not found → `status="unverified"` (do **not** drop, do
  **not** mark ok). Record which check failed in `flag_reason`.

**Done when:** every inserted row has `source_pdf`, `page`, `source_quote`, and a
`status`; values not locatable in the PDF are `unverified`, never `ok`.

### Task 2 — Extract multiple materials per PDF

The schema has a single `material_name` + one flat list, so multi-material papers get
collapsed (Material B's data filed under Material A). Change the schema to a list:

```jsonc
{
  "materials": [
    {
      "material_name": "string",
      "material_abbreviation": "string",
      "material_class": "Polymer | Fiber | Composite",   // see Task 5
      "trade_grade": "string",          // '' if absent
      "manufacturer": "string",         // '' if absent
      "matrix": "string",               // composites: '' otherwise
      "fiber": "string",                // composites
      "fiber_volume_fraction": "string",// composites, e.g. '55%'
      "properties": [ /* see Task 3 */ ]
    }
  ]
}
```

`to_rows()` iterates materials × properties. Promote `matrix`/`fiber`/`Vf` to real
columns instead of burying them in `comments`.

**Done when:** a datasheet/paper reporting ≥2 distinct materials produces ≥2 distinct
material records, each with only its own properties.

### Task 3 — Structured, unit-aware numeric values

`value` is free text and `validate()` regex-grabs the first number while ignoring the
unit entirely (so 1.2 GPa and 1.2 MPa are treated identically; "100–120" becomes 100).
Replace with structured fields per property:

```jsonc
{
  "section": "<enum, Task 8>",
  "property_name": "string",
  "value_raw": "string",      // exactly as printed, e.g. "≤ -18.0", "100–120"
  "value_num": 0.0,           // single value, else null
  "value_min": 0.0,           // range low, else null
  "value_max": 0.0,           // range high, else null
  "qualifier": "",            // one of '', '<', '<=', '>', '>=', '~', '±'
  "unit": "string",
  "test_condition": "string",
  "comments": "string",
  "source_quote": "string",   // Task 1
  "page": 0
}
```

Add unit normalization with **pint** (`pip install pint`):
- Define a canonical unit per property family (e.g. modulus→GPa, strength→MPa,
  temperature→°C, density→g/cm³, CTE→ppm/°C). Store `unit_canonical` and `value_si`
  (converted). If the unit can't be parsed or is dimensionally wrong for the property,
  set `status="unit_review"` rather than guessing.
- Rewrite `validate()` plausibility to range-check **`value_si` after conversion**, not
  the raw number. Broaden the property keyword synonyms (e.g. "modulus of elasticity"
  ≈ "tensile modulus"). Keep `value` populated (= `value_raw`) for backward compat with
  the CSV export and `page1.py`.

**Done when:** every numeric row has `value_num`/min/max + `unit_canonical` + `value_si`;
range flags fire only after unit conversion; a GPa-vs-MPa mix no longer false-flags.

### Task 4 — Stop deleting flagged rows

Today flagged rows are `continue`d — written to `review_queue.csv` and never inserted,
with no UI that reads them back. That's silent data loss. Instead:
- Insert **all** rows; carry `status` (`ok`/`unverified`/`out_of_range`/`unit_review`/
  `empty_value`/…) and `flag_reason`.
- `review_queue.csv` becomes a `SELECT … WHERE status != 'ok'` export, not a discard pile.
- (Optional, nice) add a tiny Streamlit review view or a `--promote review_queue.csv`
  path to re-admit corrected rows as `ok`.

**Done when:** no row is dropped solely for failing validation; flagged rows are in the
DB with a status and queryable.

### Task 5 — Trustworthy classification

`classify_material()` is buggy: it claims to use section names but does
`haystack += row.material_name` in the loop (repeats the name N times, never uses
`row.section`), then substring-matches. One miss routes a whole material to the wrong
table → invisible in the right search tab.
- Use the LLM's `material_class` field (constrained enum, Task 2) as the primary signal —
  it already read the paper.
- Keep a fixed keyword classifier as a **fallback only** when the field is missing, and
  fix it to actually use sections + `fiber_volume_fraction` presence (Vf ⇒ Composite).

**Done when:** classification comes from the model with a deterministic fallback; the
haystack bug is gone; a glass-filled polymer vs a CF/PEEK laminate route correctly.

### Task 6 — Per-row provenance + smarter dedup

Two problems: property rows have no link to their source PDF (the `sources` table exists
but isn't joined to rows), and `already_inserted()` keys on
`(material_abbreviation, property_name, test_condition)` — ignoring value and source, so a
second paper's identical-property measurement is dropped (you *want* independent repeats).
- Add `source_pdf`, `source_sha1` columns to every property row.
- Dedup grain = `(source_sha1, material_key, section, property_name, test_condition,
  value_raw)`: skip only a true re-ingest of the *same measurement from the same PDF*;
  keep the same property from a *different* PDF.
- Use a stable `material_key` (normalized lowercased `material_name`, fallback to abbr),
  not the lossy auto-abbreviation, as material identity.

**Done when:** re-ingesting a PDF adds 0 rows; ingesting a second PDF with the same
property keeps both, each traceable to its source.

---

## P1 — robustness & ops

### Task 7 — Retry/backoff on the Gemini call
The crawler retries 429/5xx with backoff + Retry-After; the ingester uses a bare
`requests.post`, so transient errors lose whole PDFs. Port `pdf_crawler.http_get()`'s
retry pattern into `extraction.gemini_request()`.

### Task 8 — Constrain `section` to an enum
Free-text `section` drifts ("Mechanical" vs "Mechanical Properties") and breaks
`page1.py`'s exact-match section filter. Make `section` a `responseSchema` enum aligned
with `page6.py`'s `PROPERTY_CATEGORIES` (Mechanical, Thermal, Electrical, Physical,
Optical, Rheological, Processing, Descriptive, Composition/Reinforcement,
Architecture/Structure). Normalize on ingest.

### Task 9 — Eval harness (build early, it gates the P0 work)
Add `eval/` with 3–5 hand-labeled PDFs (`gold/<name>.pdf` + `gold/<name>.json` of expected
materials/properties/values). Add `python -m eval` that runs extraction and scores
precision/recall on `(material, property)` presence and value-within-tolerance, writing a
JSON report. Run it after each P0 task to prove changes help, not hurt. (You already have
`extraction_evaluation_charts.html` / `gemini_vs_gpt4o_charts.html` in the folder — reuse
that labeled data if it's reusable.)

### Task 10 — Large/scanned PDFs
- If `len(pdf_bytes)` is large (≳15 MB) or page count is high, upload via the Gemini
  **File API** and reference it, instead of base64-inlining (inline blows the request
  size limit; the crawler allows PDFs up to 40 MB).
- Detect scanned/image-only PDFs (avg extractable text per page below a threshold) →
  `status="scanned_no_text"` and skip, or route through OCR (`ocrmypdf`/`pytesseract`) if
  installed. Don't let a scanned PDF produce confident-looking garbage.

---

## DB migration (non-destructive)

Add a `--migrate` path (or a one-off `migrate.py`) that `ALTER TABLE`s the existing
`Polymers`/`Fibers`/`Composites_materials` to add the new columns
(`material_class, trade_grade, manufacturer, matrix, fiber, fiber_volume_fraction,
value_raw, value_num, value_min, value_max, qualifier, unit_canonical, value_si,
source_pdf, source_sha1, page, source_quote, status, flag_reason, model,
prompt_version, extracted_at`). Keep the original columns (`value`, etc.) populated so
the CSV export and `page1.py` keep working. Back up the `.sqlite` first.

---

## Constraints

- **No hardcoded API keys** — `GEMINI_API_KEY` from env only. (Reminder: a real key is
  committed in the repo's `Backend/Pdf_DataExtraction.py` — flag it for rotation; don't
  edit that repo here.)
- All DB/CSV changes **additive** — must not break the connection phase or `page1.py`.
- Reuse existing code (`fitz`, `pdf_crawler.http_get`); don't add heavy deps beyond
  `pint` (and optional `ocrmypdf`).
- Keep `temperature=0`; bump `PROMPT_VERSION` on any prompt change and store it per row.

## Acceptance tests (show me)

1. A 2-material datasheet → 2 material records, no cross-contamination.
2. Every numeric row has `value_num`/`unit_canonical`/`value_si`; a GPa/MPa mix no longer
   false-flags; ranges parse into min/max.
3. Every row has `source_pdf` + `page`; rows whose value isn't in the PDF are
   `unverified`, not `ok`.
4. Flagged rows are in the DB (status != 'ok'), not deleted; `review_queue.csv` is a view
   over them.
5. Re-ingest same PDF → 0 new rows; second PDF with same property → both kept.
6. `python -m eval` reports precision/recall and improves (or holds) vs the pre-change
   baseline.
7. A scanned PDF → `scanned_no_text`, no fabricated rows.

## Deliverables

`extraction.py`, refactored `batch_ingest.py`, `migrate.py`, `eval/` harness + gold
scaffold, `EXTRACTION.md` documenting the new schema/columns/statuses, and a short note
listing manual follow-ups (rotate the committed key; sync the repo's `page6.py` to
`PROMPT_VERSION`). Commit in small steps; **don't push**.
