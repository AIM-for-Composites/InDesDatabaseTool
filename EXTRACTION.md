# Extraction pipeline (hardening phase)

All extraction logic now lives in **`extraction.py`** — the single source of
truth for the Gemini prompt, schema, grounding, unit normalization,
classification, and dedup keys. `batch_ingest.py` is just the driver + SQLite
mirror. The app repo's `page_files/categorized/page6.py` and
`Backend/Pdf_DataExtraction.py` still hold their own (now-stale) copies; they are
a *different* repo and are flagged for a follow-up sync (see
`FOLLOWUPS.md`).

```
pdf_bytes
  └─ extraction.extract_from_pdf()   PDF → Gemini structured JSON → Extraction
        ├─ scanned/large-PDF handling (PyMuPDF text probe, File API upload)
        └─ value parsing  (value_raw → value_num/min/max/qualifier)
  └─ extraction.verify_against_text()  ground each value in PDF text; set status
        └─ unit normalization (pint) + post-conversion plausibility
  └─ extraction.to_rows()            Extraction → list[PropertyRow]  (materials × properties)
  └─ batch_ingest.insert_row()       insert ALL rows (flagged carry a status)
  └─ batch_ingest.export_review_queue()  review_queue.csv = SELECT WHERE status != 'ok'
```

`PROMPT_VERSION` is currently **`"2.0"`** and is stored on every row. Bump it on
any prompt/schema change.

## Model JSON schema (`EXTRACTION_SCHEMA`)

The model returns a **list of materials**, each with its own properties:

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
          "section": "<enum>",            // see Sections below
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
          "page": 0                       // 1-based PDF page
        }
      ]
    }
  ]
}
```

`extraction.py` tolerates the **old** single-material shape
(`material_name` + flat `mechanical_properties`) for back-compat, and backfills
`value_num/min/max/qualifier` from `value_raw` whenever the model omits them.

### Sections (enum, Task 8)

`Mechanical, Thermal, Electrical, Physical, Optical, Rheological, Processing,
Descriptive, Composition/Reinforcement, Architecture/Structure` — aligned with
the app repo's `page6.py` `PROPERTY_CATEGORIES`. Free-text drift (e.g.
"Mechanical Properties", "Dielectric") is normalized to the enum on ingest by
`_norm_section`.

## Unit normalization (Task 3)

`canonicalize()` maps each property to a **family** (by `property_name`
keyword), converts the value to a canonical unit with [pint], and stores:

- `unit_canonical` — human label for the family (e.g. `GPa`, `MPa`, `°C`,
  `g/cm³`, `ppm/°C`)
- `value_si` — the value in SI base units (Pa, K, kg/m³, …)

Plausibility is range-checked **after** conversion, so e.g. a modulus given as
`1200 MPa` (= 1.2 GPa) is no longer false-flagged against the GPa range. If the
unit can't be parsed or is dimensionally wrong for the family, the row gets
`status="unit_review"` instead of a guessed number.

Families with plausibility ranges (canonical units): tensile/flexural/shear/
storage modulus (GPa), tensile/flexural/compressive/shear strength (MPa), glass
transition / melting / crystallization / decomposition / HDT (°C), CTE
(ppm/°C), density (g/cm³), specific gravity (dimensionless), elongation (%).
Properties outside these families are stored as-is with no range check.

Matching rules (2026-08 fixes):

- Short keywords (`tg`, `tm`, `hdt`, `cte`, `young`…) match on **token
  boundaries** — `tm` no longer hits "AS**TM**", `tg` no longer hits
  "ou**tg**assing".
- Dielectric / impact / tear strength are **passthrough** families that shield
  the generic word "strength" from the MPa families (unit passed through, no SI
  conversion, no range check).
- The raw families (CTE, elongation, specific gravity) honor the **printed
  unit** via an accepted-spelling map: `2.3e-5 1/K` → 23 ppm/°C, `13 ppm/°F` →
  23.4 ppm/°C, elongation `0.024` (bare/`mm/mm`) → 2.4 %. Anything not in the
  map is `unit_review:unexpected_unit`, never a silent default factor.
- `_preprocess_unit` treats any letter+2/3 as a power, so `N/mm2`, `kN/mm2`,
  `lb/in3` parse (the standard European MPa spelling used to fail).

## Per-row status (Task 4 — nothing is dropped)

Every inserted row carries a `status` and (when not `ok`) a `flag_reason`.
Precedence when several issues apply: `empty_value` → `unverified` →
`unit_review` → `out_of_range` → `ok`.

| status | meaning |
|--------|---------|
| `ok` | grounded in the PDF, unit sane, value in range |
| `unverified` | value (and `source_quote`) not locatable in the PDF text (Task 1) — **never** silently trusted |
| `unit_review` | unit missing or dimensionally wrong for the property family |
| `out_of_range` | value implausible **after** unit conversion |
| `empty_value` | value is blank / `n/a` / `-` / placeholder |

Grounding matches purely numeric values on **digit boundaries** — a value of
`3` is not "verified" by the `3` inside `ISO 527-3` or `23 °C` (2026-08 fix).
An `ok` row can additionally carry a *soft* reason in `flag_reason` that does
not change its status: `grounded_off_page` (the value is in the PDF but not on
the cited page) or `grounded_via_quote` (only the `source_quote` was found).
Reviewers can grep for these to inspect the weaker provenance chains.

`review_queue.csv` is a `SELECT … WHERE status != 'ok'` **view** over the DB, not
a discard pile. Corrected rows can be re-admitted with
`python batch_ingest.py --promote review_queue.csv` (sets `status='ok'`).

Document-level outcomes that produce **no** rows: `scanned_no_text` (image-only
PDF, Task 10), `empty_extraction`, `skipped_seen_sha1` (recorded in the run
report / `sources` table).

## Classification (Task 5)

`classify_material()` uses the model's `material_class` enum as the primary
signal. A deterministic keyword fallback runs only when that field is
missing/invalid: a non-empty `fiber_volume_fraction` ⇒ Composite, else composite
keywords ⇒ Composite, else fiber keywords ⇒ Fiber, else Polymer. The old
haystack bug (repeating `material_name` N times, never reading `section`) is
gone — the fallback now reads name + matrix + fiber + sections + property names.

## Provenance & dedup (Task 6)

Every property row carries `source_pdf`, `source_sha1`, `page`, and
`source_quote`. Dedup grain:

```
(source_sha1, material_key, section, property_name, test_condition, value_raw)
```

`material_key` is the normalized lowercased `material_name` (fallback:
abbreviation). Re-ingesting the same PDF adds 0 rows; the same property measured
in a *different* PDF is kept as an independent repeat.

## New / changed DB columns

Added (additively) to `Polymers` / `Fibers` / `Composites_materials` by
`migrate.py` / `--migrate`. The legacy columns (`value`, `unit`, `english`,
`section`, …) stay populated so the CSV export and `page1.py` keep working;
`value` mirrors `value_raw`.

```
material_key, material_class, trade_grade, manufacturer, matrix, fiber,
fiber_volume_fraction, value_raw, value_num, value_min, value_max, qualifier,
unit_canonical, value_si, source_pdf, source_sha1, page, source_quote,
status, flag_reason, model, prompt_version, extracted_at
```

`migrate.py` backs up the `.sqlite` to `<name>.sqlite.bak` first and is
idempotent (re-running adds nothing). A fresh DB created by `batch_ingest` gets
all columns up front.

## Large / scanned PDFs (Task 10)

- PDFs over ~15 MB or ~80 pages are uploaded via the **Gemini File API** and
  referenced by URI instead of base64-inlining (which would blow the request
  size limit).
- A PDF whose average extractable text/page is below `SCANNED_TEXT_PER_PAGE`
  (50 chars) is treated as scanned/image-only → `doc_status="scanned_no_text"`
  and skipped, so it can't produce confident-looking garbage. (OCR via
  `ocrmypdf`/`pytesseract` is a possible future route.)

## Reliability (Task 7)

`gemini_request()` retries 429/5xx and connection errors with bounded
exponential backoff, honoring `Retry-After` — mirrored from
`pdf_crawler.http_get()`. `temperature=0` throughout.

## Config / env

- `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) read from the environment only — no
  hardcoded keys.
- Deps beyond the crawler: `pint` (unit normalization), `PyMuPDF`/`fitz` (text
  grounding + scanned detection). Optional: `ocrmypdf` for scanned PDFs.
