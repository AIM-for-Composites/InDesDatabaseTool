# Claude Code prompt — Figure & Graph Mining Phase

**How to use:** open a terminal in the AIM Composites **project folder** (the one with
`extraction.py` and `batch_ingest.py`), run `claude`, and paste everything below the
line. This phase makes **images in the crawled PDFs first-class data**: figures are
harvested with provenance, classified, and the plots/table-images among them are mined
for property values the text pass can't see. It builds on the extraction-hardening
phase and must stay compatible with it (all DB changes additive). Verify every claim
against the real files before editing; if my description is wrong, stop and tell me.

---

## Context

Today the pipeline is text-only: `batch_ingest.process_pdf()` → `extraction.extract_from_pdf()`
(Gemini structured output) → `verify_against_text()` (grounds every value in the PDF
*text*) → `to_rows()` → SQLite/Postgres mirror. Anything reported **only in a figure** —
a stress–strain curve, a modulus-vs-temperature plot, a property table scanned as an
image — is invisible. Composites papers put a lot of their data exactly there.

Read first: `extraction.py` (whole file — reuse its machinery, don't duplicate it),
`batch_ingest.py` (`process_pdf`, the `db` backend seam, `PdfResult`, `_REVIEW_COLUMNS`),
`EXTRACTION.md` (statuses, dedup grain), `migrate.py`, and `eval/` (harness layout).

**Non-negotiable principle:** a number read off a graph is an *estimate*, not a
grounded fact. Figure-derived rows must be visibly second-class — separate `origin`,
their own status, never exported to the app CSVs, never silently mixed with
text-grounded rows.

---

## Task 1 — `figures.py`: harvest figures from PDFs (no LLM yet)

New module `figures.py`, the single source of truth for everything image-related
(mirror `extraction.py`'s role for text). Core entry:

```python
harvest_figures(pdf_bytes, source_pdf, source_sha1, out_dir, max_figures=12) -> list[Figure]
```

Two extraction routes with PyMuPDF (`fitz`, already a dep):

1. **Embedded raster images** — `page.get_images(full=True)` →
   `doc.extract_image(xref)`, bbox via `page.get_image_rects(xref)`. Covers
   micrographs, photos, scanned tables.
2. **Vector figures** — most journal plots are *drawn*, not embedded, so route 1
   misses them. Cluster the page's vector graphics into figure regions
   (`page.cluster_drawings()` if the installed PyMuPDF has it — check the version;
   otherwise fall back to caption-anchored regions) and render each region with
   `page.get_pixmap(clip=rect, dpi=200)`.

**Caption pairing:** scan text blocks (`page.get_text("blocks")`) for
`^(Fig(ure)?\.?|FIG\.?)\s*\d+` and attach the nearest caption below/beside each
region. Store the caption verbatim — it becomes the row's `source_quote` later.

**Junk filters (important, or you'll harvest logos):** skip images with either side
< ~120 px or area < ~15 kpx; skip anything whose sha256 repeats on ≥3 pages
(header/footer logos, watermarks); skip near-full-page background images; dedup
everything by sha256 of the PNG bytes. Cap at `max_figures` per PDF (config const),
preferring caption-matched regions.

Write PNGs to `<out_dir>/<source_sha1>/p<page>_<n>.png` (default out_dir
`crawl_out/figures/` — `crawl_out/` is already gitignored). `Figure` dataclass:
`figure_id` (image sha256 prefix), `source_pdf`, `source_sha1`, `page`, `bbox`,
`caption`, `image_path`, `image_sha256`, `width_px`, `height_px`, plus fields filled
later (`figure_kind`, `material_key`, mining status).

**Done when:** a text-borne PDF from `crawl_out/pdfs/` yields its real figures as
PNGs with page + bbox + caption, no logos/decorations, and re-running harvests 0 new
files (sha dedup).

## Task 2 — Classify figures (one batched vision call per PDF)

One Gemini call per PDF classifies **all** its figures at once (multiple inline image
parts, index-keyed `responseSchema`) — not one call per figure. Reuse
`extraction.gemini_request()` verbatim for retry/backoff (it already takes a generic
payload); `temperature=0`; new `FIGURE_PROMPT_VERSION = "1.0"` in `figures.py`
(do **not** bump `extraction.PROMPT_VERSION` — the text prompt is unchanged).

Per figure the model returns:

- `figure_kind` enum: `property_plot | table_image | micrograph | photo | schematic |
  chemical_structure | other`
- `material_guess` — pass the material names/abbreviations already extracted from
  this PDF's text pass as context; the model picks one, or `""` if unclear.

Downscale images before sending (longest side ≲ 1600 px, via a scaled
`get_pixmap` matrix — no new deps like Pillow). Skip classification entirely when
there are 0 figures.

## Task 3 — Mine plots and table-images for property values

For `property_plot` and `table_image` figures only (micrographs/photos are metadata,
not numbers), one vision call per figure asking for a **structured readout**:

- axes: `x_label`, `x_unit`, `y_label`, `y_unit` (plots)
- per series: `series_name`, `material_guess`, and **salient values only** — labeled
  points, peaks/plateaus, endpoints, legend-stated values (e.g. ultimate tensile
  strength = peak of a stress–strain curve); table cells for `table_image`.
- each value in the **same property shape `EXTRACTION_SCHEMA` uses** (`section`,
  `property_name`, `value_raw`, `unit`, `test_condition`, …) so it drops straight
  into the existing `Property` dataclass.

**Explicitly out of scope:** full curve digitization / point-by-point tracing
(WebPlotDigitizer territory) — record it in `FOLLOWUPS.md` as a future route. Salient
scalars only.

Mined properties then flow through the **existing** machinery — `parse_value_raw`/
`_fill_numeric`, `canonicalize()`, `plausibility_problem()` — exactly like text
properties. Do **not** run `verify_against_text()` on them (by definition they aren't
in the text); their grounding is: figure PNG exists on disk + `source_quote` = figure
caption + `page` = figure page.

**New status `figure_estimate`:** figure rows are **never** `ok`. Status precedence
for them: `empty_value` → `unit_review` → `out_of_range` → `figure_estimate`.
`flag_reason` names the figure, e.g. `read from Figure 3 (stress–strain plot)`.

## Task 4 — Persist: `figures` table + `origin` on property rows (additive)

- New columns on `Polymers`/`Fibers`/`Composites_materials`:
  **`origin`** (`'text'` default | `'figure'`) and **`figure_id`** (nullable FK-ish
  string). Extend `migrate.py` / `--migrate` (backs up the `.sqlite` first, stays
  idempotent) and the fresh-DB DDL in `batch_ingest.py`.
- New **`figures`** table: figure_id, source_pdf, source_sha1, page, bbox, caption,
  figure_kind, material_key, image_path, image_sha256, width_px, height_px,
  mining_status, model, figure_prompt_version, extracted_at.
- Dedup for figure rows = existing grain (it already includes `value_raw` +
  `test_condition`) **plus `origin`**, so a text row and a figure row reporting the
  same number both survive — that agreement is signal, not duplication.
- Respect the `db` backend seam in `process_pdf` (SQLite module vs `pg_mirror`).
  Implement SQLite fully; for `--pg` either add the additive figures support to
  `pg_mirror.py`/`pg_migrate.py` if clean, or make `--figures --pg` exit with a clear
  "not yet supported" error. Don't half-support it silently.

## Task 5 — Wire into `batch_ingest.py`

- New flags: `--figures` (opt-in master switch), `--figures-dir` (default
  `crawl_out/figures`), `--max-figures-per-pdf` (default 12), `--no-figure-mining`
  (harvest + classify only, skip Task 3 — cheap mode).
- In `process_pdf`, after the text pass succeeds: harvest → classify → mine → insert
  figure rows via the same `db.insert_row` path. A figure-stage failure must **not**
  kill the PDF's text rows — catch, log, count.
- `scanned_no_text` PDFs stay skipped **entirely** (whole-page scans aren't figures;
  OCR remains follow-up A2 in `FOLLOWUPS.md` — don't build it here).
- Extend `PdfResult` + `run_report.json` with `figures_found`, `figures_mined`,
  `figure_rows`. Add `origin` + `figure_id` to `_REVIEW_COLUMNS` — figure rows land
  in `review_queue.csv` automatically (status ≠ 'ok'), which is exactly right; the
  existing `--promote` path is how a human blesses one.

## Task 6 — Guard the exports

Wherever rows leave the mirror for the app (CSV export stage / anything reading these
tables for `page1.py`), **exclude `origin='figure'` rows unless `status='ok'`**
(i.e. promoted). Grep for the export/consumer code paths and add the filter there;
tell me which files you touched. The live app must never show an unreviewed
graph-reading as if it were a datasheet fact.

## Task 7 — Eval case

Add one gold figure case to `eval/`: a PDF (pick one from `crawl_out/pdfs/` with a
clear labeled plot or table-image) + expected mined values with a **wide tolerance
(±10–15%)** — reading plots is approximate and the eval must not pretend otherwise.
`python -m eval` includes it; `--selfcheck` still passes offline with no key. The
existing text-extraction gold cases must score exactly as before (no regression).

---

## Constraints

- **No hardcoded API keys** — `GEMINI_API_KEY`/`GOOGLE_API_KEY` from env only.
- All DB changes **additive**; legacy columns stay populated; `page1.py` and the CSV
  contract untouched except the Task 6 filter.
- Reuse `extraction.py` machinery (`gemini_request`, `canonicalize`,
  `plausibility_problem`, `parse_value_raw`, section enum). No new deps — PyMuPDF
  can render, scale, and PNG-encode on its own.
- Keep Gemini cost bounded: 1 classify call per PDF, ≤1 mining call per plot/table
  figure, hard cap per PDF. Log a per-run count of vision calls in the report.
- `temperature=0`; stamp `figure_prompt_version` on every figure row.
- Work in small commits; **don't push**. This is the local pipeline only — porting to
  the HF agent Space repo is a separate later phase (add it to `FOLLOWUPS.md`).

## Acceptance tests (show me)

1. `python batch_ingest.py --figures --input crawl_out/pdfs --limit 3 --db <local path>`
   → `figures/` PNGs on disk, `figures` table populated with page/bbox/caption/kind,
   and at least one `origin='figure'` row with `status='figure_estimate'`, a real
   `figure_id`, and `source_quote` = the caption.
2. A stress–strain (or similar) plot yields a strength/modulus row whose
   `unit_canonical`/`value_si` are filled by the existing canonicalization and whose
   value survives the plausibility check.
3. Re-run the same command → 0 new figures, 0 new rows (sha + dedup grain proven).
4. A text-only PDF (no figures) runs clean: zero figures, zero vision calls, text
   rows unaffected.
5. Junk proof: no harvested PNG is a publisher logo / header decoration (show the
   filter counts in the run report).
6. Kill the network mid-figure-stage (or simulate a Gemini 500) → the PDF's text rows
   still insert; the failure is counted, not fatal.
7. `review_queue.csv` contains the figure rows with `origin`/`figure_id` columns;
   `--promote` flips one to `ok`; only then would any export include it.
8. `python -m eval` — figure case scored, text cases unchanged.
9. `--migrate` on a copy of an existing DB adds the new columns + `figures` table,
   idempotent, `.bak` created.

## Deliverables

`figures.py`, wired `batch_ingest.py`, extended `migrate.py` (and `pg_migrate.py`/
`pg_mirror.py` if you took the --pg route), the Task 6 export filter, one eval gold
figure case, **`FIGURES.md`** documenting the figure schema/statuses/cost model, a
README pipeline-diagram update, and `FOLLOWUPS.md` entries for: full curve
digitization, OCR route (still open), and the HF Space port. Small commits, no push.
