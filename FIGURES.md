# Figure & graph mining (figure-mining phase)

All image handling lives in **`figures.py`** — the single source of truth for
figures, mirroring `extraction.py`'s role for text. Opt-in via
`batch_ingest.py --figures`.

```
pdf_bytes
  └─ figures.harvest_figures()      PyMuPDF: embedded rasters + vector clusters → PNGs
        ├─ junk filters (logos, header/footer art, tiny/full-page, text tables)
        └─ caption pairing (below / beside strong; above / lone weak) → source_quote later
  └─ figures.classify_figures()     ONE Gemini vision call per PDF (0 if no figures)
        └─ figure_kind enum + material_guess (text-pass material names as context)
  └─ figures.mine_figure()          one vision call per property_plot / table_image
        └─ axes + salient scalars in the EXTRACTION_SCHEMA property shape
  └─ figures.figure_properties_to_rows()  → PropertyRow(origin='figure', status='figure_estimate')
        └─ canonicalize + plausibility exactly like text; NOT verify_against_text
  └─ batch_ingest.insert_row()      dedup grain + origin; figures table upserted
  └─ review_queue.csv               every figure row lands here (status ≠ 'ok') → --promote
```

## The principle

**A number read off a graph is an estimate, not a grounded fact.** Figure rows
are visibly second-class:

| aspect | text row | figure row |
|---|---|---|
| `origin` | `text` | `figure` |
| `status` | `ok` when grounded | **never `ok` at insert** — `figure_estimate` (or `empty_value` / `unit_review` / `out_of_range`) |
| grounding | `verify_against_text` | PNG on disk + `source_quote` = caption + `page` = figure page |
| `figure_id` | NULL | sha256(source_sha1 + PNG sha)[:16] → row in `figures` |
| `prompt_version` | `extraction.PROMPT_VERSION` | `figures.FIGURE_PROMPT_VERSION` (`1.0`) |
| reaches the app | if `status='ok'` | only after a human `--promote` |

The Space's `data_loader.load_material_data()` publishes `WHERE
COALESCE(status,'ok')='ok'`, and `batch_ingest._row_values()` downgrades any
figure row that arrives as `ok` — so no unreviewed graph reading can ever show
up as a datasheet fact, whatever the caller does.

## Harvest (`harvest_figures`)

`harvest_figures(pdf_bytes, source_pdf, source_sha1, out_dir, max_figures=12, stats=None) -> list[Figure]`

Two routes:

1. **Embedded rasters** — `page.get_images(full=True)` → placement via
   `page.get_image_rects(xref)`. Micrographs, photos, most journal plots on
   this corpus.
2. **Vector figures** — `page.cluster_drawings()` groups a page's vector
   paths into regions (PyMuPDF ≥ 1.24; 1.27 here). Genuine vector plots
   (e.g. cure-cycle and stress–strain charts in review papers) come out with
   their captions.

Every stored PNG is a **render of the figure's page region** (bbox expanded
4 %, longest side ≤ 1600 px, ≤ 300 dpi) rather than the raw embedded bytes:
axis labels and legends are often vector text drawn *over* a raster plot and
the raw image would lose them. Geometry from `get_image_rects` /
`cluster_drawings` / `get_text` is in *unrotated* page coordinates while
`get_pixmap`'s clip is in *rotated* display coordinates, so the region is
mapped through `page.rotation_matrix` first (a `/Rotate 90` landscape page
used to render 80 % blank). The render is deterministic for a given PyMuPDF,
so `image_sha256 = sha256(PNG)` is stable across runs and
`figure_id = sha256(source_sha1 + image_sha256)[:16]` — **scoped to the PDF**,
so the same rendered figure in two different PDFs (a paper crawled twice with
different metadata, a datasheet family sharing a plot) never shares a primary
key. The Gemini payload is a JPEG (q 85) re-encode of the same pixmap, ~6×
smaller (no Pillow); the PNG stays the lossless artifact on disk.

Files: `<out_dir>/<source_sha1>/p<page>_<sha10>.png` (default
`crawl_out/figures/`, gitignored) — the name is an *identity* (page + PNG sha
prefix), not a rank, so a path can never be silently reassigned to a different
figure when the candidate set changes between runs. A file that already
exists with the same sha is not rewritten — re-running harvests **0 new
files**.

Junk filters, each counted in `HarvestStats` (surfaced in
`run_report.json["figures"]["harvest_filters"]`):

| filter | rule | catches |
|---|---|---|
| `skipped_small` | either side < 120 px or area < 15 kpx; vector cluster < 120×80 pt | icons, bullets, glyph clusters |
| `skipped_placed_tiny` | placed area < 1.2 % of the page | publisher logos rendered small |
| `skipped_fullpage` | placed area > 90 % of the page | backgrounds, whole-page scans |
| `skipped_repeated` | same image bytes / same cluster **size and position** on ≥ 3 pages, and no direct Figure caption | header/footer art, watermarks (not a series of same-size auto-exported plots) |
| `skipped_header_footer` | cluster entirely inside the top/bottom 7 % band | running heads |
| `skipped_overlap_raster` | cluster ≥ 60 % covered by a harvested raster | the frame drawn around an image |
| `skipped_text_table` | vector cluster with a `Table N` caption directly above it, or **uncaptioned** and > 3.5 chars/kpt² of **live text** | datasheet property tables and gridded paper tables — already extracted, grounded, by the text pass; mining them only buys `figure_estimate` duplicates of `ok` rows |
| `skipped_dup_sha` | same PNG bytes already kept | multi-placement of one image |
| `capped` | over `max_figures` | — |

The text-table rule was measured on the corpus: every uncaptioned vector
cluster above 3.5 chars/kpt² is a live-text table (Toray 4.9–12, Polystrand
4.2, Avient 3.8); the only real uncaptioned vector figure sits at 0.4; a
directly captioned bar chart with dense tick labels ran 4.2 and is exempt.
Set `VECTOR_TEXT_DENSITY_SKIP = 0` to disable. Net effect on the 41-PDF
corpus at the time of writing: **292** figures kept, **99 %** of the 264
paper figures carry their caption, 85 text tables skipped; 9 of 11 Toray
datasheets yield **0** figures (tc910_pa6: 1 raster, tc940_pet: 2 raster) and
5 datasheets in total have figures.

Caption pairing: text blocks matching `^(Fig(ure)?\.?|FIG\.?)\s*\d+`
(trimmed to their first non-blank line — Word-generated PDFs pad caption
blocks with blank lines, which used to swap captions on real pages) are
attached to the nearest figure **below** (≤ 180 pt, ≥ −15 pt, horizontally
overlapping), else **beside** (margin captions, ≤ 110 pt gap, vertically
overlapping, block narrower than 32 % of the page — Springer style); those
are *strong* pairings that exempt a vector cluster from the repeat and
text-table rules. Else **above** (≤ 120 pt) or the page's only caption —
*weak* pairings: attached as best-effort provenance but exempting nothing.
Stored verbatim (whitespace-collapsed, ≤ 600 chars) — it becomes the row's
`source_quote`. Priority when the cap bites: captioned figures first (page
order), then uncaptioned rasters, then uncaptioned vectors.

## Classify (`classify_figures`)

One call per PDF (split into consecutive batches only if the base64 payload
would exceed `CLASSIFY_MAX_BYTES` = 12 MB — with JPEG copies the worst corpus
PDF is 4.2 MB, so in practice one), all figures as inline JPEG parts each
preceded by `[figure i] page P — caption: …`, index-keyed `responseSchema`:

```jsonc
{"figures": [{"index": 0, "figure_kind": "property_plot", "material_guess": "PPS composite"}, …]}
```

`figure_kind ∈ property_plot | table_image | micrograph | photo | schematic |
chemical_structure | other`. `material_guess` is asked to be one of the names
the text pass already extracted (passed in the prompt), or `''`. Only
`property_plot` and `table_image` proceed to mining (`mining_status`
`not_mined` vs `skipped_kind`); a failed call marks every figure
`classify_failed`, counts it, and never raises. An index the model skipped
(or duplicated) leaves that figure `classify_failed` — pending, retried on
the next run — never a silent terminal `other`; the run report counts it as
`classify_incomplete`.

## Mine (`mine_figure`)

One call per mineable figure: caption + material names + the PNG →

```jsonc
{"figure_kind": "...", "x_label": "Strain", "x_unit": "", "y_label": "Stress", "y_unit": "MPa",
 "series": [{"series_name": "PPS", "material_guess": "PPS composite",
             "values": [{"section": "Mechanical", "property_name": "Tensile strength",
                         "value_raw": "~610", "value_num": 610, "unit": "MPa",
                         "test_condition": "", "comments": "peak of PPS curve"}, …]}]}
```

The prompt asks for **salient scalars only** — labeled points, peaks /
plateaus (UTS = peak of a stress–strain curve), endpoints (strain at break),
legend-stated values, table cells — and to use qualifier `~` when estimating
from gridlines. **Full curve digitization is explicitly out of scope**
(FOLLOWUPS A5). Each value uses the same property shape as
`EXTRACTION_SCHEMA` (minus `source_quote`/`page`, which come from the figure),
so it drops straight into `extraction.Property` and through `_fill_numeric`,
`canonicalize()` and `plausibility_problem()` unchanged.

## Rows and statuses

`figure_properties_to_rows()` attaches each value to the text-pass material
its `material_guess` names — scoring every candidate on whole tokens (exact
name/abbr/grade › all name tokens + the grade token › all name tokens › the
guess is a sub-name › longest shared token; `PEEK 450G` goes to the 450G
grade, `PA66` never to `PA6`, `PPSU` never to `PPS`) — so the row lands in
the right table with the right `material_key` — or, if nothing scores, to a
material named after the guess (class via the deterministic keyword
fallback).

Status precedence for figure rows: `empty_value` → `unit_review` →
`out_of_range` → **`figure_estimate`**. `flag_reason` names the figure, e.g.
`read from Figure 6 (property_plot, series 'PPS')`, with the unit/range
problem prepended when there is one.

Dedup grain = the text grain **plus `origin`**: a text row and a figure row
reporting the same number both survive — that agreement is signal, not
duplication. `--promote` also matches on `origin`, so promoting a text row
cannot silently bless its figure twin (or vice versa).

## Schema (additive)

`Polymers` / `Fibers` / `Composites_materials`: `origin TEXT DEFAULT 'text'`,
`figure_id TEXT` (both in `migrate.EXTRA_COLUMNS`; every pre-existing row
reads `origin='text'`).

`figures` table (SQLite; `migrate.FIGURES_DDL`, created by `--migrate` and
`init_db`, keyed on `figure_id`, upserted so a later classify/mine pass can
refresh `figure_kind` / `mining_status`):

```
figure_id, source_pdf, source_sha1, page, bbox (json), caption, figure_kind,
material_key, image_path, image_sha256, width_px, height_px, route,
mining_status, n_values, model, figure_prompt_version, extracted_at
```

`review_queue.csv` gained `origin` and `figure_id` columns; a reviewer opens
the PNG behind `figure_id`, and `python batch_ingest.py --promote review.csv`
is how a reading gets blessed.

**Postgres (`--pg`)**: `--figures --pg` exits with a clear "not yet supported"
error. **New gate on the live DB**: `origin` / `figure_id` are now in
`migrate.EXTRA_COLUMNS`, and `pg_mirror.check_schema()` requires every
column in that list — so *any* `batch_ingest --pg` (with or without
`--figures`, and `--promote --pg`) refuses to run until `python pg_migrate.py`
(dry-run) then `--apply` has added the two columns. Additive and idempotent.
Postgres still has no `figures` table and its partial unique dedup index does
not include `origin` (a figure row equal to a text row on the old grain would
violate it) — hence the refusal. Tracked in FOLLOWUPS A5.

## Wiring (`batch_ingest.py`)

| flag | meaning |
|---|---|
| `--figures` | master switch (off by default) |
| `--figures-dir DIR` | PNG root (default `crawl_out/figures`) |
| `--max-figures-per-pdf N` | hard cap (default 12) |
| `--no-figure-mining` | harvest + classify only — cheap mode |

The figure stage runs **after the PDF's text rows are committed**; anything
that goes wrong lands in `PdfResult.figure_error` and is counted
(`run_report.json["figures"]["figure_errors_by_kind"]`) — it cannot kill the
text rows. `scanned_no_text` PDFs stay skipped entirely (whole-page scans are
not figures; OCR remains FOLLOWUPS A2). A PDF ingested *before* `--figures`
existed gets a figure-only **backfill** pass on the next `--figures` run
(materials rebuilt from its rows).

**Outage recovery**: figures whose `mining_status` is `classify_failed`,
`mining_failed` (or `not_mined` after a `--no-figure-mining` run) are
*pending*; the next `--figures` run retries **only those** — figures already
`mined` / `skipped_kind` are excluded from the vision calls
(`done_figure_ids`) and keep their stored status. A PDF with nothing pending
costs 0 calls (a PDF that genuinely has zero figures is re-harvested locally,
~0.2 s, no calls).

`run_report.json["figures"]`: `figures_found`, `figures_mined`, `figure_rows`,
`figure_rows_duplicate_skipped`, `vision_calls`, `figure_errors_by_kind`,
`harvest_filters`. The top-level `rows_*` metrics (`rows_inserted`,
`insert_rate`, `flag_rate`) stay **text-only**; figure rows are counted only
under `figures`.

A `scanned_no_text` PDF is never backfilled either — the `sources` logbook
records that status and the rerun branch honours it.

## Cost model

Per PDF: **≤ 1 classify call** (0 when nothing was harvested — 9 of 11 Toray
datasheets; more only if the JPEG payload would exceed 12 MB, which no
corpus PDF does) **+ ≤ 1 mining call per `property_plot` / `table_image`**,
hard-capped by `--max-figures-per-pdf`. Every call goes through
`extraction.gemini_request` (retry/backoff, `temperature=0`). Payload per
classify call: JPEG q85 copies, worst corpus PDF 4.2 MB base64 (the lossless
PNGs would have been 23.8 MB — over Gemini's 20 MB inline limit). On the
current corpus that is 32 classify calls (27 papers + 5 datasheets with
figures) plus roughly the number of plots — an order of magnitude below the
text pass in tokens per PDF. `--no-figure-mining` bounds it at the classify
calls only.

## Eval

`eval/gold/pekk_pps_thermoforming_fig6.{pdf,json}` (`"kind": "figure"`):
Figure 6, p 7, a stress–strain plot with PPS/PEKK curves; gold = peak stress
(608 / 824 MPa, confirmed by the paper's text) and strain at break (2.7 /
3.3 %), **±15 %** — reading plots is approximate and the eval says so. `python
-m eval` runs the text pass + figure stage for figure cases and scores
**only** figure-derived rows, under `figure_aggregate` so the text aggregate
is unchanged; `--selfcheck` proves offline that a 10 %-off readout scores 1.0
and a 30 %-off one 0.0 and that figure rows are never `ok`; `--gold-check`
verifies the figure page/caption exist.

## Out of scope / follow-ups (FOLLOWUPS A5)

- Full curve digitization / point-by-point tracing (WebPlotDigitizer territory).
- OCR for `scanned_no_text` PDFs (still A2).
- Postgres: `figures` table + `origin` in the dedup index; then lift the
  `--figures --pg` guard.
- HF Space port: figure PNGs behind the review queue, per-row promote that is
  origin-aware (the agent Space's per-PDF blanket promote must exclude figure rows).
