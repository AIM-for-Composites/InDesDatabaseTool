# Figure & graph mining (figure-mining phase)

All image handling lives in **`figures.py`** — the single source of truth for
figures, mirroring `extraction.py`'s role for text. Opt-in via
`batch_ingest.py --figures`.

```
pdf_bytes
  └─ figures.harvest_figures()      PyMuPDF: embedded rasters + vector clusters → PNGs
        ├─ junk filters (logos, header/footer art, tiny/full-page, text tables)
        └─ caption pairing (below / beside / above)   → source_quote later
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
| `figure_id` | NULL | sha256(PNG)[:16] → row in `figures` |
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
the raw image would lose them. The render is deterministic for a given
PyMuPDF, so `figure_id = sha256(PNG)[:16]` is stable across runs and the PNG
doubles as the Gemini payload (no separate downscale, no Pillow).

Files: `<out_dir>/<source_sha1>/p<page>_<n>.png` (default
`crawl_out/figures/`, gitignored). A file that already exists with the same
sha is not rewritten — re-running harvests **0 new files**.

Junk filters, each counted in `HarvestStats` (surfaced in
`run_report.json["figures"]["harvest_filters"]`):

| filter | rule | catches |
|---|---|---|
| `skipped_small` | either side < 120 px or area < 15 kpx; vector cluster < 120×80 pt | icons, bullets, glyph clusters |
| `skipped_placed_tiny` | placed area < 1.2 % of the page | publisher logos rendered small |
| `skipped_fullpage` | placed area > 90 % of the page | backgrounds, whole-page scans |
| `skipped_repeated` | same image bytes / same cluster geometry on ≥ 3 pages | header/footer art, watermarks |
| `skipped_header_footer` | cluster entirely inside the top/bottom 7 % band | running heads |
| `skipped_overlap_raster` | cluster ≥ 60 % covered by a harvested raster | the frame drawn around an image |
| `skipped_text_table` | vector cluster > 4 chars/kpt² of **live text** with ≤ 20 drawing objects | datasheet property tables — already extracted, grounded, by the text pass; mining them only buys `figure_estimate` duplicates of `ok` rows |
| `skipped_dup_sha` | same PNG bytes already kept | multi-placement of one image |
| `capped` | over `max_figures` | — |

The text-table rule was measured on the corpus: Toray/Avient tables run
5–12 chars/kpt² with 1–5 ruled lines; vector plots run < 2 chars/kpt² with
26–500+ path objects. Set `VECTOR_TEXT_DENSITY_SKIP = 0` to disable. Net
effect on the 41-PDF corpus: 294 figures kept (97 % of paper figures carry
their caption), every Toray datasheet yields **0** figures and therefore
**0** vision calls.

Caption pairing: text blocks matching `^(Fig(ure)?\.?|FIG\.?)\s*\d+` are
attached to the nearest figure **below** (≤ 180 pt, horizontally
overlapping), else **beside** (margin captions, ≤ 110 pt gap, vertically
overlapping — Springer style), else **above** (≤ 120 pt), else the page's
only caption. Stored verbatim (whitespace-collapsed, ≤ 600 chars) — it
becomes the row's `source_quote`. Priority when the cap bites: captioned
figures first (page order), then uncaptioned rasters, then uncaptioned
vectors.

## Classify (`classify_figures`)

One call per PDF, all figures as inline PNG parts each preceded by
`[figure i] page P — caption: …`, index-keyed `responseSchema`:

```jsonc
{"figures": [{"index": 0, "figure_kind": "property_plot", "material_guess": "PPS composite"}, …]}
```

`figure_kind ∈ property_plot | table_image | micrograph | photo | schematic |
chemical_structure | other`. `material_guess` is asked to be one of the names
the text pass already extracted (passed in the prompt), or `''`. Only
`property_plot` and `table_image` proceed to mining (`mining_status`
`not_mined` vs `skipped_kind`); a failed call marks every figure
`classify_failed`, counts it, and never raises.

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
its `material_guess` names (exact match, then containment on name /
abbreviation / trade grade) — so the row lands in the right table with the
right `material_key` — or, if nothing matches, to a material named after the
guess (class via the deterministic keyword fallback).

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
error. The Postgres mirror gains the two columns automatically on the next
`pg_migrate.py --apply` (harmless), but it has no `figures` table and its
partial unique dedup index does not include `origin`, so a figure row equal
to a text row on the old grain would violate it. Tracked in FOLLOWUPS A5.

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
(materials rebuilt from its rows); a PDF with figures already recorded is
skipped without any call.

`run_report.json["figures"]`: `figures_found`, `figures_mined`, `figure_rows`,
`figure_rows_duplicate_skipped`, `vision_calls`, `figure_errors_by_kind`,
`harvest_filters`.

## Cost model

Per PDF: **≤ 1 classify call** (0 when nothing was harvested — every Toray
datasheet on the corpus) **+ ≤ 1 mining call per `property_plot` /
`table_image`**, hard-capped by `--max-figures-per-pdf`. Every call goes
through `extraction.gemini_request` (retry/backoff, `temperature=0`). Payload
per classify call ≈ 12 × ~0.2 MB PNG. On the current corpus (27 papers) that
is 27 classify calls plus roughly the number of plots — an order of magnitude
below the text pass in tokens per PDF. `--no-figure-mining` bounds it at 1
call per PDF with figures.

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
