# Manual follow-ups

Two groups: (A) follow-ups from the extraction-hardening phase in **this** repo,
and (B) findings from reviewing the companion **InDeS** agentic repo
(`AIM-for-Composites/InDesDatabaseTool`, branch `agentic_code`) — a different
repo that can't be edited from here.

---

## A. Extraction-hardening phase (this repo)

### A1. Run the eval baseline once a key is available
`python -m eval` needs `GEMINI_API_KEY`. Suggested flow for acceptance test #6
(precision/recall holds or improves):

```bash
# (optional) capture a pre-change baseline from the previous commit
python -m eval --report eval_baseline.json
# then on this branch:
python -m eval --report eval_report.json --baseline eval_baseline.json
```

Add 1–3 more gold cases to reach the suggested 3–5 (a bare-fiber datasheet for
the Fiber class; a journal paper with ranges). See `eval/gold/README.md`.

### A2. Optional niceties not built
- Streamlit review view over `status != 'ok'` rows (a CSV `--promote` path is
  implemented; a UI is not).
- OCR route (`ocrmypdf`/`pytesseract`) for scanned PDFs — currently they are
  detected and skipped as `scanned_no_text` rather than OCR'd.

### A3. After the 2026-08-15 bugfix batch (commits 4e9a4cf..401783f)

Eight verified defects were fixed (value parser U+2212/±, unit-family keyword
collisions + raw-unit maps, grounding digit boundaries, `material_key` grade
identity, `sources` keyed on sha1, File-API retry, crawler seen-set ordering,
Space publish gate) with 176 regression tests in `tests/`. Consequences that
need a human step:

- **Re-run `python pg_migrate.py` (dry-run) then `--apply` against the shared
  Postgres** before the next `batch_ingest --pg`. `check_schema()` now refuses
  to ingest until `sources` is re-keyed on `pdf_sha1` (the dry-run prints the
  exact change). Additive and idempotent, like the July migration.
- **Re-ingest multi-grade datasheets** once, to recover the rows the old
  `material_key` collapsed. The migration re-keys existing rows to
  `<name>|<grade>` (so the re-ingest dedups against them and only the missing
  rows are added), but `process_pdf` skips any sha1 already in `sources` — to
  re-ingest a specific PDF, `DELETE FROM sources WHERE pdf_sha1='…'` first
  (its property rows stay; only the logbook entry is removed).
- **Deploy `hf_agent_deploy/aim_agent_space.zip` and `hf_space_patch.zip`
  from their regenerated state** — both were regenerated in this batch
  (agent zip: current pipeline files + `agdb.py` round-trips the crawler's
  `failed` map; Space patch: `data_loader` publish gate). Do **not** deploy
  an older `pg_mirror.py` after `pg_migrate.py --apply` has re-keyed
  `sources`: its `record_source` conflicts on `pdf_filename`, whose UNIQUE
  constraint the migration drops.
- **`review_queue.csv` re-export**: a large share of the July flags were false
  positives from the fixed families (dielectric/impact strength, specific
  gravity, `N/mm2`, ASTM→`tm`); regenerate before anyone reviews it.
- **Run `python -m eval`** with a key: `--gold-check` proves all 4 gold cases
  are answerable from their PDFs, so tc1100's July `0.0/0.0` is an
  extraction-side miss; the harness now keeps `eval/last_run/*.extraction.json`
  so the next run explains it. Then grow gold to 8–10 (RUNBOOK P2.12).
- The agent-space `Review_Queue.py` promote is per-PDF, all-or-nothing; the
  local `--promote` now guards on section + status. Align before figure rows
  (which must never be blanket-promoted) land in that Space.

### A4. Figure & graph mining phase — landed 2026-08-15 (commits 85c515e..)

`figures.py` + `batch_ingest.py --figures` (see `FIGURES.md`). Verified
against the repo before building: the app consumer to guard is the Space's
`data_loader.py` (`load_material_data()`), not a local `page1.py`; its
publish gate (`status='ok'`) plus the insert-time invariant "a figure row is
never `ok`" is the export guard. Human steps:

- **Re-run `python pg_migrate.py` then `--apply` before the next
  `batch_ingest --pg`** — even a text-only one: `origin`/`figure_id` joined
  `EXTRA_COLUMNS`, and `check_schema()` refuses to ingest (or `--promote`)
  until they exist. The dry-run shows exactly the two ADD COLUMNs.
- **Run it with a key** — the offline suite proves the plumbing end to end
  with a scripted Gemini (harvest on the real corpus, classify/mine/rows/dedup/
  review/promote/backfill/failure modes), but the live acceptance run
  (`python batch_ingest.py --figures --input crawl_out/pdfs --limit 3 --db
  <local>`), the stress–strain → canonicalized strength row, and `python -m
  eval` (figure case scored under `figure_aggregate`) need `GEMINI_API_KEY`.
- **First `--figures` run on an already-ingested mirror** does a figure-only
  backfill for those PDFs (0 text calls) — expected.

### A5. Figure mining — deliberately out of scope, still open

- **Full curve digitization** (point-by-point tracing of stress–strain
  curves, WebPlotDigitizer territory). The mining prompt asks for salient
  scalars only. A future route: render → axis calibration → trace → emit a
  `series` blob; would need its own status/origin (`figure_trace`).
- **OCR route** for `scanned_no_text` PDFs (still A2) — whole-page scans are
  not figures and stay skipped by both passes.
- **Postgres port**: `pg_migrate.py --apply` will add `origin`/`figure_id`
  columns automatically (they are in `EXTRA_COLUMNS`), but there is no
  `figures` table on Postgres and the partial unique dedup index
  (`ix_<table>_pipeline_dedup`) does not include `origin`, so `--figures
  --pg` refuses to run. To lift: add `origin` to the index (new index name,
  drop the old), create `figures`, port `upsert_figure` /
  `figures_recorded_for` / `materials_for_source` to `pg_mirror.py`,
  and mirror the export-guard downgrade (already in `_row_values`, shared).
- **HF Space port**: the review queue page should show the PNG behind
  `figure_id`; the agent Space's per-PDF blanket promote must exclude
  `origin='figure'` (figure rows are promoted one at a time, on inspection).
  Not part of `hf_agent_deploy/aim_agent_space.zip` yet.
- Vector-route coverage: on this corpus most journal plots are embedded
  rasters; genuine vector plots are caught (review-paper cure-cycle and
  tensile charts) but the text-table skip (`VECTOR_TEXT_DENSITY_SKIP`) is a
  corpus-measured heuristic — re-check it if a new source's plots are
  text-heavy (dense legends) and start being skipped (`skipped_text_table`
  in the run report).

---

## B. InDeS agentic repo (`InDesDatabaseTool`, branch `agentic_code`)

Reviewed 2026-06 (5-subsystem code review). This is the live agentic system:
a LangGraph crawler (`crawler_graph.py`), a dual-LLM extractor
(`langchain-env/Lib/DocToDB_eval_v2.py`), and a validation agent
(`langchain-env/Lib/pdf_monitor_agent.py`). All code lives under
`langchain-env/Lib/` — which is **a committed virtualenv** (see B5).

### B1. 🔴 Rotate leaked API keys — do this first (PUBLIC repo)
Two separate leaks, both in a public repo, so both are compromised:

1. **`langchain-env/Lib/.env`** is committed and (per
   `PDF_MONITOR_AGENT_SETUP.md`) contains **`GEMINI_API_KEY` + `OPENAI_API_KEY`**.
   → Rotate **both** Google and OpenAI keys.
2. **`page_files/categorized/Backend/Pdf_DataExtraction.py:138`** hardcodes a
   Gemini key (`GEMINI_KEY = "AIza…"`) in source.

Steps for each:
- Revoke/rotate in the provider console (Google AI Studio, OpenAI dashboard).
- Move to env-only (`os.environ[...]`), as `extraction.py` does in this repo.
- They're in git history → rotating is mandatory; scrubbing the file alone does
  not un-leak. Use `git filter-repo`/BFG, then update any HF Space / CI secrets.
- Also: `DocToDB_eval_v2.py` passes the Gemini key as a **URL query parameter**
  (~lines 114, 1106), which leaks into proxy/server logs — switch to the
  `x-goog-api-key` header.

### B2. 🔴 Gemini extraction is silently disabled (dual-LLM is GPT-only)
`DocToDB_eval_v2._call_gemini` reads `_GEMINI_CONFIG["max_output_tokens"]`, a key
that is **never set** (commented out at ~line 280; absent from `_GEMINI_DEFAULTS`
~257). Every Gemini call raises a `KeyError` that is swallowed (~1140), so
`df_gemini` is always empty and the "dual-LLM consensus" degrades to GPT-only —
silently. The pick-winner / consensus logic is effectively dead for Gemini.
→ Add `max_output_tokens` to `_GEMINI_DEFAULTS` (or `.get(..., default)`).

### B3. 🟠 Prompt/schema drift across **four** copies
The Gemini prompt + JSON schema is duplicated and has drifted across:
- `langchain-env/Lib/DocToDB_eval_v2.py`
- `langchain-env/Lib/Extraction.py`
- `page_files/categorized/page6.py`
- `page_files/categorized/Backend/Pdf_DataExtraction.py`

Specific drifts found:
- In `DocToDB_eval_v2.py` the schema array is named `mechanical_properties`
  (~358) but prompts/consumers use `properties`; the cache round-trip writes back
  `mechanical_properties` (~1733) while merge reads `properties` first, so
  **cached re-runs drop all properties**.
- `Backend/Pdf_DataExtraction.py`'s active prompt asks for
  `experiment_name/measured_value/...` while its `responseSchema` demands
  `material_name/mechanical_properties` — contradictory (correct prompt is
  commented out).
- Dead `trade_grade` parsing in `DocToDB_eval_v2.py` (~1120-1133) for a field the
  schema doesn't define.

→ Consolidate to one shared module (mirror this repo's `extraction.py`, which is
the multi-material, grounded, unit-aware, enum-`section`, retry/backoff upgrade)
and stamp a single `PROMPT_VERSION`. Until then, the app, the batch ingester, and
the agentic extractor all extract with **different** schemas — outputs are not
comparable. `page1.py`'s exact-match `section` filter must stay aligned with the
shared section enum.

### B4. 🟠 Import-time side effects make modules unsafe to import
`DocToDB_eval_v2.py` runs at **import time**: a debug `print`, `load_dotenv`, a
live `get_gemini_model` HTTP call (~146), a daemon prewarm thread (~682), and
startup probes (~683). With no key these still fire (10s timeouts) and break
import/tests. Separately, the Streamlit `Backend/Pdf_DataExtraction.py` runs an
**entire Excel↔DB matching pipeline at import time** (module-level code ~214-295),
so importing it connects to a DB, reads Excel, calls Gemini, and writes files.
→ Move all network/thread/IO work into entry functions guarded by
`if __name__ == "__main__":`.

### B5. 🟠 A virtualenv is committed (repo bloat)
`langchain-env/` is a full committed Python venv (~52k files, including
`materials.db`, `pdf_extraction_cache.json`, `chroma_store/`, `downloads/`). It
bloats the repo and truncates the GitHub tree.
→ `.gitignore` the venv + DB + caches, `git rm -r --cached` them, and purge from
history. Keep only the source files (`*.py`, `Sources.json`, the setup docs).

### B6. 🟡 Lower-severity correctness (from review, verify before fixing)
- `_to_si` offset-converts °C→K then compares with a *relative* 5% tolerance
  (meaningless on an offset scale); percent / wt% / vol% / mol% all collapse to
  one ratio family and are treated as interchangeable in consensus matching.
- `_numeric_found_in_text` strips all non-digits so `2,5`→`25.0` and bare `5`
  matches any chunk containing `5` — drives `source_verified`, giving false
  confidence.
- `_ev_score` trusts LLM-judge indices with no bounds check; an `IndexError` is
  swallowed and silently switches to the token-overlap fallback while still
  labeling matches `llm_judge`.
- `build_batches` truncates oversized chunks in place (already indexed in
  ChromaDB) instead of splitting → displayed text corrupted, data lost.

---

## C. Deliverable produced here: `combined_agent.py`

A single LangGraph agent that **merges the crawler and validation agents** into
one workflow (built at the user's request; lives at
`Desktop/InDeS_combined_agent/combined_agent.py`, to be dropped into
`langchain-env/Lib/`):

```
load_frontier → fetch_papers → filter_relevance → download_pdfs →
save_papers → extract_and_validate → log_run → END
```

`extract_and_validate` runs the dual-LLM pipeline per downloaded PDF, GT-scores
when a `{stem}_gt.*` file exists (keeps the higher-F1 winner's TP rows) and falls
back to consensus + source-verified rows otherwise, then persists to a **new
`properties` table** in `materials.db` (additive). It lazy-imports
`DocToDB_eval_v2` to avoid the B4 import-time side effects and degrades
gracefully around the B2 Gemini bug. **Untested at runtime** (needs the repo,
keys, and heavy deps) — do a first run in the repo after B1/B2.
