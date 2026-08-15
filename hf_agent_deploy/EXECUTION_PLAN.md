# AIM Composites — Integration + Autonomous Agent Space: findings & execution plan

*Prepared 2026-08-11 · everything below is built and tested; deploy steps are ~15 minutes total.*

## 1. What I found

**The fresh pushes are on the HF Space repo, not GitHub.** Tejaswi
(~24 commits, Aug 10–11) rebuilt the upload pipeline in
`aim4composites/MaterialsDatabase`: a new SciBERT plot↔property mapper
(`mapper5.py`), a category push layer writing the RDS tables with your
pipeline's 33-column schema (`category_push.py`), a rewritten Upload page with
regex **DOI capture** (`doi_url`), DOI added to the Gemini extraction schema
plus per-row `source_text`/`source_page` provenance, a 3-stage figure
extraction pipeline with multi-LLM verification, and S3 crop storage. All of
it is manual-upload driven. Abhijit last pushed Aug 2–3.

**GitHub (`InDesDatabaseTool`)** hasn't moved since the June 25 agentic-code
merge you reviewed: LangGraph crawler (arXiv-only), dual-LLM extractor with
the Gemini `KeyError` bug (silently GPT-only), a scheduler that fires every
minute and self-terminates after 5 runs, the `Sources.json` case bug (fatal
on Linux), and no DOI-based dedup anywhere.

**Your local pipeline** is the only DOI-native piece (OpenAlex→Unpaywall by
DOI, sha256 dedup, `sources` provenance) and already has the Postgres bridge
(`pg_mirror.py`) into the exact tables the Space reads. Tejaswi's new
`category_push.py` schema matches your `extraction.py` columns — he built
around your schema, so the integration is natural, not forced.

⚠️ **Two leaked Google API keys remain reachable in the Space repo's public
git history** (removed from HEAD Aug 10, but history is forever). **Rotate
both in Google AI Studio now**, plus the old OpenAI key from the InDeS `.env`.

## 2. What I built (all tested)

### A. `aim_agent_space.zip` — the autonomous agent Space (new, push-ready)

Docker + Streamlit + LangGraph. One agent cycle:
`plan → discover → dedupe → download → ingest → report`

- **Discover**: rotating query frontier (editable in UI, seeded with the
  crawler's 8 intents) over OpenAlex (OA-filtered, DOI-native) with Unpaywall
  fallback, arXiv, optional Semantic Scholar; keyword relevance scoring.
- **Dedupe by DOI** *before* download, against a durable `agent_doi_seen`
  registry in the shared Postgres (then URL, then content sha256).
- **Download**: your `pdf_crawler.py` verbatim — polite, robots-aware,
  magic-byte + PyMuPDF validated, size-bounded.
- **Ingest**: your `extraction.py` v2.0 verbatim via
  `batch_ingest.process_pdf(db=pg_mirror)` — grounding, pint unit
  normalization, plausibility, per-row status; flagged rows quarantined.
- **Publish**: same RDS tables the MaterialsDatabase Space reads — results
  appear on the team site automatically.
- **UI**: Dashboard (status, rows/day, latest DOIs) · Run Control
  (autonomy toggle, interval, per-cycle budget caps, source toggles, query
  editor, run-now) · Recently Added · Review Queue (with human promote) ·
  Runs & Logs. Controls are password-gated; read-only for visitors.
- **Durability**: all state (config, crawler seen-sets, run history, cycle
  lock) lives in namespaced `agent_*` Postgres tables — container restarts
  resume cleanly; the shared tables' schema is never altered by the agent.
- Optional: Gemini **query expansion** when cycles come up dry
  (`generate_queries.py`).

**Verified in a sandbox**: full E2E cycle against a real Postgres with the
real migration — 16/16 checks passed (DOI dedupe across cycles, grounding
ok/flag routing, GPa→Pa SI conversion, class routing, run/event bookkeeping,
crawler-state persistence); all 5 pages render; app boots.

### B. `hf_space_patch.zip` — main-Space update kit

Your `hf_space_additions` **rebased onto Tejaswi's Aug 11 code**: Recently
Added page (now with DOI links) + ADDED column with 🆕 tags in Categorized
Search + `extracted_at` in `data_loader.py`. Four drop-in files + exact diff +
apply guide (`README_APPLY.md`). Tested against a live-schema Postgres.

### C. `github_integration.zip` — team-repo branch, ready to push

Two commits for `InDesDatabaseTool` as git bundle + patches: (1) the three
bug fixes (Gemini KeyError, scheduler, Frontier case), (2) `agent_space/` +
`INTEGRATION.md` so the Space code is versioned with the team repo.

## 3. Deploy checklist (in order)

1. **Rotate keys** (5 min): Google AI Studio → revoke both leaked Gemini keys,
   create a fresh one. OpenAI dashboard → rotate the old leaked key.
2. **Create the agent Space** (5 min): New Space → `aim4composites/AutonomousAgent`
   → Docker → push/upload `aim_agent_space.zip` contents → set secrets
   (`DB_HOST/PORT/NAME/USER/PASSWORD` — same as the main Space, `GEMINI_API_KEY`
   fresh key, `AGENT_ADMIN_PASSWORD` your choice).
3. **Smoke-test** (5 min): open Space → Run Control → unlock admin → *Run one
   cycle now* → watch Runs & Logs → check Recently Added + the main Space.
4. **Turn autonomy ON** (10 sec): toggle + interval (6 h default). Note: free
   Spaces pause after ~48 h without visits — visit it, add an uptime pinger,
   or upgrade to always-on hardware.
5. **Main Space patch** (5 min): apply `hf_space_patch` per its README —
   coordinate with Tejaswi since he's actively pushing.
6. **GitHub** (5 min): push the integration branch per its README, open a PR,
   tag Tejaswi + Abhijit.

## 4. Open items / paper notes

- RDS schema: already migrated by your July runs (`pg_migrate.py` is included
  and the agent refuses to write to an unmigrated schema — fails safe).
- `agent_runs` accumulates per-cycle metrics (PDFs, rows, flag rates, errors)
  — direct evidence for paper §6 Implementation and §8 Results; the
  LangGraph node structure maps 1:1 onto the §4 agentic-architecture story.
- Not in the Space by design: `matweb_discovery.py` (needs a real browser)
  and Tejaswi's figure pipeline (heavy torch deps) — both stay
  local/main-Space respectively; the DB is the integration point.
- Semantic Scholar off by default (free tier rate limits); enable with an
  `S2_API_KEY` secret.
