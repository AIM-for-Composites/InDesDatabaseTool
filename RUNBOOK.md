# Immediate-action runbook

Prioritized, do-it-now plan distilled from a full code review + a multi-lens
decision evaluation of the proposed fully-autonomous crawler. Companion to
[`FOLLOWUPS.md`](FOLLOWUPS.md) (which has the detailed findings). Most items are
in the **InDeS** repo (`AIM-for-Composites/InDesDatabaseTool`, branch
`agentic_code`) or its provider consoles; a few are local AIM fixes.

## Decision verdict (do this read first)

**Do not build the full open-web autonomous loop yet. Phase it.** The feature
that makes the database trustworthy — dual-LLM consensus — is dead code (Gemini
silently disabled), and extraction accuracy has never been measured. Automating
breadth on top of an unmeasured, single-LLM extractor multiplies an unknown
error rate by 365×/year. The crawl layer (AIM `pdf_crawler.py`) is the strong
part; the blockers are all in the InDeS code the merge would pull in. Source
supply is already adequate (41 validated PDFs from clean API lanes). **Earn
automation by proving accuracy on the clean core first.**

Two verified corrections to earlier assumptions:
- AIM `pdf_crawler.py:539` **already** validates the `%PDF` magic byte (now
  hardened — see item L1). The magic-byte gap is only in the InDeS
  `Downloader.py`.
- `Scheduler.py:25–31` is mislabeled "daily 02:00 UTC" but actually uses
  `trigger="interval", minutes=1` — it fires the **whole crawler every 60
  seconds**. Trust the code, not the comment (item P1.6).

---

## P0 — right now (the credential bleed)

Both repos are PUBLIC; assume the keys are already scraped.

- **P0.1 Revoke + rotate both leaked keys.** Gemini hardcoded at
  `page_files/categorized/Backend/Pdf_DataExtraction.py:138`; Gemini + OpenAI in
  the committed `langchain-env/Lib/.env`. Revoke in Google AI Studio / Cloud
  console + OpenAI dashboard, re-mint, set **hard budget caps + alerts**, scan
  usage logs for abuse in the exposure window.
- **P0.2 Move the key out of the URL.** `DocToDB_eval_v2.py:~114, ~1106` pass
  the Gemini key as a `?key=` query param (leaks into logs) → use the
  `x-goog-api-key` header.
- **P0.3 Freeze automation.** Don't run `Scheduler.py` or import the unsafe
  modules until P1/P2. Disable any existing scheduled task.

## P1 — within 24 hours

- **P1.4 Purge secrets from git history.** `git filter-repo`/BFG on a fresh
  `--mirror` clone (back it up first), force-push, make the repo private during
  the rewrite, enable GitHub Push Protection + secret scanning. Rotation (P0.1)
  is still mandatory — history purge alone doesn't un-leak.
- **P1.5 Stop committing the venv/DB/caches.** `.gitignore` + `git rm -r
  --cached langchain-env/ materials.db pdf_extraction_cache.json chroma_store/
  downloads/ .env` (~52k files); add a pinned `requirements.txt`.
- **P1.6 Fix `Scheduler.py`** → `CronTrigger(hour=2, minute=0)` for the daily
  job (the every-60-seconds bug).
- **P1.7 Fix the cache data-loss.** `DocToDB_eval_v2.py` schema key
  `mechanical_properties` (:358) vs `properties` everywhere else; cache
  write-back (:1733) drops all properties on cached re-runs. Standardize on
  `properties`, add a `PROMPT_VERSION` cache key, delete the poisoned
  `pdf_extraction_cache.json`.
- **P1.8 Magic-byte guard in InDeS `Downloader.download_pdf`** (it has none) —
  require `b"%PDF-"` + a PyMuPDF open before saving. (Port from AIM, item L1.)

## P2 — this week

- **P2.9 Re-enable Gemini.** `max_output_tokens` missing from `_GEMINI_DEFAULTS`
  (~257) → swallowed `KeyError` (~1098/1140) → `df_gemini` always empty → the
  "dual-LLM" is GPT-only. Add the key (or `.get(default)`), stop swallowing the
  error — **or honestly relabel output as single-model.**
- **P2.10 Make modules import-safe.** Move `DocToDB_eval_v2` import-time network
  call/threads (:146, :682) and the entire at-import Excel pipeline in
  `Backend/Pdf_DataExtraction.py` (:214–295) behind functions / `__main__`.
- **P2.11 Consolidate the 4–5 drifted prompt/schema copies** into one source of
  truth (mirror AIM `extraction.py`, `PROMPT_VERSION`-stamped). `page1.py`'s
  exact-match `section` filter must stay aligned with the section enum.
- **P2.12 Measure accuracy.** Grow the gold set (now 4 cases) toward 8–10 and
  run `python -m eval` — the precision/recall baseline that has never been run.
  Wire it as a regression gate.

## ⛔ Hold until all gates are green

- No daily scheduler (weekly/manual captures ~all the value at a fraction of the
  cost/risk; composites data moves monthly-to-yearly).
- No self-expanding DuckDuckGo discovery in an unattended loop — it's the one
  path (critical) to bulk-downloading paywalled/pirated full text under your
  name. Keep discovery a human-approved allowlist.
- No open-web journal scraping — stay on OpenAlex/arXiv/Unpaywall/S2 + a curated
  ~10-URL vendor-datasheet seed list. Store extracted facts + provenance, never
  journal PDF bytes.
- Go/no-go gate before any scheduled run: keys rotated + history purged + caps ·
  Gemini fixed or honestly relabeled · imports safe · magic-byte in · kill
  switch + per-run budget cap + daily summary alert · idempotent dedup proven ·
  3 supervised dry-runs.

---

## Local AIM fixes already applied (this repo)

- **L1 — Hardened the magic-byte check** in `pdf_crawler.py:539`: now requires
  `b"%PDF-"` in the first bytes **and** (when PyMuPDF is available) that the
  bytes open as a non-encrypted, ≥1-page document — rejecting paywall/login HTML
  mis-served as `application/pdf`. Verified against real-PDF / paywall-HTML /
  truncated / empty inputs.
- **L2 — Expanded the eval gold set** to 4 cases (`tc920_pc_abs` 3 materials,
  `tc910_pa6`, `tc1200_peek`, `tc1100_pps`) toward the 8–10 target for a
  meaningful precision/recall baseline.
- **L3 — 2026-08-15 bugfix batch** (commits `4e9a4cf..401783f`, see
  `FOLLOWUPS.md` A3 for the human steps it triggers): value parser (U+2212
  minus flipped signs; `±` mangled `1,200 ± 100`), unit families (`tm` matched
  "AS**TM**"; bare `strength` swallowed dielectric/impact strength; CTE and
  elongation ignored the printed unit; `N/mm2` unparseable; specific gravity
  false-flagged), grounding (short numbers verified against `ISO 527-3`),
  identity (`material_key` ignored `trade_grade` → multi-grade rows silently
  dropped), `sources` keyed on basename (same-name PDFs re-extracted forever),
  `--promote` over-matching, File-API path without retry + a 4xx retried 3×,
  crawler blacklisting transient failures, `generate_queries.py` ImportError,
  and the Space `data_loader` publish gate. 176 regression tests in `tests/`.

## What only you can do (needs consoles / the InDeS repo)

P0.1–P0.2, P1.4–P1.8, P2.9–P2.11 are in the InDeS repo or provider consoles and
can't be done from here. The full per-item detail (exact commands, verification,
rollback, pitfalls) was produced by the runbook review — ask to expand any item.
