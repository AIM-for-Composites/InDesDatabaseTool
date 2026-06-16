# Manual follow-ups (extraction-hardening phase)

These could not be done from this repo and need a human / a separate repo.

## 1. Rotate the committed Gemini API key (security) — do this first

`Backend/Pdf_DataExtraction.py` in the **app repo** (the live Streamlit/HF Space
project, not this one) has a real `GEMINI_API_KEY` committed in source. Treat it
as compromised:

1. Revoke/rotate the key in Google AI Studio / Cloud console.
2. Remove it from the code; read it from the environment instead
   (`os.environ["GEMINI_API_KEY"]`), the way `extraction.py` does here.
3. Because it's in git history, rotating is mandatory — scrubbing the file alone
   does not un-leak it. Consider `git filter-repo` / BFG on that repo, then force
   the new key everywhere it's used (HF Space secrets, local `.env`).

This repo never hardcodes a key (env-only) and `.gitignore` already excludes
`.env`, `*.key`, `*.pem`, `credentials.json`.

## 2. Sync the duplicated prompt/schema in the app repo to `PROMPT_VERSION`

The same Gemini prompt/schema is duplicated and has drifted across three places.
This repo's copy is now centralized in `extraction.py`
(`PROMPT_VERSION = "2.0"`). The other two are in the **app repo** and were not
edited from here:

- `page_files/categorized/page6.py`
- `Backend/Pdf_DataExtraction.py`

Action: in that repo, replace both copies with an import from a shared module
mirroring `extraction.py` (or vendor `extraction.py` over), and add at the top of
each old definition:

```python
# TODO: sync with extraction.py PROMPT_VERSION ("2.0") — schema is now a
# materials[] list with structured value_num/min/max + source_quote/page.
```

Until they're synced, the app and the batch ingester extract with **different**
schemas (single flat `mechanical_properties` list vs. multi-material structured
properties), so their outputs are not directly comparable.

`page1.py`'s exact-match `section` filter depends on the section enum — keep it
aligned with `extraction.SECTION_ENUM` when syncing.

## 3. Run the eval baseline once a key is available

`python -m eval` needs `GEMINI_API_KEY`. Suggested flow to satisfy acceptance
test #6 (precision/recall holds or improves):

```bash
# (optional) capture a pre-change baseline by checking out the previous commit
python -m eval --report eval_baseline.json
# then on this branch:
python -m eval --report eval_report.json --baseline eval_baseline.json
```

Add 1–3 more gold cases to reach the 3–5 suggested in the prompt (a bare-fiber
datasheet for the Fiber class, and a journal paper with ranges, would round out
coverage). See `eval/gold/README.md`.

## 4. Optional niceties not built

- Streamlit review view over `status != 'ok'` rows (a CSV `--promote` path is
  implemented; a UI is not).
- OCR route (`ocrmypdf`/`pytesseract`) for scanned PDFs — currently they are
  detected and skipped as `scanned_no_text` rather than OCR'd.
