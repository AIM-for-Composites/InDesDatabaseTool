# PDF Crawler — Design Rationale & Approach

How `pdf_crawler.py` was designed, what's in it, and why each piece is there. Every
claim below maps to a specific line in the actual file.

## Where it fits

The crawler isn't a standalone tool — it was built as **component 1 (source
discovery)** of the autonomous-ingestion architecture in Section 6 of the project
report, i.e. the exact piece `batch_ingest.py` deliberately skips. That framing drove
the whole design: its only job is to put *clean, relevant, de-duplicated* PDFs into a
folder (`crawl_out/pdfs/`) that plugs straight into `batch_ingest.py --input`, plus a
`sources.csv` provenance log and a `state.json` for incremental reruns. So it was
designed *around the existing pipeline*, not in isolation.

## Two design principles

Almost every feature traces to one of two goals:

1. **Protect the expensive downstream step.** Each PDF that makes it through gets sent
   to Gemini in `batch_ingest.py` — a paid API call that also writes rows to the
   database. So the crawler's job is to spend those calls only on real, relevant PDFs,
   and never on the same document twice.
2. **Survive a long run over unreliable third-party servers.** A crawl touches many
   academic APIs and manufacturer sites over a long session; the network is flaky and
   rate-limited. The crawler has to finish with complete data despite that.

Everything below is one of those two principles applied.

## The features, and why each is there

### Relevancy — *spend extraction calls only on likely-useful PDFs*
A search for "thermoplastic composite properties" returns review articles, modeling
papers, syllabi, and posters with no usable data. `relevance_score()` (line 249) scores
each candidate's title/abstract against a domain keyword list (PEEK, PPS, tensile,
modulus, datasheet…) and `MIN_SCORE = 4` (line 95) gates out the noise *before*
downloading or extracting. `JUNK_URL_PATTERNS` (line 123, applied at line 487) does the
same at the URL level — obvious non-papers are dropped without a download. The scoring
is deliberately plain keyword matching (cheap, local, no API cost); the expensive LLM
judgment is left to `batch_ingest.py`.

### Magic bytes — *confirm it's actually a PDF before paying to read it*
A URL ending in `.pdf` often returns an HTML error page, a login wall, or a Cloudflare
challenge — all with a normal HTTP 200. Line 539, `data.startswith(b"%PDF")`, checks the
PDF file signature on the downloaded bytes and rejects anything that isn't a real PDF,
backed by a minimum-size floor so tiny stub/error pages are dropped too. This stops the
pipeline from wasting a Gemini call extracting garbage.

### Dedup + resumability — *never download or extract the same document twice*
The same paper shows up across OpenAlex, Semantic Scholar, arXiv, and Unpaywall, and
datasheets get linked from multiple pages. Two layers prevent re-work: URL-level dedup
and content-level dedup via `hashlib.sha256()` of the bytes (line 543), tracked in a
`seen_hashes` set. `CrawlerState` (line 145) persists both seen-sets to `state.json`
(lines 156–163), so a rerun only fetches genuinely new material — the crawler can be run
repeatedly and stays incremental instead of redoing everything.

### Robustness — *finish the run despite flaky, rate-limited servers*
- **Retry with backoff.** Free academic APIs (especially Semantic Scholar's no-key
  tier) routinely return 429 and 5xx under load. `http_get()` (line 201) retries up to
  `MAX_RETRIES = 3` (116) on `RETRY_STATUS = {429,500,502,503,504}` (117) with
  exponential backoff, honoring the server's `Retry-After` header (187–192). One
  transient blip no longer loses a paper or aborts a query.
- **HEAD pre-check.** `head_precheck()` (line 490) sends a cheap HEAD request first and
  rejects paywalled/blocked URLs (401/403/404/410), wrong content-types (`text/html`),
  and out-of-range sizes — so a 30 MB file or a login page is never fully downloaded.
- **Politeness.** `robots_allows()` (line 412) respects robots.txt and
  `PER_DOMAIN_DELAY = 1.5` (line 110) rate-limits per domain, so the crawler doesn't get
  IP-banned mid-run and stays a good citizen — the right posture for an open-source
  project.

## How it built on the previous components

It was developed in two passes:

- **v1** shipped with the core: relevance scoring, `%PDF` verification, sha256 dedup +
  resumable `state.json`, robots.txt respect, and per-domain rate limiting — and the
  output wired directly into the existing `batch_ingest.py`.
- **v2** added a second layer of hardening adapted from the project's `agentic`-branch
  crawler (`Frontier.py`/`Downloader.py`/`Discovery.py`): Semantic Scholar as a fourth
  source, the HEAD pre-check, a minimum-size floor, extra junk-URL filters, per-source
  yield tracking, and `--from-csv` so MatWeb-discovered candidates flow through the same
  dedup/provenance machinery.

What was kept different from the agentic branch on purpose: keyword scoring stays local
(no Gemini cost *during* crawling — the AI judgment already happens once, in ingest),
plus OpenAlex + Unpaywall coverage, a provenance CSV, resumable state, and the
MatWeb-as-index workflow that avoids MatWeb's bot-blocking and content license.

## One-line summary

The algorithm is organized around a single idea: **do all the cheap filtering
(relevance, magic-byte, size, dedup, HEAD pre-check) up front so the one expensive step —
Gemini extraction — only ever runs on clean, relevant, novel PDFs, and make the whole
thing survive a long, rate-limited crawl and resume cleanly.**
