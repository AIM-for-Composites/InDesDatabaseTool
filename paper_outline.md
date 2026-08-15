# Paper Outline — AIM Composites Materials Database (v2, professor's structure)

**Working title:** From Scientific Literature to an Autonomous Composites Database: An Agentic AI Framework for Knowledge Extraction, Validation, and Retrieval

**Target:** journal/conference paper (e.g., *Composites Part B*, *Integrating Materials and Manufacturing Innovation*, or a materials-informatics venue)

**Provenance:** restructured 2026-08-10 around the professor's outline (`paper/professor_outline.docx`). v1 of this file followed our own structure; the old→new mapping is at the bottom. Companion gap analysis: `paper/GAP_ANALYSIS.md`.

> ⚠️ **Numbering note:** the professor's outline skips §3 and §5 (runs 1, 2, 4, 6…10). We inserted **3 Related Work** and **5 Database Design** — both largely written already. *Mathias confirms with the professor before the numbering is final.*

---

## Author assignments

**Mathias is primary author** — first author, merges everything, and reviews/edits all sections for a consistent voice. Meher has left the team; his old sections (unit normalization → Mathias, batch-ingestion results → Abhijit) are reabsorbed below.

| Section | Topic | Owner |
|---|---|---|
| Abstract | Problem → limitation → method → results → impact | Mathias |
| 1 | Introduction (P1 motivation · P2 limitations + gap · P3 this work) | Mathias |
| 2 | Problem Definition & System Requirements | Tejaswi |
| 3 | Related Work *(inserted)* | Mathias |
| 4.1 | Overall architecture (+ Figure 4-1) | Tejaswi |
| 4.2 | Orchestrator agent | Tejaswi |
| 4.3 | Literature-discovery agent | Abhijit |
| 4.4 | Document acquisition & processing agent | Abhijit |
| 4.5 | Relevance-classification agent | Abhijit |
| 4.6 | Information-extraction agent | Mathias |
| 4.7 | Schema agent | Tejaswi |
| 4.8 | Numerical-normalization agent | Mathias *(was Meher)* |
| 4.9 | Validation & critic agent | Mathias |
| 4.10 | Conflict-resolution agent | Mathias |
| 4.11 | Database-management agent | Tejaswi |
| 4.12 | Human-review agent / interface | Mathias |
| 5 | Database Design *(inserted)* | Tejaswi |
| 6.1 | Software framework | Tejaswi |
| 6.2 | Agent communication | Tejaswi |
| 6.3 | Confidence assessment | Mathias |
| 7 | Experimental Design & Evaluation Metrics (7.1–7.5) | Abhijit |
| 8.1 | Results: literature discovery | Abhijit |
| 8.2 | Results: information extraction | Abhijit |
| 8.3 | Results: validation | Mathias |
| 8.4 | Results: database quality & scale | Tejaswi |
| 8.5 | Results: ablations | Abhijit |
| 8.6 | Results: scalability & cost *(absorbs old 6.4)* | Abhijit |
| 9 | Discussion (9.1–9.6) | Mathias |
| 10 | Conclusions | Mathias |
| Ack. | Acknowledgments & Data Availability | Mathias |

Tally: Mathias 13 · Tejaswi 9 · Abhijit 8. Parent headings 4, 6, 8 are unowned; Mathias adds their short lead-ins at merge.

---

## Abstract — *Mathias*
Five moves, in order (professor's structure): **Problem** (scientific data scattered across a fast-growing literature) → **Limitation** (manual curation slow; automated extraction lacks validation/provenance/updating) → **Method** (agentic framework coordinating discovery, processing, extraction, validation, DB management, updating) → **Results** (accuracy, traceability, DB quality, update performance) → **Impact** (unstructured literature → continuously updated, searchable resource).

## 1. Introduction — *Mathias*
- **P1 Scientific motivation:** results live in unstructured text/tables/figures; manual extraction is expert-heavy; databases cover a fraction and go stale.
- **P2 Limitations of current approaches (mini-review):** rule-based brittleness; single-pass LLM inconsistency; RAG answers questions but builds no database; no sentence/table-level provenance; no conflict handling; manual adaptation to new formats. **End on the research gap:** isolated tasks, no integrated autonomous framework with quality + provenance + continuous updates.
- **P3 This research:** the system discovers → extracts → validates → stores → supports evidence-based reasoning. State plainly what it is *not* (just a search agent / QA system / one-time pipeline / static KG) — it is a **living scientific database** managing the full data lifecycle.

## 2. Problem Definition & System Requirements — *Tejaswi*
- **2.1 Input corpus:** journal + conference papers, preprints, technical reports, supplementary files, existing databases, vendor datasheets (our corpus's backbone — say so).
- **2.2 Target scientific record:** the professor's tuple rᵢ = (eᵢ, aᵢ, vᵢ, uᵢ, mᵢ, pᵢ, qᵢ, tᵢ) — map it explicitly onto `PropertyRow` (material/class, property, value_raw/num/min/max, unit + unit_canonical/value_si, test_condition, source_pdf/sha1/page/quote, status, extracted_at). Composites fields: matrix, reinforcement, composition, fiber volume fraction, processing, properties, uncertainty, evidence.
- **2.3 System requirements:** autonomy, schema compliance, traceability, unit consistency, dedup, conflict identification, continuous updating, human review for uncertain records, scalable retrieval, reproducibility (PROMPT_VERSION, temperature=0, idempotent re-ingest).

## 3. Related Work — *Mathias* *(inserted section — confirm)*
- 3.1 Materials property databases (MatWeb, NIST, Materials Project, CMH-17) and their composites gaps.
- 3.2 LLM/NLP extraction from scientific literature (ChemDataExtractor lineage, structured-output LLMs); agentic/multi-agent extraction systems (new — position against P2's limitation list).
- 3.3 Data infrastructure for materials informatics (EAV vs. wide schemas, FAIR).
- Seed: `literature_review_section.md` — condense, then extend with agentic-pipeline literature.

## 4. Agentic AI System Architecture
*Parent lead-in at merge (Mathias). Present as a state-based multi-agent workflow with shared memory/state; the professor explicitly allows combined agents — each subsection names its functional agent, our implementing component, and (where honest) what is future work. Pipeline overview = Figure 4-1.*
- **4.1 Overall architecture — Tejaswi:** crawl → validate → extract → ground → normalize → validate → mirror → review; shared state = DB + `state.json` + run reports.
- **4.2 Orchestrator — Tejaswi:** `batch_ingest.py` driver; per-stage error capture, retry/backoff, doc statuses; reprocessing = re-run (idempotent); dynamic agent selection = future work.
- **4.3 Literature discovery — Abhijit:** `generate_queries.py` (LLM query proposals) + `pdf_crawler.py` (OpenAlex/S2/arXiv/Unpaywall + curated vendor seeds) + `matweb_discovery.py`; keyword ranking; sha256+state dedup; incremental new-doc detection; scheduling deliberately manual (governance, cf. 9.6).
- **4.4 Acquisition & processing — Abhijit:** polite crawling (robots.txt, throttle, HEAD pre-check), `%PDF` magic byte + PyMuPDF open, page-level text, scanned detection, large-PDF File API. GAP: structural table/caption separation, semantic sectioning.
- **4.5 Relevance classification — Abhijit:** pre-download keyword gate + junk-URL filter; post-extraction `empty_extraction`. GAP: LLM doc/section-level scores.
- **4.6 Information extraction — Mathias:** Gemini structured output; multi-material schema; per-value `source_quote` + `page`; section enum; `PROMPT_VERSION`; temperature=0.
- **4.7 Schema — Tejaswi:** `_norm_section`, `classify_material`, `material_key`; EAV = open taxonomy absorbs new properties. GAP: property-name synonym ontology.
- **4.8 Numerical normalization — Mathias:** pint families → `unit_canonical`/`value_si`; originals preserved; plausibility after conversion; `unit_review`. GAP: measured-vs-predicted flag.
- **4.9 Validation & critic — Mathias:** `verify_against_text` grounding; status precedence (nothing dropped); routes to review queue. GAP: second-model re-extraction; numeric confidence (cf. 6.3).
- **4.10 Conflict resolution — Mathias:** cross-PDF repeats preserved as independent records; conflict detection/explanation pass to build (GAP_ANALYSIS item 1).
- **4.11 Database management — Tejaswi:** SQLite mirror + shared Postgres; additive migrations; sha1-idempotent re-ingest; `sources` logbook; run reports.
- **4.12 Human review — Mathias:** `review_queue.csv` view + `--promote`; GAP: UI, prioritization, correction-history feedback loop.

## 5. Database Design — *Tejaswi* *(inserted section — confirm)*
- 5.1 Material taxonomy: polymers / fibers / composites; curated property sections.
- 5.2 EAV schema and rationale (sparsity, open taxonomy, per-observation provenance); tradeoffs vs. wide tables.
- 5.3 Identity & provenance: abbreviation keys, `material_key`, `sources` table, per-row provenance columns.

## 6. System Implementation
*Parent lead-in at merge (Mathias).*
- **6.1 Software framework — Tejaswi:** OUR stack, not the outline's example: Python + requests, Gemini API (structured output), pint, PyMuPDF, SQLite + Postgres (`pg_mirror.py`), Streamlit + HF Spaces, Playwright/ddgs for discovery. Env-only credentials.
- **6.2 Agent communication — Tejaswi:** file/DB-mediated shared state; input/output formats (PropertyRow, statuses); retry + correction policies; termination; logging + error recovery (`errors_by_kind`).
- **6.3 Confidence assessment — Mathias:** define the score (status taxonomy as ordinal scale, or composite of grounding/unit/plausibility/agreement — decide per GAP_ANALYSIS item 2).

## 7. Experimental Design & Evaluation Metrics — *Abhijit*
- 7.1 RQ1 accuracy · RQ2 multi-agent vs single-pass (**riskiest — needs baseline ladder**) · RQ3 validation's effect on unsupported records · RQ4 provenance fidelity · RQ5 update effectiveness · RQ6 cost/scalability.
- 7.2 Datasets: (1) expert-annotated gold set — grow 4 → 8–10 cases; (2) large-scale corpus — 41 validated PDFs now, target a few hundred via query-gen rounds.
- 7.3 Baselines: rule-based, conventional NLP, single-pass LLM (verification off), single-agent (stage toggles), full system. One toggle harness serves baselines *and* ablations.
- 7.4 Metrics: document selection P/R/F1; entity/relation F1, numeric + unit accuracy (SI-compared, tolerance-based — harness exists); DB quality (schema compliance, duplicate rate, missing fields, conflicts, unsupported-record rate); provenance (evidence-location, citation accuracy/completeness, claim-to-source consistency); cost (time/PDF, LLM calls, tokens, $/validated record, update time, query latency).
- 7.5 Robustness: layouts, poor PDFs (scanned detection), long tables, unseen terminology, conflicting publications, different LLMs, other subdomains.

## 8. Results *(every subsection: lead with a table)*
- **8.1 Discovery — Abhijit:** papers identified, gate rates, dedup performance, coverage (from `sources.csv` + `state.json`).
- **8.2 Extraction — Abhijit:** entity/property P/R, numeric + unit accuracy, by section/data type (eval harness).
- **8.3 Validation — Mathias:** error rate before/after grounding; corrected records; unsupported-claim reduction; confidence calibration; human-review rate.
- **8.4 DB quality & scale — Tejaswi:** PDFs processed, validated records, unique entities, growth, % full provenance (18,189 legacy rows without provenance vs. pipeline rows with — the before/after story).
- **8.5 Ablations — Abhijit:** per-stage contribution to accuracy/coverage/cost/time.
- **8.6 Scalability & cost — Abhijit:** throughput (8.76 s/PDF baseline), LLM usage, $/paper, $/validated record, growth behavior.

## 9. Discussion — *Mathias*
- 9.1 Benefits of role separation (workflow control, error detection, traceability, adaptability, maintainability, reliability).
- 9.2 Why agents: map each agent to the failure it prevents (professor's list).
- 9.3 Scientific implications: reviews, materials selection, ML training data, experimental planning.
- 9.4 Generalizability: process–structure–property, inverse design, manufacturing, other domains.
- 9.5 Limitations: document quality, paywalled access, table/figure errors, ontology sensitivity, LLM drift, cost, calibration, expert review in high-risk use, genuine-contradiction vs. condition-difference ambiguity.
- 9.6 Responsible use: copyright + access restrictions, licensing, attribution, model/prompt versioning, privacy/security, human oversight, no unsupported claims — our RUNBOOK governance (key hygiene, no bulk paywalled downloads, robots politeness, store facts + provenance not PDFs, hold-until-green gates) is a strength; write it that way.

## 10. Conclusions — *Mathias*
Four restatements: end-to-end agentic framework · autonomous extraction + storage + validation + provenance + retrieval combined · measurable gains over non-agentic baselines · a living database for future AI-driven research.

## Acknowledgments / Data Availability — *Mathias*
Clemson–Delaware collaboration; HF Space (aim4composites/MaterialsDatabase) + code links.

## References
[AuthorYear] while drafting; merged, deduped, renumbered once at the end (Mathias).

---

## Old → new section mapping (v1 → v2)

| v1 (ours) | v2 (professor's) |
|---|---|
| Abstract | Abstract (5-move structure) |
| 1 Introduction | 1 (P1–P3) |
| 2 Related Work | 3 *(inserted)* + partially 1-P2 |
| 3 Database Design | 5 *(inserted)* + 2.2 (record definition) |
| 4 System Architecture | 4.1, 6.1–6.2 |
| 5.1 Discovery & crawling | 4.3, 4.4, 4.5 |
| 5.2 LLM extraction | 4.6 |
| 5.3 Grounding | 4.9 |
| 5.4 Unit normalization *(Meher)* | 4.8 *(Mathias)* |
| 5.5 Validation, dedup, review queue | 4.9, 4.10, 4.12 |
| 6.1–6.3 Eval & model comparison | 7.1–7.5, 8.2 |
| 6.4 Batch-ingestion results *(Meher)* | 8.6 *(Abhijit)* |
| 7 Discussion | 9.1–9.4 |
| 8 Limitations & future work | 9.5 (+ future work folded into 9.4/10) |
| 9 Conclusion | 10 |
| *(new, no v1 home)* | 2.1/2.3, 4.2, 4.7 (partly), 4.10, 6.3, 7.3 baselines, 8.5 ablations |

## Conventions (unchanged from v1 docs)

Literal section numbers in headings (no auto-numbering). Cite [AuthorYear] while drafting; references merged once at the end. Figures/tables numbered by section (Figure 4.3-1). Scaffolding lives in the "Guidance" and "OwnerTag" styles for bulk deletion at merge. Owner colors: Mathias blue · Tejaswi green · Abhijit orange. Regenerate the docs with `node paper/build_paper.js` (script now checked in).
