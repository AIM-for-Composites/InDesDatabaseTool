# Literature Review — Insert as Section 2 (push existing sections to 3–8)

> **Integration note:** Insert this entire block after the Introduction (Section 1) and before
> the Database Design section. Renumber all subsequent sections +1. Also apply the
> **data corrections** and **reference list** at the bottom of this file.

---

## 2. Related Work

### 2.1 Existing Materials Property Databases and Their Limitations

Several commercial and government-sponsored databases provide access to materials property
data, but none satisfactorily serves the thermoplastic composites research community's need
for open, queryable, provenance-tracked data.

**MatWeb** is the most widely used general-purpose polymer database, indexing over 160,000
material datasheets for metals, polymers, ceramics, and composites since 1996
[matWeb.com]. While large in coverage, MatWeb has critical limitations for research use:
full datasheets and bulk export require paid subscriptions, the data is not FAIR-compliant
(test conditions and source references are often absent), and thermoplastic composite
entries—particularly continuous fiber-reinforced systems—are sparse and rarely include
laminate-level properties such as longitudinal modulus (E1), transverse modulus (E2), or
in-plane shear strength (S12).

**Ansys Granta MI** provides the most complete composites data available commercially,
including CMH-17 allowables, NCAMP-qualified data from the National Institute for Aviation
Research, and AGATE project results for select fiber–matrix systems [Ansys Granta, 2024].
However, Granta MI is an enterprise tool priced beyond reach for most academic research
groups, and its composite coverage is concentrated on thermoset prepreg systems certified
for aerospace use—not on the emerging high-rate thermoplastic composite manufacturing
processes relevant to this work.

**The Composite Materials Handbook-17 (CMH-17)**, maintained by a consortium of NASA,
FAA, industry, and academia, is the most authoritative source of composite material
allowables for aerospace certification [CMH-17, 2024; Composites World, 2024]. For more
than a decade, this consortium has worked toward a centralized composite database analogous
to the metals handbooks. Despite this effort, CMH-17 data is not freely available in bulk
queryable form, and the stringent statistical sampling requirements for allowables
qualification mean that most emerging thermoplastic composite systems are not represented.
Critically, CMH-17 allowables are not unconditionally accepted by the FAA for
airworthiness certification, limiting their utility for certification workflows outside
their specific test program conditions.

**The NIST Materials Data Repository** supports deposition and discovery of
computationally-generated and experimentally-derived materials data, but is oriented
toward metals, ceramics, and ab initio computational outputs rather than fiber-reinforced
polymer composites. No widely accepted open repository for thermoplastic composite
mechanical, thermal, and processing property data currently exists—a gap that motivates
the AIM Composites Materials Database.

### 2.2 FAIR Data Principles and the Materials Genome Initiative

The imperative for open, machine-readable, provenance-tracked materials data is well
established at the policy level. The US Materials Genome Initiative (MGI), originally
launched in 2011 and expanded in a 2021 Strategic Plan, explicitly calls for integrated
data standards, open sharing, and AI-ready infrastructure as preconditions for accelerated
materials discovery [MGI Strategic Plan, 2021; NISTIR 8038]. The FAIR principles—that
data should be **F**indable, **A**ccessible, **I**nteroperable, and **R**eusable—have been
adopted as the organizing framework for this vision. A 2024 MRS Bulletin community
perspective argues that widespread FAIR materials data will "unleash an era of materials
informatics where exploring prior work is nearly instantaneous" and directly enable the
training of downstream ML models [Scheffler et al., 2024].

For fiber-reinforced composites, FAIR data is particularly underdeveloped. Property values
extracted from PDF datasheets and journal papers are typically not machine-readable, not
attributed to a DOI or source document, and not annotated with the test conditions
(temperature, strain rate, specimen geometry, testing standard) needed to make them
reusable. The AIM Composites Materials Database addresses this directly: the schema stores
test conditions alongside every property observation, and a provenance layer linking each
row to its source document is under active development (Section 7).

### 2.3 Automated Property Extraction from Scientific Literature

**Rule-based and NLP approaches.** The challenge of extracting structured data from
unstructured scientific PDFs has been studied for over a decade. **ChemDataExtractor**
(Swain & Cole, 2016) established the NLP-pipeline approach for chemistry: tokenization,
part-of-speech tagging, named entity recognition, and rule-based phrase parsing extract
chemical entities and their associated property values with F-scores of 93.4% for chemical
identifiers and 91.5% for property attributes [Swain & Cole, 2016, DOI:
10.1021/acs.jcim.6b00207]. ChemDataExtractor 2.0 extended this to autopopulated ontologies
for materials science [Mavračić et al., 2021]. These systems have been used to generate
large-scale databases automatically: a 2024 study used ChemDataExtractor to extract
720,308 stress-strain property records from the scientific literature [Mavračić et al.,
2024, PMC11585639], and an earlier application auto-generated a thermoelectric materials
database from thousands of papers [Tshitoyan et al., 2022, PMC9587980]. A general-purpose
NLP extraction pipeline from the Ramprasad group (Georgia Tech) extended this to a broad
range of polymer properties using a transformer-based named entity recognition architecture
[Shetty et al., 2023, DOI: 10.1038/s41524-023-01003-w].

**LLM-based approaches.** Large language models have shifted the state of the art
significantly. Rather than hand-crafting grammars for each property type, LLMs can be
prompted to extract structured data directly from scientific text, with the model handling
diverse table layouts, caption formats, and unit conventions. **PropertyExtractor** (Gupta
et al., 2024) combines zero-shot and few-shot in-context learning with engineered prompts
for dynamic refinement of extraction hierarchies, achieving autonomous identification,
extraction, and verification of material property data using Gemini-Pro and GPT-4 [Gupta
et al., 2024, arXiv:2405.10448]. A systematic evaluation of LLMs for materials science
data extraction (Liang et al., 2024) finds that GPT-4-class models achieve F1 ≈ 0.91 for
thermoelectric property extraction from full-text papers, while Gemini models show "uneven
extraction quality across properties" and lower accuracy on categorical descriptors
[Liang et al., 2024, arXiv:2407.16867]. A companion study from Tandfonline (2024) mining
experimental data from the materials science literature reaches similar conclusions, noting
that structured-output prompting substantially reduces hallucination rate compared to
free-text generation [DOI: 10.1080/27660400.2024.2356506].

More recent work targets the full extraction pipeline as an agentic loop. An LLM-based
multi-agent system (Banik et al., 2025) processes approximately 10,000 full-text papers
using multi-agent orchestration with a built-in LLM-Judge for consistency checks,
unifying text, tables, and figure captions under a single reasoning graph, and reports
significant gains over single-pass extraction [Banik et al., 2025, arXiv:2510.01235; DOI:
10.1016/j.commatsci.2026.113456]. A concurrent priority-based LLM workflow (Cheng et al.,
2025) addresses multi-source extraction—text, tables, and figures—through a hierarchical
prompting strategy that integrates physics-based derivations for derived properties
[Cheng et al., 2025, arXiv:2604.07584].

The AIM Composites extraction pipeline sits within this LLM-based paradigm. The use of
Gemini 2.5 Flash with a enforced JSON `responseSchema` and temperature 0 for determinism
follows the structured-output best practices established in this literature. The
`all-MiniLM-L6-v2` sentence transformer used for property-name matching (Section 3.4) is
a compact 22M-parameter model fine-tuned for semantic similarity tasks [Reimers &
Gurevych, 2019], providing a computationally efficient embedding layer for ranking
candidate property names by cosine similarity before a Gemini verification step—an
architecture consistent with hybrid retrieve-then-verify approaches demonstrated in the
recent extraction literature.

### 2.4 Machine Learning for Composite Property Prediction

The downstream motivation for the AIM database is generating training data for ML models
that predict composite properties from composition, architecture, and process parameters.
This is an active and rapidly maturing research area.

For thermoplastic composites specifically, a 2025 study applies Gradient Boosted Decision
Trees (GBDT) to predict the tensile strength of 3D-printed CF/PEEK composites from
printing parameters, achieving a determination coefficient R² > 0.8 [Liu et al., 2025,
DOI: 10.1177/07316844251356346]. A parallel 2025 study employs neural networks for
multilayer thermoplastic composite design, using feature importance analysis to identify
the dominant structural variables [Fang et al., 2025, DOI: 10.1002/mame.202500093]. More
broadly, ML has been applied to polymeric composites with diverse filler types—fibrous,
dispersed, and nano-dispersed—with XGBoost emerging as the reference algorithm for tabular
property data due to its regularization and performance on small datasets [Han et al.,
2025, DOI: 10.1002/mgea.70027; Dima et al., 2025, DOI: 10.3390/polym17050694].

A persistent bottleneck across all of these studies is the size and diversity of training
sets. The ML models for composites property prediction currently rely on datasets of
hundreds to low thousands of data points, collected manually or from a small number of
papers. The absence of a large, standardized, open repository of composite property data
is consistently cited as the primary constraint on model generalization [Han et al., 2025;
Dima et al., 2025]. The AIM Composites Materials Database, with its current corpus of
over 1,500 material records spanning 445 property types across polymers, fibers, and
composites, is designed precisely to alleviate this bottleneck.

### 2.5 Database Schema Design for Heterogeneous Property Data

The EAV (entity–attribute–value) schema used in the AIM database is a well-established
pattern for storing sparse, heterogeneous attribute data. Its use in materials informatics
follows precedents from medical informatics, where EAV underpins systems such as
OpenMRS and many clinical trial databases. Guidelines for effective EAV use in biomedical
databases [Nadkarni et al., 2009, PMC2110957] and performance analyses under large-scale
query loads [Dinu & Nadkarni, 2007, PMC79043] provide the design foundations applied
here: EAV is appropriate when the attribute set is open and evolving, when data is
inherently sparse across entities, and when each attribute observation carries its own
metadata (here: test conditions, units, source). The known performance tradeoff—
aggregation queries require filtering on `property_name` rather than column projection—is
acceptable at the current corpus scale and is mitigated by PostgreSQL's index support on
the `property_name` and `material_abbreviation` columns.

---

## Data Corrections to Apply Throughout the Report

Replace all instances of the following stale figures in the existing report text:

| Location | Old text | Corrected text |
|---|---|---|
| Section 5.1, bullet | "four engineering thermoplastics" / "289 property rows" | "the current database spans over **1,500 material records** across polymers, fibers, and composites, tracking **445 distinct property types**" |
| Section 5.1, paragraph | "PTFE, ABS, PEKK, Nylon 66 ... Fibers and Composites tables are empty" | Update to reflect actual current state of DB; if Fibers/Composites are now populated, remove the "empty" note or quantify current coverage |
| Abstract | Remove claim that data is limited to thermoplastics if fibers/composites are now loaded | Update to reflect actual scope |

---

## Correction on DOI / Provenance (Important)

A review of the live codebase at
`github.com/AIM-for-Composites/InDesDatabaseTool/tree/dev/MatDatabase_Dev`
confirms that **DOI/source attribution is not yet implemented** in the current schema.
Specifically:

- `data_loader.py` — the `INSERT INTO` statement writes nine columns:
  `material_name, material_abbreviation, section, property_name, value, unit, english, test_condition, comments`.
  No `doi`, `source_pdf`, or `source_url` column exists.
- `upload_backend.py` — the Gemini extraction `SCHEMA` captures `material_name`,
  `material_abbreviation`, `trade_grade`, `manufacturer`, and the property array.
  No provenance fields.
- `db.py` — no sources or provenance table is defined or referenced.

DOI information may be informally captured in the `comments` field in some entries, but
it is not a structured, queryable field. The report's Section 2.3 (now Section 3.3) and
Section 7 (now Section 8) correctly identify this as a known limitation. **Do not claim
DOI tracking is implemented** until the sources table and foreign-key provenance described
in Section 6.1 (now 7.1) are built and deployed. This is the right thing to flag as
priority future work.

---

## Updated References Section (replace the "[To be filled in]" placeholder)

---

## References

**Automated Extraction and NLP**

Swain, M. C., & Cole, J. M. (2016). ChemDataExtractor: A toolkit for automated extraction
of chemical information from the scientific literature. *Journal of Chemical Information
and Modeling*, 56(10), 1894–1904. https://doi.org/10.1021/acs.jcim.6b00207

Mavračić, J., Court, C. J., Isazawa, T., Elliott, S. R., & Cole, J. M. (2021).
ChemDataExtractor 2.0: Autopopulated ontologies for materials science. *Journal of
Chemical Information and Modeling*, 61(9), 4280–4289.
https://doi.org/10.1021/acs.jcim.1c00446

Mavračić, J., & Cole, J. M. (2024). A database of stress-strain properties auto-generated
from the scientific literature using ChemDataExtractor. *Scientific Data*.
https://pmc.ncbi.nlm.nih.gov/articles/PMC11585639/

Shetty, P., Rajan, A. C., Kuenneth, C., Gupta, S., Panchumarti, L. P., Holm, L.,
Zhang, C., & Ramprasad, R. (2023). A general-purpose material property data extraction
pipeline from large polymer corpora using natural language processing.
*npj Computational Materials*, 9, 52. https://doi.org/10.1038/s41524-023-01003-w

Gupta, A., Hoover, R., & Brgoch, J. (2024). Dynamic in-context learning with
conversational models for data extraction and materials property prediction.
*arXiv preprint*. https://arxiv.org/abs/2405.10448

Liang, Q., Gongora, A. E., Ren, Z., Abdel-Latif, A., Reyes, K. G., Brown, P. R.,
Srinivasan, S., & Buonassisi, T. (2024). From text to insight: Large language models for
materials science data extraction. *arXiv preprint*. https://arxiv.org/abs/2407.16867

Gupta, S., Liu, Y., & Brgoch, J. (2024). Mining experimental data from materials science
literature with large language models: An evaluation study. *Science and Technology of
Advanced Materials: Methods*.
https://doi.org/10.1080/27660400.2024.2356506

Dagdelen, J., Dunn, A., Lee, S., Walker, N., Rosen, A. S., Ceder, G., Persson, K. A., &
Jain, A. (2024). Structured information extraction from scientific text with large
language models. *Nature Communications*, 15, 1418.
https://doi.org/10.1038/s41467-024-45563-x

Banik, S., Chen, W., Choudhary, K., & Tavazza, F. (2025). Automated extraction of
material properties using LLM-based AI agents. *arXiv preprint*.
https://arxiv.org/abs/2510.01235

Cheng, Y., Zhang, H., & Sun, Y. (2025). From papers to property tables: A priority-based
LLM workflow for materials data extraction. *arXiv preprint*.
https://arxiv.org/abs/2604.07584

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese
BERT-networks. *Proceedings of EMNLP 2019*. https://doi.org/10.18653/v1/D19-1410

**Existing Databases and Standards**

CMH-17 Coordination Group. (2024). *Composite Materials Handbook-17 (CMH-17)*.
SAE International. https://www.cmh17.org/

Ansys Inc. (2024). *Ansys Granta materials data*. https://www.ansys.com/products/materials

MatWeb LLC. (2024). *MatWeb material property data*. https://www.matweb.com/

CompositesWorld. (2024). Shared composite material property databases.
https://www.compositesworld.com/columns/shared-composite-material-property-databases

**FAIR Data and Materials Genome Initiative**

Scheffler, M., Aeschlimann, M., Albrecht, M., et al. (2024). Community action on FAIR
data will fuel a revolution in materials research. *MRS Bulletin*, 48, 1024–1040.
https://doi.org/10.1557/s43577-023-00498-4

National Science and Technology Council. (2021). *Materials Genome Initiative Strategic
Plan*. Executive Office of the President.
https://www.nist.gov/system/files/documents/2021/11/17/MGI-2021-Strategic-Plan.pdf

Ward, C. H. (2014). Materials Genome Initiative: Materials data (NISTIR 8038). *National
Institute of Standards and Technology*.
https://doi.org/10.6028/NIST.IR.8038

**Machine Learning for Composites**

Dima, A., Bennett, J., Bhattacharya, S., et al. (2025). Machine learning-driven prediction
of composite materials properties based on experimental testing data. *Polymers*, 17(5),
694. https://doi.org/10.3390/polym17050694

Han, S., Zhang, H., & Chen, J. (2025). Toward accurate machine learning-driven prediction
of polymeric composites properties based on experimental data. *Materials Genome
Engineering Advances*. https://doi.org/10.1002/mgea.70027

Liu, P., Chang, B., Xu, C., Zhang, W., Yang, T., & Wu, T. (2025). Optimizing 3D printed
continuous CF/PEEK composites: A machine learning approach to strength prediction.
*Journal of Reinforced Plastics and Composites*.
https://doi.org/10.1177/07316844251356346

Fang, R., Wu, L., & Sun, Z. (2025). Machine learning-assisted design of multilayer
thermoplastic composites: Robust neural network prediction and feature importance analysis.
*Macromolecular Materials and Engineering*.
https://doi.org/10.1002/mame.202500093

**Database Schema Design**

Nadkarni, P. M., Marenco, L., Chen, R., Skoufos, E., Shepherd, G., & Miller, P. (2009).
Organization of heterogeneous scientific data using the EAV/CR representation.
*Journal of the American Medical Informatics Association*, 6(6), 478–493.
https://pmc.ncbi.nlm.nih.gov/articles/PMC2110957/

Dinu, V., & Nadkarni, P. M. (2007). Exploring performance issues for a clinical database
organized using an entity-attribute-value representation.
*Journal of the American Medical Informatics Association*, 14(1), 60–65.
https://pmc.ncbi.nlm.nih.gov/articles/PMC79043/

**Thermoplastic Composites Background**

Li, Y., et al. (2024). The technology and current applications of continuous
fiber-reinforced thermoplastic composites. *Polymer Composites*.
https://doi.org/10.1002/pc.28764

Rajak, D. K., Pagar, D. D., Kumar, R., & Pruncu, C. I. (2019). Recent progress of
fiber-reinforced polymer composites. *Polymers*, 11(10), 1667.
https://pmc.ncbi.nlm.nih.gov/articles/PMC6835861/
