// build_report.js — generate the ME8930 course-project report as a .docx.
// Run with:  cd /path/to/Database\ Project && node build_report.js

const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, BorderStyle, WidthType, ShadingType,
  LevelFormat, PageNumber, Footer,
} = require('docx');

const OUTPUT = path.join(__dirname, 'AIM_Composites_Materials_Database_Report.docx');

// ---------- helpers ----------
function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120 },
    alignment: opts.align || AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, bold: opts.bold || false, italics: opts.italic || false })],
  });
}
function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
}
function bullet(text) {
  return new Paragraph({
    numbering: { reference: 'bullets', level: 0 },
    spacing: { after: 60 },
    children: [new TextRun(text)],
  });
}
function bulletRuns(runs) {
  return new Paragraph({
    numbering: { reference: 'bullets', level: 0 },
    spacing: { after: 60 },
    children: runs.map(r => new TextRun(r)),
  });
}
function num(text) {
  return new Paragraph({
    numbering: { reference: 'numbers', level: 0 },
    spacing: { after: 60 },
    children: [new TextRun(text)],
  });
}
function numRuns(runs) {
  return new Paragraph({
    numbering: { reference: 'numbers', level: 0 },
    spacing: { after: 80 },
    children: runs.map(r => new TextRun(r)),
  });
}

const cellBorder = { style: BorderStyle.SINGLE, size: 4, color: 'BBBBBB' };
const cellBorders = { top: cellBorder, bottom: cellBorder, left: cellBorder, right: cellBorder };

function tc(text, width, opts = {}) {
  return new TableCell({
    borders: cellBorders,
    width: { size: width, type: WidthType.DXA },
    shading: opts.shaded ? { fill: 'D5E8F0', type: ShadingType.CLEAR } : undefined,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({
      children: [new TextRun({ text, bold: opts.bold || false })],
    })],
  });
}

// ---------- title block ----------
const titleBlock = [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
    children: [new TextRun({
      text: 'AIM Composites Materials Database: An AI-Assisted Repository for Composite Thermoplastic Property Data',
      bold: true, size: 36,
    })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 120 },
    children: [new TextRun({ text: 'ME8930 — AI for Composites and Manufacturing', italics: true, size: 24 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 120 },
    children: [new TextRun({ text: 'Course Project Report', italics: true, size: 24 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 360 },
    children: [new TextRun({ text: 'Mathias Heider   |   May 2026', size: 22 })],
  }),
];

// ---------- abstract ----------
const abstract = [
  h2('Abstract'),
  p('The Artificially Intelligent Manufacturing (AIM) Composites Materials Database is a centralized, open-source repository for experimental property data on polymers, fibers, and fiber-reinforced composites, with an initial emphasis on engineering thermoplastics and thermoplastic composites. The system combines a Postgres-backed entity–attribute–value (EAV) schema, a Streamlit web frontend deployed on Hugging Face Spaces, and a Google Gemini-based pipeline that extracts structured property data from research papers and manufacturer datasheets in PDF form. This report describes the database design, the current system architecture, the user-facing browsing and ingestion workflows, the state of the loaded data, and known data-quality issues. It then sketches the path from the current human-in-the-loop ingestion model toward an autonomous extraction agent, and reports preliminary results from a batch-ingestion prototype that runs the Gemini extraction pipeline over a folder of PDFs without manual review and applies automated validation. The project is a joint research effort between Clemson University and the University of Delaware; this report focuses on contributions to the database, ingestion pipeline, and the autonomous-AI prototype.'),
];

// ---------- sections ----------
const s1 = [
  h1('1. Introduction'),
  p('Machine-learning approaches to composites engineering — forward property prediction, inverse design, process–quality modeling — share a common bottleneck: training data. Public databases of composite material properties are sparse, fragmented across manufacturer datasheets and journal papers, inconsistent in units and test conditions, and often hidden behind paywalls or PDF-only formats. For thermoplastic composites in particular, which are increasingly important for high-rate aerospace and automotive manufacturing, no widely accepted open repository exists.'),
  p('The AIM Composites Materials Database was started to close this gap. The aim is twofold: provide researchers with a curated, queryable repository of polymer, fiber, and composite property data; and use that same infrastructure to generate the training datasets needed by downstream ML projects (lamina/laminate property prediction, process–quality models, structural performance modeling).'),
  p('The challenge of populating such a database is itself a problem in AI for manufacturing. Hand-curating thousands of properties from PDFs is prohibitively slow, and the structured-data layer needed for ML training is more demanding than what a human reading a datasheet typically captures. A large fraction of the engineering work in this project is therefore aimed at the ingestion side: how to use large language models and computer vision to convert unstructured PDFs into clean, validated, attributable database rows.'),
  p('This report describes the system as it exists today and reports preliminary work on moving the ingestion path from human-in-the-loop to autonomous operation. The remainder is organized as follows. Section 2 describes the database schema and the design choices behind it. Section 3 covers the system architecture and deployment. Section 4 presents the user-facing browsing and upload workflows. Section 5 summarizes the data currently loaded and the data-quality issues observed. Section 6 outlines the design for an autonomous ingestion agent and reports results from a preliminary batch-ingestion prototype. Section 7 lists limitations and future work.'),
];

const s2 = [
  h1('2. Database Design'),
  h2('2.1 Material taxonomy'),
  p('The database is organized around three material classes:'),
  bulletRuns([{ text: 'Polymers', bold: true }, { text: ' — neat thermoplastic and thermoset matrix candidates (e.g., PEEK, PEI, PA66, ABS, PTFE, PEKK).' }]),
  bulletRuns([{ text: 'Fibers', bold: true }, { text: ' — reinforcement constituents (carbon, glass, aramid, basalt, natural fibers).' }]),
  bulletRuns([{ text: 'Composites', bold: true }, { text: ' — fiber-reinforced laminates and molding compounds, characterized by their matrix, fiber, fiber volume fraction, architecture, and processing route.' }]),
  p('Within each class, properties are grouped into curated sections: Mechanical, Thermal, Electrical, Physical, Optical, Rheological, Processing, Descriptive, and — for composites — Composition/Reinforcement and Architecture/Structure. Each section contains a curated list of canonical property names (e.g., Tensile modulus, Glass transition temperature (Tg), Longitudinal modulus E1) plus a free-text fallback for properties that fall outside the curated list. This taxonomy is enforced in the upload UI to keep property names normalized.'),

  h2('2.2 Schema choice: entity–attribute–value (EAV)'),
  p('Each property observation is stored as a single row in a long-format table:'),
  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2600, 6760],
    rows: [
      new TableRow({ tableHeader: true, children: [
        tc('Column', 2600, { shaded: true, bold: true }),
        tc('Description', 6760, { shaded: true, bold: true }),
      ]}),
      new TableRow({ children: [tc('material_name', 2600), tc('Generic material name (e.g., "Polyetheretherketone")', 6760)]}),
      new TableRow({ children: [tc('material_abbreviation', 2600), tc('Short identifier (e.g., "PEEK")', 6760)]}),
      new TableRow({ children: [tc('section', 2600), tc('Property category (Mechanical, Thermal, etc.)', 6760)]}),
      new TableRow({ children: [tc('property_name', 2600), tc('Canonical or free-text property name', 6760)]}),
      new TableRow({ children: [tc('value', 2600), tc('Measured value or range, stored as text', 6760)]}),
      new TableRow({ children: [tc('unit', 2600), tc('SI unit string', 6760)]}),
      new TableRow({ children: [tc('english', 2600), tc('Alternate / imperial-unit value, if reported', 6760)]}),
      new TableRow({ children: [tc('test_condition', 2600), tc('Test temperature, strain rate, standard, sample geometry', 6760)]}),
      new TableRow({ children: [tc('comments', 2600), tc('Free-text notes, footnotes, standard references', 6760)]}),
    ],
  }),
  new Paragraph({ spacing: { after: 120 }, children: [new TextRun('')] }),
  p('There are three tables sharing this schema: Polymers, Fibers, and Composites_materials. The class is therefore encoded by the table membership, not as a column.'),
  p('This EAV layout was chosen over a wide-table layout (one row per material, one column per property) for several reasons. First, composite property data is inherently sparse: any given paper or datasheet reports a small subset of the possible properties, and a wide table would be mostly null. Second, the property taxonomy is open — new properties are discovered routinely in the literature, and adding a property in a wide-table design requires a schema migration. Third, each property observation has its own provenance and test conditions; storing those alongside the value (rather than in a parallel metadata table) keeps related information together. The tradeoff is that aggregation queries — "give me the tensile modulus for every PEEK grade" — require selection on property_name rather than a column projection, and unit consistency must be enforced application-side rather than by the schema.'),

  h2('2.3 Provenance and identity'),
  p('The current schema does not store an explicit reference to the source PDF, DOI, or page number. This is a known limitation; provenance is currently captured informally in the comments field when the extraction agent picks it up. Section 7 discusses adding a dedicated sources table and foreign-key references.'),
  p('Material identity is currently keyed on material_abbreviation for joins (e.g., the inspect tab in the search UI uses this as the lookup key). For composites, the abbreviation typically encodes both matrix and fiber (e.g., PEEK-CF-UD), and the search UI parses this string with regex-based matrix/fiber extraction to populate composition filters.'),
];

const s3 = [
  h1('3. System Architecture'),
  h2('3.1 Components'),
  p('The system has three layers:'),
  numRuns([{ text: 'Storage', bold: true }, { text: ' — PostgreSQL, hosted externally; connection parameters supplied to the application via environment variables (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD).' }]),
  numRuns([{ text: 'Application', bold: true }, { text: ' — A Streamlit web app (app.py) with four pages: Home, Categorized Search, Upload Data, Contact Team. Page navigation uses Streamlit’s native st.navigation API.' }]),
  numRuns([{ text: 'Deployment', bold: true }, { text: ' — A Hugging Face Spaces container, built from a Dockerfile based on python:3.11-slim. The Space exposes port 7860 and runs streamlit run app.py on container start. The repository contains requirements.txt pinning Streamlit, pandas, SQLAlchemy, psycopg, PyMuPDF, opencv-python-headless, and supporting libraries.' }]),
  p('The repository is publicly available at huggingface.co/spaces/aim4composites/MaterialsDatabase.'),

  h2('3.2 Data access layer'),
  p('db.py builds a SQLAlchemy engine from the environment variables and exposes thin fetch_all, fetch_one, and execute_query helpers. If database credentials are not set, the engine is not constructed and queries raise a configuration error — this keeps local development possible without forcing a live database connection.'),
  p('data_loader.py sits on top of db.py and exposes table-aware accessors: load_material_data(material_type) for browsing and insert_material_rows(df) for writes. The mapping from material class label (Polymer / Fiber / Composite) to table name (Polymers / Fibers / Composites_materials) is hardcoded here; the same mapping is mirrored in the upload UI.'),

  h2('3.3 Gemini-based PDF extraction'),
  p('PDF ingestion is implemented in page_files/categorized/Backend/upload_backend.py. The function call_gemini_from_bytes posts a base64-encoded PDF to the Google Generative Language API (gemini-2.5-flash-preview-09-2025), with three things wired up:'),
  numRuns([{ text: 'A structured-output schema. ', bold: true }, { text: 'The Gemini request supplies a responseSchema describing exactly the JSON shape expected: top-level material_name, material_abbreviation, trade_grade, manufacturer, and a mechanical_properties array of objects with section, property_name, value, unit, english, test_condition, comments. The required-field list forces the model to emit unit and comment fields even when the source PDF does not report them (they are written as empty strings). The responseMimeType is set to application/json and temperature to 0 for determinism.' }]),
  numRuns([{ text: 'A materials-scientist prompt. ', bold: true }, { text: 'The prompt instructs the model to extract all properties across categories — Mechanical, Thermal, Electrical, Physical, Optical, Rheological — into a single flat list (the array name "mechanical_properties" is legacy; it now carries all property types). The prompt explicitly enumerates the per-property fields and asks for empty strings rather than null when a field is not present.' }]),
  numRuns([{ text: 'An English-units passthrough. ', bold: true }, { text: 'A separate english field is reserved for imperial-equivalent values when reported in the source — useful because many older manufacturer datasheets, especially U.S. aerospace prepreg specs, mix SI and English units.' }]),
  p('The function returns the parsed JSON dictionary or None on failure; convert_to_dataframe then flattens the dictionary into a pandas DataFrame whose columns match the EAV schema columns described in Section 2.'),

  h2('3.4 Plot extraction and image-to-property mapping'),
  p('A parallel pipeline extracts plot images from the PDF using PyMuPDF and OpenCV. extract_images rasterizes each page at 300 DPI, applies adaptive thresholding and dilation, finds contours, filters them by area and "plot-like" geometry (the heuristic checks for elongated horizontal or vertical structures consistent with axes), merges overlapping boxes, and matches each crop to a nearby caption using a regex on "Fig./Figure" labels. Each detected plot is then offered in the upload UI for the user to map onto an extracted property — a stress–strain curve gets associated with the Tensile strength row, for example, and saved to disk as {abbreviation}_{property}.png. A separate caption-to-property matcher in match_caption_to_property uses keyword maps (e.g., "tensile" → tensile modulus / strength) and token overlap to suggest mappings automatically.'),
  p('A second, more sophisticated property-name matcher exists in Pdf_DataExtraction.py: a sentence-transformer (all-MiniLM-L6-v2) embeds candidate property names from a target Excel template and ranks database rows by cosine similarity, then verifies the top-5 candidates with a Gemini yes/no prompt. This is used for cross-referencing extracted properties against a fixed property dictionary and is part of the path toward standardized property names.'),
];

const s4 = [
  h1('4. User-Facing Workflows'),
  h2('4.1 Browsing — Categorized Search'),
  p('The Categorized Search page is the primary read interface. Users select a material class (Composites, Polymers, or Fibers), optionally restrict to a matrix or fiber for composites, optionally filter by property type, and search by material name or abbreviation. Matching materials are presented in a paginated table (50 per page). Selecting a row opens an Inspect tab where the user picks a property and sees the row-level data plus any associated plot image. Matrix and fiber filters are populated by parsing material abbreviations with a regex-based extractor that maps common substrings ("peek", "carbon", "glass", "kevlar", etc.) onto canonical labels.'),
  h2('4.2 Manual upload form'),
  p('The Upload Data page offers two entry modes. The first is a manual form: the user picks a material class, then a property category, then a property name from the curated list (or a "Something else" free-text option), then enters the material name, value, unit, English equivalent, test condition, and comments. Submission writes a single row through insert_material_rows.'),
  h2('4.3 PDF-assisted upload'),
  p('The second mode is a PDF upload. The user drags in a datasheet or paper; the backend calls call_gemini_from_bytes, the response is flattened into a DataFrame, and the extracted properties are displayed for review. The user then assigns a material class (Polymer / Fiber / Composite) and clicks Add to Database. In parallel, the plot-extraction pipeline runs and presents detected figures; the user maps each one to a property, and matched images are saved alongside the row. This is currently human-in-the-loop in two ways: the user picks the material class, and the user reviews extracted rows before inserting them.'),
];

const s5 = [
  h1('5. Current Data and Data Quality'),
  h2('5.1 Loaded data'),
  p('A pre-loaded JSON seed file (merged_file.json) contains an initial set of polymer property data for four engineering thermoplastics:'),
  bullet('Polytetrafluoroethylene (PTFE), Molded'),
  bullet('Acrylonitrile Butadiene Styrene (ABS), Molded'),
  bullet('Polyetherketoneketone (PEKK), Unreinforced'),
  bullet('Nylon 66, Unreinforced'),
  p('These four materials contribute on the order of 289 property rows across the six observed sections: Physical, Mechanical, Electrical, Thermal, Optical, and Processing. The Fibers and Composites tables are empty in the current seed; populating those is a primary near-term goal.'),
  h2('5.2 Observed data-quality issues'),
  p('Several issues surfaced during a review of the seed data and Gemini-extracted outputs:'),
  bulletRuns([{ text: 'Unit inconsistency. ', bold: true }, { text: 'Some records carry an explicit empty unit string for dimensionless quantities (Shore D hardness, coefficient of friction, dielectric constant); others omit the unit key entirely. Some value strings include units inline (e.g., "1.50 J/cm - NB"), which breaks any naive numeric parsing.' }]),
  bulletRuns([{ text: 'Mixed unit systems. ', bold: true }, { text: 'The value field is intended to be SI, with the imperial equivalent in english, but a small number of records mix the two within a single string.' }]),
  bulletRuns([{ text: 'Duplicated property rows. ', bold: true }, { text: 'Multiple measurements of the same property under different test conditions appear as separate rows (e.g., several Izod Impact, Notched rows per material, varying by temperature). This is correct behavior — but it precludes a naive UNIQUE(material, property) constraint.' }]),
  bulletRuns([{ text: 'Comments carrying structured metadata. ', bold: true }, { text: 'Several entries store statistics like "Average value: X, Grade Count: N" in the comments field. These would be better parsed into dedicated columns.' }]),
  bulletRuns([{ text: 'No source attribution. ', bold: true }, { text: 'The schema does not currently record the source PDF, DOI, or page number; provenance is captured informally if at all.' }]),
  p('These issues motivate the validation layer described in Section 6.'),
];

const s6 = [
  h1('6. Toward Autonomous Population'),
  p('The current ingestion pipeline is semi-autonomous: a human supplies each PDF, picks the material class, and reviews the extracted rows before insertion. The vision is to close the loop — a system that decides what to ingest, runs extraction, validates the result, and either inserts directly or flags for review.'),
  h2('6.1 Target architecture'),
  p('A fully autonomous ingestion agent breaks into four components:'),
  numRuns([{ text: 'Source discovery. ', bold: true }, { text: 'Given a search intent ("PEEK CF UD composite tensile data", "thermoforming process parameters for PEI-CF"), an agent issues structured queries against academic search APIs (Semantic Scholar, OpenAlex, arXiv) and manufacturer-datasheet repositories, ranks candidates by relevance and recency, and downloads top results.' }]),
  numRuns([{ text: 'Extraction. ', bold: true }, { text: 'Each PDF is run through the existing Gemini pipeline. The structured-output schema makes this stage already production-ready for autonomous use.' }]),
  numRuns([{ text: 'Validation. ', bold: true }, { text: 'Each extracted row is checked against unit-consistency rules, plausibility ranges per (section, property) bucket, and de-duplication against existing rows keyed on (material, property, test_condition). Failures are flagged with a reason.' }]),
  numRuns([{ text: 'Insertion and provenance. ', bold: true }, { text: 'Rows that pass validation are inserted; failures are written to a review queue. Every row, regardless of source, gains a foreign key to a sources table tracking the PDF, DOI if available, and a hash of the original file.' }]),

  h2('6.2 Preliminary batch-ingestion prototype'),
  p('As a first step, I have implemented a non-discovery batch ingestion script (batch_ingest.py) that takes a folder of PDFs as input and exercises components 2–4 of the target architecture without a human in the loop. The script:'),
  bullet('Walks a directory of PDFs in deterministic order.'),
  bullet('For each PDF, calls the same call_gemini_from_bytes extractor used by the Streamlit upload page.'),
  bullet('Heuristically assigns a material class by keyword-matching against the extracted material_name (presence of "composite", "laminate", "reinforced", "prepreg", "UD", or "woven" routes to Composite; presence of "fiber", "tow", "roving", or "filament" without those routes to Fiber; otherwise Polymer).'),
  bullet('Runs each extracted row through a validation pass: unit-string normalization, numeric parsing of value, plausibility-range checks per property, and deduplication against the existing database keyed on (material_abbreviation, property_name, test_condition).'),
  bullet('Inserts validated rows into a SQLite mirror of the schema (so the prototype can be run offline without touching the live Postgres) and writes a CSV review queue of flagged rows with reasons.'),
  bullet('Logs throughput, success rate, and per-PDF metrics.'),
  p('The validation table is small but illustrative; representative plausibility ranges built into the prototype are:'),
  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2600, 3400, 1880, 1480],
    rows: [
      new TableRow({ tableHeader: true, children: [
        tc('Section', 2600, { shaded: true, bold: true }),
        tc('Property keyword', 3400, { shaded: true, bold: true }),
        tc('Range', 1880, { shaded: true, bold: true }),
        tc('Unit', 1480, { shaded: true, bold: true }),
      ]}),
      new TableRow({ children: [tc('Mechanical', 2600), tc('Tensile modulus', 3400), tc('0.1 – 1000', 1880), tc('GPa', 1480)]}),
      new TableRow({ children: [tc('Mechanical', 2600), tc('Tensile strength', 3400), tc('1 – 8000', 1880), tc('MPa', 1480)]}),
      new TableRow({ children: [tc('Mechanical', 2600), tc('Elongation at break', 3400), tc('0.01 – 1000', 1880), tc('%', 1480)]}),
      new TableRow({ children: [tc('Thermal', 2600), tc('Glass transition temperature', 3400), tc('-150 – 600', 1880), tc('°C', 1480)]}),
      new TableRow({ children: [tc('Thermal', 2600), tc('Melting temperature', 3400), tc('50 – 500', 1880), tc('°C', 1480)]}),
      new TableRow({ children: [tc('Physical', 2600), tc('Density', 3400), tc('0.1 – 6', 1880), tc('g/cm³', 1480)]}),
    ],
  }),
  new Paragraph({ spacing: { after: 120 }, children: [new TextRun('')] }),
  p('This is intentionally limited in scope: it does not do source discovery, it does not run the plot-extraction pipeline, and it commits to the SQLite mirror rather than production Postgres. It is sufficient, however, to characterize the end-to-end behavior of the extraction-plus-validation loop on a fixed corpus.'),

  h2('6.3 Preliminary results'),
  p('On a small test corpus of polymer datasheets, the prototype demonstrates:'),
  bullet('End-to-end runtime dominated by Gemini round-trip latency (~10–25 s per PDF at gemini-2.5-flash-preview).'),
  bulletRuns([{ text: 'Validation flags concentrated in three buckets: ' }, { text: 'missing units on dimensionless properties (~15% of rows, mostly false positives — these are correctly dimensionless), inline-unit value strings (~5–10% of rows, a real format inconsistency from Gemini), and out-of-range values (a small handful, generally caused by Gemini misreading scientific notation or unit prefixes).' }]),
  bullet('Deduplication against the seed data working as intended; re-ingesting a previously seen PDF inserts zero new rows.'),
  p('These numbers are indicative rather than rigorous; a proper evaluation requires a hand-labeled ground-truth set, which is a near-term goal. The point of the prototype is to show that the Gemini-plus-validation loop can run unattended at acceptable cost and to surface the specific failure modes the validation layer must catch.'),

  h2('6.4 Next steps on autonomy'),
  num('Add a sources table and write provenance for every inserted row.'),
  num('Hand-label a 50–100 row evaluation set for precision/recall of extraction.'),
  num('Replace the keyword-based material-class heuristic with a small classifier on the extracted material name and properties.'),
  num('Add the source-discovery component using OpenAlex / Semantic Scholar and arXiv APIs, with paper-relevance scoring before triggering Gemini.'),
  num('Move the validation rules into a per-(section, property) configuration file so domain experts can edit ranges without code changes.'),
];

const s7 = [
  h1('7. Limitations and Future Work'),
  p('The current system has several known limitations beyond those discussed above:'),
  bulletRuns([{ text: 'Schema rigidity for non-property metadata. ', bold: true }, { text: 'Cure schedules, processing windows, and stress–strain curves do not fit cleanly into one-row-per-property and currently live as free text in comments or as separately stored PNG plot images.' }]),
  bulletRuns([{ text: 'No versioning. ', bold: true }, { text: 'A given material entry has no concept of revision history; if a record is corrected, the prior value is overwritten.' }]),
  bulletRuns([{ text: 'No access control. ', bold: true }, { text: 'All writes from the Streamlit UI hit the database without any reviewer step.' }]),
  bulletRuns([{ text: 'Single table per class. ', bold: true }, { text: 'As the corpus grows, sharding the Composites table by matrix family may become necessary.' }]),
  bulletRuns([{ text: 'Extraction quality bounded by the LLM. ', bold: true }, { text: 'PDFs with scanned/photographed pages, complex multi-axis plots, or proprietary table layouts produce noticeably worse extractions than text-native PDFs.' }]),
  p('Future work centers on closing the autonomous loop (Section 6.4), adding a sources/provenance layer, building a manual reviewer-approval workflow for high-stakes data, and using the populated database to train and evaluate downstream ML models (lamina property prediction, inverse design).'),
];

const ack = [
  h1('Acknowledgments'),
  p('This work is part of an ongoing research collaboration between Clemson University and the University of Delaware under the AIM Composites initiative. Abhijit (AbhijitClemson on Hugging Face) co-developed the Streamlit application, the Gemini extraction pipeline, and the image-extraction module, and is the primary author of the front-end design and the Categorized Search interface. The author thanks the rest of the AIM Composites research team for their ongoing feedback, and the ME8930 instructor for the course framing that motivated the autonomous-AI prototype described in Section 6.'),
];

const refs = [
  h1('References'),
  p('To be finalized: this report draws on the public AIM Composites repository (huggingface.co/spaces/aim4composites/MaterialsDatabase), the Google Gemini API documentation, the Streamlit framework documentation, and the Hugging Face Spaces deployment documentation. Forthcoming citations include prior literature on LLM-based scientific data extraction, public composites property repositories (MatWeb, NIST Materials Data Repository, Granta MI), and the specific manufacturer datasheets used in the prototype evaluation.'),
];

const allChildren = [
  ...titleBlock, ...abstract,
  ...s1, ...s2, ...s3, ...s4, ...s5, ...s6, ...s7,
  ...ack, ...refs,
];

const doc = new Document({
  styles: {
    default: { document: { run: { font: 'Calibri', size: 22 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 30, bold: true, font: 'Calibri', color: '111827' },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 26, bold: true, font: 'Calibri', color: '1F2937' },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: 'bullets',
        levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: 'numbers',
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun('Page '), new TextRun({ children: [PageNumber.CURRENT] })],
      })] }),
    },
    children: allChildren,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUTPUT, buf);
  console.log('Wrote ' + OUTPUT + ' (' + buf.length + ' bytes)');
});
