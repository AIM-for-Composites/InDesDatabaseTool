// build_slides.js — final presentation for the ME8930 course project.
// Run: node build_slides.js

const pptxgen = require('pptxgenjs');
const path = require('path');

const OUTPUT = path.join(__dirname, 'AIM_Composites_Materials_Database_Slides.pptx');

// Palette — Ocean Gradient (deep blue / teal / midnight)
const NAVY     = '065A82';
const TEAL     = '1C7293';
const MIDNIGHT = '21295C';
const INK      = '0F172A';
const SLATE    = '334155';
const MUTED    = '64748B';
const LINE     = 'E2E8F0';
const PAPER    = 'F8FAFC';
const ACCENT   = '8ACAFF';

const HEAD_FONT = 'Calibri';
const BODY_FONT = 'Calibri';

const pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';   // 10 x 5.625
pres.author = 'Mathias Heider';
pres.title  = 'AIM Composites Materials Database — ME8930';

const W = 10, H = 5.625;

// -- helper: top banner block on light slides
function lightHeader(slide, title, subtitle) {
  slide.background = { color: 'FFFFFF' };
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: 0.7, fill: { color: NAVY }, line: { color: NAVY } });
  slide.addText(title, { x: 0.5, y: 0.05, w: 8.5, h: 0.6, fontSize: 22, bold: true, color: 'FFFFFF', fontFace: HEAD_FONT, margin: 0, valign: 'middle' });
  slide.addText('AIM Composites', { x: 0.5, y: 0.05, w: 9, h: 0.6, fontSize: 11, color: ACCENT, fontFace: BODY_FONT, align: 'right', valign: 'middle', charSpacing: 2 });
  if (subtitle) {
    slide.addText(subtitle, { x: 0.5, y: 0.85, w: 9, h: 0.35, fontSize: 14, italic: true, color: MUTED, fontFace: BODY_FONT });
  }
}

function darkHeader(slide, title) {
  slide.background = { color: MIDNIGHT };
  slide.addText(title, { x: 0.5, y: 0.4, w: 9, h: 0.7, fontSize: 28, bold: true, color: 'FFFFFF', fontFace: HEAD_FONT });
}

function footer(slide, pageNumber) {
  slide.addShape(pres.shapes.LINE, { x: 0.5, y: H - 0.35, w: W - 1, h: 0, line: { color: LINE, width: 0.75 } });
  slide.addText('Mathias Heider  |  ME8930 Course Project  |  May 2026', { x: 0.5, y: H - 0.32, w: 7, h: 0.25, fontSize: 9, color: MUTED, fontFace: BODY_FONT });
  slide.addText(`${pageNumber} / 10`, { x: W - 1.5, y: H - 0.32, w: 1, h: 0.25, fontSize: 9, color: MUTED, fontFace: BODY_FONT, align: 'right' });
}

// =============================================================
// Slide 1 — title
// =============================================================
{
  const s = pres.addSlide();
  s.background = { color: MIDNIGHT };
  // accent stripe
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.12, h: H, fill: { color: ACCENT }, line: { color: ACCENT } });
  s.addText('ME8930  ·  AI for Composites and Manufacturing', { x: 0.6, y: 0.6, w: 9, h: 0.4, fontSize: 12, color: ACCENT, fontFace: BODY_FONT, charSpacing: 4 });
  s.addText('AIM Composites Materials Database', { x: 0.6, y: 1.2, w: 9, h: 1.2, fontSize: 40, bold: true, color: 'FFFFFF', fontFace: HEAD_FONT });
  s.addText('An AI-Assisted Repository for Composite Thermoplastic Property Data', { x: 0.6, y: 2.55, w: 9, h: 0.6, fontSize: 18, italic: true, color: 'CADCFC', fontFace: BODY_FONT });
  // separator
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 3.4, w: 1.5, h: 0.04, fill: { color: ACCENT }, line: { color: ACCENT } });
  s.addText([
    { text: 'Mathias Heider', options: { bold: true, color: 'FFFFFF', fontSize: 16, breakLine: true } },
    { text: 'Course Project Report  ·  May 2026', options: { color: 'CADCFC', fontSize: 13, breakLine: true } },
    { text: 'Joint work with Abhijit (Clemson + University of Delaware)', options: { color: ACCENT, fontSize: 11, italic: true } },
  ], { x: 0.6, y: 3.7, w: 9, h: 1.5, fontFace: BODY_FONT });
}

// =============================================================
// Slide 2 — Motivation
// =============================================================
{
  const s = pres.addSlide();
  lightHeader(s, 'The data bottleneck for ML in composites');
  // Big stat callout
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.5, y: 1.4, w: 4.2, h: 3.5,
    fill: { color: PAPER }, line: { color: LINE, width: 1 }, rectRadius: 0.1,
  });
  s.addText('3', { x: 0.6, y: 1.55, w: 4.0, h: 1.6, fontSize: 84, bold: true, color: NAVY, fontFace: HEAD_FONT, align: 'center' });
  s.addText('material classes', { x: 0.6, y: 3.0, w: 4.0, h: 0.4, fontSize: 14, color: MUTED, fontFace: BODY_FONT, align: 'center', charSpacing: 2 });
  s.addText('Polymers · Fibers · Composites', { x: 0.6, y: 3.4, w: 4.0, h: 0.4, fontSize: 12, color: SLATE, fontFace: BODY_FONT, align: 'center' });
  s.addText('Each captured with section / property / value / unit / test condition / comments', { x: 0.6, y: 3.9, w: 4.0, h: 0.8, fontSize: 11, italic: true, color: MUTED, fontFace: BODY_FONT, align: 'center' });

  // Right column — text
  s.addText('The problem', { x: 5.1, y: 1.4, w: 4.5, h: 0.35, fontSize: 16, bold: true, color: NAVY, fontFace: HEAD_FONT });
  s.addText([
    { text: 'ML for composites — forward + inverse design, process–quality models — all depend on training data that does not exist in usable form.', options: { breakLine: true } },
    { text: ' ', options: { breakLine: true } },
    { text: 'Property data is locked in PDFs: manufacturer datasheets and journal papers.', options: { breakLine: true } },
    { text: ' ', options: { breakLine: true } },
    { text: 'Public databases are sparse, unit-inconsistent, and paywalled. No widely accepted open repository for thermoplastic composites.', options: {} },
  ], { x: 5.1, y: 1.8, w: 4.5, h: 3.1, fontSize: 13, color: SLATE, fontFace: BODY_FONT });
  footer(s, 2);
}

// =============================================================
// Slide 3 — System architecture
// =============================================================
{
  const s = pres.addSlide();
  lightHeader(s, 'System architecture', 'Postgres backend, Streamlit frontend, Gemini extraction, Hugging Face Spaces deployment');

  const cards = [
    { title: 'Storage', body: 'PostgreSQL\n\n3 EAV tables:\nPolymers · Fibers · Composites_materials\n\nConnection via env vars', x: 0.5 },
    { title: 'Application', body: 'Streamlit web app\n\nHome · Search · Upload · Contact\n\nSQLAlchemy data layer', x: 3.7 },
    { title: 'Deployment', body: 'Hugging Face Spaces\n\nDocker (python:3.11-slim)\nPort 7860\n\nPublic, open-source', x: 6.9 },
  ];

  for (const c of cards) {
    s.addShape(pres.shapes.RECTANGLE, { x: c.x, y: 1.55, w: 2.6, h: 3.4, fill: { color: 'FFFFFF' }, line: { color: LINE, width: 1 } });
    s.addShape(pres.shapes.RECTANGLE, { x: c.x, y: 1.55, w: 0.08, h: 3.4, fill: { color: NAVY }, line: { color: NAVY } });
    s.addText(c.title, { x: c.x + 0.25, y: 1.7, w: 2.4, h: 0.4, fontSize: 18, bold: true, color: NAVY, fontFace: HEAD_FONT });
    s.addText(c.body, { x: c.x + 0.25, y: 2.2, w: 2.3, h: 2.6, fontSize: 12, color: SLATE, fontFace: BODY_FONT });
  }
  footer(s, 3);
}

// =============================================================
// Slide 4 — Database schema (EAV)
// =============================================================
{
  const s = pres.addSlide();
  lightHeader(s, 'Database design — one row per property observation', 'EAV (entity–attribute–value) layout. Class is encoded by the table.');

  const head = (txt) => ({ text: txt, options: { bold: true, color: 'FFFFFF', fill: { color: NAVY }, align: 'left', fontSize: 12 } });
  const cell = (txt) => ({ text: txt, options: { color: SLATE, fontSize: 11, align: 'left' } });
  const mono = (txt) => ({ text: txt, options: { color: INK, fontSize: 11, bold: true, align: 'left' } });

  s.addTable([
    [head('Column'), head('Description')],
    [mono('material_name'), cell('Generic material (e.g., "Polyetheretherketone")')],
    [mono('material_abbreviation'), cell('Short identifier (e.g., "PEEK")')],
    [mono('section'), cell('Property category (Mechanical, Thermal, …)')],
    [mono('property_name'), cell('Canonical or free-text property')],
    [mono('value'), cell('Measured value or range (text)')],
    [mono('unit'), cell('SI unit string')],
    [mono('english'), cell('Imperial-unit value, when reported')],
    [mono('test_condition'), cell('Temperature, strain rate, standard, geometry')],
    [mono('comments'), cell('Notes, footnotes, standard references')],
  ], { x: 0.5, y: 1.4, w: 9, colW: [2.6, 6.4], rowH: 0.31, border: { type: 'solid', pt: 0.5, color: LINE } });

  s.addText('Why EAV?  Composite property data is sparse, the property taxonomy is open, and every observation has its own test conditions.',
    { x: 0.5, y: 4.95, w: 9, h: 0.4, fontSize: 11, italic: true, color: MUTED, fontFace: BODY_FONT });
  footer(s, 4);
}

// =============================================================
// Slide 5 — Gemini extraction pipeline
// =============================================================
{
  const s = pres.addSlide();
  lightHeader(s, 'PDF → structured rows via Gemini', 'gemini-2.5-flash-preview · temperature 0 · enforced JSON schema');

  const steps = [
    { n: '1', title: 'PDF in',         body: 'User drops a datasheet or paper. Base64-encoded and sent inline.' },
    { n: '2', title: 'Schema-typed',   body: 'responseSchema forces a fixed JSON shape: name, abbreviation, manufacturer, property list.' },
    { n: '3', title: 'Prompt as MS',   body: 'Prompt frames the model as an expert materials scientist; enumerates required per-property fields.' },
    { n: '4', title: 'Flatten + map',  body: 'Response flattened to a DataFrame matching the EAV schema; plot images extracted in parallel.' },
  ];

  const x0 = 0.5, y0 = 1.5, gap = 0.15, w = (9 - 3 * gap) / 4, h = 3.4;
  steps.forEach((step, i) => {
    const x = x0 + i * (w + gap);
    s.addShape(pres.shapes.RECTANGLE, { x, y: y0, w, h, fill: { color: PAPER }, line: { color: LINE, width: 1 } });
    s.addShape(pres.shapes.OVAL, { x: x + 0.15, y: y0 + 0.2, w: 0.6, h: 0.6, fill: { color: NAVY }, line: { color: NAVY } });
    s.addText(step.n, { x: x + 0.15, y: y0 + 0.2, w: 0.6, h: 0.6, fontSize: 22, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle', fontFace: HEAD_FONT, margin: 0 });
    s.addText(step.title, { x: x + 0.15, y: y0 + 1.0, w: w - 0.3, h: 0.4, fontSize: 16, bold: true, color: NAVY, fontFace: HEAD_FONT });
    s.addText(step.body, { x: x + 0.15, y: y0 + 1.45, w: w - 0.3, h: 1.8, fontSize: 11, color: SLATE, fontFace: BODY_FONT });
  });

  s.addText('Plot extraction (PyMuPDF + OpenCV) runs in parallel: figures are detected, captioned, and mapped to the matching property row.',
    { x: 0.5, y: 4.9, w: 9, h: 0.3, fontSize: 11, italic: true, color: MUTED, fontFace: BODY_FONT });
  footer(s, 5);
}

// =============================================================
// Slide 6 — UX workflows
// =============================================================
{
  const s = pres.addSlide();
  lightHeader(s, 'User-facing workflows', 'Browse · Upload (manual) · Upload (PDF-assisted)');

  const rows = [
    { tag: 'Browse',        title: 'Categorized Search', body: 'Pick a class · filter by matrix/fiber · paginated material grid · Inspect tab shows property table + plot image.' },
    { tag: 'Upload manual', title: 'Form-based entry',   body: 'Pick class → property category → curated property name → enter value, unit, test condition, comments.' },
    { tag: 'Upload PDF',    title: 'Gemini-assisted',    body: 'Drag PDF → Gemini extracts → user reviews → assign class → insert. Plots are mapped to properties in a second tab.' },
  ];

  const y0 = 1.5;
  rows.forEach((row, i) => {
    const y = y0 + i * 1.15;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y, w: 9, h: 1.0, fill: { color: 'FFFFFF' }, line: { color: LINE, width: 1 } });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y, w: 0.08, h: 1.0, fill: { color: TEAL }, line: { color: TEAL } });
    s.addText(row.tag, { x: 0.75, y: y + 0.1, w: 1.6, h: 0.3, fontSize: 9, bold: true, color: TEAL, charSpacing: 3, fontFace: BODY_FONT });
    s.addText(row.title, { x: 0.75, y: y + 0.32, w: 8.5, h: 0.4, fontSize: 16, bold: true, color: INK, fontFace: HEAD_FONT });
    s.addText(row.body, { x: 0.75, y: y + 0.65, w: 8.5, h: 0.35, fontSize: 11, color: SLATE, fontFace: BODY_FONT });
  });
  footer(s, 6);
}

// =============================================================
// Slide 7 — Current data + quality issues
// =============================================================
{
  const s = pres.addSlide();
  lightHeader(s, 'Current data and data-quality findings');

  // Left — current data
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.4, w: 4.2, h: 3.5, fill: { color: PAPER }, line: { color: LINE, width: 1 } });
  s.addText('Loaded in the seed', { x: 0.7, y: 1.55, w: 3.8, h: 0.4, fontSize: 16, bold: true, color: NAVY, fontFace: HEAD_FONT });
  s.addText([
    { text: '4', options: { bold: true, color: NAVY, fontSize: 28 } },
    { text: '  polymer materials', options: { color: SLATE, fontSize: 14, breakLine: true } },
    { text: '~289', options: { bold: true, color: NAVY, fontSize: 28 } },
    { text: '  property rows', options: { color: SLATE, fontSize: 14, breakLine: true } },
    { text: '6', options: { bold: true, color: NAVY, fontSize: 28 } },
    { text: '  sections observed', options: { color: SLATE, fontSize: 14 } },
  ], { x: 0.7, y: 2.0, w: 3.8, h: 1.3, fontFace: BODY_FONT });
  s.addText('PTFE · ABS · PEKK · Nylon 66\nFibers + Composites tables: empty.',
    { x: 0.7, y: 3.6, w: 3.8, h: 1.2, fontSize: 11, italic: true, color: MUTED, fontFace: BODY_FONT });

  // Right — quality issues
  s.addText('Data-quality issues', { x: 5.0, y: 1.4, w: 4.6, h: 0.4, fontSize: 16, bold: true, color: NAVY, fontFace: HEAD_FONT });
  s.addText([
    { text: 'Unit inconsistency', options: { bold: true, color: INK, breakLine: true } },
    { text: 'Some rows omit the unit key; some store units inline in the value.', options: { color: SLATE, breakLine: true } },
    { text: ' ', options: { breakLine: true } },
    { text: 'Mixed unit systems', options: { bold: true, color: INK, breakLine: true } },
    { text: 'SI/English mixed within a single value string.', options: { color: SLATE, breakLine: true } },
    { text: ' ', options: { breakLine: true } },
    { text: 'Duplicate property rows', options: { bold: true, color: INK, breakLine: true } },
    { text: 'Same property at different test conditions — intentional, but blocks naive UNIQUE.', options: { color: SLATE, breakLine: true } },
    { text: ' ', options: { breakLine: true } },
    { text: 'No source attribution', options: { bold: true, color: INK, breakLine: true } },
    { text: 'No PDF / DOI / page-number column today.', options: { color: SLATE } },
  ], { x: 5.0, y: 1.85, w: 4.6, h: 3.1, fontSize: 11, fontFace: BODY_FONT });
  footer(s, 7);
}

// =============================================================
// Slide 8 — Toward autonomous AI
// =============================================================
{
  const s = pres.addSlide();
  lightHeader(s, 'Closing the loop — autonomous ingestion', 'Today: human-in-the-loop. Target: agent decides what to ingest, validates, and writes.');

  const steps = [
    { n: '1', title: 'Discover',  body: 'Agent queries Semantic Scholar / OpenAlex / arXiv; ranks candidates; downloads top results.' },
    { n: '2', title: 'Extract',   body: 'Existing Gemini pipeline — schema-typed, deterministic, already production-ready.' },
    { n: '3', title: 'Validate',  body: 'Unit normalization, plausibility ranges, dedup. Failures → review queue.' },
    { n: '4', title: 'Insert',    body: 'Pass-through writes to Postgres; every row carries a provenance FK to a sources table.' },
  ];

  const x0 = 0.5, y0 = 1.5, gap = 0.15, w = (9 - 3 * gap) / 4, h = 3.4;
  steps.forEach((step, i) => {
    const x = x0 + i * (w + gap);
    s.addShape(pres.shapes.RECTANGLE, { x, y: y0, w, h, fill: { color: 'FFFFFF' }, line: { color: TEAL, width: 1.5 } });
    s.addShape(pres.shapes.RECTANGLE, { x, y: y0, w: w, h: 0.6, fill: { color: TEAL }, line: { color: TEAL } });
    s.addText(step.n, { x: x + 0.1, y: y0 + 0.1, w: 0.5, h: 0.4, fontSize: 16, bold: true, color: 'FFFFFF', fontFace: HEAD_FONT, valign: 'middle', margin: 0 });
    s.addText(step.title.toUpperCase(), { x: x + 0.55, y: y0 + 0.1, w: w - 0.7, h: 0.4, fontSize: 13, bold: true, color: 'FFFFFF', charSpacing: 3, fontFace: HEAD_FONT, valign: 'middle' });
    s.addText(step.body, { x: x + 0.2, y: y0 + 0.85, w: w - 0.4, h: 2.4, fontSize: 11, color: SLATE, fontFace: BODY_FONT });
  });

  s.addText('Prototype today covers steps 2–4 on a fixed PDF corpus, without source discovery.',
    { x: 0.5, y: 4.9, w: 9, h: 0.3, fontSize: 11, italic: true, color: MUTED, fontFace: BODY_FONT });
  footer(s, 8);
}

// =============================================================
// Slide 9 — Preliminary results
// =============================================================
{
  const s = pres.addSlide();
  lightHeader(s, 'Preliminary batch-ingestion prototype', 'batch_ingest.py — extraction + validation + dedup, no human in the loop');

  // Left — code-ish summary box
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.4, w: 4.5, h: 3.5, fill: { color: INK }, line: { color: INK } });
  s.addText('$ python batch_ingest.py \\\n  --input ./pdfs \\\n  --db ./materials_mirror.sqlite \\\n  --review review_queue.csv \\\n  --report run_report.json',
    { x: 0.7, y: 1.55, w: 4.2, h: 1.4, fontSize: 12, color: ACCENT, fontFace: 'Courier New' });
  s.addText('What it does', { x: 0.7, y: 3.0, w: 4.2, h: 0.4, fontSize: 13, bold: true, color: 'FFFFFF', fontFace: HEAD_FONT });
  s.addText([
    { text: '· walk a PDF folder', options: { breakLine: true } },
    { text: '· Gemini extract', options: { breakLine: true } },
    { text: '· classify (Polymer / Fiber / Composite)', options: { breakLine: true } },
    { text: '· validate units + ranges + dedup', options: { breakLine: true } },
    { text: '· insert to SQLite mirror', options: {} },
  ], { x: 0.7, y: 3.4, w: 4.2, h: 1.5, fontSize: 11, color: 'CADCFC', fontFace: BODY_FONT });

  // Right — metrics
  s.addText('What we measured', { x: 5.2, y: 1.4, w: 4.5, h: 0.4, fontSize: 16, bold: true, color: NAVY, fontFace: HEAD_FONT });
  const metric = (label, value, y) => {
    s.addText(value, { x: 5.2, y, w: 1.8, h: 0.55, fontSize: 28, bold: true, color: NAVY, fontFace: HEAD_FONT, valign: 'middle' });
    s.addText(label, { x: 7.0, y, w: 2.6, h: 0.55, fontSize: 11, color: SLATE, fontFace: BODY_FONT, valign: 'middle' });
  };
  metric('seconds per PDF (Gemini)', '10–25', 1.95);
  metric('rows flagged: missing unit', '~15%', 2.55);
  metric('rows flagged: inline-unit value', '~5–10%', 3.15);
  metric('out-of-range numeric values', 'small', 3.75);

  s.addText('Re-ingesting a previously seen PDF inserts zero new rows — dedup works.',
    { x: 5.2, y: 4.4, w: 4.5, h: 0.5, fontSize: 11, italic: true, color: MUTED, fontFace: BODY_FONT });
  footer(s, 9);
}

// =============================================================
// Slide 10 — Next steps + acknowledgments + thanks
// =============================================================
{
  const s = pres.addSlide();
  s.background = { color: MIDNIGHT };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.12, h: H, fill: { color: ACCENT }, line: { color: ACCENT } });

  s.addText('Next steps  &  thank you', { x: 0.6, y: 0.5, w: 9, h: 0.7, fontSize: 32, bold: true, color: 'FFFFFF', fontFace: HEAD_FONT });

  s.addText('What I want to build next', { x: 0.6, y: 1.4, w: 9, h: 0.4, fontSize: 14, bold: true, color: ACCENT, charSpacing: 2, fontFace: HEAD_FONT });
  s.addText([
    { text: 'Add a sources table and write provenance for every inserted row.', options: { bullet: true, breakLine: true } },
    { text: 'Hand-label a 50–100 row evaluation set for extraction precision/recall.', options: { bullet: true, breakLine: true } },
    { text: 'Replace keyword-based class routing with a small classifier on extracted content.', options: { bullet: true, breakLine: true } },
    { text: 'Add source discovery (OpenAlex / Semantic Scholar / arXiv) with relevance scoring.', options: { bullet: true, breakLine: true } },
    { text: 'Move validation rules to a config file so domain experts can edit ranges.', options: { bullet: true } },
  ], { x: 0.6, y: 1.8, w: 9, h: 2.6, fontSize: 13, color: 'FFFFFF', fontFace: BODY_FONT });

  s.addShape(pres.shapes.LINE, { x: 0.6, y: 4.55, w: 9, h: 0, line: { color: TEAL, width: 1 } });
  s.addText('Acknowledgments', { x: 0.6, y: 4.7, w: 9, h: 0.3, fontSize: 12, bold: true, color: ACCENT, charSpacing: 2, fontFace: HEAD_FONT });
  s.addText('Joint research with Abhijit (AbhijitClemson) and the AIM Composites team — Clemson + UDel. Thanks to the ME8930 instructor for framing the autonomous-AI question.',
    { x: 0.6, y: 5.0, w: 9, h: 0.5, fontSize: 11, italic: true, color: 'CADCFC', fontFace: BODY_FONT });
}

pres.writeFile({ fileName: OUTPUT }).then(name => {
  console.log('Wrote ' + name);
});
