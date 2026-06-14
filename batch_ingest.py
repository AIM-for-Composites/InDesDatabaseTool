r"""
batch_ingest.py — preliminary autonomous batch-ingestion prototype for the
AIM Composites Materials Database.

This script takes a folder of PDFs as input and exercises the extraction +
validation + insertion stages of the target autonomous-ingestion architecture
described in Section 6 of the project report:

  PDFs  ->  Gemini extraction  ->  validation  ->  dedup  ->  SQLite mirror
                                                          \-> review_queue.csv

It intentionally does NOT do source discovery (component 1) and does NOT run
the plot-extraction / image-mapping pipeline. It is a minimal closed loop
sufficient to characterize the cost, throughput, and failure modes of the
extraction-plus-validation stage on a fixed corpus.

Usage:
    export GEMINI_API_KEY=...
    python batch_ingest.py --input ./pdfs --db ./materials_mirror.sqlite \
        --review review_queue.csv --report run_report.json

Author: Mathias Heider, ME8930 course project, May 2026.
Extraction prompt and schema adapted from the live Streamlit app
(page_files/categorized/Backend/upload_backend.py) co-developed with
Abhijit (AbhijitClemson) on the AIM Composites HF Space.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import logging
import os
import re
import sqlite3
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"
GEMINI_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)
REQUEST_TIMEOUT_S = 300

# Same JSON schema the live Streamlit app uses, so behavior matches production.
EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "material_name": {"type": "STRING"},
        "material_abbreviation": {"type": "STRING"},
        "trade_grade": {"type": "STRING"},
        "manufacturer": {"type": "STRING"},
        "mechanical_properties": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "section": {"type": "STRING"},
                    "property_name": {"type": "STRING"},
                    "value": {"type": "STRING"},
                    "unit": {"type": "STRING"},
                    "english": {"type": "STRING"},
                    "test_condition": {"type": "STRING"},
                    "comments": {"type": "STRING"},
                },
                "required": [
                    "section",
                    "property_name",
                    "value",
                    "english",
                    "comments",
                ],
            },
        },
    },
}

EXTRACTION_PROMPT = (
    "You are an expert materials scientist. From the attached PDF, extract:\n"
    "- material_name (generic material, e.g., isotactic polypropylene)\n"
    "- material_abbreviation\n"
    "- trade_grade (commercial or trade name; '' if not provided)\n"
    "- manufacturer (company; '' if not provided)\n\n"
    "Extract ALL properties across categories (Mechanical, Thermal, Electrical, "
    "Physical, Optical, Rheological, Processing) and return them as a single "
    "'mechanical_properties' list.\n\n"
    "For each property you MUST extract: property_name, value (or range), unit, "
    "english (alternate units, '' if absent), test_condition, comments "
    "('' if none).\n"
    "Respond ONLY with valid JSON following the schema."
)

# Heuristic plausibility ranges per (section, property) bucket. Wide enough
# to flag obvious errors (e.g. units off by 10^3) without false-positiving
# normal data. Units are the natural SI unit for the property; the validator
# accepts values within these ranges OR with empty/unknown unit (to avoid
# punishing dimensionless properties).
PLAUSIBILITY: dict[tuple[str, str], tuple[float, float, str]] = {
    # (section_lower, property_keyword_lower) -> (min, max, expected_unit)
    ("mechanical", "tensile modulus"): (0.1, 1000.0, "GPa"),
    ("mechanical", "tensile strength"): (1.0, 8000.0, "MPa"),
    ("mechanical", "flexural modulus"): (0.1, 800.0, "GPa"),
    ("mechanical", "flexural strength"): (1.0, 4000.0, "MPa"),
    ("mechanical", "elongation"): (0.01, 1000.0, "%"),
    ("mechanical", "impact"): (0.01, 5000.0, "J/m"),
    ("mechanical", "shear modulus"): (0.05, 500.0, "GPa"),
    ("thermal", "glass transition"): (-150.0, 600.0, "°C"),
    ("thermal", "melting"): (50.0, 500.0, "°C"),
    ("thermal", "crystallization"): (0.0, 500.0, "°C"),
    ("thermal", "decomposition"): (100.0, 1200.0, "°C"),
    ("thermal", "cte"): (-50.0, 500.0, "ppm/°C"),
    ("physical", "density"): (0.1, 6.0, "g/cm³"),
}

# Material-class routing keywords (applied to extracted material_name).
# Strong composite signals: structural / format / processing terms that imply
# a matrix + reinforcement system. "Carbon fiber" / "glass fiber" by themselves
# are ambiguous (a Toray T700S datasheet is a fiber, not a composite), so they
# are not in this list — they only push toward Composite when paired with one
# of the strong signals below.
COMPOSITE_KEYWORDS = (
    "composite", "laminate", "reinforced", "prepreg",
    "cf/", "gf/", "fiber-reinforced", "fibre-reinforced",
    "ud ", "uni-directional", "unidirectional", "woven",
)
FIBER_KEYWORDS = (
    "fiber", "fibre", "yarn", "tow", "roving", "filament",
)


# ---------------------------------------------------------------------------
# SQLite mirror of the Postgres schema
# ---------------------------------------------------------------------------

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS Polymers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    material_name TEXT, material_abbreviation TEXT, section TEXT,
    property_name TEXT, value TEXT, unit TEXT, english TEXT,
    test_condition TEXT, comments TEXT
);
CREATE TABLE IF NOT EXISTS Fibers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    material_name TEXT, material_abbreviation TEXT, section TEXT,
    property_name TEXT, value TEXT, unit TEXT, english TEXT,
    test_condition TEXT, comments TEXT
);
CREATE TABLE IF NOT EXISTS Composites_materials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    material_name TEXT, material_abbreviation TEXT, section TEXT,
    property_name TEXT, value TEXT, unit TEXT, english TEXT,
    test_condition TEXT, comments TEXT
);
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pdf_filename TEXT UNIQUE,
    pdf_sha1 TEXT,
    ingested_at TEXT,
    material_class TEXT,
    material_abbreviation TEXT
);
"""

TABLE_FOR_CLASS = {
    "Polymer": "Polymers",
    "Fiber": "Fibers",
    "Composite": "Composites_materials",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PropertyRow:
    material_name: str
    material_abbreviation: str
    section: str
    property_name: str
    value: str
    unit: str
    english: str
    test_condition: str
    comments: str


@dataclasses.dataclass
class Flag:
    row: PropertyRow
    pdf: str
    reason: str


@dataclasses.dataclass
class PdfResult:
    pdf: str
    elapsed_s: float
    extracted: int
    inserted: int
    flagged: int
    duplicates: int
    material_class: Optional[str]
    material_abbreviation: Optional[str]
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def call_gemini(pdf_bytes: bytes, api_key: str) -> Optional[dict[str, Any]]:
    """POST a PDF to Gemini with the structured-output schema, return parsed JSON."""
    url = GEMINI_URL_TEMPLATE.format(model=GEMINI_MODEL, key=api_key)
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": EXTRACTION_PROMPT},
                    {
                        "inlineData": {
                            "mimeType": "application/pdf",
                            "data": base64.b64encode(pdf_bytes).decode("utf-8"),
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": EXTRACTION_SCHEMA,
        },
    }

    response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        return None
    parts = candidates[0].get("content", {}).get("parts", [])
    for part in parts:
        text = part.get("text", "").strip()
        if text.startswith("{"):
            return json.loads(text)
    return None


def flatten_extraction(extraction: dict[str, Any]) -> list[PropertyRow]:
    """Flatten the Gemini response into PropertyRow objects."""
    name = extraction.get("material_name") or ""
    abbr = extraction.get("material_abbreviation") or _autoabbr(name)
    rows: list[PropertyRow] = []
    for item in extraction.get("mechanical_properties") or []:
        rows.append(
            PropertyRow(
                material_name=name,
                material_abbreviation=abbr,
                section=(item.get("section") or "").strip() or "Mechanical",
                property_name=(item.get("property_name") or "").strip()
                or "Unknown property",
                value=(item.get("value") or "").strip(),
                unit=(item.get("unit") or "").strip(),
                english=(item.get("english") or "").strip(),
                test_condition=(item.get("test_condition") or "").strip(),
                comments=(item.get("comments") or "").strip(),
            )
        )
    return rows


def _autoabbr(name: str) -> str:
    if not name:
        return "UNKNOWN"
    parts = [p[0] for p in name.split() if p and p[0].isalpha()]
    return "".join(parts).upper() or name[:6].upper()


# ---------------------------------------------------------------------------
# Material-class routing
# ---------------------------------------------------------------------------


def classify_material(name: str, rows: Iterable[PropertyRow]) -> str:
    """Keyword-based classifier; returns 'Polymer' | 'Fiber' | 'Composite'.

    Looks at the material name plus the section names extracted. Composite
    keywords beat fiber keywords beat the polymer default.
    """
    haystack = (name or "").lower()
    for row in rows:
        haystack += " " + (row.material_name or "").lower()

    if any(k in haystack for k in COMPOSITE_KEYWORDS):
        return "Composite"
    # A plain "fiber" (e.g. T700S carbon fiber datasheet) is a Fiber, not a
    # Composite, *unless* it's described as reinforcement of a matrix.
    if any(k in haystack for k in FIBER_KEYWORDS):
        return "Fiber"
    return "Polymer"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?")


def parse_numeric(value: str) -> Optional[float]:
    """Pull the first number out of a value string. Returns None if absent."""
    if not value:
        return None
    match = _NUMBER_RE.search(value)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def validate(row: PropertyRow) -> Optional[str]:
    """Return a flag reason string if the row should be flagged; else None."""
    # Empty / placeholder values
    if not row.value or row.value.strip().lower() in {"n/a", "na", "-", "--"}:
        return "empty_value"

    # Inline-unit detection: value contains an alphabetic unit sequence and the
    # unit field is empty. This catches "1.50 J/cm - NB" style entries.
    if (not row.unit) and re.search(r"\d.*[a-zA-Z]{1,}", row.value):
        # tolerate plain "yes"/"no"/textual descriptors as comments-but-value
        if not re.match(r"^[a-zA-Z][a-zA-Z\s\-]+$", row.value.strip()):
            return "inline_unit_in_value"

    # Plausibility-range check
    numeric = parse_numeric(row.value)
    if numeric is None:
        return None  # non-numeric properties (descriptive) skip range check
    sec = row.section.lower()
    name = row.property_name.lower()
    for (sec_key, prop_key), (lo, hi, _unit) in PLAUSIBILITY.items():
        if sec_key in sec and prop_key in name:
            if not (lo <= numeric <= hi):
                return f"out_of_range[{lo},{hi}]"
            break
    return None


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------


def init_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA_DDL)
    conn.commit()
    return conn


def already_inserted(
    conn: sqlite3.Connection,
    table: str,
    row: PropertyRow,
) -> bool:
    cur = conn.execute(
        f"SELECT 1 FROM {table} "
        f"WHERE material_abbreviation = ? AND property_name = ? "
        f"AND IFNULL(test_condition,'') = IFNULL(?, '') "
        f"LIMIT 1",
        (row.material_abbreviation, row.property_name, row.test_condition),
    )
    return cur.fetchone() is not None


def insert_row(conn: sqlite3.Connection, table: str, row: PropertyRow) -> None:
    conn.execute(
        f"INSERT INTO {table} "
        f"(material_name, material_abbreviation, section, property_name, "
        f" value, unit, english, test_condition, comments) "
        f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            row.material_name,
            row.material_abbreviation,
            row.section,
            row.property_name,
            row.value,
            row.unit,
            row.english,
            row.test_condition,
            row.comments,
        ),
    )


def record_source(
    conn: sqlite3.Connection,
    pdf_path: Path,
    sha1: str,
    material_class: Optional[str],
    abbr: Optional[str],
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO sources "
        "(pdf_filename, pdf_sha1, ingested_at, material_class, material_abbreviation) "
        "VALUES (?, ?, datetime('now'), ?, ?)",
        (pdf_path.name, sha1, material_class, abbr),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def process_pdf(
    pdf_path: Path,
    conn: sqlite3.Connection,
    api_key: str,
    flags: list[Flag],
) -> PdfResult:
    started = time.time()
    pdf_bytes = pdf_path.read_bytes()
    import hashlib

    sha1 = hashlib.sha1(pdf_bytes).hexdigest()

    # Skip if we already ingested this exact file
    already = conn.execute(
        "SELECT 1 FROM sources WHERE pdf_sha1 = ? LIMIT 1", (sha1,)
    ).fetchone()
    if already:
        return PdfResult(
            pdf=pdf_path.name,
            elapsed_s=time.time() - started,
            extracted=0,
            inserted=0,
            flagged=0,
            duplicates=0,
            material_class=None,
            material_abbreviation=None,
            error="skipped_seen_sha1",
        )

    try:
        extraction = call_gemini(pdf_bytes, api_key)
    except requests.RequestException as exc:
        return PdfResult(
            pdf=pdf_path.name,
            elapsed_s=time.time() - started,
            extracted=0,
            inserted=0,
            flagged=0,
            duplicates=0,
            material_class=None,
            material_abbreviation=None,
            error=f"gemini_error:{exc}",
        )

    if not extraction:
        return PdfResult(
            pdf=pdf_path.name,
            elapsed_s=time.time() - started,
            extracted=0,
            inserted=0,
            flagged=0,
            duplicates=0,
            material_class=None,
            material_abbreviation=None,
            error="empty_extraction",
        )

    rows = flatten_extraction(extraction)
    material_class = classify_material(extraction.get("material_name", ""), rows)
    table = TABLE_FOR_CLASS[material_class]
    abbr = rows[0].material_abbreviation if rows else None

    record_source(conn, pdf_path, sha1, material_class, abbr)

    inserted = 0
    flagged = 0
    duplicates = 0
    for row in rows:
        reason = validate(row)
        if reason is not None:
            flags.append(Flag(row=row, pdf=pdf_path.name, reason=reason))
            flagged += 1
            continue
        if already_inserted(conn, table, row):
            duplicates += 1
            continue
        insert_row(conn, table, row)
        inserted += 1
    conn.commit()

    return PdfResult(
        pdf=pdf_path.name,
        elapsed_s=time.time() - started,
        extracted=len(rows),
        inserted=inserted,
        flagged=flagged,
        duplicates=duplicates,
        material_class=material_class,
        material_abbreviation=abbr,
    )


def write_review_queue(flags: Iterable[Flag], path: Path) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "pdf",
                "reason",
                "material_abbreviation",
                "section",
                "property_name",
                "value",
                "unit",
                "test_condition",
                "comments",
            ]
        )
        for flag in flags:
            r = flag.row
            writer.writerow(
                [
                    flag.pdf,
                    flag.reason,
                    r.material_abbreviation,
                    r.section,
                    r.property_name,
                    r.value,
                    r.unit,
                    r.test_condition,
                    r.comments,
                ]
            )


def summarize(results: list[PdfResult]) -> dict[str, Any]:
    total_pdfs = len(results)
    successes = [r for r in results if not r.error]
    extracted = sum(r.extracted for r in successes)
    inserted = sum(r.inserted for r in successes)
    flagged = sum(r.flagged for r in successes)
    duplicates = sum(r.duplicates for r in successes)
    total_elapsed = sum(r.elapsed_s for r in results)
    avg_elapsed = total_elapsed / total_pdfs if total_pdfs else 0.0
    error_breakdown: dict[str, int] = {}
    for r in results:
        if r.error:
            key = r.error.split(":", 1)[0]
            error_breakdown[key] = error_breakdown.get(key, 0) + 1
    return {
        "pdfs_seen": total_pdfs,
        "pdfs_ok": len(successes),
        "errors_by_kind": error_breakdown,
        "rows_extracted": extracted,
        "rows_inserted": inserted,
        "rows_flagged": flagged,
        "rows_duplicate": duplicates,
        "insert_rate": inserted / extracted if extracted else 0.0,
        "flag_rate": flagged / extracted if extracted else 0.0,
        "duplicate_rate": duplicates / extracted if extracted else 0.0,
        "avg_seconds_per_pdf": round(avg_elapsed, 2),
        "total_seconds": round(total_elapsed, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Folder of PDFs to ingest")
    parser.add_argument("--db", default=Path("materials_mirror.sqlite"),
                        type=Path, help="SQLite mirror path")
    parser.add_argument("--review", default=Path("review_queue.csv"),
                        type=Path, help="CSV of flagged rows")
    parser.add_argument("--report", default=Path("run_report.json"),
                        type=Path, help="JSON run summary")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N PDFs (for testing)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger("batch_ingest")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        log.error("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")
        return 2

    pdfs = sorted(p for p in args.input.rglob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        log.error("No PDFs found under %s", args.input)
        return 2

    log.info("Ingesting %d PDFs into %s", len(pdfs), args.db)
    conn = init_db(args.db)
    flags: list[Flag] = []
    results: list[PdfResult] = []
    for i, pdf in enumerate(pdfs, start=1):
        log.info("[%d/%d] %s", i, len(pdfs), pdf.name)
        result = process_pdf(pdf, conn, api_key, flags)
        results.append(result)
        log.info(
            "  -> class=%s abbr=%s extracted=%d inserted=%d flagged=%d "
            "duplicates=%d elapsed=%.1fs error=%s",
            result.material_class,
            result.material_abbreviation,
            result.extracted,
            result.inserted,
            result.flagged,
            result.duplicates,
            result.elapsed_s,
            result.error,
        )

    write_review_queue(flags, args.review)
    summary = summarize(results)
    args.report.write_text(json.dumps(summary, indent=2))
    log.info("Run summary written to %s", args.report)
    log.info("Review queue (%d rows) written to %s", len(flags), args.review)

    conn.close()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
