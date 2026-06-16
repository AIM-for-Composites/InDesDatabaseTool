r"""
extraction.py — single source of truth for the AIM Composites extraction pipeline.

This module consolidates the Gemini prompt/schema and all post-extraction
processing that used to live (and drift) inside ``batch_ingest.py`` and the
Streamlit app repo's ``page_files/categorized/page6.py`` /
``Backend/Pdf_DataExtraction.py``.

Pipeline shape::

    pdf_bytes
       -> extract_from_pdf()        # Gemini structured output -> Extraction
       -> verify_against_text()     # ground every value in the PDF text (Task 1)
       -> to_rows()                 # Extraction -> list[PropertyRow] (Task 2/3/5/6)
       -> (caller inserts rows, carrying `status`/`flag_reason`)

Design notes
------------
* The model is asked for a *list* of materials, each with structured,
  unit-aware property values plus a verbatim ``source_quote`` and ``page``.
* Numbers are parsed into ``value_num`` / ``value_min`` / ``value_max`` /
  ``qualifier`` and normalized to a canonical unit (``unit_canonical`` +
  ``value_si``) with :mod:`pint`. Plausibility is range-checked *after*
  conversion, so a GPa-vs-MPa mix no longer false-flags.
* Nothing is silently dropped: every row carries a ``status`` and, when not
  ``ok``, a ``flag_reason``.

Keep ``temperature=0`` and bump :data:`PROMPT_VERSION` on any prompt change.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import logging
import re
import time
import unicodedata
from typing import Any, Optional

import requests

try:  # PyMuPDF — used for text grounding (Task 1) and scanned detection (Task 10)
    import fitz  # type: ignore
except Exception:  # pragma: no cover - import guard
    fitz = None  # type: ignore

try:
    import pint  # unit normalization (Task 3)
except Exception:  # pragma: no cover - import guard
    pint = None  # type: ignore

log = logging.getLogger("extraction")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"
PROMPT_VERSION = "2.0"

GEMINI_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)
GEMINI_UPLOAD_URL = (
    "https://generativelanguage.googleapis.com/upload/v1beta/files?key={key}"
)
GEMINI_FILE_STATUS_URL = (
    "https://generativelanguage.googleapis.com/v1beta/{name}?key={key}"
)
REQUEST_TIMEOUT_S = 300

# Transient-failure retry policy, mirrored from pdf_crawler.http_get(). The
# Gemini free tier returns 429s under load; a request that fails once is often
# fine moments later. Retry connection errors and 429/5xx, honoring Retry-After.
MAX_RETRIES = 3
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
BACKOFF_BASE = 2.0  # seconds; exponential per attempt

# Inline a PDF up to this size; above it, upload via the Gemini File API so the
# base64-inflated request stays under the generateContent size limit (Task 10).
INLINE_PDF_LIMIT_BYTES = 15 * 1024 * 1024
INLINE_PAGE_LIMIT = 80
# Average extractable characters/page below this => treat as scanned/image-only.
SCANNED_TEXT_PER_PAGE = 50


# --- enums (Task 8) --------------------------------------------------------

# Aligned with the app repo's page6.py PROPERTY_CATEGORIES.
SECTION_ENUM = [
    "Mechanical",
    "Thermal",
    "Electrical",
    "Physical",
    "Optical",
    "Rheological",
    "Processing",
    "Descriptive",
    "Composition/Reinforcement",
    "Architecture/Structure",
]

MATERIAL_CLASS_ENUM = ["Polymer", "Fiber", "Composite"]


# --- schema (Tasks 2/3/8) --------------------------------------------------

_PROPERTY_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "section": {"type": "STRING", "enum": SECTION_ENUM},
        "property_name": {"type": "STRING"},
        "value_raw": {"type": "STRING"},
        "value_num": {"type": "NUMBER", "nullable": True},
        "value_min": {"type": "NUMBER", "nullable": True},
        "value_max": {"type": "NUMBER", "nullable": True},
        "qualifier": {"type": "STRING"},
        "unit": {"type": "STRING"},
        "test_condition": {"type": "STRING"},
        "comments": {"type": "STRING"},
        "source_quote": {"type": "STRING"},
        "page": {"type": "INTEGER"},
    },
    "required": ["section", "property_name", "value_raw", "unit", "source_quote", "page"],
}

EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "materials": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "material_name": {"type": "STRING"},
                    "material_abbreviation": {"type": "STRING"},
                    "material_class": {"type": "STRING", "enum": MATERIAL_CLASS_ENUM},
                    "trade_grade": {"type": "STRING"},
                    "manufacturer": {"type": "STRING"},
                    "matrix": {"type": "STRING"},
                    "fiber": {"type": "STRING"},
                    "fiber_volume_fraction": {"type": "STRING"},
                    "properties": {"type": "ARRAY", "items": _PROPERTY_SCHEMA},
                },
                "required": ["material_name", "material_class", "properties"],
            },
        }
    },
    "required": ["materials"],
}

EXTRACTION_PROMPT = (
    "You are an expert materials scientist. From the attached PDF, extract every "
    "distinct material it characterizes as a list under `materials`. A datasheet "
    "or paper may describe MORE THAN ONE material (e.g. a neat resin and its "
    "composite, or several grades) — return each as its own entry with only its "
    "own properties. Do NOT merge two materials into one.\n\n"
    "For each material provide:\n"
    "- material_name (generic material, e.g. 'isotactic polypropylene')\n"
    "- material_abbreviation\n"
    "- material_class: one of Polymer | Fiber | Composite. Choose Composite when a "
    "matrix is reinforced with fibers (laminate, prepreg, CF/PEEK, glass-filled, a "
    "reported fiber volume fraction); Fiber for a bare fiber/yarn/tow datasheet; "
    "Polymer otherwise.\n"
    "- trade_grade (commercial/trade name; '' if absent)\n"
    "- manufacturer (company; '' if absent)\n"
    "- matrix, fiber, fiber_volume_fraction (e.g. '55%') — for composites; '' otherwise\n\n"
    "Extract ALL numeric and descriptive properties across every category. For "
    "each property return:\n"
    "- section: one of " + ", ".join(SECTION_ENUM) + "\n"
    "- property_name\n"
    "- value_raw: the value EXACTLY as printed, including ranges and qualifiers "
    "(e.g. '100-120', '≤ -18.0', '~3.5')\n"
    "- value_num: the single numeric value, or null if it is a range/non-numeric\n"
    "- value_min, value_max: the low/high of a range, else null\n"
    "- qualifier: one of '', '<', '<=', '>', '>=', '~', '±'\n"
    "- unit: the unit exactly as printed ('' if dimensionless)\n"
    "- test_condition (e.g. '23 °C, 50% RH'; '' if none)\n"
    "- comments ('' if none)\n"
    "- source_quote: the VERBATIM sentence or table cell from the PDF that states "
    "this value. Copy it character-for-character; do not paraphrase.\n"
    "- page: the 1-based PDF page number the value appears on.\n\n"
    "Never invent values. If a value is not in the document, do not include it. "
    "Respond ONLY with valid JSON following the schema."
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Property:
    section: str
    property_name: str
    value_raw: str
    unit: str = ""
    value_num: Optional[float] = None
    value_min: Optional[float] = None
    value_max: Optional[float] = None
    qualifier: str = ""
    test_condition: str = ""
    comments: str = ""
    source_quote: str = ""
    page: Optional[int] = None
    # filled by verify_against_text() / canonicalization:
    unit_canonical: str = ""
    value_si: Optional[float] = None
    status: str = "ok"
    flag_reason: str = ""


@dataclasses.dataclass
class Material:
    material_name: str
    material_abbreviation: str = ""
    material_class: str = ""
    trade_grade: str = ""
    manufacturer: str = ""
    matrix: str = ""
    fiber: str = ""
    fiber_volume_fraction: str = ""
    properties: list[Property] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Extraction:
    materials: list[Material] = dataclasses.field(default_factory=list)
    model: str = GEMINI_MODEL
    prompt_version: str = PROMPT_VERSION
    doc_status: str = "ok"  # "ok" | "scanned_no_text" | "empty_extraction"


@dataclasses.dataclass
class PropertyRow:
    """One flattened, fully-resolved row ready for the SQLite mirror.

    Carries the legacy columns (`value`, `unit`, `english`, ...) so the CSV
    export and page1.py keep working, plus the new structured/provenance
    columns added in this phase.
    """

    # identity / routing
    material_name: str
    material_abbreviation: str
    material_key: str
    material_class: str
    # legacy columns (kept populated for backward compat)
    section: str
    property_name: str
    value: str
    unit: str
    english: str
    test_condition: str
    comments: str
    # composite descriptors promoted to real columns (Task 2)
    trade_grade: str = ""
    manufacturer: str = ""
    matrix: str = ""
    fiber: str = ""
    fiber_volume_fraction: str = ""
    # structured numeric value (Task 3)
    value_raw: str = ""
    value_num: Optional[float] = None
    value_min: Optional[float] = None
    value_max: Optional[float] = None
    qualifier: str = ""
    unit_canonical: str = ""
    value_si: Optional[float] = None
    # provenance (Tasks 1/6)
    source_pdf: str = ""
    source_sha1: str = ""
    page: Optional[int] = None
    source_quote: str = ""
    # status (Tasks 1/3/4)
    status: str = "ok"
    flag_reason: str = ""
    # bookkeeping
    model: str = GEMINI_MODEL
    prompt_version: str = PROMPT_VERSION


# ---------------------------------------------------------------------------
# Gemini request with retry/backoff (Task 7)
# ---------------------------------------------------------------------------


def _retry_after(resp: requests.Response) -> Optional[float]:
    val = resp.headers.get("Retry-After")
    if not val:
        return None
    try:
        return max(0.0, float(val))
    except ValueError:
        return None


def gemini_request(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: int = REQUEST_TIMEOUT_S,
    _sleep=time.sleep,
) -> Optional[requests.Response]:
    """POST `payload` to `url`, retrying 429/5xx and connection errors.

    Mirrors pdf_crawler.http_get()'s policy: bounded retries with exponential
    backoff, honoring a Retry-After header. Returns the 200 Response, or None
    if retries are exhausted / the status is non-retryable.
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code == 200:
                return resp
            if resp.status_code in RETRY_STATUS and attempt < MAX_RETRIES:
                delay = _retry_after(resp) or BACKOFF_BASE * (2 ** attempt)
                log.warning(
                    "Gemini -> %s; retry %d/%d in %.1fs",
                    resp.status_code, attempt + 1, MAX_RETRIES, delay,
                )
                _sleep(delay)
                continue
            log.error("Gemini -> %s: %s", resp.status_code, resp.text[:300])
            resp.raise_for_status()
            return None
        except requests.RequestException as exc:
            if attempt < MAX_RETRIES:
                delay = BACKOFF_BASE * (2 ** attempt)
                log.warning("Gemini request failed: %s; retry %d/%d in %.1fs",
                            exc, attempt + 1, MAX_RETRIES, delay)
                _sleep(delay)
                continue
            raise
    return None


def _upload_pdf_file(pdf_bytes: bytes, filename: str, api_key: str) -> Optional[str]:
    """Upload a PDF via the Gemini File API (resumable protocol); return file URI.

    Used for large/long PDFs (Task 10) where base64 inlining would blow the
    generateContent request-size limit.
    """
    start_url = GEMINI_UPLOAD_URL.format(key=api_key)
    start = requests.post(
        start_url,
        headers={
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(len(pdf_bytes)),
            "X-Goog-Upload-Header-Content-Type": "application/pdf",
            "Content-Type": "application/json",
        },
        json={"file": {"display_name": filename}},
        timeout=REQUEST_TIMEOUT_S,
    )
    start.raise_for_status()
    upload_url = start.headers.get("X-Goog-Upload-URL")
    if not upload_url:
        log.error("File API did not return an upload URL")
        return None

    up = requests.post(
        upload_url,
        headers={
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
            "Content-Length": str(len(pdf_bytes)),
        },
        data=pdf_bytes,
        timeout=REQUEST_TIMEOUT_S,
    )
    up.raise_for_status()
    info = up.json().get("file", {})
    name = info.get("name")
    uri = info.get("uri")
    state = info.get("state")

    # Wait for the file to become ACTIVE before referencing it.
    for _ in range(30):
        if state == "ACTIVE":
            return uri
        if state == "FAILED":
            log.error("File API processing failed for %s", filename)
            return None
        time.sleep(1.0)
        poll = requests.get(
            GEMINI_FILE_STATUS_URL.format(name=name, key=api_key),
            timeout=REQUEST_TIMEOUT_S,
        )
        poll.raise_for_status()
        info = poll.json()
        state = info.get("state")
        uri = info.get("uri", uri)
    return uri if state == "ACTIVE" else None


def _parse_extraction_json(resp: requests.Response) -> Optional[dict[str, Any]]:
    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        return None
    parts = candidates[0].get("content", {}).get("parts", [])
    for part in parts:
        text = (part.get("text") or "").strip()
        if text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                log.error("Gemini returned non-JSON text")
                return None
    return None


# ---------------------------------------------------------------------------
# PDF text (grounding + scanned detection)
# ---------------------------------------------------------------------------


def pdf_page_texts(pdf_bytes: bytes) -> list[str]:
    """Return per-page extractable text. Empty list if PyMuPDF is unavailable."""
    if fitz is None:
        return []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # pragma: no cover
        log.warning("Could not open PDF for text extraction: %s", exc)
        return []
    try:
        return [p.get_text() for p in doc]
    finally:
        doc.close()


def _is_scanned(page_texts: list[str]) -> bool:
    if not page_texts:
        return False  # can't tell without PyMuPDF; don't falsely flag
    total = sum(len(t.strip()) for t in page_texts)
    return (total / max(1, len(page_texts))) < SCANNED_TEXT_PER_PAGE


# ---------------------------------------------------------------------------
# Extraction entry point
# ---------------------------------------------------------------------------


def extract_from_pdf(pdf_bytes: bytes, filename: str, api_key: str) -> Extraction:
    """Extract structured materials from a PDF (Tasks 2/3/8/10).

    Returns an Extraction whose ``doc_status`` is ``scanned_no_text`` (and
    ``materials`` empty) when the PDF has no extractable text, so the caller
    can record the source without fabricating rows.
    """
    page_texts = pdf_page_texts(pdf_bytes)
    if _is_scanned(page_texts):
        log.info("%s looks scanned/image-only; skipping extraction", filename)
        return Extraction(doc_status="scanned_no_text")

    parts: list[dict[str, Any]] = [{"text": EXTRACTION_PROMPT}]
    use_file_api = (
        len(pdf_bytes) > INLINE_PDF_LIMIT_BYTES or len(page_texts) > INLINE_PAGE_LIMIT
    )
    if use_file_api:
        uri = _upload_pdf_file(pdf_bytes, filename, api_key)
        if not uri:
            return Extraction(doc_status="empty_extraction")
        parts.append({"fileData": {"mimeType": "application/pdf", "fileUri": uri}})
    else:
        parts.append(
            {
                "inlineData": {
                    "mimeType": "application/pdf",
                    "data": base64.b64encode(pdf_bytes).decode("utf-8"),
                }
            }
        )

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": EXTRACTION_SCHEMA,
        },
    }
    url = GEMINI_URL_TEMPLATE.format(model=GEMINI_MODEL, key=api_key)
    resp = gemini_request(url, payload)
    if resp is None:
        return Extraction(doc_status="empty_extraction")
    raw = _parse_extraction_json(resp)
    if not raw:
        return Extraction(doc_status="empty_extraction")
    return _extraction_from_json(raw)


def _extraction_from_json(raw: dict[str, Any]) -> Extraction:
    """Coerce raw model JSON into a typed Extraction (tolerates the old shape)."""
    materials_json = raw.get("materials")
    if materials_json is None:
        # Back-compat: a single-material response with a flat property list.
        flat = raw.get("mechanical_properties") or raw.get("properties") or []
        materials_json = [
            {
                "material_name": raw.get("material_name", ""),
                "material_abbreviation": raw.get("material_abbreviation", ""),
                "material_class": raw.get("material_class", ""),
                "trade_grade": raw.get("trade_grade", ""),
                "manufacturer": raw.get("manufacturer", ""),
                "properties": flat,
            }
        ]

    materials: list[Material] = []
    for mj in materials_json or []:
        props: list[Property] = []
        for pj in mj.get("properties") or []:
            value_raw = (pj.get("value_raw") or pj.get("value") or "").strip()
            prop = Property(
                section=_norm_section(pj.get("section")),
                property_name=(pj.get("property_name") or "").strip() or "Unknown property",
                value_raw=value_raw,
                unit=(pj.get("unit") or "").strip(),
                value_num=_as_float(pj.get("value_num")),
                value_min=_as_float(pj.get("value_min")),
                value_max=_as_float(pj.get("value_max")),
                qualifier=(pj.get("qualifier") or "").strip(),
                test_condition=(pj.get("test_condition") or "").strip(),
                comments=(pj.get("comments") or "").strip(),
                source_quote=(pj.get("source_quote") or "").strip(),
                page=_as_int(pj.get("page")),
            )
            # Fill structured numeric fields from value_raw if the model omitted them.
            _fill_numeric(prop)
            props.append(prop)
        materials.append(
            Material(
                material_name=(mj.get("material_name") or "").strip(),
                material_abbreviation=(mj.get("material_abbreviation") or "").strip(),
                material_class=(mj.get("material_class") or "").strip(),
                trade_grade=(mj.get("trade_grade") or "").strip(),
                manufacturer=(mj.get("manufacturer") or "").strip(),
                matrix=(mj.get("matrix") or "").strip(),
                fiber=(mj.get("fiber") or "").strip(),
                fiber_volume_fraction=(mj.get("fiber_volume_fraction") or "").strip(),
                properties=props,
            )
        )
    return Extraction(materials=materials)


def _as_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_int(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _norm_section(section: Optional[str]) -> str:
    """Normalize free-text section drift to the enum (Task 8)."""
    s = (section or "").strip()
    if not s:
        return "Descriptive"
    low = s.lower()
    for canon in SECTION_ENUM:
        if low == canon.lower():
            return canon
    # substring / synonym normalization, e.g. "Mechanical Properties" -> "Mechanical"
    synonyms = {
        "mechanical": "Mechanical",
        "thermal": "Thermal",
        "electrical": "Electrical",
        "dielectric": "Electrical",
        "physical": "Physical",
        "optical": "Optical",
        "rheolog": "Rheological",
        "viscosity": "Rheological",
        "processing": "Processing",
        "descriptive": "Descriptive",
        "composition": "Composition/Reinforcement",
        "reinforcement": "Composition/Reinforcement",
        "architecture": "Architecture/Structure",
        "structure": "Architecture/Structure",
    }
    for key, canon in synonyms.items():
        if key in low:
            return canon
    return "Descriptive"


# ---------------------------------------------------------------------------
# Value parsing (Task 3)
# ---------------------------------------------------------------------------

_QUALIFIER_MAP = [
    ("≤", "<="), ("≥", ">="), ("≈", "~"),
    ("<=", "<="), (">=", ">="), ("<", "<"), (">", ">"),
    ("~", "~"), ("±", "±"),
]
_NUM_RE = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_RANGE_SEP_RE = re.compile(r"\s*(?:–|—|−|-|to|…|\.\.\.)\s*")


def _clean_number(tok: str) -> Optional[float]:
    try:
        return float(tok.replace(",", ""))
    except ValueError:
        return None


def parse_value_raw(value_raw: str) -> tuple[Optional[float], Optional[float], Optional[float], str]:
    """Parse a printed value into (value_num, value_min, value_max, qualifier).

    Handles ranges ('100-120', '100 to 120'), qualifiers ('<= -18', '~3.5'),
    plus-minus ('5.0 +/- 0.2'), thousands separators and scientific notation.
    """
    if not value_raw:
        return None, None, None, ""
    s = unicodedata.normalize("NFKC", value_raw).strip()

    qualifier = ""
    for needle, canon in _QUALIFIER_MAP:
        if needle in s:
            qualifier = canon
            break

    # plus/minus -> midpoint value with min/max
    pm = re.search(r"([-+]?\d*\.?\d+)\s*(?:±|\+/-|\+-)\s*(\d*\.?\d+)", s)
    if pm:
        base = _clean_number(pm.group(1))
        delta = _clean_number(pm.group(2))
        if base is not None and delta is not None:
            return base, base - delta, base + delta, "±"

    # Range: two numbers separated by a dash/"to". Guard against a leading sign
    # being misread as a separator by working on the sign-stripped remainder.
    body = s
    lead_sign = ""
    if body[:1] in "+-":
        lead_sign, body = body[0], body[1:]
    range_match = re.match(
        r"\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r"\s*(?:–|—|-|to)\s*"
        r"([-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        body,
    )
    if range_match:
        lo = _clean_number(lead_sign + range_match.group(1))
        hi = _clean_number(range_match.group(2))
        if lo is not None and hi is not None:
            if lo > hi:
                lo, hi = hi, lo
            return None, lo, hi, qualifier

    nums = _NUM_RE.findall(s)
    if not nums:
        return None, None, None, qualifier
    val = _clean_number(nums[0])
    return val, None, None, qualifier


def _fill_numeric(prop: Property) -> None:
    """Backfill value_num/min/max/qualifier from value_raw when model omitted them."""
    has_any = (
        prop.value_num is not None
        or prop.value_min is not None
        or prop.value_max is not None
    )
    if has_any and prop.qualifier:
        return
    num, lo, hi, qual = parse_value_raw(prop.value_raw)
    if prop.value_num is None and prop.value_min is None and prop.value_max is None:
        prop.value_num, prop.value_min, prop.value_max = num, lo, hi
    if not prop.qualifier:
        prop.qualifier = qual


def _representative_value(prop: Property) -> Optional[float]:
    if prop.value_num is not None:
        return prop.value_num
    if prop.value_min is not None and prop.value_max is not None:
        return (prop.value_min + prop.value_max) / 2.0
    return prop.value_min if prop.value_min is not None else prop.value_max


# ---------------------------------------------------------------------------
# Unit normalization + plausibility (Task 3)
# ---------------------------------------------------------------------------

if pint is not None:  # pragma: no branch
    _UREG = pint.UnitRegistry()
    _UREG.define("ksi = 1000 * psi")
    _UREG.define("Msi = 1000000 * psi")
else:  # pragma: no cover
    _UREG = None


@dataclasses.dataclass
class _Family:
    name: str
    keywords: tuple[str, ...]
    canonical: str          # pint unit string (or display for non-pint families)
    display: str            # human-facing unit label
    lo: float               # plausibility low, in `display` units
    hi: float               # plausibility high, in `display` units
    kind: str               # "physical" | "temperature" | "raw"
    si_factor: float = 1.0  # for "raw" families: value_si = value * si_factor


# Ordered: specific keywords before generic ones (first match wins).
PROPERTY_FAMILIES: list[_Family] = [
    _Family("tensile_modulus",
            ("tensile modulus", "modulus of elasticity", "young", "elastic modulus"),
            "GPa", "GPa", 0.01, 1000.0, "physical"),
    _Family("flexural_modulus", ("flexural modulus", "bending modulus"),
            "GPa", "GPa", 0.01, 800.0, "physical"),
    _Family("shear_modulus", ("shear modulus", "modulus of rigidity"),
            "GPa", "GPa", 0.005, 500.0, "physical"),
    _Family("storage_modulus", ("storage modulus", "compressive modulus", "modulus"),
            "GPa", "GPa", 0.001, 1000.0, "physical"),
    _Family("tensile_strength",
            ("tensile strength", "ultimate tensile", "strength at break",
             "yield strength", "tensile stress"),
            "MPa", "MPa", 0.5, 10000.0, "physical"),
    _Family("flexural_strength", ("flexural strength", "bending strength"),
            "MPa", "MPa", 0.5, 4000.0, "physical"),
    _Family("compressive_strength", ("compressive strength", "compression strength"),
            "MPa", "MPa", 0.5, 6000.0, "physical"),
    _Family("shear_strength", ("shear strength", "strength"),
            "MPa", "MPa", 0.5, 4000.0, "physical"),
    _Family("glass_transition", ("glass transition", "tg"),
            "degC", "°C", -150.0, 600.0, "temperature"),
    _Family("melting", ("melting", "melt temperature", "tm"),
            "degC", "°C", 50.0, 500.0, "temperature"),
    _Family("crystallization", ("crystallization",),
            "degC", "°C", 0.0, 500.0, "temperature"),
    _Family("decomposition", ("decomposition", "degradation temperature"),
            "degC", "°C", 100.0, 1200.0, "temperature"),
    _Family("hdt", ("heat deflection", "deflection temperature", "hdt",
                    "heat distortion"),
            "degC", "°C", 0.0, 600.0, "temperature"),
    _Family("cte", ("thermal expansion", "cte", "expansion coefficient"),
            "ppm/degC", "ppm/°C", -50.0, 500.0, "raw", si_factor=1e-6),
    _Family("density", ("density", "specific gravity"),
            "g/cm**3", "g/cm³", 0.1, 12.0, "physical"),
    _Family("elongation", ("elongation", "strain at break"),
            "%", "%", 0.001, 2000.0, "raw", si_factor=0.01),
]


def _match_family(prop: Property) -> Optional[_Family]:
    name = prop.property_name.lower()
    for fam in PROPERTY_FAMILIES:
        for kw in fam.keywords:
            if kw in name:
                return fam
    return None


def _preprocess_unit(unit: str) -> str:
    u = unicodedata.normalize("NFKC", unit).strip()
    u = u.replace("·", "*").replace("−", "-")
    u = u.replace("³", "**3").replace("²", "**2")
    u = u.replace("µ", "u").replace("μ", "u")
    u = u.replace("^", "**")
    u = u.replace("cm3", "cm**3").replace("cm2", "cm**2").replace("m3", "m**3")
    u = u.replace("g/cc", "g/cm**3")
    return u


_TEMP_UNITS = {
    "": "degC", "c": "degC", "degc": "degC", "°c": "degC", "celsius": "degC",
    "k": "kelvin", "kelvin": "kelvin",
    "f": "degF", "degf": "degF", "°f": "degF", "fahrenheit": "degF",
}


def canonicalize(prop: Property) -> tuple[str, Optional[float], Optional[str]]:
    """Return (unit_canonical, value_si, problem).

    `problem` is None on success, or a short reason ("unit_review:...") when the
    unit is missing/dimensionally wrong for the property family.
    """
    fam = _match_family(prop)
    rep = _representative_value(prop)
    if fam is None:
        # No known family: pass the unit through, no SI conversion, no check.
        return (prop.unit, None, None)
    if rep is None:
        return (fam.display, None, None)

    if fam.kind == "raw":
        return (fam.display, rep * fam.si_factor, None)

    if _UREG is None:  # pragma: no cover
        return (fam.display, None, None)

    try:
        if fam.kind == "temperature":
            key = unicodedata.normalize("NFKC", prop.unit).strip().lower()
            src = _TEMP_UNITS.get(key)
            if src is None:
                return (fam.display, None, f"unit_review:bad_temp_unit:{prop.unit}")
            q = _UREG.Quantity(rep, src)
            value_si = q.to("kelvin").magnitude
            return (fam.display, value_si, None)

        # physical (pressure, density, ...)
        pre = _preprocess_unit(prop.unit)
        if not pre:
            return (fam.display, None, "unit_review:missing_unit")
        q = rep * _UREG(pre)
        target = _UREG(fam.canonical)
        if q.dimensionality != target.dimensionality:
            return (fam.display, None,
                    f"unit_review:dim_mismatch:{prop.unit}!~{fam.display}")
        value_si = q.to_base_units().magnitude
        return (fam.display, value_si, None)
    except Exception as exc:  # pint parse failure, undefined unit, etc.
        return (fam.display, None, f"unit_review:unparseable:{prop.unit}:{exc}")


def _canonical_value(prop: Property, fam: _Family) -> Optional[float]:
    """Representative value expressed in the family's `display` unit (for range check)."""
    rep = _representative_value(prop)
    if rep is None:
        return None
    if fam.kind == "raw":
        return rep
    if _UREG is None:  # pragma: no cover
        return rep
    try:
        if fam.kind == "temperature":
            key = unicodedata.normalize("NFKC", prop.unit).strip().lower()
            src = _TEMP_UNITS.get(key)
            if src is None:
                return None
            return _UREG.Quantity(rep, src).to(fam.canonical).magnitude
        pre = _preprocess_unit(prop.unit)
        if not pre:
            return None
        q = rep * _UREG(pre)
        if q.dimensionality != _UREG(fam.canonical).dimensionality:
            return None
        return q.to(fam.canonical).magnitude
    except Exception:
        return None


def plausibility_problem(prop: Property) -> Optional[str]:
    """Range-check the value *after* unit conversion. Returns reason or None."""
    fam = _match_family(prop)
    if fam is None:
        return None
    cval = _canonical_value(prop, fam)
    if cval is None:
        return None  # couldn't convert -> handled as unit_review elsewhere
    if not (fam.lo <= cval <= fam.hi):
        return f"out_of_range[{fam.lo},{fam.hi}{fam.display}]:{cval:.4g}"
    return None


_PLACEHOLDER_VALUES = {"", "n/a", "na", "-", "--", "n.a.", "none", "tbd"}


def _empty_value(prop: Property) -> bool:
    return prop.value_raw.strip().lower() in _PLACEHOLDER_VALUES


# ---------------------------------------------------------------------------
# Text grounding (Task 1)
# ---------------------------------------------------------------------------


def _normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = re.sub(r"\s+", " ", s)
    return s.lower().strip()


def _grounded(needle: str, haystacks: list[str]) -> bool:
    n = _normalize_text(needle)
    if not n:
        return False
    return any(n in h for h in haystacks)


def verify_against_text(extraction: Extraction, page_texts: list[str]) -> Extraction:
    """Ground each property's value in the PDF text (Task 1).

    For every property: if `value_raw` (or a chunk of `source_quote`) appears on
    the cited page (falling back to any page), mark `status='ok'`; otherwise
    `status='unverified'`. Also folds in empty-value, unit, and range checks,
    using this precedence:
        empty_value > unverified > unit_review > out_of_range > ok
    """
    if not page_texts:
        norm_pages: list[str] = []
    else:
        norm_pages = [_normalize_text(t) for t in page_texts]

    for material in extraction.materials:
        for prop in material.properties:
            # always compute canonicalization so rows carry unit_canonical/value_si
            unit_canonical, value_si, unit_problem = canonicalize(prop)
            prop.unit_canonical = unit_canonical
            prop.value_si = value_si

            reasons: list[str] = []
            status = "ok"

            if _empty_value(prop):
                status = "empty_value"
                reasons.append("empty_value")
            else:
                # grounding
                if norm_pages:
                    page_idx = (prop.page - 1) if prop.page else None
                    cited = (
                        [norm_pages[page_idx]]
                        if page_idx is not None and 0 <= page_idx < len(norm_pages)
                        else []
                    )
                    found = _grounded(prop.value_raw, cited or norm_pages)
                    if not found and cited:
                        found = _grounded(prop.value_raw, norm_pages)  # fallback any page
                    if not found and prop.source_quote:
                        chunk = prop.source_quote[:40]
                        found = _grounded(chunk, norm_pages)
                    if not found:
                        status = "unverified"
                        reasons.append("value_not_in_pdf_text")
                # unit problem
                if status == "ok" and unit_problem:
                    status = "unit_review"
                    reasons.append(unit_problem)
                # plausibility (only meaningful once unit is sane)
                if status == "ok":
                    rng = plausibility_problem(prop)
                    if rng:
                        status = "out_of_range"
                        reasons.append(rng)

            prop.status = status
            prop.flag_reason = "; ".join(reasons)
    return extraction


# ---------------------------------------------------------------------------
# Classification (Task 5)
# ---------------------------------------------------------------------------

COMPOSITE_KEYWORDS = (
    "composite", "laminate", "reinforced", "prepreg",
    "cf/", "gf/", "fiber-reinforced", "fibre-reinforced",
    "ud ", "uni-directional", "unidirectional", "woven",
    "glass-filled", "glass filled", "carbon-filled",
)
FIBER_KEYWORDS = ("fiber", "fibre", "yarn", "tow", "roving", "filament")


def classify_material(material: Material) -> str:
    """Return 'Polymer' | 'Fiber' | 'Composite' (Task 5).

    Primary signal is the model's `material_class` enum; a deterministic
    keyword fallback (using sections + Vf presence) only runs when the field is
    missing or invalid. The old haystack bug (repeating material_name N times,
    never reading section) is gone.
    """
    field = (material.material_class or "").strip().capitalize()
    if field in MATERIAL_CLASS_ENUM:
        return field

    # --- deterministic fallback ---
    # A reported fiber volume fraction is a strong composite signal.
    if material.fiber_volume_fraction.strip():
        return "Composite"

    haystack = " ".join(
        [
            material.material_name or "",
            material.material_abbreviation or "",
            material.trade_grade or "",
            material.matrix or "",
            material.fiber or "",
        ]
        + [p.section or "" for p in material.properties]
        + [p.property_name or "" for p in material.properties]
    ).lower()

    if any(k in haystack for k in COMPOSITE_KEYWORDS):
        return "Composite"
    if any(k in haystack for k in FIBER_KEYWORDS):
        return "Fiber"
    return "Polymer"


def _autoabbr(name: str) -> str:
    if not name:
        return "UNKNOWN"
    parts = [p[0] for p in name.split() if p and p[0].isalpha()]
    return "".join(parts).upper() or name[:6].upper()


def material_key(material: Material) -> str:
    """Stable material identity: normalized lowercased name, fallback abbr (Task 6)."""
    name = (material.material_name or "").strip().lower()
    name = re.sub(r"\s+", " ", name)
    if name:
        return name
    return (material.material_abbreviation or "unknown").strip().lower()


# ---------------------------------------------------------------------------
# Flatten to DB rows (Tasks 2/5/6)
# ---------------------------------------------------------------------------


def to_rows(extraction: Extraction, source_pdf: str, source_sha1: str) -> list[PropertyRow]:
    """Flatten an Extraction into PropertyRows, iterating materials x properties."""
    rows: list[PropertyRow] = []
    for material in extraction.materials:
        mclass = classify_material(material)
        mkey = material_key(material)
        abbr = material.material_abbreviation or _autoabbr(material.material_name)
        for prop in material.properties:
            rows.append(
                PropertyRow(
                    material_name=material.material_name,
                    material_abbreviation=abbr,
                    material_key=mkey,
                    material_class=mclass,
                    section=prop.section,
                    property_name=prop.property_name,
                    value=prop.value_raw,  # legacy column == value_raw (back-compat)
                    unit=prop.unit,
                    english="",  # legacy alt-units column; model no longer emits it
                    test_condition=prop.test_condition,
                    comments=prop.comments,
                    trade_grade=material.trade_grade,
                    manufacturer=material.manufacturer,
                    matrix=material.matrix,
                    fiber=material.fiber,
                    fiber_volume_fraction=material.fiber_volume_fraction,
                    value_raw=prop.value_raw,
                    value_num=prop.value_num,
                    value_min=prop.value_min,
                    value_max=prop.value_max,
                    qualifier=prop.qualifier,
                    unit_canonical=prop.unit_canonical,
                    value_si=prop.value_si,
                    source_pdf=source_pdf,
                    source_sha1=source_sha1,
                    page=prop.page,
                    source_quote=prop.source_quote,
                    status=prop.status,
                    flag_reason=prop.flag_reason,
                    model=extraction.model,
                    prompt_version=extraction.prompt_version,
                )
            )
    return rows
