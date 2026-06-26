"""
DocToDB — Materials Science PDF Extractor (Dual LLM Consensus Pipeline)
=======================================================================
PRIMARY:  Gemini — auto-selected from live model list (preference order configurable in sidebar)
SECONDARY: GPT — auto-selected from live model list

FLOW:
  1. Extract all chunks (tables + text)
  2. Index into ChromaDB for semantic ranking
  3. Rank ALL chunks via single schema-derived retrieval query
  4. Build batches (size driven by probed token limits of selected models)
  5. Run Gemini + GPT in PARALLEL (ThreadPoolExecutor)
  6. Consensus filter
  7. DOI auto-extracted; rendered as clickable link
  8. 📐 Evaluate tab — Precision/Recall/F1 vs uploaded ground truth
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import fitz
import numpy as np
import pandas as pd
import pdfplumber
import requests
from dotenv import load_dotenv
import threading
import sys
print(sys.executable)
load_dotenv()

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logging.warning("chromadb not installed — pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    logging.warning("sentence-transformers not installed — pip install sentence-transformers")

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger(__name__)

if OCR_AVAILABLE:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

GEMINI_PREFERRED_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash",
]

OPENAI_PREFERRED_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
]

_cached_gemini_model: Optional[str] = None


def get_gemini_model(api_key: str) -> str:
    global _cached_gemini_model
    if _cached_gemini_model:
        return _cached_gemini_model
    try:
        resp = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
            timeout=10,
        )
        if not resp.ok:
            _cached_gemini_model = "gemini-2.5-flash"
            return _cached_gemini_model
        models = resp.json().get("models", [])
        available = [
            m["name"].replace("models/", "")
            for m in models
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]
        for pref in GEMINI_PREFERRED_MODELS:
            for name in available:
                if name.startswith(pref):
                    log.info(f"Gemini auto-selected: {name}")
                    _cached_gemini_model = name
                    return _cached_gemini_model
        flash_models = [
            n for n in available
            if "flash" in n
            and "1.5" not in n and "2.0" not in n
            and "tts" not in n and "audio" not in n and "image" not in n
        ]
        _cached_gemini_model = flash_models[0] if flash_models else "gemini-2.5-flash"
        return _cached_gemini_model
    except Exception as e:
        log.warning(f"Gemini model auto-detect failed: {e}")
        _cached_gemini_model = "gemini-2.5-flash"
        return _cached_gemini_model


GEMINI_MODEL: str = get_gemini_model(GEMINI_API_KEY)
GEMINI_API_URL: str = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

GPT_MODEL: str   = "gpt-4o"
GPT_API_URL: str = "https://api.openai.com/v1/chat/completions"


def _rebuild_gemini_url(model: str, key: str) -> str:
    return (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={key}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# LIVE MODEL FETCHERS
# ─────────────────────────────────────────────────────────────────────────────

_GEMINI_FALLBACK_MODELS = [
    ("gemini-3.5-flash",        1_000_000),
    ("gemini-3.1-flash-lite",   1_000_000),
    ("gemini-3.1-pro-preview",  1_000_000),
    ("gemini-2.5-flash",        1_000_000),
    ("gemini-2.0-flash",        1_000_000),
]

_OPENAI_FALLBACK_MODELS = [
    ("gpt-4o",           128_000),
    ("gpt-4o-mini",      128_000),
    ("gpt-4-turbo",      128_000),
    ("gpt-4",              8_192),
    ("gpt-3.5-turbo",    16_385),
]


def _fetch_gemini_models_live(api_key: str) -> List[Tuple[str, int]]:
    try:
        resp = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
            timeout=10,
        )
        if not resp.ok:
            return _GEMINI_FALLBACK_MODELS
        raw = resp.json().get("models", [])
        results: List[Tuple[str, int]] = []
        for m in raw:
            name  = m["name"].replace("models/", "")
            limit = int(m.get("inputTokenLimit", 0))
            if (
                "generateContent" in m.get("supportedGenerationMethods", [])
                and "gemini" in name.lower()
                and "embedding" not in name.lower()
                and "aqa" not in name.lower()
                and "tts" not in name.lower()
                and "audio" not in name.lower()
                and "image" not in name.lower()
                and limit > 32_768
            ):
                results.append((name, limit))
        def _sort_key(item):
            name = item[0]
            for i, pref in enumerate(GEMINI_PREFERRED_MODELS):
                if name.startswith(pref):
                    return (0, i, name)
            return (1, 0, name)
        results.sort(key=_sort_key)
        return results or _GEMINI_FALLBACK_MODELS
    except Exception:
        return _GEMINI_FALLBACK_MODELS


def _fetch_openai_models_live(api_key: str) -> List[Tuple[str, int]]:
    try:
        resp = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if not resp.ok:
            return _OPENAI_FALLBACK_MODELS
        raw = resp.json().get("data", [])
        results: List[Tuple[str, int]] = []
        for m in raw:
            mid = m.get("id", "")
            ctx = int(m.get("context_window", 0))
            if mid.startswith("gpt-") and ctx > 0:
                results.append((mid, ctx))
        def _sort_key(item):
            name = item[0]
            for i, pref in enumerate(OPENAI_PREFERRED_MODELS):
                if name == pref or name.startswith(pref):
                    return (0, i, name)
            return (1, 0, name)
        results.sort(key=_sort_key)
        return results or _OPENAI_FALLBACK_MODELS
    except Exception:
        return _OPENAI_FALLBACK_MODELS


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE BATCH CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
CHROMA_COLLECTION  = "doctodb_chunks"
EMBED_MODEL_NAME   = "allenai/scibert_scivocab_uncased"
NEIGHBOR_OVERLAP   = 0

_GEMINI_DEFAULTS = {"max_input_chars": 60_000, "batch_delay": 5}
_GPT_DEFAULTS    = {"max_input_chars": 50_000, "batch_delay": 2}
_config_lock     = threading.Lock()
_GEMINI_CONFIG   = dict(_GEMINI_DEFAULTS)
_GPT_CONFIG      = dict(_GPT_DEFAULTS)


def _probe_gemini_limits(model_name: str) -> None:
    try:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}"
            f"?key={GEMINI_API_KEY}"
        )
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return
        info       = r.json()
        input_tok  = info.get("inputTokenLimit",  200_000)
        output_tok = info.get("outputTokenLimit",  32_768)
        rpm        = info.get("rpm", 15)
        max_chars  = min(int(input_tok * 3 * 0.70), 500_000)
        with _config_lock:
            _GEMINI_CONFIG["max_input_chars"]   = max_chars
            #_GEMINI_CONFIG["max_output_tokens"] = min(output_tok, 32_768)
            _GEMINI_CONFIG["batch_delay"]        = max(1, round(60 / rpm))
        log.info("Gemini limits probed: input=%d tokens → %d chars, delay=%ds",
                 input_tok, max_chars, _GEMINI_CONFIG["batch_delay"])
    except Exception as exc:
        log.warning("Gemini limit probe failed (%s) — using defaults", exc)


def _probe_gpt_limits() -> None:
    try:
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        body    = {"model": GPT_MODEL, "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}
        r       = requests.post("https://api.openai.com/v1/chat/completions",
                                headers=headers, json=body, timeout=15)
        tpm       = int(r.headers.get("x-ratelimit-limit-tokens",  30_000))
        rpm       = int(r.headers.get("x-ratelimit-limit-requests", 500))
        max_chars = int(tpm * 0.50 * 3)
        with _config_lock:
            _GPT_CONFIG["max_input_chars"] = max_chars
            _GPT_CONFIG["batch_delay"]     = max(1, round(60 / rpm))
        log.info("GPT limits probed: tpm=%d → %d chars, rpm=%d, delay=%ds",
                 tpm, max_chars, rpm, _GPT_CONFIG["batch_delay"])
    except Exception as exc:
        log.warning("GPT limit probe failed (%s) — using defaults", exc)


def _run_startup_probes(model_name: str) -> None:
    with ThreadPoolExecutor(max_workers=2) as ex:
        ex.submit(_probe_gemini_limits, model_name)
        ex.submit(_probe_gpt_limits)


def _get_max_batch_chars() -> int:
    return min(_GEMINI_CONFIG["max_input_chars"], _GPT_CONFIG["max_input_chars"])


def _prewarm_embed_model() -> None:
    try:
        _get_embed_model()
        log.info("Embedding model pre-warmed.")
    except Exception as e:
        log.warning(f"Embed model pre-warm failed: {e}")



GEMINI_RETRY_DELAYS = [60, 120, 180]
GPT_RETRY_DELAYS    = [30, 60, 90]

MIN_TABLE_ROWS  = 2
MIN_TABLE_COLS  = 2
MIN_CELL_CHARS  = 2
MIN_TEXT_CHARS  = 40
OCR_THRESHOLD   = 50
CONSENSUS_VALUE_TOL = 0.05
CACHE_FILE = "pdf_extraction_cache.json"

RETRIEVAL_QUERY = (
    "Material property data including section category, property name, "
    "measured value, SI unit, imperial unit, test condition standard "
    "such as ASTM ISO DIN, and comments. Properties include mechanical "
    "thermal electrical physical rheological optical categories. "
    "Values in MPa GPa percent density conductivity temperature modulus "
    "strength elongation hardness viscosity flammability. "
    "Material name, manufacturer, trade grade, abbreviation. "
    "DOI digital object identifier paper reference link."
)

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "material_name":         {"type": "STRING"},
        "material_abbreviation": {"type": "STRING"},
        "manufacturer":          {"type": "STRING", "maxLength": 100},
        "doi":                   {"type": "STRING"},
        "mechanical_properties": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "section":        {"type": "STRING"},
                    "property_name":  {"type": "STRING"},
                    "value":          {"type": "STRING"},
                    "unit":           {"type": "STRING"},
                    "english":        {"type": "STRING"},
                    "test_condition": {"type": "STRING"},
                    "comments":       {"type": "STRING"},
                    "material_name":  {"type": "STRING"},
                    "source_page":    {"type": "STRING"},
                    "chunk_type":     {"type": "STRING"},
                    "source_text":    {"type": "STRING"},
                },
                "required": [
                    "section", "property_name", "value", "unit",
                    "english", "test_condition", "comments",
                    "material_name", "source_page", "chunk_type", "source_text",
                ],
            },
        },
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS  (updated — symmetric, range-preserving, table source_text fix)
# ─────────────────────────────────────────────────────────────────────────────

EXTRACTION_PROMPT = (
    "You are an expert materials scientist and information extraction engine.\n"
    "The content below is extracted from a materials datasheet or research paper.\n"
    "Each block is tagged with its page number and type (TABLE or TEXT).\n\n"
    "Extract ANY quantity that has a numeric value in the source. "
    "If a number appears in the text or table and describes any characteristic — "
    "of the material, its processing, its geometry, its performance, its environment, "
    "its testing conditions, or its behavior — extract it. "
    "Do not filter by property type. If it has a number and a context, it belongs in the output.\n\n"
    "Extract the following TOP-LEVEL fields (one value for the whole document):\n"
    "  - material_name        : material name only, not a full description paragraph.\n"
    "  - material_abbreviation: short abbreviation (e.g. 'PLA', 'WF/PLA').\n"
    "  - manufacturer         : company name only. Leave '' if none.\n"
    "  - doi                  : DOI identifier only. Strip https://doi.org/ prefix. Leave '' if not found.\n\n"
    "For each property record extract:\n"
    "  - section        : classify into the most appropriate category. "
    "Use standard categories where they fit — Mechanical, Thermal, Electrical, Physical, "
    "Rheological, Optical, Chemical, Barrier, Morphological, Surface. "
    "If none of these fit, describe the category yourself in one word. "
    "Never force a property into a wrong category just to use a standard name.\n\n"
    "  - property_name  : copy the property name EXACTLY as written in the source.\n"
    "    If the source shows a recognizable abbreviation or symbol (e.g. 'M100', 'Shore A', 'Tg'), "
    "preserve it exactly.\n"
    "    If the source shows a corrupted or unreadable character (single letters like 's', "
    "fragments like '\" rel', '\" res'), infer the most likely property name from "
    "context — surrounding column headers, units, and values.\n"
    "    Example: column header 's' with unit 'MPa' next to elongation data -> 'Tensile strength'.\n"
    "    Example: '\" rel' with unit '%' -> 'Elongation at break'.\n"
    "  - value          : copy the numeric value EXACTLY as written in the source.\n"
    "    - Use period as decimal separator (e.g. '45.3' not '45,3').\n"
    "    - No thousands separators (e.g. '1500' not '1,500').\n"
    "    - If the source shows a range like '40-52' or '40-52', put '40-52' here.\n"
    "    - Strip leading qualifier symbols (e.g. '~38' -> '38', '>15' -> '15', '~3.0' -> '3.0').\n"
    "    - If a value has a condition embedded like '15.0 MPa at 23C', put '15.0' in value "
    "and '23C' in test_condition.\n"
    "    - Never leave blank if a number exists for this property.\n"
    "  - unit           : copy the unit exactly as written, preserving capitalisation. "
    "Do not convert or normalise.\n"
    "  - english        : imperial equivalent if shown in the source, else ''.\n"
    "  - test_condition : ASTM/ISO/DIN standard or test conditions. "
    "Leave '' if none. Never embed conditions inside value.\n"
    "  - comments       : footnotes, qualifications, or uncertainty, else ''.\n"
    "  - material_name  : copy the exact full material name for THIS specific property "
    "as written in the source. "
    "NEVER collapse multiple materials into one generic name. "
    "If a table cell contains values for multiple materials, create one row per material. "
    "If the source shows a corrupted or fragmented material name, infer the most complete "
    "and accurate name from surrounding context. Never leave blank.\n"
    "  - source_page    : page number digits only from the block header.\n"
    "  - chunk_type     : 'table' if from a TABLE block, 'text' if from a TEXT block.\n"
    "  - source_text    :\n"
    "    For TEXT blocks: copy the exact sentence this property was extracted from.\n"
    "    For TABLE blocks: copy the column header row followed by the data row "
    "separated by a newline.\n"
    "    Never leave blank.\n\n"
    "ALSO EXTRACT formulation and composition data:\n"
    "  - Ingredient quantities in phr, wt%, vol%, mol% are valid properties.\n"
    "  - Use the ingredient name as property_name, quantity as value, unit as phr/wt% etc.\n"
    "  - Assign section = 'Chemical' for these rows.\n"
    "  - Example: 'Sulphur = 2.0 phr' -> property_name='Sulphur', value='2.0', unit='phr', section='Chemical'.\n\n"
    "RULES:\n"
    "  - Extract from BOTH tables and prose — never skip text blocks.\n"
    "  - Extract values explicitly stated in the text — do not invent values not present.\n"
    "  - If multiple materials appear, create SEPARATE entries for each — never merge them.\n"
    "  - If a single table cell contains values for multiple materials, create one row per material.\n"
    "  - For unknown or missing fields use '' — never use 'N/A', 'unknown', or null.\n"
    "  - NEVER return an empty properties array if any numeric value exists.\n"
    "  - Preserve source_page exactly from the block header as digits only.\n"
    "  - Respond ONLY with valid JSON matching the schema.\n"
    "\n\nCONTENT:\n"
)

GPT_SYSTEM_PROMPT = (
    "You are an expert materials-science information extraction engine.\n"
    "Your task is to exhaustively extract ALL numeric data from scientific PDF text.\n\n"
    "The PDF may contain narrative prose, tables, captions, figure descriptions, "
    "comparative statements, and ranges. Extract from ALL of them.\n\n"
    "OUTPUT RULES:\n"
    "- Return ONLY valid JSON. No markdown, no explanations, no comments, no trailing commas.\n"
    "- Output MUST exactly match the schema below.\n"
    "- Never omit required fields. Use \"\" for unknown or missing values — "
    "never use 'N/A', 'unknown', or null.\n\n"
    "JSON SCHEMA:\n"
    "{\n"
    "  \"material_name\": \"\",\n"
    "  \"material_abbreviation\": \"\",\n"
    "  \"manufacturer\": \"\",\n"
    "  \"doi\": \"\",\n"
    "  \"properties\": [\n"
    "    {\n"
    "      \"section\": \"\",\n"
    "      \"property_name\": \"\",\n"
    "      \"value\": \"\",\n"
      "      \"unit\": \"\",\n"
    "      \"test_condition\": \"\",\n"
    "      \"comments\": \"\",\n"
    "      \"source_text\": \"\",\n"
    "      \"material_name\": \"\",\n"
    "      \"source_page\": \"\",\n"
    "      \"chunk_type\": \"\"\n"
    "    }\n"
    "  ]\n"
    "}\n\n"
    "FIELD RULES:\n"
    "- property_name: copy the property name EXACTLY as written in the source. "
    "Preserve abbreviations, symbols, Greek letters exactly. "
    "If corrupted, infer from context (column headers, units, values).\n"
    "- value: copy EXACTLY. Period as decimal separator. No thousands separators. "
    "Ranges stay as ranges. Strip qualifier symbols. Never blank if a number exists.\n"
    "- unit: copy exactly as written, preserving capitalisation. Do not convert or normalise.\n"
    "- material_name (per property): exact full name for THIS property as written in source. "
    "Never collapse multiple materials. Never leave blank.\n"
    "- section: classify into the most appropriate category. "
    "Use standard categories where they fit — Mechanical, Thermal, Electrical, Physical, "
    "Rheological, Optical, Chemical, Barrier, Morphological, Surface. "
    "If none of these fit, describe the category yourself in one word. "
    "Never force a property into a wrong category just to use a standard name.\n"
    "- source_page: digits only from the block header.\n"
    "- chunk_type: 'table' or 'text' from the block header.\n"
    "- source_text: exact sentence (text blocks) or header+data row (table blocks). Never blank.\n"
    "- test_condition: standard or test conditions. Leave '' if none. Never embed in value.\n"
    "- comments: footnotes, qualifications, uncertainty. Leave '' if none.\n\n"
    "CRITICAL EXTRACTION RULES:\n"
    "- Extract ANY quantity that has a numeric value. If a number appears in the text or table "
    "and describes any characteristic — of the material, its processing, its geometry, "
    "its performance, its environment, its testing conditions, or its behavior — extract it. "
    "Do not filter by property type. If it has a number and a context, it belongs in the output.\n"
    "- Also extract formulation and composition data: ingredient quantities in "
    "phr, wt%, vol%, mol% are valid properties. Use ingredient name as property_name, "
    "quantity as value, appropriate unit, and section = 'Chemical'.\n"
    "- Example: 'Sulphur = 2.0 phr' -> property_name='Sulphur', value='2.0', unit='phr', section='Chemical'.\n"
    "- Extract from BOTH prose and tables — never skip either.\n"
    "- If multiple materials appear, create SEPARATE entries for each — never merge them.\n"
    "- If a single table cell contains values for multiple materials, create one row per material.\n"
    "- NEVER return an empty properties array if any numeric value exists.\n\n"
    "NORMALIZATION RULES:\n"
    "- Preserve original units and property names exactly as written.\n"
    "- Do NOT convert units, compute averages, infer missing values, or hallucinate standards.\n\n"
    "ANTI-HALLUCINATION:\n"
    "- Only extract values explicitly stated in the provided text.\n"
    "- If a value is not present, do not invent it.\n"
    "- Missing a property is acceptable. Inventing one is not.\n"
)

# ─────────────────────────────────────────────────────────────────────────────
# BOILERPLATE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

_SKIP_HEADING_RE = re.compile(
    r"^(references|bibliography|acknowledgements?|table\s+of\s+contents|"
    r"copyright|legal\s+notice|disclaimer|index|appendix\s+[a-z]$)",
    re.IGNORECASE,
)

def _is_boilerplate(text: str) -> bool:
    return bool(_SKIP_HEADING_RE.match(text.strip().split("\n")[0].strip()))

# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    page_num:   int
    chunk_type: str
    source:     str
    raw_rows:   Optional[List[List[str]]] = None
    raw_text:   Optional[str] = None
    text:       str = field(init=False)
    score:      float = 0.0
    relevant:   bool  = False

    def __post_init__(self):
        if self.chunk_type == "table" and self.raw_rows:
            self.text = _rows_to_text(self.raw_rows)
        elif self.raw_text:
            self.text = self.raw_text.strip()
        else:
            self.text = ""

    @property
    def block_header(self) -> str:
        return (
            f"\n\n{'─'*60}\n"
            f"[{self.chunk_type.upper()} | Page {self.page_num} | "
            f"score={self.score:.3f}]\n"
            f"{'─'*60}\n"
        )

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fix_spaced_text(text: str) -> str:
    return re.sub(r'\b([A-Z]) (?=[A-Z] |[A-Z]\b)', r'\1', text)

def _rows_to_text(rows: List[List[Any]]) -> str:
    lines = []
    for row in rows:
        cells = [str(c).strip() if c is not None else "" for c in row]
        if any(len(c) >= MIN_CELL_CHARS for c in cells):
            lines.append(" | ".join(cells))
    return _fix_spaced_text("\n".join(lines))

def _is_valid_table(rows: List[List[Any]]) -> bool:
    if not rows or len(rows) < MIN_TABLE_ROWS:
        return False
    if max((len(r) for r in rows), default=0) < MIN_TABLE_COLS:
        return False
    for row in rows[1:]:
        if any(re.search(r"\d", str(c)) for c in row):
            return True
    return False

def _split_paragraphs(raw_text: str) -> List[str]:
    paragraphs: List[str] = []
    current: List[str] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
        else:
            current.append(stripped)
    if current:
        paragraphs.append(" ".join(current))
    return [_fix_spaced_text(p) for p in paragraphs if len(p) >= MIN_TEXT_CHARS]

def _normalise_doi(raw: str) -> str:
    if not raw:
        return ""
    doi = raw.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi.org/", "DOI:", "doi:"):
        if doi.lower().startswith(prefix.lower()):
            doi = doi[len(prefix):]
    return doi.strip()

def _doi_url(doi: str) -> str:
    doi = _normalise_doi(doi)
    if not doi:
        return ""
    return f"https://doi.org/{doi}"

# ─────────────────────────────────────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────────────────────────────────────

def _pdf_hash(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]

def _load_cache() -> Dict:
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_cache(cache: Dict):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception as e:
        log.warning(f"Cache save failed: {e}")

def cache_get(pdf_bytes: bytes) -> Optional[Dict]:
    return _load_cache().get(_pdf_hash(pdf_bytes))

def cache_set(pdf_bytes: bytes, result: Dict):
    cache = _load_cache()
    cache[_pdf_hash(pdf_bytes)] = result
    _save_cache(cache)

# ─────────────────────────────────────────────────────────────────────────────
# CHROMADB CLIENT
# ─────────────────────────────────────────────────────────────────────────────

_chroma_client: Optional[Any] = None
_chroma_collection: Optional[Any] = None
_embed_model: Optional[Any] = None

def _get_embed_model() -> Any:
    global _embed_model
    if _embed_model is None:
        if not ST_AVAILABLE:
            raise ImportError("sentence-transformers not installed.\nRun: pip install sentence-transformers")
        log.info(f"Loading embedding model '{EMBED_MODEL_NAME}' …")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        log.info("Embedding model ready.")
    return _embed_model

threading.Thread(target=_prewarm_embed_model, daemon=True).start()
_run_startup_probes(GEMINI_MODEL)

def _get_chroma_collection() -> Any:
    global _chroma_client, _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection
    if not CHROMA_AVAILABLE:
        raise ImportError("chromadb not installed. Run: pip install chromadb")
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    _chroma_collection = _chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
    log.info(f"ChromaDB ready — '{CHROMA_COLLECTION}' ({_chroma_collection.count()} existing vectors)")
    return _chroma_collection

def _chroma_pdf_exists(pdf_hash: str) -> bool:
    try:
        col = _get_chroma_collection()
        res = col.get(where={"pdf_hash": pdf_hash}, limit=1)
        return len(res["ids"]) > 0
    except Exception:
        return False

def _chroma_store_chunks(chunks: List[Chunk], pdf_hash: str) -> None:
    col   = _get_chroma_collection()
    model = _get_embed_model()
    texts     = [c.text for c in chunks]
    metadatas = [{"pdf_hash": pdf_hash, "page_num": c.page_num,
                  "chunk_type": c.chunk_type, "source": c.source} for c in chunks]
    ids = [f"{pdf_hash}_{i}" for i in range(len(chunks))]
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
    col.upsert(ids=ids, documents=texts, embeddings=vecs.tolist(), metadatas=metadatas)
    log.info(f"ChromaDB: stored {len(chunks)} chunks for pdf_hash={pdf_hash}")



def _chroma_semantic_verify(
    query: str,
    pdf_hash: str,
    top_k: int = 5,
) -> List[Dict]:
    """
    Semantic search across ALL chunks for this pdf_hash.
    Returns top_k chunk dicts ranked by similarity to query.
    No page filter — page number from LLM is unreliable.
    """
    try:
        col   = _get_chroma_collection()
        model = _get_embed_model()
        query_vec = model.encode([query], normalize_embeddings=True)[0].tolist()
        results = col.query(
            query_embeddings=[query_vec],
            n_results=min(top_k, max(col.count(), 1)),
            where={"pdf_hash": pdf_hash},
            include=["documents", "metadatas", "distances"],
        )
        docs      = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        chunks_out = []
        for doc, meta, dist in zip(docs, metadatas, distances):
            chunks_out.append({
                "text":       doc,
                "page_num":   meta.get("page_num", ""),
                "chunk_type": meta.get("chunk_type", ""),
                "similarity": round(1.0 - float(dist), 4),
            })
        return chunks_out
    except Exception as e:
        log.warning(f"ChromaDB semantic verify failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION — TABLES
# ─────────────────────────────────────────────────────────────────────────────

def _extract_tables_docling(pdf_bytes: bytes) -> List[Chunk]:
    chunks: List[Chunk] = []
    if not DOCLING_AVAILABLE:
        return chunks
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes); tmp_path = tmp.name
    try:
        # converter = DocumentConverter()
        # result    = converter.convert(tmp_path)
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat

        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = 1.0
        pipeline_options.generate_page_images = False

        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        result = converter.convert(tmp_path)
        doc       = result.document
        for table in doc.tables:
            rows     = [[cell.text for cell in row] for row in table.data.grid]
            page_num = table.prov[0].page_no if table.prov else 1
            if _is_valid_table(rows):
                chunks.append(Chunk(page_num=page_num, chunk_type="table", source="docling", raw_rows=rows))
        log.info(f"Docling tables: {len(chunks)}")
    except Exception as e:
        log.error(f"Docling table extraction failed: {e}")
    finally:
        os.unlink(tmp_path)
    return chunks

def _extract_tables_pdfplumber(pdf_bytes: bytes) -> List[Chunk]:
    chunks: List[Chunk] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                for strategy in (
                    {"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict",
                     "snap_tolerance": 3, "join_tolerance": 3},
                    {"vertical_strategy": "text", "horizontal_strategy": "text"},
                ):
                    tables = page.extract_tables(table_settings=strategy) or []
                    for rows in tables:
                        cleaned = [[str(c).strip() if c else "" for c in row] for row in rows]
                        if _is_valid_table(cleaned):
                            chunks.append(Chunk(page_num=page_num, chunk_type="table",
                                                source="pdfplumber", raw_rows=cleaned))
                    if tables:
                        break
        log.info(f"pdfplumber tables: {len(chunks)}")
    except Exception as e:
        log.error(f"pdfplumber table extraction failed: {e}")
    return chunks

def _extract_tables_camelot(pdf_bytes: bytes) -> List[Chunk]:
    chunks: List[Chunk] = []
    if not CAMELOT_AVAILABLE:
        return chunks
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes); tmp_path = tmp.name
    try:
        for flavor in ("lattice", "stream"):
            try:
                tables = camelot.read_pdf(tmp_path, pages="all", flavor=flavor)
                if flavor == "stream":
                    tables = [t for t in tables if t.parsing_report.get("accuracy", 0) > 65]
                for table in tables:
                    rows = [[str(c).strip() for c in row] for row in table.df.values.tolist()]
                    if _is_valid_table(rows):
                        chunks.append(Chunk(page_num=table.page, chunk_type="table",
                                            source=f"camelot-{flavor}", raw_rows=rows))
                if chunks:
                    break
            except Exception as e:
                log.warning(f"camelot {flavor}: {e}")
    finally:
        os.unlink(tmp_path)
    log.info(f"camelot tables: {len(chunks)}")
    return chunks

# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION — TEXT
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text_docling(pdf_bytes: bytes) -> List[Chunk]:
    chunks: List[Chunk] = []
    if not DOCLING_AVAILABLE:
        return chunks
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes); tmp_path = tmp.name
    try:
        # converter = DocumentConverter()
        # result    = converter.convert(tmp_path)
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat

        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = 1.0
        pipeline_options.generate_page_images = False

        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        result = converter.convert(tmp_path)
        doc       = result.document
        for item, _ in doc.iterate_items():
            text     = getattr(item, "text", "").strip()
            page_num = item.prov[0].page_no if item.prov else 1
            if not text or len(text) < MIN_TEXT_CHARS or _is_boilerplate(text):
                continue
            chunks.append(Chunk(page_num=page_num, chunk_type="text", source="docling", raw_text=text))
        log.info(f"Docling text chunks: {len(chunks)}")
    except Exception as e:
        log.error(f"Docling text extraction failed: {e}")
    finally:
        os.unlink(tmp_path)
    return chunks

def _extract_text_pymupdf(pdf_bytes: bytes) -> List[Chunk]:
    chunks: List[Chunk] = []
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page_idx, page in enumerate(doc):
                raw = page.get_text("text", sort=True, flags=fitz.TEXT_PRESERVE_LIGATURES) or ""
                raw = raw.encode('utf-8', errors='replace').decode('utf-8')
                for para in _split_paragraphs(raw):
                    if not _is_boilerplate(para):
                        chunks.append(Chunk(page_num=page_idx+1, chunk_type="text",
                                            source="pymupdf", raw_text=para))
        log.info(f"pymupdf text chunks: {len(chunks)}")
    except Exception as e:
        log.error(f"pymupdf text extraction failed: {e}")
    return chunks

def _extract_text_pdfplumber(pdf_bytes: bytes) -> List[Chunk]:
    chunks: List[Chunk] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                for para in _split_paragraphs(raw):
                    if not _is_boilerplate(para):
                        chunks.append(Chunk(page_num=page_num, chunk_type="text",
                                            source="pdfplumber-text", raw_text=para))
        log.info(f"pdfplumber text chunks: {len(chunks)}")
    except Exception as e:
        log.error(f"pdfplumber text extraction failed: {e}")
    return chunks

def _ocr_page(pdf_bytes: bytes, page_num: int) -> str:
    if not OCR_AVAILABLE:
        return ""
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            page = doc[page_num - 1]
            mat  = fitz.Matrix(300 / 72, 300 / 72)
            pix  = page.get_pixmap(matrix=mat)
            img  = Image.open(io.BytesIO(pix.tobytes("png")))
            return pytesseract.image_to_string(img, lang="eng") or ""
    except Exception as e:
        log.warning(f"OCR page {page_num}: {e}")
        return ""

def _verify_page_coverage(pdf_bytes: bytes, chunks: List[Chunk]) -> List[Chunk]:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        total_pages = set(range(1, len(doc) + 1))
    covered = {c.page_num for c in chunks}
    missing = total_pages - covered
    if not missing:
        return chunks
    log.warning(f"Page coverage: {len(missing)} pages missing — OCRing: {sorted(missing)}")
    for page_num in sorted(missing):
        for para in _split_paragraphs(_ocr_page(pdf_bytes, page_num)):
            if not _is_boilerplate(para):
                chunks.append(Chunk(page_num=page_num, chunk_type="text",
                                    source="ocr-fallback", raw_text=para))
    return chunks

def _dedup(chunks: List[Chunk]) -> List[Chunk]:
    seen: set = set()
    unique: List[Chunk] = []
    for c in chunks:
        key = (c.chunk_type, re.sub(r"\s+", " ", c.text.strip())[:200])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique

def _is_corrupted_chunk(chunk: Chunk, threshold: float = 0.20) -> bool:
    """Flag chunk as corrupted if >threshold fraction of tokens are single characters."""
    if not chunk.text or len(chunk.text) < 10:
        return False
    tokens = chunk.text.split()
    if not tokens:
        return False
    single_char = sum(1 for t in tokens if len(t.strip('|-.,:;')) <= 1)
    return (single_char / len(tokens)) > threshold


def extract_all_chunks(pdf_bytes: bytes) -> List[Chunk]:
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {
            ex.submit(_extract_tables_docling,    pdf_bytes): "docling_tables",
            ex.submit(_extract_tables_pdfplumber, pdf_bytes): "pdfplumber_tables",
            ex.submit(_extract_tables_camelot,    pdf_bytes): "camelot_tables",
            ex.submit(_extract_text_pymupdf,      pdf_bytes): "pymupdf_text",
            ex.submit(_extract_text_pdfplumber,   pdf_bytes): "pdfplumber_text",
        }
        results = {}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                log.error(f"{name} failed: {e}")
                results[name] = []

    docling_tables    = results.get("docling_tables", [])
    pdfplumber_tables = results.get("pdfplumber_tables", [])
    camelot_tables    = results.get("camelot_tables", [])
    pymupdf_text      = results.get("pymupdf_text", [])
    pdfplumber_text   = results.get("pdfplumber_text", [])

    # Docling first — prefer its tables, others fill gaps for pages docling missed
    if docling_tables:
        docling_pages = {c.page_num for c in docling_tables}
        fallback_tables = [c for c in pdfplumber_tables + camelot_tables
                           if c.page_num not in docling_pages]
        table_chunks = _dedup(docling_tables + fallback_tables)
    else:
        table_chunks = _dedup(pdfplumber_tables + camelot_tables)

    text_chunks: List[Chunk] = []
    if DOCLING_AVAILABLE and docling_tables:
        text_chunks.extend(_extract_text_docling(pdf_bytes))
    text_chunks.extend(pymupdf_text)
    if sum(len(c.text) for c in pymupdf_text) < 500:
        log.warning("pymupdf sparse — supplementing with pdfplumber text")
        text_chunks.extend(pdfplumber_text)
    text_chunks = _dedup(text_chunks)

    all_chunks = table_chunks + text_chunks
    # Corruption check — if a chunk has >20% single-char tokens, attempt OCR on that page
    corrupted_pages = {c.page_num for c in all_chunks if _is_corrupted_chunk(c)}
    if corrupted_pages:
        log.warning(f"Corruption detected on pages {sorted(corrupted_pages)} — attempting OCR")
        for page_num in sorted(corrupted_pages):
            ocr_text = _ocr_page(pdf_bytes, page_num)
            for para in _split_paragraphs(ocr_text):
                if not _is_boilerplate(para):
                    all_chunks.append(Chunk(page_num=page_num, chunk_type="text",
                                           source="ocr-corruption-fix", raw_text=para))
    all_chunks = _verify_page_coverage(pdf_bytes, all_chunks)
    log.info(f"Total chunks: {len(all_chunks)} ({len(table_chunks)} tables + {len(text_chunks)} text)")
    return all_chunks

# ─────────────────────────────────────────────────────────────────────────────
# CHROMADB INDEX
# ─────────────────────────────────────────────────────────────────────────────

def _build_overlapping_chunks(chunks: List[Chunk], neighbor_window: int = NEIGHBOR_OVERLAP) -> List[Chunk]:
    if neighbor_window <= 0:
        return chunks
    eligible = [(i, c) for i, c in enumerate(chunks) if not _is_boilerplate(c.text)]
    overlap_chunks: List[Chunk] = []
    seen_content: set = set()
    for pos, (_, centre) in enumerate(eligible):
        if centre.chunk_type == "table":
            continue
        parts: List[str] = []
        for offset in range(-neighbor_window, neighbor_window + 1):
            nb_pos = pos + offset
            if nb_pos < 0 or nb_pos >= len(eligible):
                continue
            _, nb_chunk = eligible[nb_pos]
            if nb_chunk.chunk_type == "table":
                continue
            parts.append(nb_chunk.text.strip())
        merged = " \n\n ".join(p for p in parts if p)
        if not merged or merged in seen_content:
            continue
        seen_content.add(merged)
        overlap_chunks.append(Chunk(page_num=centre.page_num, chunk_type="text",
                                    source=f"{centre.source}+overlap", raw_text=merged))
    return chunks + overlap_chunks

def index_chunks_in_chroma(chunks: List[Chunk], pdf_hash: str) -> None:
    if _chroma_pdf_exists(pdf_hash):
        log.info(f"ChromaDB: pdf_hash={pdf_hash} already indexed — skipping.")
        return
    all_chunks = _build_overlapping_chunks(chunks, neighbor_window=NEIGHBOR_OVERLAP)
    storable = [c for c in all_chunks if not _is_boilerplate(c.text) and c.text.strip()]
    _chroma_store_chunks(storable, pdf_hash)

# ─────────────────────────────────────────────────────────────────────────────
# BATCH BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_batches(chunks: List[Chunk]) -> List[str]:
    if not chunks:
        return []
    batches: List[str] = []
    current_batch: List[Chunk] = []
    current_chars: int = 0
    for chunk in chunks:
        entry_len = len(chunk.block_header) + len(chunk.text)
        _max = _get_max_batch_chars()
        if entry_len > _max:
            chunk.text = chunk.text[: _max - len(chunk.block_header) - 20]
            entry_len  = len(chunk.block_header) + len(chunk.text)
        if current_chars + entry_len > _max and current_batch:
            current_batch.sort(key=lambda c: c.page_num)
            batches.append("\n".join(c.block_header + c.text for c in current_batch))
            current_batch = []
            current_chars = 0
        current_batch.append(chunk)
        current_chars += entry_len
    if current_batch:
        current_batch.sort(key=lambda c: c.page_num)
        batches.append("\n".join(c.block_header + c.text for c in current_batch))
    log.info(f"Built {len(batches)} batch(es) from {len(chunks)} chunks")
    return batches

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _call_gemini(text_payload: str) -> Tuple[Optional[Dict], str]:
    payload = {
        "contents": [{"parts": [{"text": EXTRACTION_PROMPT + text_payload}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": SCHEMA,
            "maxOutputTokens": _GEMINI_CONFIG["max_output_tokens"],
        },
    }
    for attempt, wait in enumerate([0] + GEMINI_RETRY_DELAYS):
        if wait:
            log.warning(f"Gemini 429 — waiting {wait}s (retry {attempt})")
            time.sleep(wait)
        try:
            resp = requests.post(GEMINI_API_URL, json=payload, timeout=600)
            if resp.status_code == 429:
                continue
            if not resp.ok:
                return None, f"Gemini HTTP {resp.status_code}: {resp.text[:2000]}"
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return None, f"Gemini no candidates: {json.dumps(data)[:400]}"
            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                text = part.get("text", "").strip()
                if text.startswith("{"):
                    try:
                        text = re.sub(
                            r'("(?:trade_grade|manufacturer)"\s*:\s*")([^"]{150})[^"]*"',
                            r'\1\2"', text)
                        parsed = json.loads(text)
                        tg = parsed.get("trade_grade", "")
                        if tg and (len(tg) > 30 or tg.count(" ") > 3 or
                                   any(w in tg.lower() for w in
                                       ["study","used","modified","provided","analysis",
                                        "document","without","unless","specified","otherwise",
                                        "percent","weight","fiber","composite","matrix"])):
                            parsed["trade_grade"] = ""
                        mfr = parsed.get("manufacturer", "")
                        if mfr and len(mfr) > 100:
                            parsed["manufacturer"] = mfr[:100].rsplit(" ", 1)[0]
                        return parsed, ""
                    except json.JSONDecodeError as e:
                        return None, f"Gemini JSON parse error: {e}"
            return None, "No JSON in Gemini response."
        except requests.Timeout:
            return None, "Gemini timeout after 600s."
        except Exception as e:
            return None, f"Gemini unexpected: {e}"
    return None, "Gemini max retries exceeded."

def run_gemini_batches(batches: List[str], progress_callback=None) -> Tuple[List[Dict], List[str]]:
    results: List[Dict] = []
    errors:  List[str]  = []
    total = len(batches)
    for idx, batch_text in enumerate(batches):
        msg = f"[Gemini] batch {idx+1}/{total}…"
        log.info(msg)
        if progress_callback:
            progress_callback(msg, 0.60 + 0.17 * (idx / max(total, 1)))
        if idx > 0:
            time.sleep(_GEMINI_CONFIG["batch_delay"])
        result, err = _call_gemini(batch_text)
        if result:
            results.append(result)
        if err:
            errors.append(f"Gemini batch {idx+1}: {err}")
            log.error(f"Gemini batch {idx+1} error: {err}")
    return results, errors

# ─────────────────────────────────────────────────────────────────────────────
# GPT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _call_gpt(text_payload: str) -> Tuple[Optional[Dict], str]:
    if not OPENAI_API_KEY:
        return None, "OPENAI_API_KEY not set."
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GPT_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": GPT_SYSTEM_PROMPT},
            {"role": "user",   "content": EXTRACTION_PROMPT + text_payload},
        ],
    }
    for attempt, wait in enumerate([0] + GPT_RETRY_DELAYS):
        if wait:
            log.warning(f"GPT 429 — waiting {wait}s (retry {attempt})")
            time.sleep(wait)
        try:
            resp = requests.post(GPT_API_URL, headers=headers, json=payload, timeout=300)
            if resp.status_code == 429:
                continue
            if not resp.ok:
                return None, f"GPT HTTP {resp.status_code}: {resp.text[:2000]}"
            data    = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            if content.startswith("{"):
                try:
                    return json.loads(content), ""
                except json.JSONDecodeError as e:
                    return None, f"GPT JSON parse error: {e} | raw: {content[:300]}"
            return None, f"GPT response not JSON: {content[:300]}"
        except requests.Timeout:
            return None, "GPT timeout after 300s."
        except Exception as e:
            return None, f"GPT unexpected: {e}"
    return None, "GPT max retries exceeded."

def run_gpt_batches(batches: List[str], progress_callback=None) -> Tuple[List[Dict], List[str]]:
    results: List[Dict] = []
    errors:  List[str]  = []
    total = len(batches)
    for idx, batch_text in enumerate(batches):
        msg = f"[GPT] batch {idx+1}/{total}…"
        log.info(msg)
        if progress_callback:
            progress_callback(msg, 0.60 + 0.17 * (idx / max(total, 1)))
        if idx > 0:
            time.sleep(_GPT_CONFIG["batch_delay"])
        result, err = _call_gpt(batch_text)
        if result:
            results.append(result)
        if err:
            errors.append(f"GPT batch {idx+1}: {err}")
            log.error(f"GPT batch {idx+1} error: {err}")
    return results, errors

# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL DUAL-LLM RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_dual_llm_parallel(
    batches: List[str],
    progress_callback=None,
) -> Tuple[List[Dict], List[Dict], List[str]]:
    gemini_results: List[Dict] = []
    gpt_results:    List[Dict] = []
    all_errors:     List[str]  = []

    if progress_callback:
        progress_callback("Running Gemini + GPT in parallel…", 0.62)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_gemini = executor.submit(run_gemini_batches, batches, None)
        future_gpt    = executor.submit(run_gpt_batches,    batches, None)
        for future in as_completed([future_gemini, future_gpt]):
            label = "Gemini" if future is future_gemini else "GPT"
            try:
                res, errs = future.result()
                (gemini_results if future is future_gemini else gpt_results).extend(res)
                all_errors.extend(errs)
                log.info(f"{label} done — {len(res)} batch result(s)")
            except Exception as e:
                all_errors.append(f"{label} thread error: {e}")

    return gemini_results, gpt_results, all_errors

# ─────────────────────────────────────────────────────────────────────────────
# MERGE + DEDUP
# ─────────────────────────────────────────────────────────────────────────────

FIXED_COLS = ["material_name", "material_abbreviation", "manufacturer", "doi"]

def _make_abbreviation(name: str) -> str:
    abbr = "".join(c for c in name if c.isupper())
    return abbr or name[:4].upper()

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower().strip())

def _richness(row: Dict) -> int:
    return sum(1 for v in row.values() if str(v).strip() not in ("", "N/A"))

def _fuzzy_dedup(rows: List[Dict]) -> List[Dict]:
    seen_fuzzy: Dict[tuple, int] = {}
    kept: List[Dict] = []
    for row in rows:
        key = (_norm(row.get("material_name", "")),
               _norm(row.get("property_name", "")),
               _norm(row.get("section", "")))
        if key in seen_fuzzy:
            existing_idx = seen_fuzzy[key]
            existing     = kept[existing_idx]
            if _richness(row) > _richness(existing):
                merged = dict(row)
                for k, v in existing.items():
                    if str(merged.get(k, "")).strip() in ("", "N/A") and str(v).strip() not in ("", "N/A"):
                        merged[k] = v
                kept[existing_idx] = merged
        else:
            seen_fuzzy[key] = len(kept)
            kept.append(row)
    return kept

def merge_to_dataframe(results: List[Dict]) -> Tuple[pd.DataFrame, str]:
    all_rows:   List[Dict] = []
    seen_exact: set        = set()
    fallback = {k: "" for k in FIXED_COLS}
    for r in results:
        for k in FIXED_COLS:
            if not fallback[k] and r.get(k):
                fallback[k] = r[k]
    if not fallback["material_abbreviation"] and fallback["material_name"]:
        fallback["material_abbreviation"] = _make_abbreviation(fallback["material_name"])
    doi = _normalise_doi(fallback.get("doi", ""))
    for r in results:
        r_doi = _normalise_doi(r.get("doi", ""))
        if r_doi and not doi:
            doi = r_doi
        r_identity = {
            "material_name":         r.get("material_name", "") or fallback["material_name"],
            "material_abbreviation": r.get("material_abbreviation", "") or fallback["material_abbreviation"],
            "manufacturer":          r.get("manufacturer", "") or fallback["manufacturer"],
        }
        if not r_identity["material_abbreviation"] and r_identity["material_name"]:
            r_identity["material_abbreviation"] = _make_abbreviation(r_identity["material_name"])
        for item in r.get("properties", r.get("mechanical_properties", [])):
            prop_mat = item.get("material_name", "").strip()
            identity = dict(r_identity)
            if prop_mat:
                identity["material_name"]         = prop_mat
                identity["material_abbreviation"] = _make_abbreviation(prop_mat)
            raw_page   = str(item.get("source_page", "")).strip()
            page_label = (f"Page {raw_page}" if raw_page.isdigit()
                          else raw_page if raw_page else "Unknown")
            chunk_type = item.get("chunk_type", "").strip() or "unknown"
            key = (_norm(identity["material_name"]),
                   _norm(item.get("section", "")),
                   _norm(item.get("property_name", "")),
                   _norm(item.get("value", "")))
            if key in seen_exact:
                continue
            seen_exact.add(key)
            # Simple value extraction — prompts now say preserve as-is, no min/max needed
            value = item.get("value", "").strip()
            all_rows.append({
                **identity,
                "section":        item.get("section", "")        or "General",
                "property_name":  item.get("property_name", "")  or "Unknown",
                "value":          value                           or "N/A",
                "unit":           item.get("unit", "")            or "",
                "english":        item.get("english", "")         or "",
                "test_condition": item.get("test_condition", "")  or "",
                "comments":       item.get("comments", "")        or "",
                "source_text":    item.get("source_text", "")     or "",
                "source_page":    page_label,
                "chunk_type":     chunk_type,
            })
    all_rows = _fuzzy_dedup(all_rows)
    df = pd.DataFrame(all_rows)
    if not df.empty:
        base_cols = [c for c in df.columns if c not in ("source_page", "chunk_type")]
        df = df[base_cols + ["source_page", "chunk_type"]]
    return df, doi

# ─────────────────────────────────────────────────────────────────────────────
# CONSENSUS FILTER  — updated functions
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# CONSENSUS FILTER
# ─────────────────────────────────────────────────────────────────────────────

_current_pdf_hash: str = ""


def _first_num(s: str) -> Optional[float]:
    m = re.search(r"[\d.]+", str(s))
    return float(m.group()) if m else None


def _to_si(value: float, unit: str):
    _UNIT_SI = {
        "pa": ("pressure", 1.0), "hpa": ("pressure", 1e2), "kpa": ("pressure", 1e3),
        "mpa": ("pressure", 1e6), "gpa": ("pressure", 1e9),
        "bar": ("pressure", 1e5), "psi": ("pressure", 6_894.757),
        "n/mm2": ("pressure", 1e6), "n/m2": ("pressure", 1.0),
        "kg/m3": ("density", 1.0), "g/cm3": ("density", 1e3),
        "g/cc": ("density", 1e3), "g/ml": ("density", 1e3),
        "k": ("temperature", 1.0), "c": ("temperature", 1.0), "degc": ("temperature", 1.0),
        "%": ("ratio", 1.0), "wt%": ("ratio", 1.0), "vol%": ("ratio", 1.0),
        "mol%": ("ratio", 1.0),
        "j/m2": ("surface_energy", 1.0), "kj/m2": ("surface_energy", 1e3),
        "w/mk": ("therm_cond", 1.0), "w/(mk)": ("therm_cond", 1.0),
        "pas": ("viscosity", 1.0), "mpas": ("viscosity", 1e-3), "cp": ("viscosity", 1e-3),
        "g/10min": ("melt_flow", 1.0),
        "g/mol": ("mol_weight", 1.0), "kg/mol": ("mol_weight", 1e3),
        "da": ("mol_weight", 1.0), "kda": ("mol_weight", 1e3),
    }
    uk = re.sub(r"[^a-z0-9/()]+", "", str(unit).lower().strip())
    entry = _UNIT_SI.get(uk)
    if entry is None:
        return value, None
    family, factor = entry
    if family == "temperature":
        if uk in ("c", "degc"):
            return value + 273.15, "temperature"
        return value, "temperature"
    if factor is None:
        return value, None
    return value * factor, family


def _value_match(val_a: str, val_b: str, unit_a: str = "", unit_b: str = "",
                 tol: float = CONSENSUS_VALUE_TOL) -> bool:
    na, nb = _norm(val_a), _norm(val_b)
    if na == nb:
        return True

    range_re = re.compile(r"([\d.]+)\s*[-\u2013\u2014]\s*([\d.]+)")

    def _safe_range(m):
        lo, hi = float(m.group(1)), float(m.group(2))
        return (min(lo, hi), max(lo, hi))

    rm_a = range_re.search(str(val_a))
    rm_b = range_re.search(str(val_b))

    if rm_a:
        lo_a, hi_a = _safe_range(rm_a)
    else:
        fa = _first_num(na)
        lo_a, hi_a = (fa, fa) if fa is not None else (None, None)

    if rm_b:
        lo_b, hi_b = _safe_range(rm_b)
    else:
        fb = _first_num(nb)
        lo_b, hi_b = (fb, fb) if fb is not None else (None, None)

    if lo_a is None or lo_b is None:
        return False

    if unit_a and unit_b and _norm(unit_a) != _norm(unit_b):
        try:
            lo_a_si, fam_a = _to_si(lo_a, unit_a)
            hi_a_si, _     = _to_si(hi_a, unit_a)
            lo_b_si, fam_b = _to_si(lo_b, unit_b)
            hi_b_si, _     = _to_si(hi_b, unit_b)
            if fam_a is not None and fam_b is not None and fam_a == fam_b:
                if lo_a_si <= hi_b_si and lo_b_si <= hi_a_si:
                    return True
                mid_a = (lo_a_si + hi_a_si) / 2
                mid_b = (lo_b_si + hi_b_si) / 2
                return mid_b != 0 and abs(mid_a - mid_b) / max(abs(mid_a), abs(mid_b)) <= tol
        except Exception:
            pass

    if lo_a <= hi_b and lo_b <= hi_a:
        return True
    mid_a = (lo_a + hi_a) / 2
    mid_b = (lo_b + hi_b) / 2
    return mid_b != 0 and abs(mid_a - mid_b) / max(abs(mid_a), abs(mid_b)) <= tol


def _prop_key(row: Dict) -> Tuple[str, str]:
    return (_norm(row.get("property_name", "")), _norm(row.get("material_name", "")))


def _token_overlap(a: str, b: str) -> float:
    sa = {t for t in re.split(r"[^a-z0-9]+", _norm(a)) if len(t) >= 2}
    sb = {t for t in re.split(r"[^a-z0-9]+", _norm(b)) if len(t) >= 2}
    if not sa and not sb: return 1.0
    if not sa or not sb:  return 0.0
    return len(sa & sb) / len(sa | sb)


def _embed_similarity(a: str, b: str) -> float:
    try:
        model = _get_embed_model()
        vecs  = model.encode([a, b], normalize_embeddings=True)
        return float(np.dot(vecs[0], vecs[1]))
    except Exception:
        return 0.0


def _llm_adjudicate(pairs: List[Tuple[str, str]]) -> List[bool]:
    if not pairs:
        return []
    lines  = "\n".join(f'{i+1}. Gemini: "{a}" | GPT: "{b}"' for i, (a, b) in enumerate(pairs))
    prompt = (
        "You are a materials science expert.\n"
        "For each numbered pair, decide if both sides refer to the SAME material property "
        "(accounting for abbreviations, synonyms, Greek letters, or different word order).\n"
        "Reply ONLY with a JSON array of booleans, one per pair.\n"
        f"Example for 3 pairs: [true, false, true]\n\n{lines}"
    )
    try:
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        body    = {"model": GPT_MODEL, "temperature": 0,
                   "max_tokens": max(len(pairs) * 15 + 100, 500),
                   "messages": [{"role": "user", "content": prompt}]}
        resp = requests.post(GPT_API_URL, headers=headers, json=body, timeout=30)
        if not resp.ok:
            return [False] * len(pairs)
        content = resp.json()["choices"][0]["message"]["content"].strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        result  = json.loads(content)
        if isinstance(result, list) and len(result) == len(pairs):
            return [bool(v) for v in result]
    except Exception as e:
        log.warning(f"LLM adjudication failed: {e}")
    return [False] * len(pairs)


def _key_str(row: Dict) -> str:
    return f"{row.get('property_name', '')} {row.get('material_name', '')}".strip()


_EMBED_SANITY_FLOOR = 0.50


def _cascade_match(gem_row: Dict, gpt_rows: List[Dict]) -> Optional[Dict]:
    gem_key = _prop_key(gem_row)
    gem_str = _key_str(gem_row)

    # Layer 1 — exact
    for gpt_row in gpt_rows:
        if _prop_key(gpt_row) == gem_key:
            return {**gpt_row, "_match_layer": "exact"}

    # Layer 2 — token overlap >= 0.50
    best_tok, best_tok_row = 0.0, None
    for gpt_row in gpt_rows:
        score = _token_overlap(gem_str, _key_str(gpt_row))
        if score > best_tok:
            best_tok, best_tok_row = score, gpt_row
    if best_tok >= 0.50:
        return {**best_tok_row, "_match_layer": "token"}

    # Layer 3 — embedding: above sanity floor pass, below send to LLM
    best_emb, best_emb_row = 0.0, None
    for gpt_row in gpt_rows:
        score = _embed_similarity(gem_str, _key_str(gpt_row))
        if score > best_emb:
            best_emb, best_emb_row = score, gpt_row

    if best_emb_row is not None:
        if best_emb >= _EMBED_SANITY_FLOOR:
            return {**best_emb_row, "_match_layer": "embedding"}
        else:
            return {**best_emb_row, "_match_layer": "llm_needed"}

    return None


def _resolve_value_disagreement(gem_row: Dict, gpt_row: Dict, layer: str,
                                tol: float = CONSENSUS_VALUE_TOL) -> List[Dict]:
    gem_val  = str(gem_row.get("value", "")).strip()
    gpt_val  = str(gpt_row.get("value", "")).strip()
    gem_unit = str(gem_row.get("unit", "")).strip()
    gpt_unit = str(gpt_row.get("unit", "")).strip()
    gem_cond = _norm(gem_row.get("test_condition", ""))
    gpt_cond = _norm(gpt_row.get("test_condition", ""))
    gem_null = not gem_val or gem_val == "N/A"
    gpt_null = not gpt_val or gpt_val == "N/A"

    def _clean(row: Dict) -> Dict:
        return {k: v for k, v in row.items() if not k.startswith("_")}

    def _make_row(base: Dict, other: Dict) -> Dict:
        row = dict(base)
        for k, v in other.items():
            if k.startswith("_"):
                continue
            if str(row.get(k, "")).strip() in ("", "N/A") and str(v).strip() not in ("", "N/A"):
                row[k] = v
        row["confirmed_by_gpt"] = True
        row["match_layer"]      = layer
        return row

    if gem_null and not gpt_null:
        row = _make_row(gem_row, gpt_row)
        row["value"] = gpt_val
        return [row]
    if gpt_null and not gem_null:
        return [_make_row(gem_row, gpt_row)]
    if gem_null and gpt_null:
        return []

    if _value_match(gem_val, gpt_val, gem_unit, gpt_unit, tol):
        return [_make_row(gem_row, gpt_row)]

    if gem_cond and gpt_cond and gem_cond != gpt_cond:
        return [
            {**_clean(gem_row), "confirmed_by_gpt": True, "match_layer": layer},
            {**_clean(gpt_row), "confirmed_by_gpt": True, "match_layer": layer},
        ]

    if gem_unit and gpt_unit and _norm(gem_unit) != _norm(gpt_unit):
        return [{**_clean(gem_row), "confirmed_by_gpt": True, "match_layer": layer}]

    return [{**_clean(gem_row), "confirmed_by_gpt": True, "match_layer": layer}]


def consensus_filter(df_gemini: pd.DataFrame, df_gpt: pd.DataFrame,
                     tol: float = CONSENSUS_VALUE_TOL) -> pd.DataFrame:
    if df_gemini.empty and df_gpt.empty:
        return pd.DataFrame()
    if df_gemini.empty:
        return df_gpt.assign(confirmed_by_gpt=False, match_layer="none")
    if df_gpt.empty:
        return df_gemini.assign(confirmed_by_gpt=False, match_layer="none")

    from collections import defaultdict

    gem_rows = df_gemini.to_dict(orient="records")
    gpt_rows = df_gpt.to_dict(orient="records")

    def _group_by_material(rows):
        groups: Dict[str, List[Dict]] = defaultdict(list)
        for row in rows:
            groups[_norm(row.get("material_name", ""))].append(row)
        return groups

    gem_by_mat = _group_by_material(gem_rows)
    gpt_by_mat = _group_by_material(gpt_rows)

    def _best_material_match(mat_key, other_groups):
        if mat_key in other_groups:
            return mat_key
        best_key, best_score = None, 0.0
        for other_key in other_groups:
            score = _token_overlap(mat_key, other_key)
            if score > best_score:
                best_score, best_key = score, other_key
        if best_score >= 0.25:
            return best_key
        best_key, best_score = None, 0.0
        for other_key in other_groups:
            score = _embed_similarity(mat_key, other_key)
            if score > best_score:
                best_score, best_key = score, other_key
        return best_key if best_score >= 0.70 else None

    confirmed:     List[Dict] = []
    llm_needed:    List[Tuple] = []
    main_resolved: List[Dict] = []

    # Direction A: Gemini → GPT
    for gem_mat_key, gem_mat_rows in gem_by_mat.items():
        matched_gpt_mat_key = _best_material_match(gem_mat_key, gpt_by_mat)
        if matched_gpt_mat_key is None:
            log.info(f"Material '{gem_mat_key}' found only in Gemini — skipping")
            continue
        gpt_pool = gpt_by_mat[matched_gpt_mat_key]
        for gem_row in gem_mat_rows:
            candidate = _cascade_match(gem_row, gpt_pool)
            if candidate is None:
                continue
            layer = candidate.get("_match_layer", "")
            if layer == "llm_needed":
                llm_needed.append((len(confirmed), gem_row, candidate))
                confirmed.append(None)
            else:
                main_resolved.extend(_resolve_value_disagreement(gem_row, candidate, layer, tol))

    def _row_key(r: Dict) -> tuple:
        return (_norm(r.get("property_name", "")),
                _norm(r.get("material_name", "")),
                _norm(str(r.get("value", ""))))

    matched_keys: set = set()
    for r in main_resolved:
        matched_keys.add(_row_key(r))
    for r in confirmed:
        if r is not None:
            matched_keys.add(_row_key(r))

    # Direction B: GPT → Gemini
    for gpt_mat_key, gpt_mat_rows in gpt_by_mat.items():
        matched_gem_mat_key = _best_material_match(gpt_mat_key, gem_by_mat)
        if matched_gem_mat_key is None:
            continue
        gem_pool = gem_by_mat[matched_gem_mat_key]
        for gpt_row in gpt_mat_rows:
            if _row_key(gpt_row) in matched_keys:
                continue
            candidate = _cascade_match(gpt_row, gem_pool)
            if candidate is None:
                continue
            layer = candidate.get("_match_layer", "")
            if layer == "llm_needed":
                continue
            resolved = _resolve_value_disagreement(candidate, gpt_row, layer, tol)
            for r in resolved:
                if _row_key(r) not in matched_keys:
                    main_resolved.append(r)
                    matched_keys.add(_row_key(r))

    # LLM adjudication
    if llm_needed:
        pairs    = [(_key_str(g), _key_str(c)) for _, g, c in llm_needed]
        verdicts = _llm_adjudicate(pairs)
        for (slot_idx, gem_row, candidate), verdict in zip(llm_needed, verdicts):
            if verdict:
                resolved = _resolve_value_disagreement(gem_row, candidate, "llm", tol)
                confirmed[slot_idx] = resolved[0] if resolved else None
                if len(resolved) == 2:
                    main_resolved.append(resolved[1])
            else:
                confirmed[slot_idx] = None

    matched = [r for r in confirmed if r is not None] + main_resolved
    if not matched:
        log.warning("Consensus: no rows agreed between Gemini and GPT.")
        return pd.DataFrame()

    df = pd.DataFrame(matched)
    if "match_layer" not in df.columns:
        df["match_layer"] = ""
    layer_counts = df["match_layer"].value_counts().to_dict()
    log.info(f"Consensus: {len(df_gemini)} Gemini + {len(df_gpt)} GPT → {len(df)} agreed {layer_counts}")
    return df



def resolve_doi(doi_gemini: str, doi_gpt: str, doi_manual: str = "") -> str:
    for candidate in (doi_manual, doi_gemini, doi_gpt):
        d = _normalise_doi(candidate)
        if d:
            return d
    return ""

FIXED_COLS_CACHE = ["material_name", "material_abbreviation", "manufacturer", "doi"]

def _df_to_cache_dict(df: pd.DataFrame, doi: str = "") -> Dict:
    if df.empty:
        return {}
    row0 = df.iloc[0]
    return {
        "material_name":         str(row0.get("material_name", "")),
        "material_abbreviation": str(row0.get("material_abbreviation", "")),
        "manufacturer":          str(row0.get("manufacturer", "")),
        "doi":                   doi,
        "mechanical_properties": df.drop(
            columns=[c for c in FIXED_COLS if c in df.columns], errors="ignore"
        ).to_dict(orient="records"),
    }

# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE TEXT VERIFICATION
# Matrix approach: BERTScore + soft cosine + exact numeric + adjacent sentence
# ─────────────────────────────────────────────────────────────────────────────

def _clean_to_float(s: str) -> Optional[float]:
    """
    Remove all non-numeric characters except decimal point,
    then parse as float. Works for any formatting variant.
    """
    # Keep only digits and dot
    cleaned = re.sub(r"[^\d.]", "", s.strip())
    # Handle multiple dots (take first valid number)
    parts = cleaned.split(".")
    if len(parts) > 2:
        cleaned = parts[0] + "." + "".join(parts[1:])
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def _extract_all_floats(text: str) -> set:
    """Extract every number from text as a normalized float."""
    # Find all token-like number patterns first, then clean each
    tokens = re.findall(r"[\d][\d\s\.,]*[\d]|[\d]", text)
    result = set()
    for token in tokens:
        val = _clean_to_float(token)
        if val is not None:
            result.add(val)
    return result


def _numeric_found_in_text(num_str: str, text: str) -> bool:
    target = _clean_to_float(num_str)
    if target is None:
        return False
    candidates = _extract_all_floats(text)
    return target in candidates



def _verify_extractions(df: pd.DataFrame, pdf_hash: str) -> pd.DataFrame:
    """
    For each extracted row, verify it is grounded in the source PDF.
    Uses: ChromaDB semantic search -> sentence splitting -> similarity matrix
    (BERTScore + soft cosine) -> exact numeric match in best or adjacent sentence.
    Adds a 'source_verified' boolean column.
    """
    if df.empty or not pdf_hash:
        if not df.empty:
            df = df.copy()
            df["source_verified"] = False
        return df

    df = df.copy()
    verified = []

    for _, row in df.iterrows():
        prop  = str(row.get("property_name", "") or "").strip()
        mat   = str(row.get("material_name",  "") or "").strip()
        val   = str(row.get("value",          "") or "").strip()
        ctype = str(row.get("chunk_type",     "") or "").strip().lower()

        # Strip qualifier symbols for numeric check
        val_clean = re.sub(r"^[~><=\u2248\u2265\u2264\u00b1\s]+", "", val).strip()
        num_m = re.search(r"[\d.]+", val_clean)
        if not num_m:
            verified.append(False)
            continue
        num_str = num_m.group()

        # Query ChromaDB for relevant chunks
        query   = f"{prop} {mat}".strip()
        top_k   = 8 if ctype == "table" else 6
        chunks  = _chroma_semantic_verify(query, pdf_hash, top_k=top_k)

        if not chunks:
            verified.append(False)
            continue

        # Split chunks into sentences, build flat sentence list
        # Check numeric value across ALL retrieved chunks
        numeric_found = any(_numeric_found_in_text(num_str, ch.get("text", "")) for ch in chunks)

        # Semantic score — ChromaDB similarity already computed by SciBERT
        best_score = max(ch.get("similarity", 0.0) for ch in chunks)

        SCORE_THRESHOLD = 0.35
        if ctype == "table":
            passed = numeric_found
        else:
            passed = best_score >= SCORE_THRESHOLD and numeric_found

        verified.append(passed)

        if not passed:
            log.info(
                f"Verification failed — prop='{prop}' val='{num_str}' "
                f"best_score={best_score:.3f} numeric_found={numeric_found}"
            )

    df["source_verified"] = verified
    n_pass = sum(verified)
    log.info(f"Source verification: {n_pass}/{len(df)} rows verified for pdf_hash={pdf_hash}")
    return df

def run_pipeline(
    pdf_bytes:         bytes,
    doi_override:      str = "",
    progress_callback: Any = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Chunk], List[str], Dict]:
    def _prog(msg: str, pct: float):
        log.info(f"[{pct*100:.0f}%] {msg}")
        if progress_callback:
            progress_callback(msg, pct)

    meta: Dict[str, Any] = {}
    pdf_hash = _pdf_hash(pdf_bytes)

    _prog("Checking cache…", 0.0)
    cached = cache_get(pdf_bytes)
    if cached:
        _prog("Cache hit.", 1.0)
        meta["path"] = "cache"
        df, doi = merge_to_dataframe([cached])
        doi = resolve_doi(doi, "", doi_override)
        df["doi_url"] = _doi_url(doi)
        return df, df.copy(), df.copy(), [], [], meta

    _prog("Stage 1 — extracting tables + text…", 0.05)
    all_chunks = extract_all_chunks(pdf_bytes)
    meta["chunks_total"]  = len(all_chunks)
    meta["chunks_tables"] = sum(1 for c in all_chunks if c.chunk_type == "table")
    meta["chunks_text"]   = sum(1 for c in all_chunks if c.chunk_type == "text")

    if not all_chunks:
        _prog("No content extracted.", 1.0)
        meta["path"] = "failed"
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], ["No content extracted."], meta

    _prog(f"Stage 1 done — {meta['chunks_tables']} tables, {meta['chunks_text']} text.", 0.20)

    _prog("Stage 2 — indexing into ChromaDB…", 0.25)
    try:
        index_chunks_in_chroma(all_chunks, pdf_hash)
    except Exception as e:
        log.error(f"ChromaDB indexing failed: {e}")

    _prog("Stage 3 — ordering all chunks by page…", 0.40)
    # No ranking — send all chunks ordered by page number
    ordered_chunks = sorted(all_chunks, key=lambda c: c.page_num)
    for c in ordered_chunks:
        c.relevant = True
    meta["chunks_ranked"] = len(ordered_chunks)

    _prog("Stage 4 — building batches…", 0.55)
    batches = build_batches(ordered_chunks)
    meta["batches"] = len(batches)

    _prog("Stage 5 — Gemini + GPT running in parallel…", 0.60)
    gemini_raw, gpt_raw, api_errors = run_dual_llm_parallel(batches, progress_callback)

    if not gemini_raw and not gpt_raw:
        meta["path"] = "failed"
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), all_chunks, api_errors, meta

    _prog("Stage 6 — merging results…", 0.90)
    df_gemini, doi_gemini = merge_to_dataframe(gemini_raw)
    df_gpt,    doi_gpt    = merge_to_dataframe(gpt_raw)
    doi     = resolve_doi(doi_gemini, doi_gpt, doi_override)
    doi_url = _doi_url(doi)
    meta["gemini_properties"] = len(df_gemini)
    meta["gpt_properties"]    = len(df_gpt)
    meta["doi"]               = doi

    _prog("Stage 7 — source text verification…", 0.90)
    df_gemini = _verify_extractions(df_gemini, pdf_hash)
    df_gpt    = _verify_extractions(df_gpt,    pdf_hash)
    meta["gemini_verified"] = int(df_gemini.get("source_verified", pd.Series(dtype=bool)).sum()) if not df_gemini.empty else 0
    meta["gpt_verified"]    = int(df_gpt.get("source_verified",    pd.Series(dtype=bool)).sum()) if not df_gpt.empty else 0

    global _current_pdf_hash
    _current_pdf_hash = pdf_hash

    _prog("Stage 8 — consensus filtering…", 0.95)
    df_consensus = consensus_filter(df_gemini, df_gpt)
    meta["consensus_properties"] = len(df_consensus)
    meta["path"] = "dual-llm-consensus"

    for df in (df_gemini, df_gpt, df_consensus):
        if not df.empty:
            df["doi_url"] = doi_url

    if not df_consensus.empty:
        cache_set(pdf_bytes, _df_to_cache_dict(df_consensus, doi))

    _prog(f"Done — {len(df_consensus)} consensus, {len(df_gemini)} Gemini, {len(df_gpt)} GPT.", 1.0)
    return df_consensus, df_gemini, df_gpt, all_chunks, api_errors, meta


# =============================================================================
# EVALUATOR EXTENSION
# Precision / Recall / F1 scoring against an uploaded ground truth file.
# =============================================================================

import io as _io
import re as _re
import pandas as _pd
from difflib import SequenceMatcher

# ─────────────────────────────────────────────────────────────────────────────
# UNIT NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────








# ─────────────────────────────────────────────────────────────────────────────
# VALUE / RANGE PARSING
# ─────────────────────────────────────────────────────────────────────────────






# ─────────────────────────────────────────────────────────────────────────────
# NAME NORMALISATION + SIMILARITY
# ─────────────────────────────────────────────────────────────────────────────

def _norm_name(text: str) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower().strip()
    text = _re.sub(r"[\s\-_/]+", " ", text)
    text = _re.sub(r"[^\w\s]", "", text)
    return _re.sub(r"\s+", " ", text).strip()


def _ev_token_set(text: str) -> set:
    return {t for t in _norm_name(text).split() if len(t) >= 2}



# ─────────────────────────────────────────────────────────────────────────────
# VALUE MATCHING
# ─────────────────────────────────────────────────────────────────────────────






# ─────────────────────────────────────────────────────────────────────────────
# CASCADE MATCH SCORE
# ─────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────
# COLUMN DETECTION
# ─────────────────────────────────────────────────────────────────────────────

_NAME_HINTS = ["property_name","property","prop_name","name","parameter","metric","variable"]
_VAL_HINTS  = ["value","val","measured_value","result","amount","number"]
_UNIT_HINTS = ["unit","units","si_unit","uom","measure"]


def _find_col(df: _pd.DataFrame, hints: list):
    cl = {str(c).lower().strip(): c for c in df.columns}
    for h in hints:
        if h in cl:
            return cl[h]
    for h in hints:
        for k, v in cl.items():
            if h in k:
                return v
    return None




# ─────────────────────────────────────────────────────────────────────────────
# CORE SCORER — bidirectional matching
# ─────────────────────────────────────────────────────────────────────────────





def _ev_score(
    model_df: _pd.DataFrame,
    gt_df:    _pd.DataFrame,
    nc: str, vc: str, uc: str,
    gnc: str, gvc: str, guc: str,
    nmc: Optional[str] = None,
    gnmc: Optional[str] = None,
    min_conf: int = 1,
):
    """
    GT vs model matching using LLM as judge.
    LLM directly returns confirmed matched pairs as a list.
    Fallback to greedy token overlap if LLM fails.
    """

    # ── filter GT by confidence ──────────────────────────────────────────
    gt_rows = []
    for gt_idx, r in gt_df.iterrows():
        try:
            conf = int(r.get("confidence", 3))
        except Exception:
            conf = 3
        if conf >= min_conf:
            row_dict = r.to_dict()
            gt_name  = str(row_dict.get(gnc, "") or "").strip()
            gt_value = str(row_dict.get(gvc, "") or "").strip()
            if gt_name and gt_value:
                gt_rows.append((gt_idx, row_dict))

    # ── filter model rows ────────────────────────────────────────────────
    model_rows = []
    for m_idx, r in model_df.iterrows():
        row_dict   = r.to_dict()
        model_name = str(row_dict.get(nc, "") or "").strip()
        model_val  = str(row_dict.get(vc, "") or "").strip()
        if model_name and model_val:
            model_rows.append((m_idx, row_dict))

    empty_met = {
        "TP": 0, "FP": len(model_rows), "FN": len(gt_rows),
        "Precision": 0.0, "Recall": 0.0, "F1": 0.0,
        "GT rows (filtered)": len(gt_rows),
        "Model rows": len(model_rows),
    }
    if not gt_rows or not model_rows:
        ann = model_df.copy()
        ann["eval_result"]         = "FP"
        ann["matched_gt_property"] = ""
        ann["matched_gt_value"]    = ""
        ann["matched_gt_unit"]     = ""
        ann["match_score"]         = 0.0
        ann["match_layer"]         = ""
        return ann, empty_met

    # ── Build prompt for LLM as judge ────────────────────────────────────
    gt_lines = []
    for i, (_, r) in enumerate(gt_rows):
        gt_lines.append(
            f"GT{i}: property=\"{r.get(gnc,'')}\", value=\"{r.get(gvc,'')}\", "
            f"unit=\"{r.get(guc,'')}\", material=\"{r.get(gnmc or '','')}\""
        )

    model_lines = []
    for j, (_, r) in enumerate(model_rows):
        model_lines.append(
            f"M{j}: property=\"{r.get(nc,'')}\", value=\"{r.get(vc,'')}\", "
            f"unit=\"{r.get(uc,'')}\", material=\"{r.get(nmc or '','')}\""
        )

    prompt = (
        "You are a materials science expert evaluating an extraction system.\n"
        "Below are Ground Truth (GT) rows and Model (M) rows.\n"
        "For each GT row, identify the single best matching Model row if one exists.\n"
        "A match means: same property concept (even if named differently — abbreviations, "
        "synonyms, Greek letters, corrupted text are fine), same numeric value within 10%, "
        "same unit family, same material.\n"
        "Only include a pair if you are fully confident it is the same measurement. "
        "If you are not sure, leave it out.\n"
        "Return ONLY a JSON array. Omit unmatched GT rows entirely. No explanation, no markdown:\n"
        "[{\"gt\": 0, \"model\": 3}, {\"gt\": 1, \"model\": 7}]\n\n"
        "GROUND TRUTH:\n" + "\n".join(gt_lines) + "\n\n"
        "MODEL:\n" + "\n".join(model_lines)
    )

    # ── Call LLM ─────────────────────────────────────────────────────────
    matched_pairs: dict = {}
    matched_gt:    set  = set()

    llm_success = False
    try:
        import requests as _req
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        body = {
            "model": GPT_MODEL,
            "temperature": 0,
            "max_tokens": max(len(gt_rows) * 20 + 500, 2000),
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = _req.post(GPT_API_URL, headers=headers, json=body, timeout=120)
        if resp.ok:
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            pairs_list = json.loads(raw)
            if isinstance(pairs_list, list):
                for pair in pairs_list:
                    gt_i  = int(pair["gt"])
                    m_j   = int(pair["model"])
                    if gt_i not in matched_gt:
                        gt_idx = gt_rows[gt_i][0]
                        m_idx  = model_rows[m_j][0]
                        matched_pairs[m_idx] = (gt_idx, 1.0)
                        matched_gt.add(gt_i)
                llm_success = True
    except Exception as e:
        log.warning(f"LLM judge call failed: {e}")

    # ── Fallback: greedy token overlap if LLM fails ───────────────────────
    if not llm_success:
        log.warning("LLM judge fallback to token overlap")
        used_model: set = set()
        for gt_i, (gt_idx, gr) in enumerate(gt_rows):
            gt_name = _norm_name(str(gr.get(gnc, "") or ""))
            best_score, best_m_idx = 0.0, None
            for m_idx, mr in model_rows:
                if m_idx in used_model:
                    continue
                m_name = _norm_name(str(mr.get(nc, "") or ""))
                ta = {t for t in gt_name.split() if len(t) >= 2}
                tb = {t for t in m_name.split() if len(t) >= 2}
                sc = len(ta & tb) / len(ta | tb) if (ta or tb) else 0.0
                if sc > best_score:
                    best_score, best_m_idx = sc, m_idx
            if best_m_idx is not None and best_score > 0:
                matched_pairs[best_m_idx] = (gt_idx, best_score)
                matched_gt.add(gt_i)
                used_model.add(best_m_idx)

    # ── Annotate model dataframe ──────────────────────────────────────────
    ann = model_df.copy()
    eval_results, matched_names, matched_vals, matched_units, scores, layers = \
        [], [], [], [], [], []

    gt_lookup = {idx: row for idx, row in gt_rows}

    for idx, _ in ann.iterrows():
        if idx in matched_pairs:
            gt_idx, sc = matched_pairs[idx]
            gt_r = gt_lookup.get(gt_idx, {})
            eval_results.append("TP")
            matched_names.append(gt_r.get(gnc, ""))
            matched_vals.append(gt_r.get(gvc, ""))
            matched_units.append(gt_r.get(guc, ""))
            scores.append(round(sc, 4))
            layers.append("llm_judge")
        else:
            eval_results.append("FP")
            matched_names.append("")
            matched_vals.append("")
            matched_units.append("")
            scores.append(0.0)
            layers.append("")

    ann["eval_result"]         = eval_results
    ann["matched_gt_property"] = matched_names
    ann["matched_gt_value"]    = matched_vals
    ann["matched_gt_unit"]     = matched_units
    ann["match_score"]         = scores
    ann["match_layer"]         = layers

    tp = int((ann["eval_result"] == "TP").sum())
    fp = int((ann["eval_result"] == "FP").sum())
    fn = len(gt_rows) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)

    metrics = {
        "TP": tp, "FP": fp, "FN": fn,
        "Precision": round(precision, 4),
        "Recall":    round(recall,    4),
        "F1":        round(f1,        4),
        "GT rows (filtered)": len(gt_rows),
        "Model rows": len(model_df),
    }
    return ann, metrics



def _ev_style(df):
    return df.style.apply(
        lambda col: col.map(
            {"TP": "background-color:#d4edda", "FP": "background-color:#f8d7da"}
        ) if col.name == "eval_result" else col.map(lambda _: ""),
        axis=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT EVALUATION TAB
# ─────────────────────────────────────────────────────────────────────────────

def render_evaluate_tab(df_consensus, df_gemini, df_gpt):
    import streamlit as st

    st.subheader("📐 Precision / Recall Evaluation")
    st.caption(
        "Upload a ground truth CSV or Excel.  \n"
        "**Matching:** LLM as judge · direct pair matching · "
        "fallback to greedy token overlap if LLM fails."
    )

    gt_file = st.file_uploader(
        "Ground Truth file (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        key="ev_gt_upload_cascade",
    )

    if gt_file is None:
        st.info(
            "Upload a ground truth file.  \n"
            "Expected columns: `material_name`, `property_name`, `value`, `unit`.  \n"
            "Optional: `confidence` (1/2/3) to filter by annotation confidence."
        )
        return

    try:
        gt_df = (
            _pd.read_csv(gt_file)
            if gt_file.name.lower().endswith(".csv")
            else _pd.read_excel(gt_file)
        )
    except Exception as e:
        st.error(f"Could not read ground truth file: {e}")
        return

    st.success(f"Ground truth loaded — {len(gt_df)} rows.")

    gt_cols = list(gt_df.columns)

    def _find(hints):
        cl = {str(c).lower().strip(): c for c in gt_df.columns}
        for h in hints:
            if h in cl: return cl[h]
        for h in hints:
            for k, v in cl.items():
                if h in k: return v
        return gt_cols[0]

    gnc  = _find(["property_name","property","prop_name","name","parameter"])
    gvc  = _find(["value","val","measured_value","result","amount"])
    guc  = _find(["unit","units","si_unit","uom","measure"])
    gnmc = _find(["material_name","material","mat_name","sample"])

    st.markdown(f"**Auto-detected GT columns:** property=`{gnc}` · "
                f"value=`{gvc}` · unit=`{guc}` · material=`{gnmc}`")

    has_conf = any(str(c).lower() == "confidence" for c in gt_df.columns)
    min_conf = 1
    if has_conf:
        conf_col = next(c for c in gt_df.columns if str(c).lower() == "confidence")
        gt_df    = gt_df.rename(columns={conf_col: "confidence"})
        min_conf = st.select_slider(
            "Minimum confidence to include",
            options=[1, 2, 3], value=2,
            format_func=lambda x: {
                1: "🔴 1 — ALL",
                2: "🟡 2 — Medium+High",
                3: "🟢 3 — High only",
            }[x],
            key="ev_minconf_cascade",
        )
        try:
            included = int((gt_df["confidence"].astype(int) >= min_conf).sum())
        except Exception:
            included = len(gt_df)
        st.caption(f"Using **{included}** / {len(gt_df)} GT rows.")

    with st.expander("Preview Ground Truth", expanded=False):
        st.dataframe(gt_df.head(20), use_container_width=True, hide_index=True)

    st.divider()

    if st.button("📐 Run Evaluation", type="primary",
                 use_container_width=True,
                 key="run_evaluation_cascade_unique_button"):

        nc  = "property_name"
        vc  = "value"
        uc  = "unit"
        nmc = "material_name"

        def _safe_score(df):
            try:
                gt_filtered = int((gt_df["confidence"].astype(int) >= min_conf).sum())
            except Exception:
                gt_filtered = len(gt_df)
            empty_met = {
                "TP": 0, "FP": 0, "FN": gt_filtered,
                "Precision": 0.0, "Recall": 0.0, "F1": 0.0,
                "GT rows (filtered)": gt_filtered, "Model rows": 0,
            }
            if df is None or df.empty:
                return _pd.DataFrame(), empty_met
            if nc not in df.columns or vc not in df.columns:
                return _pd.DataFrame(), empty_met
            return _ev_score(
                df, gt_df,
                nc, vc, uc if uc in df.columns else "",
                gnc, gvc, guc,
                nmc if nmc in df.columns else None,
                gnmc if gnmc else None,
                min_conf,
            )

        with st.spinner("Running evaluation (LLM as judge)…"):
            ann_con, met_con = _safe_score(df_consensus)
            ann_gem, met_gem = _safe_score(df_gemini)
            ann_gpt, met_gpt = _safe_score(df_gpt)

        st.divider()
        st.subheader("📊 Results")

        rc1, rc2, rc3 = st.columns(3)
        for col, label, met in [
            (rc1, "✅ Consensus", met_con),
            (rc2, "🟦 Gemini",   met_gem),
            (rc3, "🟩 GPT",      met_gpt),
        ]:
            with col:
                st.markdown(f"### {label}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Precision", f"{met['Precision']:.1%}")
                m2.metric("Recall",    f"{met['Recall']:.1%}")
                m3.metric("F1",        f"{met['F1']:.1%}")
                st.caption(
                    f"TP:{met['TP']} | FP:{met['FP']} | FN:{met['FN']} | "
                    f"GT:{met['GT rows (filtered)']}"
                )

        st.divider()
        t_gt, t_con, t_gem, t_gpt, t_ver = st.tabs([
            "Ground Truth",
            "✅ Consensus Scored",
            "🟦 Gemini Scored",
            "🟩 GPT Scored",
            "🔍 Verified Scored",

        ])
        with t_gt:
            st.dataframe(gt_df, use_container_width=True, hide_index=True)
        with t_con:
            st.dataframe(
                _ev_style(ann_con) if not ann_con.empty else ann_con,
                use_container_width=True, hide_index=True)
        with t_gem:
            st.dataframe(
                _ev_style(ann_gem) if not ann_gem.empty else ann_gem,
                use_container_width=True, hide_index=True)
        with t_gpt:
            st.dataframe(
                _ev_style(ann_gpt) if not ann_gpt.empty else ann_gpt,
                use_container_width=True, hide_index=True)
        with t_ver:
            df_gem_ver = df_gemini[df_gemini["source_verified"] == True] if "source_verified" in df_gemini.columns else df_gemini
            df_gpt_ver = df_gpt[df_gpt["source_verified"] == True] if "source_verified" in df_gpt.columns else df_gpt
            ann_gem_ver, met_gem_ver = _safe_score(df_gem_ver)
            ann_gpt_ver, met_gpt_ver = _safe_score(df_gpt_ver)
            rv1, rv2 = st.columns(2)
            for col, label, met in [
                (rv1, "🟦 Gemini Verified", met_gem_ver),
                (rv2, "🟩 GPT Verified",    met_gpt_ver),
            ]:
                with col:
                    st.markdown(f"### {label}")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Precision", f"{met['Precision']:.1%}")
                    m2.metric("Recall",    f"{met['Recall']:.1%}")
                    m3.metric("F1",        f"{met['F1']:.1%}")
                    st.caption(
                        f"TP:{met['TP']} | FP:{met['FP']} | FN:{met['FN']} | "
                        f"GT:{met['GT rows (filtered)']}"
                    )
            st.divider()
            tv1, tv2 = st.tabs(["🟦 Gemini Verified Scored", "🟩 GPT Verified Scored"])
            with tv1:
                st.dataframe(_ev_style(ann_gem_ver) if not ann_gem_ver.empty else ann_gem_ver,
                            use_container_width=True, hide_index=True)
            with tv2:
                st.dataframe(_ev_style(ann_gpt_ver) if not ann_gpt_ver.empty else ann_gpt_ver,
                            use_container_width=True, hide_index=True)
        st.divider()
        xlsx = _ev_excel(
            gt_df,
            "Consensus", ann_con if not ann_con.empty else _pd.DataFrame(), met_con,
            "Gemini",    ann_gem if not ann_gem.empty else _pd.DataFrame(), met_gem,
            "GPT",       ann_gpt if not ann_gpt.empty else _pd.DataFrame(), met_gpt,
        )
        st.download_button(
            "⬇️ Download Scored Excel",
            data=xlsx,
            file_name="evaluation_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="download_evaluation_results_cascade",
        )


def _ev_excel(gt_df, la, aa, ma, lb, ab, mb, lc, ac, mc):
    buf = _io.BytesIO()
    with _pd.ExcelWriter(buf, engine="openpyxl") as w:
        _pd.DataFrame(
            {"Metric": list(ma.keys()),
             la: list(ma.values()),
             lb: list(mb.values()),
             lc: list(mc.values())}
        ).to_excel(w, sheet_name="Summary", index=False)
        gt_df.to_excel(w, sheet_name="Ground Truth", index=False)
        aa.to_excel(w, sheet_name=f"{la[:28]} Scored", index=False)
        ab.to_excel(w, sheet_name=f"{lb[:28]} Scored", index=False)
        ac.to_excel(w, sheet_name=f"{lc[:28]} Scored", index=False)
    return buf.getvalue()

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

def _run_streamlit():
    import streamlit as st

    st.set_page_config(
        page_title="DocToDB — Dual LLM Consensus Extractor",
        page_icon="🧬",
        layout="wide",
    )
    st.title("🧬 DocToDB — Dual LLM Consensus Extractor")

    with st.sidebar:
        st.header("⚙️ Settings")
        st.divider()

        @st.cache_data(show_spinner=False)
        def _cached_gemini_models(key: str):
            return _fetch_gemini_models_live(key)

        @st.cache_data(show_spinner=False)
        def _cached_openai_models(key: str):
            return _fetch_openai_models_live(key)

        if GEMINI_API_KEY:
            gemini_model_data = _cached_gemini_models(GEMINI_API_KEY)
            st.caption(f"✅ {len(gemini_model_data)} Gemini models available")
        else:
            gemini_model_data = _GEMINI_FALLBACK_MODELS
            st.caption("⚠️ Enter GEMINI_API_KEY to load live model list")

        gemini_names  = [m[0] for m in gemini_model_data]
        gemini_limits = {m[0]: m[1] for m in gemini_model_data}
        default_g_idx = next((i for i, n in enumerate(gemini_names) if n == GEMINI_MODEL), 0)
        selected_gemini = st.selectbox(
            "Gemini model", gemini_names, index=default_g_idx,
            help="Models sorted by GEMINI_PREFERRED_MODELS preference list."
        )
        g_token_limit = gemini_limits.get(selected_gemini, 1_000_000)
        g_batch_chars = min(int(g_token_limit * 3 * 0.70), 500_000)
        st.info(f"📐 **{selected_gemini}**  \nContext: {g_token_limit:,} tokens  \n"
                f"Batch size: **{g_batch_chars:,} chars** (70 %)")

        st.divider()

        if OPENAI_API_KEY:
            openai_model_data = _cached_openai_models(OPENAI_API_KEY)
            st.caption(f"✅ {len(openai_model_data)} OpenAI models available")
        else:
            openai_model_data = _OPENAI_FALLBACK_MODELS
            st.caption("⚠️ Enter OPENAI_API_KEY to load live model list")

        openai_names  = [m[0] for m in openai_model_data]
        openai_limits = {m[0]: m[1] for m in openai_model_data}
        default_o_idx = next((i for i, n in enumerate(openai_names) if n == GPT_MODEL), 0)
        selected_gpt  = st.selectbox(
            "GPT model", openai_names, index=default_o_idx,
            help="Models sorted by OPENAI_PREFERRED_MODELS preference list."
        )
        o_token_limit = openai_limits.get(selected_gpt, 128_000)
        o_batch_chars = min(int(o_token_limit * 3 * 0.50), 450_000)
        st.info(f"📐 **{selected_gpt}**  \nContext: {o_token_limit:,} tokens  \n"
                f"Batch size: **{o_batch_chars:,} chars** (50 %)")

        st.divider()
        st.markdown(f"**Embedder:** `{EMBED_MODEL_NAME}`")
        st.markdown(f"**ChromaDB:** {'✅' if CHROMA_AVAILABLE else '❌ not installed'}")
        st.markdown(f"**Docling:**  {'✅' if DOCLING_AVAILABLE else '❌'}")
        st.markdown(f"**Camelot:**  {'✅' if CAMELOT_AVAILABLE else '❌'}")
        st.markdown(f"**OCR:**      {'✅' if OCR_AVAILABLE else '❌'}")
        st.divider()
        st.markdown(f"**Consensus tolerance:** ±{CONSENSUS_VALUE_TOL*100:.0f}%")
        st.divider()

        doi_manual = st.text_input(
            "DOI override (optional)",
            placeholder="10.1016/j.mat.2023.01.001",
        )
        st.divider()
        if st.button("🗑 Clear JSON Cache"):
            if os.path.exists(CACHE_FILE):
                os.remove(CACHE_FILE)
                st.success("JSON cache cleared.")
        if st.button("🗑 Clear ChromaDB Collection"):
            try:
                col = _get_chroma_collection()
                col.delete(where={"pdf_hash": {"$ne": ""}})
                st.success("ChromaDB cleared.")
            except Exception as e:
                st.error(f"ChromaDB clear failed: {e}")

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if not uploaded:
        st.info("Upload a PDF to get started.")
        return

    pdf_bytes = uploaded.getvalue()
    stem      = uploaded.name.rsplit(".", 1)[0]

    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY environment variable not set.")
        return
    if not OPENAI_API_KEY:
        st.warning("OPENAI_API_KEY not set — GPT will be skipped; no consensus possible.")

    if st.button("🚀 Run Extraction", type="primary", use_container_width=True):
        global GEMINI_MODEL, GEMINI_API_URL, GPT_MODEL
        GEMINI_MODEL   = selected_gemini
        GEMINI_API_URL = _rebuild_gemini_url(selected_gemini, GEMINI_API_KEY)
        GPT_MODEL      = selected_gpt
        with _config_lock:
            _GEMINI_CONFIG["max_input_chars"] = g_batch_chars
            _GPT_CONFIG["max_input_chars"]    = o_batch_chars
        threading.Thread(target=_probe_gemini_limits, args=(selected_gemini,), daemon=True).start()

        bar    = st.progress(0.0)
        status = st.empty()

        def cb(msg, pct):
            bar.progress(min(pct, 1.0))
            status.text(msg)

        with st.spinner("Running dual-LLM pipeline…"):
            df_consensus, df_gemini, df_gpt, chunks, errors, meta = run_pipeline(
                pdf_bytes, doi_override=doi_manual, progress_callback=cb
            )

        bar.progress(1.0)
        status.empty()

        st.session_state["df_consensus"] = df_consensus
        st.session_state["df_gemini"]    = df_gemini
        st.session_state["df_gpt"]       = df_gpt
        st.session_state["chunks"]       = chunks
        st.session_state["errors"]       = errors
        st.session_state["meta"]         = meta
        st.session_state["stem"]         = stem
        st.session_state["extraction_done"] = True

    if not st.session_state.get("extraction_done", False):
        st.info("Click **🚀 Run Extraction** to start.")
        return
    df_consensus = st.session_state["df_consensus"]
    df_gemini    = st.session_state["df_gemini"]
    df_gpt       = st.session_state["df_gpt"]
    chunks       = st.session_state["chunks"]
    errors       = st.session_state["errors"]
    meta         = st.session_state["meta"]
    stem         = st.session_state["stem"]

    doi     = meta.get("doi", "")
    doi_url = _doi_url(doi)
    if doi_url:
        st.markdown(f"**DOI:** [{doi}]({doi_url})")
    else:
        st.info("No DOI found in this PDF.")

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Tables",       meta.get("chunks_tables", "—"))
    c2.metric("Text blocks",  meta.get("chunks_text", "—"))
    c3.metric("Chunks sent",  meta.get("chunks_ranked", "—"))
    c4.metric("Batches",      meta.get("batches", "—"))
    c5.metric("🟦 Gemini",   meta.get("gemini_properties", len(df_gemini)))
    c6.metric("🟩 GPT",      meta.get("gpt_properties", len(df_gpt)))
    c7.metric("✅ Consensus", meta.get("consensus_properties", len(df_consensus)))

    if errors:
        with st.expander(f"⚠️ {len(errors)} API error(s)"):
            for e in errors:
                st.code(e)

    def _show_df(df: pd.DataFrame):
        if df.empty:
            st.info("No data.")
            return
        display = df.drop(columns=["doi_url"], errors="ignore")
        doi_url_val = df["doi_url"].iloc[0] if "doi_url" in df.columns and not df.empty else ""
        if doi_url_val:
            doi_clean = _normalise_doi(doi_url_val.replace("https://doi.org/", ""))
            st.markdown(f"🔗 **Paper DOI:** [{doi_clean}]({doi_url_val})")
        st.dataframe(display, use_container_width=True, hide_index=True)

    tab_consensus, tab_gemini, tab_gpt, tab_verified,tab_pages, tab_export, tab_evaluate = st.tabs([
    f"✅ Consensus ({len(df_consensus)})",
    f"🟦 Gemini ({len(df_gemini)})",
    f"🟩 GPT ({len(df_gpt)})",
    "🔎 Verified",
    "🗂 By Page",
    "📤 Export",
    "📐 Evaluate",
    ])

    with tab_consensus:
        st.caption(f"**{len(df_consensus)} properties** agreed by both LLMs "
                f"(fuzzy match ±{CONSENSUS_VALUE_TOL*100:.0f}%)")
        if df_consensus.empty:
            st.warning("No consensus rows found.")
        else:
            t_tbl, t_txt, t_all = st.tabs(["📋 Tables", "📄 Text", "🔢 All"])
            with t_tbl:
                mask = df_consensus.get("chunk_type", pd.Series(dtype=str)) == "table"
                _show_df(df_consensus[mask] if "chunk_type" in df_consensus.columns else pd.DataFrame())
            with t_txt:
                mask = df_consensus.get("chunk_type", pd.Series(dtype=str)) == "text"
                _show_df(df_consensus[mask] if "chunk_type" in df_consensus.columns else pd.DataFrame())
            with t_all:
                _show_df(df_consensus)

    with tab_gemini:
        st.caption(f"{len(df_gemini)} properties extracted by Gemini ({meta.get('gemini_model', GEMINI_MODEL)})")
        _show_df(df_gemini)

    with tab_gpt:
        st.caption(f"{len(df_gpt)} properties extracted by GPT ({meta.get('gpt_model', GPT_MODEL)})")
        _show_df(df_gpt)

    with tab_verified:
        st.caption("Rows where source text verification passed in either LLM output.")
        gem_verified = df_gemini[df_gemini.get("source_verified", pd.Series(dtype=bool)) == True].copy() if not df_gemini.empty and "source_verified" in df_gemini.columns else pd.DataFrame()
        gpt_verified = df_gpt[df_gpt.get("source_verified", pd.Series(dtype=bool)) == True].copy() if not df_gpt.empty and "source_verified" in df_gpt.columns else pd.DataFrame()
        if not gem_verified.empty:
            gem_verified["source"] = "Gemini"
        if not gpt_verified.empty:
            gpt_verified["source"] = "GPT"
        combined = pd.concat([gem_verified, gpt_verified], ignore_index=True)
        if combined.empty:
            st.warning("No source-verified rows found.")
        else:
            st.caption(f"{len(gem_verified)} Gemini + {len(gpt_verified)} GPT = {len(combined)} verified rows total.")
            _show_df(combined)

    with tab_pages:
        df_src = df_gemini if not df_gemini.empty else df_gpt
        if "source_page" in df_src.columns:
            def _page_sort_key(label: str) -> int:
                m = re.search(r"\d+", label)
                return int(m.group()) if m else 9999
            for pg in sorted(df_src["source_page"].unique(), key=_page_sort_key):
                pg_df = df_src[df_src["source_page"] == pg]
                tbl_n = int((pg_df.get("chunk_type", pd.Series(dtype=str)) == "table").sum())
                txt_n = int((pg_df.get("chunk_type", pd.Series(dtype=str)) == "text").sum())
                with st.expander(
                    f"📄 {pg} — {len(pg_df)} propert{'y' if len(pg_df)==1 else 'ies'} "
                    f"({tbl_n} table · {txt_n} text)", expanded=False
                ):
                    _show_df(pg_df)

    with tab_export:
        export_df = df_consensus if not df_consensus.empty else df_gemini
        export_df = export_df.drop(columns=["doi_url"], errors="ignore")
        col1, col2 = st.columns(2)
        col1.download_button("⬇️ Consensus CSV",
            export_df.to_csv(index=False).encode(),
            f"{stem}_consensus.csv", "text/csv", use_container_width=True)
        col2.download_button("⬇️ Consensus JSON",
            export_df.to_json(orient="records", indent=2).encode(),
            f"{stem}_consensus.json", "application/json", use_container_width=True)
        st.divider()
        col3, col4 = st.columns(2)
        col3.download_button("⬇️ Gemini CSV",
            df_gemini.drop(columns=["doi_url"], errors="ignore").to_csv(index=False).encode()
            if not df_gemini.empty else b"",
            f"{stem}_gemini.csv", "text/csv", use_container_width=True,
            disabled=df_gemini.empty)
        col4.download_button("⬇️ GPT CSV",
            df_gpt.drop(columns=["doi_url"], errors="ignore").to_csv(index=False).encode()
            if not df_gpt.empty else b"",
            f"{stem}_gpt.csv", "text/csv", use_container_width=True,
            disabled=df_gpt.empty)

    with tab_evaluate:
        render_evaluate_tab(df_consensus, df_gemini, df_gpt)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _in_streamlit = False
    try:
        import streamlit.runtime.scriptrunner as _sr
        if _sr.get_script_run_ctx() is not None:
            _in_streamlit = True
    except Exception:
        pass

    if _in_streamlit:
        _run_streamlit()
    else:
        print(
            "\nUsage:\n"
            "  streamlit run PDF_DataExtraction_updated.py\n"
            "\nEnvironment variables required:\n"
            "  GEMINI_API_KEY   — Google AI Studio key\n"
            "  OPENAI_API_KEY   — OpenAI key\n"
        )
