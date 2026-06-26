"""
DocToDB — Materials Science PDF Extractor (ChromaDB RAG Pipeline)
=================================================================
PRIMARY:  Native PDF → Gemini (1 call, base64)
FALLBACK: Docling extraction → ChromaDB vector store → single
          schema-derived semantic query → ALL non-boilerplate chunks
          ranked by relevance → batched Gemini calls

No top_k limit. No similarity cutoff. Every non-boilerplate chunk
is sent to Gemini, ranked so highest relevance fills first batch.
Batching is governed only by Gemini's 80k char input limit.

Rate limit handling:
  - Exponential backoff on 429 (30s → 60s → 90s)
  - 4s delay between fallback batch calls
  - ChromaDB persistence (no re-embedding for same PDF)
  - JSON cache (no re-calling Gemini for same PDF)
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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import fitz          # pymupdf
import numpy as np
import pandas as pd
import pdfplumber
import requests
from pathlib import Path
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

API_KEY = os.getenv("GEMINI_API_KEY", "")

_cached_model: Optional[str] = None
 

def get_available_model(api_key: str) -> str:
    """Fetch available models and return best flash model found."""
    global _cached_model
    if _cached_model:
        return _cached_model
    try:
        resp = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
            timeout=10
        )
        if not resp.ok:
            _cached_model = "gemini-1.5-flash"
            return _cached_model
        models = resp.json().get("models", [])
        model_names = [m["name"].replace("models/", "") for m in models
                      if "generateContent" in m.get("supportedGenerationMethods", [])]
        preferred = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-preview",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
        ]
        for pref in preferred:
            for name in model_names:
                if name.startswith(pref):
                    log.info(f"Auto-selected model: {name}")
                    _cached_model = name
                    return _cached_model
        flash_models = [n for n in model_names if "flash" in n]
        _cached_model = flash_models[0] if flash_models else "gemini-1.5-flash"
        return _cached_model
    except Exception as e:
        log.warning(f"Model auto-detect failed: {e}")
        _cached_model = "gemini-1.5-flash"
        return _cached_model
 
GEMINI_MODEL = get_available_model(API_KEY)
API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/"
    f"models/{GEMINI_MODEL}:generateContent"
    f"?key={API_KEY}"
)

# ChromaDB
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
CHROMA_COLLECTION  = "doctodb_chunks"
EMBED_MODEL_NAME   = "all-MiniLM-L6-v2"
NEIGHBOR_OVERLAP   = 1           # adjacent chunks merged for cross-boundary context

# Batching


MAX_BATCH_CHARS  = 1_500_000   # splits 1078 chunks into ~2 batches, each fast enough
BATCH_CALL_DELAY = 65        # 65s between batches, safely under free tier RPM
NEIGHBOR_OVERLAP = 0   
RETRY_DELAYS     = [60, 120, 180]  # longer backoff if 429 hits
# Extraction
MIN_TABLE_ROWS  = 2
MIN_TABLE_COLS  = 2
MIN_CELL_CHARS  = 2
MIN_TEXT_CHARS  = 40
OCR_THRESHOLD   = 50

CACHE_FILE = "pdf_extraction_cache.json"

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE RETRIEVAL QUERY  (Option A)
# Derived from schema + prompt — describes what a relevant chunk looks like.
# No seed queries. No top_k. No cutoff.
# ChromaDB ranks ALL chunks by similarity to this query.
# Every non-boilerplate chunk is sent to Gemini in ranked order.
# ─────────────────────────────────────────────────────────────────────────────

RETRIEVAL_QUERY = (
    "Material property data including section category, property name, "
    "measured value, SI unit, imperial unit, test condition standard "
    "such as ASTM ISO DIN, and comments. Properties include mechanical "
    "thermal electrical physical rheological optical categories. "
    "Values in MPa GPa percent density conductivity temperature modulus "
    "strength elongation hardness viscosity flammability. "
    "Material name, manufacturer, trade grade, abbreviation."
)

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI SCHEMA + PROMPT
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "material_name":         {"type": "STRING"},
        "material_abbreviation": {"type": "STRING"},
        "trade_grade":           {"type": "STRING"},
        "manufacturer":          {"type": "STRING"},
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
                },
                "required": [
                    "section", "property_name", "value", "unit",
                    "english", "test_condition", "comments",
                    "material_name", "source_page", "chunk_type",
                ],
            },
        },
    },
}

PROMPT = (
    "You are an expert materials scientist. "
    "The content below is extracted from a materials datasheet or research paper. "
    "Each block is tagged with its page number and type (TABLE or TEXT).\n\n"
    "Extract every material property. For each property record:\n"
    "  - section        : category (Mechanical, Thermal, Electrical, Physical, Rheological, etc.)\n"
    "  - property_name  : exact name as written\n"
    "  - value          : exact value or range\n"
    "  - unit           : SI unit\n"
    "  - english        : imperial equivalent if shown, else ''\n"
    "  - test_condition : ASTM/ISO/DIN standard or conditions, else ''\n"
    "  - comments       : footnotes or qualifications, else ''\n"
    "  - material_name  : exact material name (never leave blank)\n"
    "  - source_page    : page number from the block header (digits only, e.g. '3')\n"
    "  - chunk_type     : 'table' if from a TABLE block, 'text' if from a TEXT block\n\n"
    "RULES:\n"
    "  - Extract ONLY measured/specified properties, NOT equations or simulation params.\n"
    "  - If multiple materials appear, create separate entries for each.\n"
    "  - Preserve source_page exactly from the block header.\n"
    "  - Respond ONLY with valid JSON matching the schema.\n"
    "\n\nCONTENT:\n"
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
    chunk_type: str                                # "table" | "text"
    source:     str                                # extractor name
    raw_rows:   Optional[List[List[str]]] = None  # tables only
    raw_text:   Optional[str] = None              # text only
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
    def source_label(self) -> str:
        return f"Page {self.page_num} ({self.chunk_type})"

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

def _rows_to_text(rows: List[List[Any]]) -> str:
    lines = []
    for row in rows:
        cells = [str(c).strip() if c is not None else "" for c in row]
        if any(len(c) >= MIN_CELL_CHARS for c in cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


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
    return [p for p in paragraphs if len(p) >= MIN_TEXT_CHARS]

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
            raise ImportError(
                "sentence-transformers not installed.\n"
                "Run: pip install sentence-transformers"
            )
        log.info(f"Loading embedding model '{EMBED_MODEL_NAME}' …")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        log.info("Embedding model ready.")
    return _embed_model


def _get_chroma_collection() -> Any:
    global _chroma_client, _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection
    if not CHROMA_AVAILABLE:
        raise ImportError("chromadb not installed. Run: pip install chromadb")
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    _chroma_collection = _chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    log.info(
        f"ChromaDB ready — '{CHROMA_COLLECTION}' "
        f"({_chroma_collection.count()} existing vectors)"
    )
    return _chroma_collection


def _chroma_pdf_exists(pdf_hash: str) -> bool:
    try:
        col = _get_chroma_collection()
        res = col.get(where={"pdf_hash": pdf_hash}, limit=1)
        return len(res["ids"]) > 0
    except Exception:
        return False


def _chroma_store_chunks(chunks: List[Chunk], pdf_hash: str) -> None:
    """Embed all chunks and upsert into ChromaDB."""
    col   = _get_chroma_collection()
    model = _get_embed_model()

    texts     = [c.text for c in chunks]
    metadatas = [
        {
            "pdf_hash":   pdf_hash,
            "page_num":   c.page_num,
            "chunk_type": c.chunk_type,
            "source":     c.source,
        }
        for c in chunks
    ]
    ids = [f"{pdf_hash}_{i}" for i in range(len(chunks))]

    # Embed in batches of 64
    all_embeddings: List[List[float]] = []
    for start in range(0, len(texts), 64):
        batch = texts[start : start + 64]
        vecs  = model.encode(batch, normalize_embeddings=True)
        all_embeddings.extend(vecs.tolist())

    col.upsert(
        ids=ids,
        documents=texts,
        embeddings=all_embeddings,
        metadatas=metadatas,
    )
    log.info(f"ChromaDB: stored {len(chunks)} chunks for pdf_hash={pdf_hash}")


def _chroma_rank_all(pdf_hash: str) -> List[Chunk]:
    """
    Embed the single RETRIEVAL_QUERY, query ChromaDB filtered to this PDF,
    return ALL non-boilerplate chunks ranked by cosine similarity (highest first).
    No top_k. No cutoff. Everything goes to Gemini in ranked order.
    """
    col   = _get_chroma_collection()
    model = _get_embed_model()

    # How many chunks does this PDF have in ChromaDB?
    all_ids = col.get(where={"pdf_hash": pdf_hash}, include=[])
    n_total = len(all_ids["ids"])

    if n_total == 0:
        log.warning(f"No chunks in ChromaDB for pdf_hash={pdf_hash}")
        return []

    # Embed the single retrieval query
    query_vec = model.encode(
        [RETRIEVAL_QUERY], normalize_embeddings=True
    )[0].tolist()

    # Query for ALL chunks (n_results = n_total so nothing is dropped)
    results = col.query(
        query_embeddings=[query_vec],
        n_results=n_total,
        where={"pdf_hash": pdf_hash},
        include=["documents", "metadatas", "distances"],
    )

    ids_list       = results.get("ids",       [[]])[0]
    docs_list      = results.get("documents", [[]])[0]
    metas_list     = results.get("metadatas", [[]])[0]
    distances_list = results.get("distances", [[]])[0]

    ranked_chunks: List[Chunk] = []
    for doc, meta, dist in zip(docs_list, metas_list, distances_list):
        # ChromaDB cosine distance = 1 - cosine_similarity
        similarity = 1.0 - float(dist)
        chunk = Chunk(
            page_num=int(meta.get("page_num", 1)),
            chunk_type=meta.get("chunk_type", "text"),
            source=meta.get("source", ""),
            raw_text=doc,
            score=similarity,
            relevant=True,
        )
        ranked_chunks.append(chunk)

    table_n = sum(1 for c in ranked_chunks if c.chunk_type == "table")
    text_n  = sum(1 for c in ranked_chunks if c.chunk_type == "text")
    log.info(
        f"ChromaDB ranked {len(ranked_chunks)} chunks "
        f"({table_n} tables + {text_n} text) — all sent to Gemini"
    )
    return ranked_chunks

# ─────────────────────────────────────────────────────────────────────────────
# OVERLAPPING NEIGHBOR CHUNKS
# ─────────────────────────────────────────────────────────────────────────────

def _build_overlapping_chunks(
    chunks: List[Chunk],
    neighbor_window: int = NEIGHBOR_OVERLAP,
) -> List[Chunk]:
    """
    For every text chunk, create an additional overlapping chunk whose content
    is the concatenation of its neighbors. Tables are never merged into
    neighbor windows — they keep their structure intact.
    Returns original chunks PLUS the new overlapping chunks.
    """
    if neighbor_window <= 0:
        return chunks

    eligible = [
        (i, c) for i, c in enumerate(chunks)
        if not _is_boilerplate(c.text)
    ]

    overlap_chunks: List[Chunk] = []
    seen_content: set = set()

    for pos, (_, centre) in enumerate(eligible):
        # Tables keep their own vector — no neighbor merging
        if centre.chunk_type == "table":
            continue

        parts: List[str] = []
        for offset in range(-neighbor_window, neighbor_window + 1):
            nb_pos = pos + offset
            if nb_pos < 0 or nb_pos >= len(eligible):
                continue
            _, nb_chunk = eligible[nb_pos]
            # Don't blend table text into prose windows
            if nb_chunk.chunk_type == "table":
                continue
            parts.append(nb_chunk.text.strip())

        merged = " \n\n ".join(p for p in parts if p)
        if not merged or merged in seen_content:
            continue
        seen_content.add(merged)

        overlap_chunks.append(Chunk(
            page_num=centre.page_num,
            chunk_type="text",
            source=f"{centre.source}+overlap",
            raw_text=merged,
        ))

    log.info(
        f"Overlap chunks: {len(overlap_chunks)} created "
        f"(window={neighbor_window}, base={len(chunks)})"
    )
    return chunks + overlap_chunks

# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION — TABLES
# ─────────────────────────────────────────────────────────────────────────────

def _extract_tables_docling(pdf_bytes: bytes) -> List[Chunk]:
    chunks: List[Chunk] = []
    if not DOCLING_AVAILABLE:
        return chunks
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        converter = DocumentConverter()
        result    = converter.convert(tmp_path)
        doc       = result.document
        for table in doc.tables:
            rows     = [[cell.text for cell in row] for row in table.data.grid]
            page_num = table.prov[0].page_no if table.prov else 1
            if _is_valid_table(rows):
                chunks.append(Chunk(
                    page_num=page_num, chunk_type="table",
                    source="docling", raw_rows=rows,
                ))
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
                    {
                        "vertical_strategy":   "lines_strict",
                        "horizontal_strategy": "lines_strict",
                        "snap_tolerance": 3,
                        "join_tolerance": 3,
                    },
                    {"vertical_strategy": "text", "horizontal_strategy": "text"},
                ):
                    tables = page.extract_tables(table_settings=strategy) or []
                    for rows in tables:
                        cleaned = [
                            [str(c).strip() if c else "" for c in row]
                            for row in rows
                        ]
                        if _is_valid_table(cleaned):
                            chunks.append(Chunk(
                                page_num=page_num, chunk_type="table",
                                source="pdfplumber", raw_rows=cleaned,
                            ))
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
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        for flavor in ("lattice", "stream"):
            try:
                tables = camelot.read_pdf(tmp_path, pages="all", flavor=flavor)
                if flavor == "stream":
                    tables = [
                        t for t in tables
                        if t.parsing_report.get("accuracy", 0) > 65
                    ]
                for table in tables:
                    rows = [
                        [str(c).strip() for c in row]
                        for row in table.df.values.tolist()
                    ]
                    if _is_valid_table(rows):
                        chunks.append(Chunk(
                            page_num=table.page, chunk_type="table",
                            source=f"camelot-{flavor}", raw_rows=rows,
                        ))
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
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        converter = DocumentConverter()
        result    = converter.convert(tmp_path)
        doc       = result.document
        for item, _ in doc.iterate_items():
            text     = getattr(item, "text", "").strip()
            page_num = item.prov[0].page_no if item.prov else 1
            if not text or len(text) < MIN_TEXT_CHARS or _is_boilerplate(text):
                continue
            chunks.append(Chunk(
                page_num=page_num, chunk_type="text",
                source="docling", raw_text=text,
            ))
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
                page_num = page_idx + 1
                raw      = page.get_text("text", sort=True) or ""
                for para in _split_paragraphs(raw):
                    if not _is_boilerplate(para):
                        chunks.append(Chunk(
                            page_num=page_num, chunk_type="text",
                            source="pymupdf", raw_text=para,
                        ))
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
                        chunks.append(Chunk(
                            page_num=page_num, chunk_type="text",
                            source="pdfplumber-text", raw_text=para,
                        ))
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
    """Force-OCR any PDF pages that produced zero chunks."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        total_pages = set(range(1, len(doc) + 1))

    covered   = {c.page_num for c in chunks}
    missing   = total_pages - covered

    if not missing:
        return chunks

    log.warning(f"Page coverage: {len(missing)} pages missing — OCRing: {sorted(missing)}")
    for page_num in sorted(missing):
        ocr_text = _ocr_page(pdf_bytes, page_num)
        for para in _split_paragraphs(ocr_text):
            if not _is_boilerplate(para):
                chunks.append(Chunk(
                    page_num=page_num, chunk_type="text",
                    source="ocr-fallback", raw_text=para,
                ))
    return chunks

# ─────────────────────────────────────────────────────────────────────────────
# COMBINED EXTRACTION + DEDUP
# ─────────────────────────────────────────────────────────────────────────────

def _dedup(chunks: List[Chunk]) -> List[Chunk]:
    seen: set = set()
    unique: List[Chunk] = []
    for c in chunks:
        key = (c.chunk_type, re.sub(r"\s+", " ", c.text.strip())[:200])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def extract_all_chunks(pdf_bytes: bytes) -> List[Chunk]:
    """
    Extract tables AND text. Returns all chunks tagged with
    page_num, chunk_type, source.
    """
    table_chunks: List[Chunk] = []
    text_chunks:  List[Chunk] = []

    # Tables
    docling_tables = _extract_tables_docling(pdf_bytes)
    table_chunks.extend(docling_tables)
    table_chunks.extend(_extract_tables_pdfplumber(pdf_bytes))
    table_chunks.extend(_extract_tables_camelot(pdf_bytes))
    table_chunks = _dedup(table_chunks)

    # Text — Docling if tables found, always pymupdf
    if DOCLING_AVAILABLE and docling_tables:
        text_chunks.extend(_extract_text_docling(pdf_bytes))

    pymupdf_text = _extract_text_pymupdf(pdf_bytes)
    text_chunks.extend(pymupdf_text)

    if sum(len(c.text) for c in pymupdf_text) < 500:
        log.warning("pymupdf sparse — supplementing with pdfplumber")
        text_chunks.extend(_extract_text_pdfplumber(pdf_bytes))

    text_chunks = _dedup(text_chunks)

    all_chunks = table_chunks + text_chunks
    all_chunks = _verify_page_coverage(pdf_bytes, all_chunks)

    log.info(
        f"Total chunks: {len(all_chunks)} "
        f"({len(table_chunks)} tables + {len(text_chunks)} text)"
    )
    return all_chunks

# ─────────────────────────────────────────────────────────────────────────────
# INDEX INTO CHROMADB
# ─────────────────────────────────────────────────────────────────────────────

def index_chunks_in_chroma(chunks: List[Chunk], pdf_hash: str) -> None:
    """
    Build overlapping neighbor chunks, filter boilerplate,
    store everything in ChromaDB. Skips if already indexed.
    """
    if _chroma_pdf_exists(pdf_hash):
        log.info(f"ChromaDB: pdf_hash={pdf_hash} already indexed — skipping.")
        return

    # Add neighbor-overlap chunks before indexing
    all_chunks = _build_overlapping_chunks(chunks, neighbor_window=NEIGHBOR_OVERLAP)

    # Filter boilerplate
    storable = [
        c for c in all_chunks
        if not _is_boilerplate(c.text) and c.text.strip()
    ]
    _chroma_store_chunks(storable, pdf_hash)

# ─────────────────────────────────────────────────────────────────────────────
# BUILD GEMINI BATCHES
# Chunks arrive pre-ranked by ChromaDB similarity score.
# Highest relevance fills first batch — no content is dropped.
# ─────────────────────────────────────────────────────────────────────────────

def build_batches(chunks: List[Chunk]) -> List[str]:
    """
    Pack chunks into Gemini-safe batches (80k char limit).
    Input chunks are already sorted by relevance score (highest first)
    from ChromaDB. Within each batch, chunks are re-sorted by page
    number so Gemini reads coherent context.
    """
    if not chunks:
        return []

    batches:       List[str]   = []
    current_batch: List[Chunk] = []
    current_chars: int         = 0

    for chunk in chunks:
        entry_len = len(chunk.block_header) + len(chunk.text)

        # Single chunk too large — truncate it
        if entry_len > MAX_BATCH_CHARS:
            log.warning(
                f"Page {chunk.page_num} {chunk.chunk_type} too large "
                f"({entry_len} chars) — truncating"
            )
            chunk.text = chunk.text[: MAX_BATCH_CHARS - len(chunk.block_header) - 20]
            entry_len  = len(chunk.block_header) + len(chunk.text)

        # Flush current batch if adding this chunk would exceed limit
        if current_chars + entry_len > MAX_BATCH_CHARS and current_batch:
            # Re-sort by page for coherent reading before flushing
            current_batch.sort(key=lambda c: c.page_num)
            batches.append(
                "\n".join(c.block_header + c.text for c in current_batch)
            )
            current_batch = []
            current_chars = 0

        current_batch.append(chunk)
        current_chars += entry_len

    # Flush remaining
    if current_batch:
        current_batch.sort(key=lambda c: c.page_num)
        batches.append(
            "\n".join(c.block_header + c.text for c in current_batch)
        )

    log.info(
        f"Built {len(batches)} batch(es) from {len(chunks)} chunks "
        f"(no chunks dropped)"
    )
    return batches

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _call_gemini(text_payload: str) -> Tuple[Optional[Dict], str]:
    payload = {
        "contents": [{"parts": [{"text": PROMPT + text_payload}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": SCHEMA,
            "maxOutputTokens": 65536,
        },
    }
    for attempt, wait in enumerate([0] + RETRY_DELAYS):
        if wait:
            log.warning(f"429 rate limit — waiting {wait}s (retry {attempt})")
            time.sleep(wait)
        try:
            resp = requests.post(API_URL, json=payload, timeout=600)
            if resp.status_code == 429:
                continue
            if not resp.ok:
                return None, f"HTTP {resp.status_code}: {resp.text[:2000]}"

            data       = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return None, f"No candidates: {json.dumps(data)[:400]}"

            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                text = part.get("text", "").strip()
                if text.startswith("{"):
                    try:
                        return json.loads(text), ""
                    except json.JSONDecodeError as e:
                        return None, f"JSON parse error: {e}"
            return None, "No JSON in Gemini response."

        except requests.Timeout:
            return None, "Timeout after 180s."
        except Exception as e:
            return None, f"Unexpected: {e}"

    return None, "Max retries exceeded."


def _run_gemini_batches(
    batches: List[str],
    progress_callback=None,
) -> Tuple[List[Dict], List[str]]:
    results: List[Dict] = []
    errors:  List[str]  = []
    total = len(batches)

    for idx, batch_text in enumerate(batches):
        pct = 0.60 + 0.35 * (idx / max(total, 1))
        msg = f"Gemini batch {idx+1}/{total}…"
        log.info(msg)
        if progress_callback:
            progress_callback(msg, pct)

        if idx > 0:
            time.sleep(BATCH_CALL_DELAY)

        result, err = _call_gemini(batch_text)
        if result:
            results.append(result)
        if err:
            errors.append(f"Batch {idx+1}: {err}")
            log.error(f"Batch {idx+1} error: {err}")

    return results, errors

# ─────────────────────────────────────────────────────────────────────────────
# MERGE + DEDUP → DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────

FIXED_COLS = ["material_name", "material_abbreviation", "trade_grade", "manufacturer"]


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
        key = (
            _norm(row.get("material_name", "")),
            _norm(row.get("property_name", "")),
            _norm(row.get("section", "")),
        )
        if key in seen_fuzzy:
            existing_idx = seen_fuzzy[key]
            existing     = kept[existing_idx]
            if _richness(row) > _richness(existing):
                merged = dict(row)
                for k, v in existing.items():
                    if (
                        str(merged.get(k, "")).strip() in ("", "N/A")
                        and str(v).strip() not in ("", "N/A")
                    ):
                        merged[k] = v
                kept[existing_idx] = merged
        else:
            seen_fuzzy[key] = len(kept)
            kept.append(row)
    return kept


def merge_to_dataframe(results: List[Dict]) -> pd.DataFrame:
    all_rows:   List[Dict] = []
    seen_exact: set        = set()

    fallback = {k: "" for k in FIXED_COLS}
    for r in results:
        for k in FIXED_COLS:
            if not fallback[k] and r.get(k):
                fallback[k] = r[k]
    if not fallback["material_abbreviation"] and fallback["material_name"]:
        fallback["material_abbreviation"] = _make_abbreviation(fallback["material_name"])

    for r in results:
        r_identity = {
            "material_name":
                r.get("material_name", "") or fallback["material_name"],
            "material_abbreviation":
                r.get("material_abbreviation", "") or fallback["material_abbreviation"],
            "trade_grade":
                r.get("trade_grade", "") or fallback["trade_grade"],
            "manufacturer":
                r.get("manufacturer", "") or fallback["manufacturer"],
        }
        if not r_identity["material_abbreviation"] and r_identity["material_name"]:
            r_identity["material_abbreviation"] = _make_abbreviation(
                r_identity["material_name"]
            )

        for item in r.get("mechanical_properties", []):
            prop_mat = item.get("material_name", "").strip()
            identity = dict(r_identity)
            if prop_mat:
                identity["material_name"]         = prop_mat
                identity["material_abbreviation"] = _make_abbreviation(prop_mat)

            raw_page   = str(item.get("source_page", "")).strip()
            page_label = (
                f"Page {raw_page}" if raw_page.isdigit()
                else raw_page if raw_page
                else "Unknown"
            )
            chunk_type = item.get("chunk_type", "").strip() or "unknown"

            key = (
                _norm(identity["material_name"]),
                _norm(item.get("section", "")),
                _norm(item.get("property_name", "")),
                _norm(item.get("value", "")),
            )
            if key in seen_exact:
                continue
            seen_exact.add(key)

            all_rows.append({
                **identity,
                "section":        item.get("section", "")        or "General",
                "property_name":  item.get("property_name", "")  or "Unknown",
                "value":          item.get("value", "")          or "N/A",
                "unit":           item.get("unit", "")           or "",
                "english":        item.get("english", "")        or "",
                "test_condition": item.get("test_condition", "") or "",
                "comments":       item.get("comments", "")       or "",
                "source_page":    page_label,
                "chunk_type":     chunk_type,
            })

    all_rows = _fuzzy_dedup(all_rows)
    df = pd.DataFrame(all_rows)

    if not df.empty:
        base_cols = [c for c in df.columns if c not in ("source_page", "chunk_type")]
        df = df[base_cols + ["source_page", "chunk_type"]]

    return df


def _df_to_cache_dict(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {}
    row0 = df.iloc[0]
    return {
        "material_name":         str(row0.get("material_name", "")),
        "material_abbreviation": str(row0.get("material_abbreviation", "")),
        "trade_grade":           str(row0.get("trade_grade", "")),
        "manufacturer":          str(row0.get("manufacturer", "")),
        "mechanical_properties": df.drop(columns=FIXED_COLS, errors="ignore")
                                   .to_dict(orient="records"),
    }

# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    pdf_bytes:         bytes,
    progress_callback: Any = None,
) -> Tuple[pd.DataFrame, List[Chunk], List[str], Dict]:
    """
    Full extraction pipeline.

    Returns
    -------
    df         : extracted properties DataFrame
    all_chunks : all Chunk objects with scores
    api_errors : list of error strings
    meta       : pipeline stats dict
    """
    def _prog(msg: str, pct: float):
        log.info(f"[{pct*100:.0f}%] {msg}")
        if progress_callback:
            progress_callback(msg, pct)

    meta:      Dict[str, Any] = {}
    pdf_hash = _pdf_hash(pdf_bytes)

    # ── Cache check ───────────────────────────────────────────────────────────
    _prog("Checking cache…", 0.0)
    cached = cache_get(pdf_bytes)
    if cached:
        _prog("Cache hit.", 1.0)
        meta["path"] = "cache"
        return merge_to_dataframe([cached]), [], [], meta

    # ── Stage 1: Extract ──────────────────────────────────────────────────────
    _prog("Stage 1 — extracting tables + text…", 0.05)
    all_chunks = extract_all_chunks(pdf_bytes)
    meta["chunks_total"]  = len(all_chunks)
    meta["chunks_tables"] = sum(1 for c in all_chunks if c.chunk_type == "table")
    meta["chunks_text"]   = sum(1 for c in all_chunks if c.chunk_type == "text")

    if not all_chunks:
        _prog("No content extracted.", 1.0)
        meta["path"] = "failed"
        return pd.DataFrame(), [], ["No content extracted — PDF may be image-only."], meta

    _prog(
        f"Stage 1 done — {meta['chunks_tables']} tables, "
        f"{meta['chunks_text']} text blocks.", 0.20,
    )

    # ── Stage 2: Index into ChromaDB ─────────────────────────────────────────
    _prog("Stage 2 — indexing into ChromaDB (with neighbor overlap)…", 0.25)
    try:
        index_chunks_in_chroma(all_chunks, pdf_hash)
    except Exception as e:
        log.error(f"ChromaDB indexing failed: {e} — sending all chunks directly")
        api_errors: List[str] = [f"ChromaDB index error: {e}"]
        for c in all_chunks:
            c.relevant = True
        batches  = build_batches(all_chunks)
        results, errs = _run_gemini_batches(batches, progress_callback)
        api_errors.extend(errs)
        df = merge_to_dataframe(results)
        meta["path"] = "no-chroma"
        if not df.empty:
            cache_set(pdf_bytes, _df_to_cache_dict(df))
        return df, all_chunks, api_errors, meta

    # ── Stage 3: Rank ALL chunks via single retrieval query ───────────────────
    _prog("Stage 3 — ranking all chunks via schema-derived query…", 0.40)
    ranked_chunks = _chroma_rank_all(pdf_hash)
    meta["chunks_ranked"] = len(ranked_chunks)

    if not ranked_chunks:
        meta["path"] = "failed"
        return pd.DataFrame(), all_chunks, ["ChromaDB returned no chunks."], meta

    _prog(f"Stage 3 done — {len(ranked_chunks)} chunks ranked.", 0.50)

    # ── Stage 4: Build batches ────────────────────────────────────────────────
    _prog("Stage 4 — building Gemini batches…", 0.55)
    batches = build_batches(ranked_chunks)
    meta["batches"] = len(batches)
    _prog(f"Stage 4 done — {len(batches)} batch(es) queued.", 0.60)

    # ── Stage 5: Gemini extraction ────────────────────────────────────────────
    _prog(f"Stage 5 — Gemini extraction ({len(batches)} batch(es))…", 0.60)
    results, api_errors = _run_gemini_batches(batches, progress_callback)

    if not results:
        meta["path"] = "failed"
        return pd.DataFrame(), all_chunks, api_errors, meta

    # ── Stage 6: Merge + dedup ────────────────────────────────────────────────
    _prog("Stage 6 — merging & deduplicating…", 0.95)
    df = merge_to_dataframe(results)
    meta["properties_extracted"] = len(df)
    meta["path"] = "rag-chroma"

    if not df.empty:
        cache_set(pdf_bytes, _df_to_cache_dict(df))

    _prog(f"Done — {len(df)} properties extracted.", 1.0)
    return df, all_chunks, api_errors, meta

# ─────────────────────────────────────────────────────────────────────────────
# BATCH EXTRACTION FOR CRAWLER PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_downloaded_pdfs(progress_callback=None) -> pd.DataFrame:
    """
    Extract material properties from all PDFs in downloads folder.
    Processes one-by-one sequentially with rate limiting.
    
    Args:
        progress_callback: Optional callable(msg: str, pct: float) for progress updates
    
    Returns:
        Combined DataFrame with all extracted properties
    """
    DOWNLOADS_DIR = Path(__file__).parent / "downloads"
    all_pdfs = sorted(DOWNLOADS_DIR.rglob("*.pdf")) if DOWNLOADS_DIR.exists() else []
    
    if not all_pdfs:
        log.warning("No PDFs in ./downloads/")
        return pd.DataFrame()
    
    all_results = []
    total = len(all_pdfs)
    
    for idx, pdf_path in enumerate(all_pdfs):
        stem = pdf_path.stem
        pdf_bytes = pdf_path.read_bytes()
        
        msg = f"Extracting {idx+1}/{total}: {pdf_path.name}"
        if progress_callback:
            progress_callback(msg, idx / total)
        print(f"\n[Extraction] {msg}")
        
        try:
            df, chunks, errors, meta = run_pipeline(pdf_bytes)
            if not df.empty:
                df["pdf_name"] = stem
                all_results.append(df)
                print(f"  ✓ Extracted {len(df)} properties")
            else:
                print(f"  ⚠ No properties extracted")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        
        # Rate limit between PDFs
        if idx < total - 1:
            time.sleep(10)
    
    if progress_callback:
        progress_callback("Done", 1.0)
    
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        print(f"\n[Extraction] Total: {len(combined)} properties from {total} PDFs")
        return combined
    else:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

def _run_streamlit():
    import streamlit as st

    st.set_page_config(
        page_title="DocToDB — Materials Extractor",
        page_icon="🧬",
        layout="wide",
    )
    st.title("🧬 DocToDB — PDF RAG Extractor")
    st.caption(
        "Extracts material properties from **tables AND prose text** "
        "using ChromaDB semantic ranking. All non-boilerplate chunks "
        "sent to Gemini — no arbitrary limits."
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        st.divider()
        st.markdown(f"**Model:** `{GEMINI_MODEL}`")
        st.markdown(f"**Embedder:** `{EMBED_MODEL_NAME}`")
        st.markdown(f"**ChromaDB:** {'✅' if CHROMA_AVAILABLE else '❌ not installed'}")
        st.markdown(f"**Docling:** {'✅' if DOCLING_AVAILABLE else '❌'}")
        st.markdown(f"**Camelot:** {'✅' if CAMELOT_AVAILABLE else '❌'}")
        st.markdown(f"**OCR:** {'✅' if OCR_AVAILABLE else '❌'}")
        st.divider()
        st.markdown("**Retrieval:** Single schema-derived query")
        st.markdown("**Chunks sent:** ALL non-boilerplate (ranked)")
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

    DOWNLOADS_DIR = Path(__file__).parent / "downloads"
    all_pdfs = sorted(DOWNLOADS_DIR.rglob("*.pdf")) if DOWNLOADS_DIR.exists() else []

    if not all_pdfs:
        st.info("No PDFs in ./downloads/ yet. Run the crawler first.")
        return

    for pdf_path in all_pdfs:
        stem      = pdf_path.stem
        pdf_bytes = pdf_path.read_bytes()
        st.write(f"Processing: {pdf_path.name}")
        # everything below this line stays exactly as it was
        # (the if st.button block, run_pipeline call, tabs, etc.)

        bar    = st.progress(0.0)
        status = st.empty()

        def cb(msg, pct):
            bar.progress(min(pct, 1.0))
            status.text(msg)

        with st.spinner("Running pipeline…"):
            df, chunks, errors, meta = run_pipeline(pdf_bytes, progress_callback=cb)
        
        # Rate limit between PDFs (respect Gemini free tier)
        time.sleep(10)  # Wait 10s before next PDF

        bar.progress(1.0)
        status.empty()

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Tables",       meta.get("chunks_tables", "—"))
        c2.metric("Text blocks",  meta.get("chunks_text", "—"))
        c3.metric("Ranked",       meta.get("chunks_ranked", "—"))
        c4.metric("Batches",      meta.get("batches", "—"))
        c5.metric("Properties",   meta.get("properties_extracted", len(df)))
        c6.metric("Path",         meta.get("path", "—"))

        if errors:
            with st.expander(f"⚠️ {len(errors)} API error(s)"):
                for e in errors:
                    st.code(e)

        if df.empty:
            st.error("No properties extracted.")
            return

        tab_table, tab_text, tab_pages, tab_all, tab_export = st.tabs([
            "📋 From Tables",
            "📄 From Text",
            "🗂 By Page",
            "🔢 All",
            "📤 Export",
        ])

        with tab_table:
            tbl_df = df[df["chunk_type"] == "table"]
            st.caption(f"{len(tbl_df)} properties from tables")
            st.dataframe(tbl_df, use_container_width=True, hide_index=True)

        with tab_text:
            txt_df = df[df["chunk_type"] == "text"]
            st.caption(f"{len(txt_df)} properties from prose text")
            st.dataframe(txt_df, use_container_width=True, hide_index=True)

        with tab_pages:
            if "source_page" in df.columns:
                def _page_sort_key(label: str) -> int:
                    m = re.search(r"\d+", label)
                    return int(m.group()) if m else 9999
                for pg in sorted(df["source_page"].unique(), key=_page_sort_key):
                    pg_df = df[df["source_page"] == pg]
                    tbl_n = (pg_df["chunk_type"] == "table").sum()
                    txt_n = (pg_df["chunk_type"] == "text").sum()
                    with st.expander(
                        f"📄 {pg} — {len(pg_df)} "
                        f"propert{'y' if len(pg_df)==1 else 'ies'} "
                        f"({tbl_n} table · {txt_n} text)",
                        expanded=False,
                    ):
                        st.dataframe(pg_df, use_container_width=True, hide_index=True)

        with tab_all:
            st.caption(f"{len(df)} total properties")
            st.dataframe(df, use_container_width=True, hide_index=True)

        with tab_export:
            col1, col2 = st.columns(2)
            col1.download_button(
                "⬇️ CSV",
                df.to_csv(index=False).encode(),
                f"{stem}_extracted.csv",
                "text/csv",
                use_container_width=True,
            )
            col2.download_button(
                "⬇️ JSON",
                df.to_json(orient="records", indent=2).encode(),
                f"{stem}_extracted.json",
                "application/json",
                use_container_width=True,
            )
            st.divider()
            grouped: Dict[str, Any] = {}
            for pg in df["source_page"].unique():
                pg_df = df[df["source_page"] == pg]
                grouped[pg] = (
                    pg_df.drop(columns=["source_page"], errors="ignore")
                    .to_dict(orient="records")
                )
            st.download_button(
                "⬇️ JSON grouped by page",
                json.dumps(grouped, indent=2).encode(),
                f"{stem}_by_page.json",
                "application/json",
                use_container_width=True,
            )

            st.divider()
            with st.expander("🔎 Chunk inspector (top-30 by score)"):
                show_type = st.radio("Show", ["all", "table", "text"], horizontal=True)
                shown = [
                    c for c in chunks
                    if show_type == "all" or c.chunk_type == show_type
                ]
                shown = sorted(shown, key=lambda c: c.score, reverse=True)[:30]
                for c in shown:
                    st.code(
                        f"[{c.chunk_type.upper()}] Page {c.page_num} | "
                        f"score={c.score:.3f}\n\n{c.text[:400]}",
                        language=None,
                    )

# ─────────────────────────────────────────────────────────────────────────────
# PRECISION / RECALL EVALUATION HARNESS
# ─────────────────────────────────────────────────────────────────────────────
#
# Usage:
#   python doctodb_rag.py --eval \
#       --pdf path/to/material.pdf \
#       --ground-truth path/to/gt.json \
#       [--baseline-json path/to/baseline_model_output.json]
#
# Ground-truth JSON format:
# [
#   {
#     "section": "Mechanical",
#     "property_name": "Tensile Strength",
#     "value": "85",
#     "unit": "MPa",
#     "material_name": "ABS"
#   }, ...
# ]
# ─────────────────────────────────────────────────────────────────────────────

def _norm_eval(s: str) -> str:
    s = str(s).lower().strip()
    return re.sub(r"[^a-z0-9.\-]", "", s)


def _property_key(
    row: Dict,
    fields: tuple = ("section", "property_name", "material_name"),
) -> tuple:
    return tuple(_norm_eval(row.get(f, "")) for f in fields)


def _value_match(pred_val: str, gt_val: str, tol: float = 0.05) -> bool:
    pv = _norm_eval(pred_val)
    gv = _norm_eval(gt_val)
    if pv == gv:
        return True

    def _first_num(s: str) -> Optional[float]:
        m = re.search(r"[\d.]+", s)
        return float(m.group()) if m else None

    pn, gn = _first_num(pv), _first_num(gv)
    if pn is not None and gn is not None and gn != 0:
        if abs(pn - gn) / abs(gn) <= tol:
            return True

    range_m = re.match(r"([\d.]+)[^\d.]+([\d.]+)", gv)
    if range_m and pn is not None:
        lo, hi = float(range_m.group(1)), float(range_m.group(2))
        if lo <= pn <= hi:
            return True

    return False


def _load_records(path: str) -> List[Dict]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    records  = []
    identity = {k: data.get(k, "") for k in FIXED_COLS}
    for item in data.get("mechanical_properties", []):
        records.append({**identity, **item})
    return records


def compute_metrics(
    predicted:    List[Dict],
    ground_truth: List[Dict],
    value_tol:    float = 0.05,
) -> Dict[str, Any]:
    matched_gt:   set = set()
    matched_pred: set = set()

    for pi, pred_row in enumerate(predicted):
        pk = _property_key(pred_row)
        for gi, gt_row in enumerate(ground_truth):
            if gi in matched_gt:
                continue
            if _property_key(gt_row) != pk:
                continue
            if _value_match(
                pred_row.get("value", ""),
                gt_row.get("value", ""),
                tol=value_tol,
            ):
                matched_gt.add(gi)
                matched_pred.add(pi)
                break

    tp = len(matched_pred)
    fp = len(predicted) - tp
    fn = len(ground_truth) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    sections = sorted({r.get("section", "General") for r in ground_truth})
    per_section: Dict[str, Dict] = {}
    for sec in sections:
        per_section[sec] = compute_metrics(
            [r for r in predicted    if r.get("section") == sec],
            [r for r in ground_truth if r.get("section") == sec],
            value_tol,
        )

    materials = sorted({r.get("material_name", "") for r in ground_truth})
    per_material: Dict[str, Dict] = {}
    for mat in materials:
        per_material[mat] = compute_metrics(
            [r for r in predicted    if r.get("material_name") == mat],
            [r for r in ground_truth if r.get("material_name") == mat],
            value_tol,
        )

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "per_section":  per_section,
        "per_material": per_material,
    }


def print_eval_report(
    metrics:          Dict[str, Any],
    label:            str = "Pipeline",
    baseline_metrics: Optional[Dict[str, Any]] = None,
    baseline_label:   str = "Baseline",
) -> None:
    SEP = "═" * 70
    print(f"\n{SEP}")
    print(f"  EVALUATION REPORT — {label}")
    print(SEP)
    for metric in ("precision", "recall", "f1"):
        val  = metrics[metric]
        base = baseline_metrics[metric] if baseline_metrics else None
        diff = f"   Δ {val - base:+.4f}" if base is not None else ""
        print(f"  {metric.capitalize():<12} {val:.4f}{diff}")
    print(f"\n  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}")

    if metrics.get("per_section"):
        print(f"\n  {'Section':<20} {'P':>7} {'R':>7} {'F1':>7}")
        print(f"  {'-'*45}")
        for sec, sm in sorted(metrics["per_section"].items()):
            base_f1 = (
                baseline_metrics.get("per_section", {}).get(sec, {}).get("f1")
                if baseline_metrics else None
            )
            diff = f"   Δ {sm['f1'] - base_f1:+.3f}" if base_f1 is not None else ""
            print(
                f"  {sec:<20} {sm['precision']:>7.3f} "
                f"{sm['recall']:>7.3f} {sm['f1']:>7.3f}{diff}"
            )

    if metrics.get("per_material"):
        print(f"\n  {'Material':<30} {'P':>7} {'R':>7} {'F1':>7}")
        print(f"  {'-'*55}")
        for mat, mm in sorted(metrics["per_material"].items()):
            base_f1 = (
                baseline_metrics.get("per_material", {}).get(mat, {}).get("f1")
                if baseline_metrics else None
            )
            diff = f"   Δ {mm['f1'] - base_f1:+.3f}" if base_f1 is not None else ""
            print(
                f"  {mat:<30} {mm['precision']:>7.3f} "
                f"{mm['recall']:>7.3f} {mm['f1']:>7.3f}{diff}"
            )
    print(f"\n{SEP}\n")


def run_eval_cli(
    pdf_path:      str,
    gt_path:       str,
    baseline_path: Optional[str] = None,
    value_tol:     float         = 0.05,
) -> None:
    print(f"\n  PDF     : {pdf_path}")
    print(f"  GT JSON : {gt_path}")
    if baseline_path:
        print(f"  Baseline: {baseline_path}")

    ground_truth = _load_records(gt_path)
    print(f"\n  Ground truth: {len(ground_truth)} properties")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    def _cli_prog(msg, pct):
        bar = "█" * int(pct * 40) + "░" * (40 - int(pct * 40))
        print(f"  [{bar}] {pct*100:5.1f}%  {msg}")

    df, chunks, api_errors, meta = run_pipeline(
        pdf_bytes, progress_callback=_cli_prog
    )

    if df.empty:
        print("\n  ❌ Pipeline returned no results.")
        return

    predicted = df.to_dict(orient="records")
    print(f"\n  Pipeline extracted: {len(predicted)} properties (path={meta.get('path')})")

    pipeline_metrics = compute_metrics(predicted, ground_truth, value_tol)

    baseline_metrics = None
    if baseline_path:
        baseline_records = _load_records(baseline_path)
        baseline_metrics = compute_metrics(baseline_records, ground_truth, value_tol)
        print_eval_report(baseline_metrics, label="Baseline Model")

    print_eval_report(
        pipeline_metrics,
        label="DocToDB RAG Pipeline",
        baseline_metrics=baseline_metrics,
        baseline_label="Baseline Model",
    )

    out_path = pdf_path.rsplit(".", 1)[0] + "_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "pipeline_metrics":     pipeline_metrics,
                "baseline_metrics":     baseline_metrics,
                "pipeline_predictions": predicted,
                "ground_truth":         ground_truth,
            },
            f, indent=2,
        )
    print(f"  Results saved → {out_path}\n")

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    _in_streamlit = False
    try:
        import streamlit.runtime.scriptrunner as _sr
        if _sr.get_script_run_ctx() is not None:
            _in_streamlit = True
    except Exception:
        pass

    if _in_streamlit:
        _run_streamlit()
    elif "--eval" in sys.argv:
        import argparse
        parser = argparse.ArgumentParser(description="DocToDB RAG — Evaluator")
        parser.add_argument("--eval",          action="store_true")
        parser.add_argument("--pdf",           required=True)
        parser.add_argument("--ground-truth",  required=True)
        parser.add_argument("--baseline-json", default=None)
        parser.add_argument("--value-tol",     type=float, default=0.05)
        args = parser.parse_args()
        run_eval_cli(
            pdf_path=args.pdf,
            gt_path=args.ground_truth,
            baseline_path=args.baseline_json,
            value_tol=args.value_tol,
        )
    else:
        print(
            "\nUsage:\n"
            "  streamlit run doctodb_rag.py          # Streamlit UI\n"
            "  python doctodb_rag.py --eval          # Evaluator\n"
            "    --pdf path/to/material.pdf\n"
            "    --ground-truth path/to/gt.json\n"
            "    [--baseline-json path/to/baseline.json]\n"
            "    [--value-tol 0.05]\n"
        )
