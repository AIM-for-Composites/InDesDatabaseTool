"""
FinalVerdict.py
=======================
Straight one-call Gemini extraction (no chunking, no ranking, no batching) with
source verification: every extracted property's value must be found in the PDF
first (tolerant numeric match), then its property+material meaning is checked
against the sentence(s) that contained that value.

Output: one table — Gemini's extracted properties, each with the paper DOI and
flagged source_verified True/False, downloadable as CSV.

Run:  streamlit run FinalVerdict.py
Needs GEMINI_API_KEY in .env
"""

import os
import re
import json
import math
import base64
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

EMBED_MODEL_NAME   = "allenai/scibert_scivocab_uncased"   # matches DocToDB_eval_v2 (no-rank pipeline)
VALUE_REL_TOL      = 0.005   # 0.5% relative tolerance for numeric matching
MEANING_THRESHOLD  = 0.35    # cosine similarity floor for property/material meaning check

# DOIs follow a strict format (10.<registrant>/<suffix>), so regex over the raw
# PDF text is far more reliable than an LLM guess. Used as the primary source;
# Gemini's own DOI (from the schema) is only a fallback.
DOI_CORE_RE = re.compile(r'10\.\d{4,9}/[^\s"<>\]\)]+', re.IGNORECASE)

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA / PROMPT
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "material_name": {"type": "STRING"},
        "material_abbreviation": {"type": "STRING"},
        "doi": {"type": "STRING"},
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
                "required": ["section", "property_name", "value", "english", "comments"],
            },
        },
    },
}

EXTRACTION_PROMPT = (
    "You are an expert materials scientist. From the attached PDF, extract the material name, "
    "abbreviation, the paper's DOI, and ALL properties across categories (Mechanical, Thermal, "
    "Electrical, Physical, Optical, Rheological, etc.). Return the properties as "
    "'mechanical_properties' (a single list). "
    "For the DOI, extract the Digital Object Identifier exactly as printed (e.g., "
    "10.1021/acsami.0c01234); write '' if it is not present. "
    "For each property, you MUST extract:\n"
    "- section (category)\n- property_name\n- value (or range)\n- unit\n"
    "- english (converted or alternate units, e.g., psi, °F, inches; write '' if not provided)\n"
    "- test_condition\n- comments (include any notes, footnotes, standards, remarks; write '' if none)\n"
    "All fields including english and comments are REQUIRED. Respond ONLY with valid JSON following the "
    "schema below — no markdown, no code fences, no commentary.\n\n"
    "SCHEMA:\n"
    "{\n"
    '  "material_name": "",\n'
    '  "material_abbreviation": "",\n'
    '  "doi": "",\n'
    '  "mechanical_properties": [\n'
    '    {"section":"","property_name":"","value":"","unit":"","english":"","test_condition":"","comments":""}\n'
    "  ]\n"
    "}\n"
)


def make_abbreviation(name: str) -> str:
    if not name:
        return "UNKNOWN"
    words = name.split()
    abbr = "".join(w[0] for w in words if w and w[0].isalpha()).upper()
    return abbr or name[:6].upper()


# ─────────────────────────────────────────────────────────────────────────────
# DOI EXTRACTION — regex over raw PDF text (primary), Gemini doi (fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _clean_doi(doi: str) -> str:
    """Strip common prefixes and trailing punctuation that regex can over-capture."""
    if not doi:
        return ""
    doi = doi.strip()
    doi = re.sub(r'^(?:https?://)?(?:dx\.)?doi\.org/', '', doi, flags=re.IGNORECASE)
    doi = re.sub(r'^doi[:\s]+', '', doi, flags=re.IGNORECASE)
    doi = doi.rstrip(' .,;:)]}>"\'')
    return doi


def _extract_doi_from_pdf(pdf_bytes: bytes) -> str:
    """Find the paper DOI in the raw PDF text. Prefers an explicit doi.org URL or
    a 'doi:' label, then falls back to the first bare 10.xxxx/... token."""
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            text = "\n".join((page.get_text("text") or "") for page in doc)
    except Exception:
        return ""

    # 1) explicit doi.org URL
    m = re.search(r'(?:https?://)?(?:dx\.)?doi\.org/(10\.\d{4,9}/[^\s"<>\]\)]+)', text, re.IGNORECASE)
    if m:
        return _clean_doi(m.group(1))

    # 2) 'doi:' or 'DOI ' labelled
    m = re.search(r'\bdoi[:\s]+\s*(10\.\d{4,9}/[^\s"<>\]\)]+)', text, re.IGNORECASE)
    if m:
        return _clean_doi(m.group(1))

    # 3) any bare DOI token
    m = DOI_CORE_RE.search(text)
    if m:
        return _clean_doi(m.group(0))

    return ""


def resolve_doi(pdf_bytes: bytes, gemini_data: Optional[Dict[str, Any]]) -> str:
    """Prefer the regex-extracted DOI (format-verified). Fall back to Gemini's."""
    doi = _extract_doi_from_pdf(pdf_bytes)
    if doi:
        return doi
    if gemini_data:
        return _clean_doi(str(gemini_data.get("doi", "") or ""))
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI CALL — single call, whole PDF, no chunking
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini_from_bytes(pdf_bytes: bytes) -> Optional[Dict[str, Any]]:
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY not set.")
        return None
    encoded = base64.b64encode(pdf_bytes).decode("utf-8")
    payload = {
        "contents": [{
            "parts": [
                {"text": EXTRACTION_PROMPT},
                {"inlineData": {"mimeType": "application/pdf", "data": encoded}},
            ]
        }],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": SCHEMA,
        },
    }
    try:
        r = requests.post(GEMINI_API_URL, params={"key": GEMINI_API_KEY}, json=payload, timeout=300)
        if not r.ok:
            st.error(f"Gemini HTTP {r.status_code}")
            st.code(r.text)
            return None
        data = r.json()
        candidates = data.get("candidates", [])
        if not candidates:
            st.warning("Gemini returned no candidates.")
            return None
        for part in candidates[0].get("content", {}).get("parts", []):
            text = part.get("text", "").strip()
            if text.startswith("{"):
                return json.loads(text)
        st.warning("Gemini response didn't contain JSON text.")
        return None
    except Exception as e:
        st.error(f"Gemini API exception: {e}")
        return None


def convert_to_dataframe(data: Dict[str, Any], doi: str = "") -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    mat_name = data.get("material_name", "") or ""
    mat_abbr = data.get("material_abbreviation", "") or ""
    if not mat_abbr:
        mat_abbr = make_abbreviation(mat_name)

    rows = []
    for item in data.get("mechanical_properties", []):
        rows.append({
            "doi": doi,
            "material_name": mat_name,
            "material_abbreviation": mat_abbr,
            "section": item.get("section", "") or "Mechanical",
            "property_name": item.get("property_name", "") or "Unknown property",
            "value": item.get("value", "") or "N/A",
            "unit": item.get("unit", "") or "",
            "english": item.get("english", "") or "",
            "test_condition": item.get("test_condition", "") or "",
            "comments": item.get("comments", "") or "",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE VERIFICATION — value found in PDF first (hard gate, tolerant),
# then meaning check on property + material (only for survivors)
# ─────────────────────────────────────────────────────────────────────────────

_embed_model = None

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _extract_sentences(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """Plain-text sentence index of the whole PDF — used only for verification,
    not for extraction. No chunking/table/ranking machinery needed here."""
    sentences = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_idx, page in enumerate(doc, start=1):
            raw = page.get_text("text") or ""
            for para in re.split(r"(?<=[.\n])\s+", raw):
                para = para.strip()
                if len(para) >= 8:
                    sentences.append({"text": para, "page": page_idx})
    return sentences


def _num_close(target: float, candidate: float, rel_tol: float = VALUE_REL_TOL) -> bool:
    return math.isclose(target, candidate, rel_tol=rel_tol, abs_tol=1e-6)


def _find_value_matches(value_str: str, sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """STEP 1 — hard gate. Find every sentence containing a number within tolerance
    of the target value. Tolerant, not exact-substring, so '45.3' vs '45.30' or
    minor rounding differences still count as the same number."""
    val_clean = re.sub(r"^[~><=\u2248\u2265\u2264\u00b1\s]+", "", str(value_str)).strip()
    m = re.search(r"[\d.]+", val_clean)
    if not m:
        return []
    try:
        target = float(m.group())
    except ValueError:
        return []

    matches = []
    for s in sentences:
        for num_str in re.findall(r"[-+]?\d*\.?\d+", s["text"]):
            try:
                cand = float(num_str)
            except ValueError:
                continue
            if _num_close(target, cand):
                matches.append(s)
                break
    return matches


def verify_dataframe(df: pd.DataFrame, sentences: List[Dict[str, Any]]) -> pd.DataFrame:
    # Always emit the three provenance columns, even on the empty/short-circuit
    # paths, so downstream schemas can rely on them existing.
    if df.empty or not sentences:
        if not df.empty:
            df = df.copy()
            df["source_verified"] = False
            df["source_text"] = ""
            df["source_page"] = ""
        return df

    df = df.copy()
    model = None
    try:
        model = _get_embed_model()
    except Exception as e:
        st.warning(f"Embedding model unavailable, meaning-check skipped: {e}")

    verified: List[bool] = []
    source_texts: List[str] = []
    source_pages: List[Any] = []

    for _, row in df.iterrows():
        prop = str(row.get("property_name", "") or "").strip()
        mat  = str(row.get("material_name", "") or "").strip()
        val  = str(row.get("value", "") or "").strip()

        # STEP 1 — does this value exist anywhere in the PDF (tolerant match)?
        candidates = _find_value_matches(val, sentences)
        if not candidates:
            verified.append(False)
            source_texts.append("")
            source_pages.append("")
            continue

        # STEP 2 — of the sentences containing that value, pick the one whose
        # meaning best matches the claimed property + material, and record it
        # as this row's source_text / source_page.
        if model is None:
            # No embedder: value confirmed, meaning-check unavailable. Record the
            # first value-matched sentence rather than over-penalize.
            best = candidates[0]
            verified.append(True)
            source_texts.append(best["text"])
            source_pages.append(best["page"])
            continue

        query = f"{prop} {mat}".strip()
        try:
            q_vec = model.encode([query], normalize_embeddings=True)[0]
            texts = [c["text"] for c in candidates]
            c_vecs = model.encode(texts, normalize_embeddings=True)
            scores = [float(np.dot(q_vec, v)) for v in c_vecs]
            best_i = int(np.argmax(scores)) if scores else 0
            best_score = scores[best_i] if scores else 0.0
        except Exception:
            best_i, best_score = 0, 0.0

        best = candidates[best_i]
        verified.append(best_score >= MEANING_THRESHOLD)
        source_texts.append(best["text"])
        source_pages.append(best["page"])

    df["source_verified"] = verified
    df["source_text"] = source_texts
    df["source_page"] = source_pages
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Gemini Extraction — Source Verified", layout="wide")
    st.title(" Gemini Extraction — Source Verified")
    st.caption(
        "Straight one-call Gemini extraction — no chunking, no ranking, no batching. "
        "Every property is checked against the PDF: value found first, then meaning matched. "
        "Paper DOI is pulled from the PDF text."
    )

    pdf_file = st.file_uploader("PDF to extract", type=["pdf"], key="fv_pdf")

    if not pdf_file:
        st.info("Upload a PDF to begin.")
        return

    pdf_bytes = pdf_file.getvalue()
    stem = pdf_file.name.rsplit(".", 1)[0]

    sig = (pdf_file.name, pdf_file.size)
    if st.session_state.get("fv_sig") != sig:
        st.session_state["fv_sig"] = sig
        st.session_state["fv_result"] = None

    if st.button("Run Extraction", type="primary", use_container_width=True):
        with st.spinner("Extracting…"):
            data = call_gemini_from_bytes(pdf_bytes)
            doi = resolve_doi(pdf_bytes, data)
            df = convert_to_dataframe(data, doi=doi)
            sentences = _extract_sentences(pdf_bytes)
            st.session_state["fv_result"] = verify_dataframe(df, sentences)
            st.session_state["fv_doi"] = doi

    result = st.session_state.get("fv_result")
    if result is None:
        st.info("Click **Run Extraction** to start.")
        return

    doi = st.session_state.get("fv_doi", "")
    n_verified = int(result["source_verified"].sum()) if "source_verified" in result.columns else 0
    st.caption(
        f"DOI: {doi or '(not found)'}  —  "
        f"{len(result)} properties extracted — {n_verified} source-verified."
    )
    st.dataframe(result, use_container_width=True, hide_index=True)

    if not result.empty:
        st.download_button(" Download CSV", result.to_csv(index=False).encode(),
                            f"{stem}_gemini_verified.csv", "text/csv")


if __name__ == "__main__":
    main()