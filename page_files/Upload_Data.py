import logging
import sys
import os

log = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import io
import json
import tempfile
import base64
import zipfile
import re
from io import BytesIO
import time
import cv2
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

from dotenv import load_dotenv
load_dotenv()

_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not _GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment")

# ── Data extraction (Code 1) ─────────────────────────────────────────────────
# NOTE: DOI is injected locally below (see _extract_doi_from_pdf). If you want
# the DOI produced inside PDF_DataExtraction itself, port the same regex there.
from categorized.Backend.PDF_DataExtraction import (
    call_gemini_from_bytes,
    convert_to_dataframe,
    _extract_sentences,
    verify_dataframe,
)

# ── New integrated stack: image2 → mapper5 → category_push ────────────────────
# These are imported as top-level modules (the same way mapper5 imports image2
# and category_push). Ensure image2.py / mapper5.py / category_push.py are on
# the import path (same directory or added to sys.path above). If you keep them
# under categorized/Backend/, change these three lines to:
#   from categorized.Backend import image2, mapper5, category_push
from categorized.Backend import Pdf_ImageExtraction as image2  # image2.py renamed
from categorized.Backend import mapper5, category_push

# Manual-entry path (unchanged — writes via the existing data_loader).
from data_loader import insert_material_rows


# ─────────────────────────────────────────────────────────────────────────────
# DOI extraction — regex over raw PDF text (format-verified, precision-first)
# ─────────────────────────────────────────────────────────────────────────────

_DOI_CORE_RE = re.compile(r'10\.\d{4,9}/[^\s"<>\]\)]+', re.IGNORECASE)


def _clean_doi(doi: str) -> str:
    if not doi:
        return ""
    doi = doi.strip()
    doi = re.sub(r'^(?:https?://)?(?:dx\.)?doi\.org/', '', doi, flags=re.IGNORECASE)
    doi = re.sub(r'^doi[:\s]+', '', doi, flags=re.IGNORECASE)
    return doi.rstrip(' .,;:)]}>"\'')


def _extract_doi_from_pdf(pdf_bytes: bytes) -> str:
    """Prefer a doi.org URL, then a 'doi:' label, then any bare 10.xxxx/ token."""
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            text = "\n".join((p.get_text("text") or "") for p in doc)
    except Exception:
        return ""
    m = re.search(r'(?:https?://)?(?:dx\.)?doi\.org/(10\.\d{4,9}/[^\s"<>\]\)]+)', text, re.IGNORECASE)
    if m:
        return _clean_doi(m.group(1))
    m = re.search(r'\bdoi[:\s]+\s*(10\.\d{4,9}/[^\s"<>\]\)]+)', text, re.IGNORECASE)
    if m:
        return _clean_doi(m.group(1))
    m = _DOI_CORE_RE.search(text)
    return _clean_doi(m.group(0)) if m else ""


# ─────────────────────────────────────────────────────────────────────────────
# Metadata helper
# ─────────────────────────────────────────────────────────────────────────────

def _df_to_meta(df: pd.DataFrame) -> dict:
    """Re-create the flat metadata dict the UI expects."""
    if df.empty:
        return {}
    row0 = df.iloc[0]
    props = df.to_dict(orient="records")
    return {
        "material_name":         str(row0.get("material_name", "")),
        "material_abbreviation": str(row0.get("material_abbreviation", "")),
        "trade_grade":           str(row0.get("trade_grade", "")),
        "manufacturer":          str(row0.get("manufacturer", "")),
        "doi_url":               str(row0.get("doi_url", "")),
        "mechanical_properties": props,
    }


# ─────────────────────────────────────────────────────────────────────────────
# extract_images adapter — now a thin wrapper over image2.extract_and_verify_plots
# Returns image2's native shape:
#   [{caption, page, image_data:[{array, bytes, filename, subplot_label,
#                                 subplot_caption, source, verification}]}]
# which is exactly what mapper5.map_plots_to_properties and the display expect.
# ─────────────────────────────────────────────────────────────────────────────

def extract_images(pdf_path: str, engines=None, check_missed: bool = True) -> list:
    """Detect + (optionally) recover + crop-verify plots.
    `engines` selects the verifier LLMs: ["gemini"] (default, fast) or
    ["gemini","gpt","claude"] for a majority-vote cross-check that rejects more
    non-plots (needs OPENAI_API_KEY / ANTHROPIC_API_KEY; missing keys just make
    that engine abstain). `check_missed` toggles the per-page Gemini recovery scan."""
    engines = engines or ["gemini"]
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        plot_results, coverage = image2.extract_and_verify_plots(
            pdf_bytes,
            check_missed_plots=check_missed,
            verify_crops=True,
            verify_engines=engines,
        )
        # Drop captions naming photos/micrographs/logos and crops the verifier
        # marked "discard" (keeps "keep"/"recrop"/"unverified"). Non-mutating.
        plot_results = mapper5.select_real_plots(plot_results)
        st.session_state["plot_coverage"] = coverage
    except Exception as e:
        log.error(f"extract_images failed: {e}")
        st.session_state["plot_coverage"] = {}
        return []
    return plot_results


# ─────────────────────────────────────────────────────────────────────────────
# Styles
# ─────────────────────────────────────────────────────────────────────────────

def inject_upload_page_styles():
    st.markdown(
        """
        <style>
            @import url("https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap");

            [data-testid="stHeader"] { display: none !important; }
            .stApp { background: #f3f6fb !important; }
            html, body, [class*="css"] { font-family: "DM Sans", sans-serif !important; }

            .block-container {
                max-width: 980px !important;
                padding-top: 1rem !important;
                padding-bottom: 2rem !important;
            }

            .st-emotion-cache-tn0cau { background: #ffffff !important; }

            div[class*="st-key-ud_main_card"] > div[data-testid="stVerticalBlockBorderWrapper"] > div {
                background: #ffffff !important;
                border: 1px solid #dbe3ee !important;
                border-radius: 16px !important;
                padding: 28px 32px 32px 32px !important;
                box-shadow: 0 4px 24px rgba(15, 23, 42, 0.08) !important;
            }

            div[class*="st-key-ud_main_card"] [data-testid="stVerticalBlockBorderWrapper"] {
                background: #ffffff !important;
                border: 1px solid #dbe3ee !important;
                border-radius: 16px !important;
                box-shadow: 0 4px 24px rgba(15, 23, 42, 0.08) !important;
            }

            span.st-emotion-cache-epvm6 {
                display: flex !important;
                justify-content: center !important;
                width: 100% !important;
            }

            div[class*="st-key-material_ident_card"] [data-testid="stVerticalBlockBorderWrapper"],
            div[class*="st-key-material_form_card"] [data-testid="stVerticalBlockBorderWrapper"] {
                background: transparent !important;
                border: 0 !important;
                border-radius: 0 !important;
                padding: 0 !important;
                box-shadow: none !important;
            }

            div[class*="st-key-material_ident_card"] label p {
                color: #1f2937 !important;
                font-size: 0.95rem !important;
                font-weight: 600 !important;
            }

            div[class*="st-key-material_ident_card"] div[data-baseweb="select"] > div,
            div[class*="st-key-material_ident_card"] div[data-baseweb="input"] > div {
                min-height: 46px !important;
                border-radius: 10px !important;
                border: 1px solid #d6dee8 !important;
                background: #f8fafc !important;
            }

            [data-testid="stFileUploaderDropzone"] {
                background: #f8fbff !important;
                border: 2px dashed #d4deea !important;
                border-radius: 14px !important;
                min-height: 230px !important;
                padding: 1.4rem !important;
                position: relative !important;
                display: flex !important;
                flex-direction: column !important;
                align-items: center !important;
                justify-content: center !important;
            }

            [data-testid="stFileUploaderDropzone"] > div {
                display: flex !important;
                flex-direction: column !important;
                align-items: center !important;
                justify-content: center !important;
                text-align: center !important;
                gap: 10px !important;
                width: 100% !important;
            }

            [data-testid="stFileUploaderDropzone"] button,
            [data-testid="stFileUploaderDropzone"] > div button {
                background: #2f6fe4 !important;
                color: #ffffff !important;
                border: 0 !important;
                border-radius: 9px !important;
                font-weight: 700 !important;
                padding: 0.45rem 1.25rem !important;
                display: block !important;
                margin: 0 auto !important;
            }

            [data-testid="stFileUploaderDropzone"] > span {
                display: flex !important;
                justify-content: center !important;
                width: 100% !important;
                margin-top: 0.5rem !important;
            }

            [data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] {
                width: 100% !important;
                display: flex !important;
                flex-direction: column !important;
                align-items: center !important;
                justify-content: center !important;
                text-align: center !important;
            }

            [data-testid="stFileUploaderDropzone"] small {
                font-size: 0.96rem !important;
                text-align: center !important;
                display: block !important;
            }

            [data-testid="stFileUploaderDropzone"] p,
            [data-testid="stFileUploaderDropzone"] div > p {
                text-align: center !important;
                width: 100% !important;
            }

            .ud-topbar {
                display: flex;
                align-items: center;
                gap: 10px;
                background: #bae1fc;
                border: 4px solid #d7e4f2;
                border-radius: 20px;
                color: #111827;
                font-size: 1.05rem;
                font-weight: 700;
                padding: 12px 14px;
                margin-bottom: 7px;
            }

            .ud-topbar img { width: 20px; height: 20px; object-fit: contain; border-radius: 4px; }

            .ud-ident-title {
                color: #111827; font-size: 2rem; font-weight: 800;
                margin: 4px 0 8px 2px; display: flex; align-items: center; gap: 8px;
            }

            .ud-upload-title {
                color: #111827; font-size: 1.9rem; font-weight: 800;
                margin: 12px 0 8px 0; display: flex; align-items: center; gap: 8px;
            }

            .ud-sec-icon {
                width: 18px; height: 18px; border-radius: 999px;
                background: #2563eb; color: #ffffff; display: inline-flex;
                align-items: center; justify-content: center;
                font-size: 0.72rem; font-weight: 700; line-height: 1;
            }

            .conf-badge {
                display: inline-block;
                padding: 2px 10px;
                border-radius: 99px;
                font-size: 0.78rem;
                font-weight: 700;
                color: #fff;
            }

            .plot-card-meta {
                font-size: 0.82rem;
                color: #64748b;
                margin-bottom: 4px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_bar():
    logo_html = ""
    try:
        with open("logo.png", "rb") as fh:
            logo_b64 = base64.b64encode(fh.read()).decode()
        logo_html = f"<img src='data:image/png;base64,{logo_b64}' alt='AIM'/>"
    except Exception:
        pass
    st.markdown(
        f"<div class='ud-topbar'>{logo_html}<span>AIM Composites</span></div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for the mapping UI
# ─────────────────────────────────────────────────────────────────────────────

def _confidence_badge(conf: str) -> str:
    colors = {"high": "#16a34a", "medium": "#d97706", "low": "#dc2626"}
    c = colors.get((conf or "low").lower(), "#6b7280")
    return f"<span class='conf-badge' style='background:{c}'>{(conf or '').upper()}</span>"


def _score_confidence(score) -> str:
    """Map a numeric match_score to a coarse confidence label for the badge."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "low"
    if s >= 0.75:
        return "high"
    if s >= 0.45:
        return "medium"
    return "low"


# ─────────────────────────────────────────────────────────────────────────────
# Manual input form (unchanged — writes via data_loader.insert_material_rows)
# ─────────────────────────────────────────────────────────────────────────────

def input_form():
    property_categories = {
        "Polymer":   ["Thermal", "Mechanical", "Processing", "Physical", "Descriptive"],
        "Fiber":     ["Mechanical", "Physical", "Thermal", "Descriptive"],
        "Composite": [
            "Mechanical", "Thermal", "Processing", "Physical", "Descriptive",
            "Composition / Reinforcement", "Architecture / Structure",
        ],
    }

    property_names = {
        "Polymer": {
            "Thermal":     ["Glass transition temperature (Tg)", "Melting temperature (Tm)",
                            "Crystallization temperature (Tc)", "Degree of crystallinity",
                            "Decomposition temperature"],
            "Mechanical":  ["Tensile modulus", "Tensile strength", "Elongation at break",
                            "Flexural modulus", "Impact strength"],
            "Processing":  ["Melt flow index (MFI)", "Processing temperature",
                            "Cooling rate", "Mold shrinkage"],
            "Physical":    ["Density", "Specific gravity"],
            "Descriptive": ["Material grade", "Manufacturer"],
        },
        "Fiber": {
            "Mechanical":  ["Tensile modulus", "Tensile strength", "Strain to failure"],
            "Physical":    ["Density", "Fiber diameter"],
            "Thermal":     ["Decomposition temperature"],
            "Descriptive": ["Fiber type", "Surface treatment"],
        },
        "Composite": {
            "Mechanical":  ["Longitudinal modulus (E1)", "Transverse modulus (E2)",
                            "Shear modulus (G12)", "Poissons ratio (V12)",
                            "Tensile strength (fiber direction)", "Interlaminar shear strength"],
            "Thermal":     ["Glass transition temperature (matrix)",
                            "Coefficient of thermal expansion (CTE)"],
            "Processing":  ["Curing temperature", "Curing pressure"],
            "Physical":    ["Density"],
            "Descriptive": ["Laminate type"],
            "Composition / Reinforcement": ["Fiber volume fraction", "Fiber weight fraction",
                                            "Fiber type", "Matrix type"],
            "Architecture / Structure":    ["Weave type", "Ply orientation",
                                            "Number of plies", "Stacking sequence"],
        },
    }

    with st.container(border=False, key="material_ident_card"):
        st.markdown(
            "<div class='ud-ident-title'>"
            "<span class='ud-sec-icon'>i</span>Material Identification</div>",
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            material_class = st.selectbox(
                "Material Class", ("Polymer", "Fiber", "Composite"),
                index=None, placeholder="Choose material class",
                key="manual_material_class",
            )
        with col_b:
            if material_class:
                property_category = st.selectbox(
                    "Property Type", property_categories[material_class],
                    index=None, placeholder="Choose property type",
                    key="manual_property_category",
                )
            else:
                property_category = None
                st.selectbox(
                    "Property Type", ["Choose material class first"],
                    index=0, disabled=True,
                    key="manual_property_category_disabled",
                )

        property_name = None
        if material_class and property_category:
            property_options = property_names[material_class][property_category] + ["Something else"]
            property_name = st.selectbox(
                "Property Name", property_options,
                index=None, placeholder="Choose property",
                key="manual_property_name",
            )

        custom_property_name = ""
        if property_name == "Something else":
            custom_property_name = st.text_input(
                "Custom Property Name", placeholder="Type property name",
                key="manual_custom_property_name",
            ).strip()

        selected_property_name = (
            custom_property_name if property_name == "Something else" else property_name
        )

    if material_class and property_category and selected_property_name:
        with st.container(border=False, key="material_form_card"):
            with st.form("user_input"):
                st.subheader("Enter Data")
                material_name  = st.text_input("Material Name")
                material_abbr  = st.text_input("Material Abbreviation")
                value          = st.text_input("Value")
                unit           = st.text_input("Unit (SI)")
                english        = st.text_input("English Units")
                test_condition = st.text_input("Test Condition")
                comments       = st.text_area("Comments")
                submitted      = st.form_submit_button("Submit")

                if submitted:
                    if not (material_name and value):
                        st.error("Material name and value are required.")
                        return False

                    input_db = pd.DataFrame([{
                        "material_class":        material_class,
                        "material_name":         material_name,
                        "material_abbreviation": material_abbr,
                        "section":               property_category,
                        "property_name":         selected_property_name,
                        "value":                 value,
                        "unit":                  unit,
                        "english":               english,
                        "test_condition":        test_condition,
                        "comments":              comments,
                    }])

                    try:
                        inserted = insert_material_rows(input_db)
                    except Exception as exc:
                        st.error(f"Failed to save to PostgreSQL: {exc}")
                        return False

                    if inserted <= 0:
                        st.error("No rows were inserted into PostgreSQL.")
                        return False

                    st.cache_data.clear()
                    st.success("Property added successfully to PostgreSQL.")
                    st.dataframe(input_db)
                    return True

                return False

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Material Data  (extraction + DOI + RDS category selection)
# ─────────────────────────────────────────────────────────────────────────────

_STAGE_LABELS = {
    0.00: ("Checking cache",           2),
    0.30: ("Extracting via Gemini",   20),
    0.70: ("Verifying against source", 8),
    1.00: ("Done",                     0),
}


def _nearest_stage_label(pct: float):
    best_key = min(_STAGE_LABELS, key=lambda k: abs(k - pct))
    return _STAGE_LABELS[best_key]


def render_material_data_tab(pdf_path: str):
    st.subheader("Material Properties Data")

    if not st.session_state.pdf_data_extracted:
        bar    = st.progress(0.0)
        status = st.empty()
        timer  = st.empty()
        start_ts = time.time()

        def _cb(msg: str, pct: float):
            elapsed = time.time() - start_ts
            label, est_remaining = _nearest_stage_label(pct)
            bar.progress(min(pct, 1.0))
            status.markdown(
                f"**{label}** &nbsp;·&nbsp; <span style='color:#64748b'>{msg}</span>",
                unsafe_allow_html=True,
            )
            timer.caption(
                f"⏱ Elapsed: {elapsed:.0f}s"
                + (f" · Est. remaining: ~{est_remaining}s" if est_remaining > 0 else "")
            )

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        _cb("Extracting via Gemini…", 0.30)
        data = call_gemini_from_bytes(pdf_bytes)
        df = convert_to_dataframe(data)

        # DOI — regex over the PDF text, stored as a full URL (single column).
        doi = _extract_doi_from_pdf(pdf_bytes)
        doi_url = f"https://doi.org/{doi}" if doi else ""
        if not df.empty:
            df["doi_url"] = doi_url

        if not df.empty:
            _cb("Verifying against source PDF…", 0.70)
            sentences = _extract_sentences(pdf_bytes)
            df = verify_dataframe(df, sentences)

        _cb("Done.", 1.0)
        elapsed_total = time.time() - start_ts
        bar.progress(1.0)
        status.empty()
        timer.empty()

        if not df.empty:
            st.session_state.pdf_extracted_df   = df
            st.session_state.pdf_data_extracted = True
            st.session_state.pdf_extracted_meta = _df_to_meta(df)
            st.session_state.pdf_doi            = doi_url
            st.success(f" Extracted {len(df)} properties in {elapsed_total:.0f}s")
        else:
            st.warning("No data extracted from PDF.")
            return

    df = st.session_state.pdf_extracted_df
    if df.empty:
        return

    meta = st.session_state.get("pdf_extracted_meta", {})
    doi  = st.session_state.get("pdf_doi", "")

    c1, c2, c3 = st.columns(3)
    c1.metric("Material",     meta.get("material_name",         "N/A"))
    c2.metric("Abbreviation", meta.get("material_abbreviation", "N/A"))
    c3.metric("DOI",          doi or "—")

    st.dataframe(df, use_container_width=True, height=400)

    st.subheader("Assign Material Category")
    st.selectbox(
        "Category table (routes to the RDS table on push)",
        category_push.CATEGORY_TABLES,   # Composites_materials / Fibers / Polymers
        index=None,
        placeholder="Required before pushing to the database",
        key="push_category",
        help="This is the RDS table the mapped rows are pushed into (from the "
             "Extracted Plots tab).",
    )
    if st.session_state.get("push_category"):
        st.caption(
            f"Category **{st.session_state['push_category']}** selected. "
            "Go to the **Extracted Plots** tab to map figures and push to RDS."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Extracted Plots → map (mapper5) → store (SQLite) → push (category_push)
# ─────────────────────────────────────────────────────────────────────────────

_OUT_DIR = os.getenv("AIM_OUT_DIR", os.path.abspath("./aim_efrc_db"))
_SOURCE_TABLE = "gemini_verified"   # provenance label for the stored rows


def _links_for_group(links_df: pd.DataFrame, gi: int) -> pd.DataFrame:
    if links_df is None or links_df.empty or "group_idx" not in links_df.columns:
        return pd.DataFrame()
    return links_df[links_df["group_idx"] == gi]


def _reset_plot_mapping():
    """Structural edits (removing a crop/figure) shift the positional group_idx
    that links_df is keyed on, so any edit invalidates the current mapping.
    Clearing it forces a clean re-map on the pruned set."""
    st.session_state.mapping_done = False
    st.session_state.links_df     = pd.DataFrame()
    st.session_state.df_aug       = pd.DataFrame()
    st.session_state.store        = None

import os
from categorized.Backend import category_push as cp
st.write({
    "S3_BUCKET_env": os.getenv("S3_BUCKET"),
    "cp.S3_BUCKET": cp.S3_BUCKET,
    "_s3_ready": cp._s3_ready(),
})


def render_plots_tab(pdf_path: str, paper_id: str):
    st.subheader("Extracted Plot Images & Property Mapping")

    # ── Extraction settings: multi-LLM cross-check + recovery toggle ──────────
    with st.expander("⚙ Extraction settings", expanded=not st.session_state.pdf_processed):
        multi = st.checkbox(
            "Cross-check each figure with GPT + Claude (majority vote — rejects more "
            "non-plots, but ~3× slower and needs OPENAI_API_KEY / ANTHROPIC_API_KEY)",
            value=st.session_state.get("verify_multi", False),
            key="verify_multi",
        )
        recover = st.checkbox(
            "Missed-plot recovery (per-page Gemini scan — catches plots OpenCV missed, slower)",
            value=st.session_state.get("recover_missed", True),
            key="recover_missed",
        )
        if multi and not (os.getenv("OPENAI_API_KEY") and os.getenv("ANTHROPIC_API_KEY")):
            st.caption(" One or both of OPENAI_API_KEY / ANTHROPIC_API_KEY is not set — "
                       "that engine will abstain; the vote falls back to the ones available.")
        if st.session_state.pdf_processed and st.button(" Re-extract with these settings"):
            st.session_state.pdf_processed = False
            _reset_plot_mapping()
            st.rerun()

    # 1) Extract plots once (detection + recovery + crop verification)
    if not st.session_state.pdf_processed:
        engines = ["gemini", "gpt", "claude"] if st.session_state.get("verify_multi") else ["gemini"]
        with st.spinner("Extracting plots from PDF…"
                        + (" (cross-checking with GPT + Claude — this is slower)"
                           if st.session_state.get("verify_multi") else "")):
            st.session_state.plot_results  = extract_images(
                pdf_path, engines=engines,
                check_missed=st.session_state.get("recover_missed", True),
            )
            st.session_state.pdf_processed = True
            _reset_plot_mapping()

    plot_results = st.session_state.plot_results
    if not plot_results:
        st.warning("No plots found in this PDF.")
        return

    plot_results = st.session_state.plot_results
    if not plot_results:
        st.warning("No plots found in this PDF.")
        return

    df = st.session_state.pdf_extracted_df
    has_data = not df.empty
    n_imgs = sum(len(g.get("image_data", [])) for g in plot_results)

    cov = st.session_state.get("plot_coverage", {}) or {}
    if has_data:
        mat_abbr = df.iloc[0]["material_abbreviation"]
        st.info(
            f"**{len(plot_results)} figures / {n_imgs} crops** extracted  |  "
            f"Material: **{mat_abbr}**  |  {df['property_name'].nunique()} properties  |  "
            f"recovered: {cov.get('total_recovered', 0)}"
        )
    else:
        st.warning("Extract material data in the **Material Data** tab first to enable mapping.")

    # Downloads (image2's native zipper)
    d1, d2 = st.columns(2)
    d1.download_button(
        " Images + metadata (ZIP)",
        data=image2.create_plot_zip(plot_results, include_json=True),
        file_name=f"{paper_id}_plots.zip", mime="application/zip",
        use_container_width=True, key="dl_plots_zip",
    )
    d2.download_button(
        " Images only (ZIP)",
        data=image2.create_plot_zip(plot_results, include_json=False),
        file_name=f"{paper_id}_images.zip", mime="application/zip",
        use_container_width=True, key="dl_images_zip",
    )

    st.divider()

    # 2) Map properties → figures (mapper5 four-signal cascade)
    if has_data:
        cA, cB = st.columns([0.6, 0.4])
        run_map = cA.button(
            " Map properties → figures",
            type="primary",
            disabled=st.session_state.get("mapping_done", False),
            use_container_width=True,
        )
        if st.session_state.get("mapping_done"):
            if cB.button("↺ Re-run mapping", use_container_width=True):
                st.session_state.mapping_done = False
                st.session_state.links_df = pd.DataFrame()
                st.session_state.df_aug = pd.DataFrame()
                st.session_state.store = None
                st.rerun()

        if run_map:
            with st.spinner("Fusing figure-citation → page → SciBERT → token signals…"):
                links_df, df_aug = mapper5.map_plots_to_properties(df, plot_results)
            st.session_state.links_df     = links_df
            st.session_state.df_aug       = df_aug
            st.session_state.mapping_done = True
            n_mapped = int((df_aug.get("map_score", pd.Series(dtype=str)).astype(str) != "").sum()) \
                if not df_aug.empty else 0
            st.success(f" {n_mapped}/{len(df)} property rows linked to a figure "
                       f"({len(links_df)} total link(s)).")
            st.rerun()

    links_df = st.session_state.get("links_df", pd.DataFrame())
    mapping_done = st.session_state.get("mapping_done", False)

    st.divider()

    # 3) Figure-centric review — each figure with delete/remove controls and
    #    (after mapping) the property rows linked to it.
    st.caption("Review the crops below. Remove a bad crop with ** **, or drop a whole "
               "figure with ** **. Editing figures clears the current mapping — re-run "
               "mapping afterwards.")
    for gi, group in enumerate(plot_results):
        caption = group.get("caption", f"Figure {gi+1}")
        page    = group.get("page", "?")
        imgs    = group.get("image_data", [])

        with st.container(border=True):
            head, dele = st.columns([0.85, 0.15])
            head.markdown(f"**Page {page}** — {caption}")
            if dele.button(" Delete figure", key=f"delfig_{gi}", use_container_width=True):
                plot_results.pop(gi)
                st.session_state.plot_results = plot_results
                _reset_plot_mapping()
                st.rerun()

            icols = st.columns(min(len(imgs), 4) or 1)
            for pos, im in enumerate(imgs):
                with icols[pos % len(icols)]:
                    arr = im.get("array")
                    if arr is not None:
                        sub = im.get("subplot_label") or ""
                        st.image(arr, channels="BGR", width=200,
                                 caption=(sub or None))

                    v = im.get("verification") or {}
                    act = v.get("majority_action") if isinstance(v, dict) else None
                    if act and act != "keep":
                        st.caption(f"crop verdict: **{act}**")
                    # per-engine detail when the multi-LLM cross-check ran
                    by = v.get("by_engine") if isinstance(v, dict) else None
                    if isinstance(by, dict):
                        flagged = [e for e, vd in by.items()
                                   if isinstance(vd, dict) and vd.get("recommended_action") == "discard"]
                        if flagged:
                            st.caption("flagged non-plot by: " + ", ".join(flagged))
                    elif act in (None, "unverified"):
                        st.caption("crop verdict: unverified")

                    if st.button("✕ Remove", key=f"rmimg_{gi}_{pos}", use_container_width=True):
                        imgs.pop(pos)
                        if not imgs:
                            plot_results.pop(gi)
                        st.session_state.plot_results = plot_results
                        _reset_plot_mapping()
                        st.rerun()

            if mapping_done:
                grp = _links_for_group(links_df, gi)
                if grp.empty:
                    st.caption("No property linked to this figure.")
                else:
                    hdr = st.columns([3, 1.4, 1, 2, 1.2, 1, 0.7])
                    for h, t in zip(hdr, ["property", "value", "unit", "material",
                                          "subplot", "score", ""]):
                        h.caption(t)
                    for _, l in grp.iterrows():
                        row = st.columns([3, 1.4, 1, 2, 1.2, 1, 0.7])
                        row[0].write(str(l.get("property_name", "")))
                        row[1].write(str(l.get("value", "")))
                        row[2].write(str(l.get("unit", "")))
                        row[3].write(str(l.get("material_name", "")))
                        row[4].write(str(l.get("matched_subplot") or ""))
                        row[5].markdown(
                            _confidence_badge(_score_confidence(l.get("match_score"))),
                            unsafe_allow_html=True,
                        )
                        key = f"rmlink_{gi}_{int(l.get('prop_row', 0))}_{int(l.get('match_rank', 0))}"
                        if row[6].button("✕", key=key, help="Remove this link"):
                            mask = ~(
                                (links_df["group_idx"] == gi)
                                & (links_df["prop_row"] == l["prop_row"])
                                & (links_df["match_rank"] == l["match_rank"])
                            )
                            st.session_state.links_df = links_df[mask].reset_index(drop=True)
                            # rebuild mapped_* columns from the pruned links
                            st.session_state.df_aug = mapper5._apply_links(
                                df, st.session_state.links_df
                            )
                            st.rerun()

    if not (has_data and mapping_done):
        return

    st.divider()

    # 4) Store to SQLite (saves crops + attaches plot_image_path), then push to RDS
    st.markdown("**Store the mapped result, then push to RDS**")
    st.caption("Store writes the mapped rows + saved crops to a local SQLite DB and "
               "attaches each crop's path. Push conforms those rows to the chosen "
               "category table's 33-column schema and uploads them.")

    if st.button(" Store to database (SQLite)", use_container_width=True):
        with st.spinner("Saving crops and writing SQLite…"):
            st.session_state.store = mapper5.store_properties_with_plots(
                st.session_state.df_aug,
                st.session_state.links_df,
                plot_results,
                out_dir=_OUT_DIR,
                pdf_stem=paper_id,
                source_table=_SOURCE_TABLE,
            )
        s = st.session_state.store
        st.success(
            f"Stored {s['n_properties']} row(s) → {s['db_filename']}; "
            f"{s['n_properties_with_plot']} with a plot, {s['n_images_saved']} crop(s)."
        )
        st.rerun()

    store = st.session_state.get("store")

    st.markdown("** Push to RDS database**")
    if not mapper5.db_configured():
        st.caption("Set DB_NAME / DB_USER / DB_PASSWORD in your `.env` "
                   "(host is already configured) to enable the RDS push.")
        return

    category = st.session_state.get("push_category")
    if not category:
        category = st.selectbox(
            "Category table (routes by material type)",
            category_push.CATEGORY_TABLES,
            key="push_category_tab2",
            help="Rows conform to this table's 33-col schema; columns it can't hold "
                 "go to <table>_extras; matched crops copy to rds_plots/<category>/<stem>/.",
        )
    embed_img = st.checkbox("Embed matched plot into the table's image column", value=True)

    cT, cP = st.columns(2)
    if cT.button(" Test connection", use_container_width=True):
        try:
            mapper5.db_healthcheck()
            st.success("Connected to RDS.")
        except Exception as e:
            st.error(f"Connection failed: {e}")

    push_ready = bool(store) and store.get("n_properties", 0) > 0
    if cP.button(f" Push to {category}", type="primary",
                 use_container_width=True, disabled=not push_ready):
        try:
            with st.spinner(f"Conforming + writing rows to '{category}'…"):
                res = category_push.push_by_category(
                    store, category, mapper5.get_db_engine(), embed_image=embed_img
                )
            st.success(
                f"Pushed {res['pushed']} row(s) to '{res['table']}', "
                f"{res['extras']} to '{res['extras_table']}', "
                f"{res['plots_copied']} crop(s) → {res['plots_dir']}."
            )
            st.cache_data.clear() 
        except Exception as e:
            st.error(f"Push failed: {e}")
    if not push_ready:
        st.caption("Click ** Store to database (SQLite)** first — the push uploads those stored rows.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    inject_upload_page_styles()
    render_top_bar()

    st.subheader("Submit Scientific Material")
    st.caption("Provide technical data and research documentation for the central repository.")

    defaults = {
        "plot_results":       [],
        "plot_coverage":      {},
        "links_df":           pd.DataFrame(),
        "df_aug":             pd.DataFrame(),
        "store":              None,
        "pdf_processed":      False,
        "mapping_done":       False,
        "current_pdf_name":   None,
        "form_submitted":     False,
        "pdf_data_extracted": False,
        "pdf_extracted_df":   pd.DataFrame(),
        "pdf_extracted_meta": {},
        "pdf_doi":            "",
        "verify_multi":       False,
        "recover_missed":     True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    with st.container(border=True, key="ud_main_card"):
        if input_form():
            st.session_state.form_submitted = True

        st.markdown(
            "<div class='ud-upload-title'>"
            "<span class='ud-sec-icon'>i</span>Research Documentation</div>",
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload PDF (Material Datasheet or Research Paper)", type=["pdf"]
        )

        if not uploaded_file:
            st.info("Upload a PDF to extract material data and plots")

    if not uploaded_file:
        for k, v in defaults.items():
            st.session_state[k] = v
        return

    paper_id = os.path.splitext(uploaded_file.name)[0].replace(" ", "_")

    if st.session_state.current_pdf_name != uploaded_file.name:
        for k, v in defaults.items():
            st.session_state[k] = v
        st.session_state.current_pdf_name = uploaded_file.name

    if st.session_state.form_submitted:
        st.session_state.form_submitted = False
        st.info("Form submitted. Upload again to process a new PDF.")
        st.tabs(["Material Data", "Extracted Plots"])
        return

    tab1, tab2 = st.tabs([" Material Data", " Extracted Plots"])

    tmp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, prefix="matdb_")
    try:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_file.flush()
        tmp_file.close()
        pdf_path = tmp_file.name

        with tab1:
            render_material_data_tab(pdf_path)
        with tab2:
            render_plots_tab(pdf_path, paper_id)
    finally:
        try:
            os.unlink(tmp_file.name)
        except Exception:
            pass


main()