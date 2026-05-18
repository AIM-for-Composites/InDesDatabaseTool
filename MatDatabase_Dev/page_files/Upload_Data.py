from logging import log
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import io
import json
import tempfile
import base64
import zipfile
import re
from io import BytesIO

import cv2
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

# ── imports from doctodb_rag (data extraction) ────────────────────────────────
from categorized.Backend.Pdf_DataExtraction import run_pipeline

# ── imports from figure_extractor (image extraction) ─────────────────────────
from categorized.Backend.Pdf_ImageExtraction import (
    init_gemini,
    get_plot_data_from_llm,
    extract_plots,
)

from data_loader import insert_material_rows
from categorized.Backend.plot_property_mapper import (
    batch_map_plots,
    fetch_properties_for_material,
    save_plot_image_mapping,
    save_plot_image_to_db,
)
from db import fetch_all


# ─────────────────────────────────────────────────────────────────────────────
# Helpers that were previously in upload_backend
# ─────────────────────────────────────────────────────────────────────────────

def _df_to_meta(df: pd.DataFrame) -> dict:
    """Re-create the flat metadata dict that the UI previously got from Gemini."""
    if df.empty:
        return {}
    row0 = df.iloc[0]
    props = df.to_dict(orient="records")
    return {
        "material_name":         str(row0.get("material_name", "")),
        "material_abbreviation": str(row0.get("material_abbreviation", "")),
        "trade_grade":           str(row0.get("trade_grade", "")),
        "manufacturer":          str(row0.get("manufacturer", "")),
        "mechanical_properties": props,
    }


def create_zip(image_results: list, include_json: bool = True) -> bytes:
    """
    Pack extracted plot images (and optional JSON metadata) into a ZIP.
    Each item in image_results has: caption, page, image_data (list of dicts
    with 'array' (BGR ndarray) and 'filename').
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        meta = []
        for item in image_results:
            caption = item.get("caption", "")
            page    = item.get("page", "?")
            for img_dict in item.get("image_data", []):
                bgr      = img_dict.get("array")
                filename = img_dict.get("filename", "plot.png")
                if bgr is not None:
                    ok, enc = cv2.imencode(".png", bgr)
                    if ok:
                        zf.writestr(filename, enc.tobytes())
            if include_json:
                meta.append({
                    "caption":     caption,
                    "page":        page,
                    "image_count": len(item.get("image_data", [])),
                    "images":      [d.get("filename") for d in item.get("image_data", [])],
                })
        if include_json and meta:
            zf.writestr("metadata.json", json.dumps(meta, indent=4))
    return buf.getvalue()


def save_matched_images(
    df: pd.DataFrame,
    image_results: list,
    save_dir: str = "images",
) -> list:
    """
    Heuristically match extracted plot captions to property names in df and
    save matched images to disk. Returns list of match-info dicts.
    """
    os.makedirs(save_dir, exist_ok=True)
    saved = []
    props = df["property_name"].str.lower().tolist() if "property_name" in df.columns else []

    for item in image_results:
        caption  = (item.get("caption") or "").lower()
        best_prop = None
        best_score = 0
        for prop in props:
            # simple overlap score: shared words
            cap_words  = set(re.findall(r"\w+", caption))
            prop_words = set(re.findall(r"\w+", prop))
            score = len(cap_words & prop_words)
            if score > best_score:
                best_score = score
                best_prop  = prop

        if best_prop and best_score > 0:
            for idx, img_dict in enumerate(item.get("image_data", [])):
                bgr = img_dict.get("array")
                if bgr is None:
                    continue
                safe_prop = re.sub(r"[^\w\-]", "_", best_prop)
                filename  = f"{safe_prop}_{idx}.png"
                filepath  = os.path.join(save_dir, filename)
                cv2.imwrite(filepath, bgr)
                saved.append({
                    "property": best_prop,
                    "caption":  item.get("caption", ""),
                    "path":     filepath,
                })
    return saved


def save_single_image_with_property(
    bgr: np.ndarray,
    property_name: str,
    save_dir: str = "images",
) -> str:
    """Save a single BGR image tagged with a property name. Returns filepath."""
    os.makedirs(save_dir, exist_ok=True)
    safe = re.sub(r"[^\w\-]", "_", property_name)
    filepath = os.path.join(save_dir, f"{safe}.png")
    cv2.imwrite(filepath, bgr)
    return filepath


# ─────────────────────────────────────────────────────────────────────────────
# extract_images adapter
# Bridges figure_extractor's extract_plots API to the image_results list shape
# expected by the rest of the UI (list of {caption, page, image_data}).
# ─────────────────────────────────────────────────────────────────────────────

_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBzyMFKEqcjsWpR-OGAY42T250o1O39v3Y")

def extract_images(pdf_path: str) -> list:
    """
    Use figure_extractor to detect and crop plot images from a PDF path.
    Returns a list compatible with the image_results shape used throughout the UI:
      [{ "caption": str, "page": int, "image_data": [{"array": bgr_ndarray, "filename": str}] }]
    """
    try:
        gemini_model = init_gemini(_GEMINI_API_KEY)
        plot_data    = get_plot_data_from_llm(gemini_model, pdf_path)
        raw_plots    = extract_plots(
            pdf_path=pdf_path,
            plot_data=plot_data,
            pad=22,
            score_thresh=0.35,
        )
    except Exception as e:
        log.error(f"extract_images failed: {e}")
        return []




    # raw_plots items: {caption, page, path, plot_score, plot_type}
    # Convert to image_results shape
    image_results = []
    for item in raw_plots:
        bgr = cv2.imread(item["path"]) if item.get("path") else None
        # clean up temp file written by extract_plots
        if item.get("path") and os.path.exists(item["path"]):
            try:
                os.remove(item["path"])
            except Exception:
                pass

        page    = item.get("page", 1)
        caption = item.get("caption", f"Figure (page {page})")
        safe    = re.sub(r"[^\w\-]", "_", caption)[:40]
        filename = f"page{page}_{safe}.png"

        image_results.append({
            "caption":    caption,
            "page":       page,
            "image_data": [{"array": bgr, "filename": filename}] if bgr is not None else [],
        })

    return image_results


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
# Helpers for tab2 mapping UI
# ─────────────────────────────────────────────────────────────────────────────

def _confidence_badge(conf: str) -> str:
    colors = {"high": "#16a34a", "medium": "#d97706", "low": "#dc2626"}
    c = colors.get((conf or "low").lower(), "#6b7280")
    return (
        f"<span class='conf-badge' style='background:{c}'>"
        f"{conf.upper()}</span>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Manual input form
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
# Tab 1: Material Data
# Uses run_pipeline from doctodb_rag instead of call_gemini_from_bytes
# ─────────────────────────────────────────────────────────────────────────────

def render_material_data_tab(pdf_path: str):
    st.subheader("Material Properties Data")

    if not st.session_state.pdf_data_extracted:
        with st.spinner("Extracting material data…"):
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            # run_pipeline returns (df, chunks, api_errors, meta)
            df, _chunks, api_errors, meta = run_pipeline(pdf_bytes)

            if api_errors:
                for err in api_errors:
                    st.warning(err)

            if not df.empty:
                # Build the metadata dict that the rest of the UI expects
                data = _df_to_meta(df)
                st.session_state.pdf_extracted_df   = df
                st.session_state.pdf_data_extracted = True
                st.session_state.pdf_extracted_meta = data
            else:
                st.warning("No data extracted from PDF.")

    df = st.session_state.pdf_extracted_df

    if df.empty:
        return

    meta = st.session_state.get("pdf_extracted_meta", {})
    st.success(f"Extracted {len(df)} properties")

    col1, col2 = st.columns(2)
    col1.metric("Material",     meta.get("material_name",         "N/A"))
    col2.metric("Abbreviation", meta.get("material_abbreviation", "N/A"))

    st.dataframe(df, use_container_width=True, height=400)
    st.subheader("Assign Material Category")

    extracted_material_class = st.selectbox(
        "Select category for this material",
        ["Polymer", "Fiber", "Composite"],
        index=None,
        placeholder="Required before adding to database",
        key="tab1_material_class",
    )

    if st.button("+ Add to Database"):
        if not extracted_material_class:
            st.error("Please select a material category before adding.")
            return

        df["material_class"] = extracted_material_class
        df["material_type"]  = extracted_material_class

        if st.session_state.image_results:
            with st.spinner("Saving matched plot images…"):
                saved_images = save_matched_images(
                    df, st.session_state.image_results, save_dir="images"
                )
                if saved_images:
                    st.success(f"Saved {len(saved_images)} plot image(s)")
                    with st.expander("View saved images"):
                        for img_info in saved_images:
                            st.write(f"**{img_info['property']}** → {img_info['caption']}")
                            st.write(f"Saved to: `{img_info['path']}`")
                else:
                    st.info("No plots matched the extracted properties automatically.")

        st.session_state.setdefault("user_uploaded_data", pd.DataFrame())
        st.session_state["user_uploaded_data"] = pd.concat(
            [st.session_state["user_uploaded_data"], df], ignore_index=True
        )
        st.success(f"Added to {extracted_material_class} database!")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Extracted Plots + AI Property Mapping
# Uses extract_images (adapter above) instead of upload_backend's version
# ─────────────────────────────────────────────────────────────────────────────

def render_plots_tab(pdf_path: str, paper_id: str):
    st.subheader("Extracted Plot Images & Property Mapping")

   
    if not st.session_state.pdf_processed:
        with st.spinner("Extracting plots from PDF…"):
            st.session_state.image_results = extract_images(pdf_path)
            st.session_state.pdf_processed = True
            st.session_state.mapping_done  = False

    image_results = st.session_state.image_results

    if not image_results:
        st.warning("No plots found in this PDF.")
        return

    has_data = not st.session_state.pdf_extracted_df.empty

    if has_data:
        mat_abbr      = st.session_state.pdf_extracted_df.iloc[0]["material_abbreviation"]
        property_list = st.session_state.pdf_extracted_df["property_name"].unique().tolist()
        st.info(
            f"**{len(image_results)} plots** extracted  |  "
            f"Material: **{mat_abbr}**  |  "
            f"{len(property_list)} properties available for mapping"
        )
    else:
        st.warning(
            "Extract material data in the **Material Data** tab first "
            "to enable AI property mapping."
        )

    subtab_images, subtab_json = st.tabs(["🖼 Images & Mapping", "{ } JSON Preview"])

    # ════════════════════════════════════════════════════════════════════════
    with subtab_images:

        col_img, col_json_dl, col_all = st.columns(3)
        with col_img:
            st.download_button(
                "⬇ Images Only",
                data=create_zip(image_results, include_json=False),
                file_name=f"{paper_id}_images.zip",
                mime="application/zip",
                use_container_width=True,
                key="dl_images",
            )
        with col_json_dl:
            json_meta = [
                {"caption": r["caption"], "page": r["page"],
                 "image_count": len(r["image_data"])}
                for r in image_results
            ]
            st.download_button(
                "⬇ JSON",
                data=json.dumps(json_meta, indent=4),
                file_name=f"{paper_id}_metadata.json",
                mime="application/json",
                use_container_width=True,
                key="dl_json",
            )
        with col_all:
            st.download_button(
                "⬇ Download All",
                data=create_zip(image_results, include_json=True),
                file_name=f"{paper_id}_complete.zip",
                mime="application/zip",
                use_container_width=True,
                key="dl_all",
            )

        st.divider()

        if has_data:
            col_cls, col_btn = st.columns([0.45, 0.55])

            with col_cls:
                map_class = st.selectbox(
                    "Material class for DB lookup",
                    ["Polymer", "Fiber", "Composite"],
                    key="mapping_material_class",
                    help="Routes to the correct PostgreSQL table.",
                )

            with col_btn:
                st.write("")
                st.write("")
                run_mapping = st.button(
                    "🤖  Run AI Property Mapping",
                    type="primary",
                    disabled=st.session_state.get("mapping_done", False),
                    use_container_width=True,
                )

            if run_mapping:
                df           = st.session_state.pdf_extracted_df
                mat_abbr     = df.iloc[0]["material_abbreviation"]
                extracted_json = st.session_state.get("pdf_extracted_meta", {})

                with st.spinner("Fetching properties from PostgreSQL…"):
                    try:
                        db_properties = fetch_properties_for_material(
                            mat_abbr, map_class, fetch_all
                        )
                    except Exception as exc:
                        st.error(f"DB error: {exc}")
                        db_properties = []

                if not db_properties:
                    st.warning(
                        f"No DB rows found for **{mat_abbr}** in the **{map_class}** table. "
                        "Mapping will use all available properties from the extracted data."
                    )

                prog = st.progress(0, text="Starting…")

                def _on_progress(i, total, caption):
                    pct = int((i / max(total, 1)) * 100)
                    prog.progress(pct, text=f"Mapping {i+1}/{total}: {caption[:55]}…")

                with st.spinner("AI is analysing plots…"):
                    mapped = batch_map_plots(
                        image_results=image_results,
                        extracted_json=extracted_json,
                        db_properties=db_properties,
                        progress_callback=_on_progress,
                    )

                prog.progress(100, text="Done ✓")
                st.session_state.mapped_results = mapped
                st.session_state.mapping_done   = True
                st.success(f"✅  Mapped {len(mapped)} plots — review below.")
                st.rerun()

            if st.session_state.get("mapping_done"):
                col_info, col_reset = st.columns([0.78, 0.22])
                col_info.caption(
                    "AI mapping complete. The dropdown for each plot is pre-filled "
                    "with the suggestion — override freely, then hit **Save**."
                )
                if col_reset.button("↺ Re-run Mapping", use_container_width=True):
                    st.session_state.mapping_done   = False
                    st.session_state.mapped_results = []
                    st.rerun()

        st.divider()

        use_mapped = (
            has_data
            and st.session_state.get("mapping_done", False)
            and bool(st.session_state.get("mapped_results"))
        )
        display_list = (
            st.session_state.mapped_results if use_mapped else image_results
        )

        for idx in range(len(display_list)):
            if idx >= len(display_list):
                break

            item     = display_list[idx]
            caption  = item.get("caption", f"Figure {idx+1}")
            page     = item.get("page", "?")
            img_list = item.get("image_data", [])
            mapping  = item.get("mapping_result") if use_mapped else None

            with st.container(border=True):

                col_cap, col_del = st.columns([0.87, 0.13])
                col_cap.markdown(f"**Page {page}** — {caption}")
                if col_del.button("🗑", key=f"del_grp_{idx}", help="Delete this figure"):
                    display_list.pop(idx)
                    if use_mapped:
                        st.session_state.mapped_results = display_list
                    else:
                        st.session_state.image_results  = display_list
                    st.rerun()

                if mapping:
                    prop_name  = mapping.get("property_name", "")
                    section    = mapping.get("section", "")
                    confidence = mapping.get("confidence", "low")
                    reasoning  = mapping.get("reasoning", "")
                    db_row     = mapping.get("db_row")
                    candidates = mapping.get("all_candidates", [])

                    if prop_name:
                        badge = _confidence_badge(confidence)
                        st.markdown(
                            f"🔗 **AI Match:** `{section}` › **{prop_name}** &nbsp; {badge}",
                            unsafe_allow_html=True,
                        )
                        if reasoning:
                            st.caption(f"💬 {reasoning}")

                        if db_row:
                            with st.expander("📋 Matched DB row", expanded=False):
                                c1, c2, c3 = st.columns(3)
                                c1.metric("Value",     db_row.get("value",          "—"))
                                c2.metric("Unit",      db_row.get("unit",           "—"))
                                c3.metric("Condition", db_row.get("test_condition", "—"))
                                if db_row.get("comments"):
                                    st.caption(f"Comments: {db_row['comments']}")
                                if db_row.get("english"):
                                    st.caption(f"English units: {db_row['english']}")

                        if candidates:
                            with st.expander("🔄 All candidates", expanded=False):
                                for c in candidates:
                                    st.markdown(
                                        f"{c.get('rank','?')}. `{c.get('section','?')}` › "
                                        f"**{c.get('property_name','?')}** &nbsp; "
                                        f"{_confidence_badge(c.get('confidence','low'))}",
                                        unsafe_allow_html=True,
                                    )
                    else:
                        st.warning("⚠️ AI could not match this plot to any DB property.")

                for p_idx in range(len(img_list)):
                    if p_idx >= len(item.get("image_data", [])):
                        break

                    img_data = item["image_data"][p_idx]
                    bgr      = img_data.get("array")
                    if bgr is None:
                        continue

                    img_key = f"{idx}_{p_idx}_{page}"
                    st.image(bgr, channels="BGR", width=420)

                    if has_data:
                        df            = st.session_state.pdf_extracted_df
                        mat_abbr      = df.iloc[0]["material_abbreviation"]
                        property_list = df["property_name"].unique().tolist()
                        options       = ["— Select property —"] + property_list

                        ai_prop     = mapping.get("property_name", "") if mapping else ""
                        ai_section  = mapping.get("section", "")        if mapping else ""
                        default_idx = (
                            property_list.index(ai_prop) + 1
                            if ai_prop in property_list else 0
                        )

                        col_sel, col_sec, col_save, col_rem = st.columns(
                            [0.40, 0.20, 0.20, 0.20]
                        )

                        with col_sel:
                            selected = st.selectbox(
                                "Property",
                                options=options,
                                index=default_idx,
                                key=f"prop_sel_{img_key}",
                                label_visibility="collapsed",
                            )

                        with col_sec:
                            section_options = [
                                "Mechanical",
                                "Thermal",
                                "Processing",
                                "Physical",
                                "Descriptive",
                                "Composition / Reinforcement",
                                "Architecture / Structure",
                            ]
                            section_default = (
                                section_options.index(ai_section)
                                if ai_section in section_options
                                else 0
                            )
                            section_val = st.selectbox(
                                "Section",
                                options=section_options,
                                index=section_default,
                                key=f"sec_{img_key}",
                                label_visibility="collapsed",
                            )

                        with col_save:
                            if st.button("💾 Save", key=f"save_{img_key}",
                                         use_container_width=True):
                                if selected and selected != "— Select property —":

                                    filepath = save_plot_image_mapping(
                                        mat_abbr, selected, section_val,
                                        bgr, save_dir="images",
                                    )

                                    try:
                                        from db import execute_query
                                        saved_to_db = save_plot_image_to_db(
                                            material_abbr=mat_abbr,
                                            property_name=selected,
                                            image_bgr=bgr,
                                            material_class=st.session_state.get(
                                                "mapping_material_class", "Polymer"
                                            ),
                                            execute_query_fn=execute_query,
                                        )
                                        if saved_to_db:
                                            st.success(
                                                f"✅ Saved to DB & disk → "
                                                f"`{os.path.basename(filepath)}`"
                                            )
                                        else:
                                            st.warning(
                                                "⚠️ Saved to disk only — "
                                                "no matching DB row found for this property."
                                            )
                                    except Exception as e:
                                        st.error(f"DB save failed: {e}")
                                        st.info(f"Saved locally → `{os.path.basename(filepath)}`")

                                    st.session_state.saved_image_mapping[img_key] = {
                                        "property": selected,
                                        "section":  section_val,
                                        "caption":  caption,
                                        "filename": os.path.basename(filepath),
                                        "path":     filepath,
                                    }
                                    st.rerun()
                                else:
                                    st.warning("Select a property first.")

                        with col_rem:
                            if st.button("✕", key=f"rem_{img_key}",
                                         use_container_width=True, help="Remove image"):
                                if img_key in st.session_state.saved_image_mapping:
                                    del st.session_state.saved_image_mapping[img_key]
                                item["image_data"].pop(p_idx)
                                if not item["image_data"]:
                                    display_list.pop(idx)
                                    if use_mapped:
                                        st.session_state.mapped_results = display_list
                                    else:
                                        st.session_state.image_results  = display_list
                                st.rerun()

                        if img_key in st.session_state.saved_image_mapping:
                            saved_m = st.session_state.saved_image_mapping[img_key]
                            st.info(
                                f"✅ Saved as **{saved_m['property']}** → "
                                f"`{saved_m['filename']}`"
                            )

                    else:
                        col_msg, col_rem = st.columns([0.80, 0.20])
                        col_msg.caption(
                            "Go to **Material Data** tab to extract properties and enable mapping."
                        )
                        if col_rem.button("✕", key=f"rem_nd_{img_key}", help="Remove"):
                            item["image_data"].pop(p_idx)
                            if not item["image_data"]:
                                st.session_state.image_results.pop(idx)
                            st.rerun()

                    st.divider()

        saved_map = st.session_state.saved_image_mapping
        if saved_map:
            with st.expander(f"📁 Saved mappings ({len(saved_map)})", expanded=False):
                for key, info in saved_map.items():
                    st.markdown(
                        f"**{info['property']}** &nbsp;›&nbsp; `{info['filename']}`  \n"
                        f"<small style='color:#64748b'>Caption: {info['caption']}</small>",
                        unsafe_allow_html=True,
                    )

    # ════════════════════════════════════════════════════════════════════════
    with subtab_json:
        st.subheader("Metadata Preview")
        json_data = [
            {
                "caption":     r["caption"],
                "page":        r["page"],
                "image_count": len(r["image_data"]),
                "images":      [img["filename"] for img in r["image_data"]],
            }
            for r in image_results
        ]
        st.download_button(
            "⬇ Download JSON",
            data=json.dumps(json_data, indent=4),
            file_name="metadata.json",
            mime="application/json",
            key="dl_json_bottom",
        )
        st.json(json_data)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    inject_upload_page_styles()
    render_top_bar()

    st.subheader("Submit Scientific Material")
    st.caption("Provide technical data and research documentation for the central repository.")

    defaults = {
        "image_results":       [],
        "mapped_results":      [],
        "pdf_processed":       False,
        "mapping_done":        False,
        "current_pdf_name":    None,
        "form_submitted":      False,
        "pdf_data_extracted":  False,
        "pdf_extracted_df":    pd.DataFrame(),
        "pdf_extracted_meta":  {},
        "saved_image_mapping": {},
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
        st.info(
            "Form submitted. Previously extracted data has been saved. "
            "Upload again to process a new PDF."
        )
        st.tabs(["Material Data", "Extracted Plots"])
        return

    tab1, tab2 = st.tabs(["📊 Material Data", "🖼 Extracted Plots"])

    # Write to a stable temp file (avoids Windows WinError 267 on cleanup)
    tmp_file = tempfile.NamedTemporaryFile(
        suffix=".pdf", delete=False, prefix="matdb_"
    )
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

