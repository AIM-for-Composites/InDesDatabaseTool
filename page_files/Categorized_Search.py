import base64
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from streamlit_scroll_to_top import scroll_to_here

from data_loader import get_all_sections, load_material_data
import streamlit.components.v1 as components
import time
t0 = time.time()
import os
from urllib.parse import urlparse

S3_BUCKET = os.getenv("S3_BUCKET", "imagesstorageaims")
S3_REGION = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION", "us-east-2"))

_s3_client = None
def _s3():
    global _s3_client
    if _s3_client is None:
        import boto3
        _s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=S3_REGION,
        )
    return _s3_client

def _s3_bytes(ref: str):
    """ref = S3 key, or full s3:// / https URL -> object bytes, or None."""
    if not ref:
        return None
    try:
        if ref.startswith("s3://"):
            p = urlparse(ref); bucket, key = p.netloc, p.path.lstrip("/")
        elif ref.startswith("http"):
            p = urlparse(ref); bucket = p.netloc.split(".s3")[0]; key = p.path.lstrip("/")
        else:
            bucket, key = S3_BUCKET, ref
        return _s3().get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as e:
        st.caption(f"(S3 fetch failed: {e})")
        return None

# _class -> RDS category table
_CLASS_TABLE = {"Composites": "Composites_materials", "Fibers": "Fibers", "Polymers": "Polymers"}


def _load_plot_bytes(selected_abbr, property_choice, result_df):
    """Return the PNG bytes stored in the category table's `image` column for this
    material+property, or None. Reads BYTEA from RDS (no local files, no URLs).

    Matching is by property_name first, narrowed by material_name/abbreviation when
    available — abbreviation alone is fragile (case/whitespace/mismatch) and was
    causing rows that DO have image bytes to return nothing."""
    table = None
    material_name = None
    if not result_df.empty:
        if "_class" in result_df.columns:
            table = _CLASS_TABLE.get(str(result_df.iloc[0]["_class"]))
        if "material_name" in result_df.columns:
            material_name = str(result_df.iloc[0]["material_name"])
    tables = [table] if table else list(_CLASS_TABLE.values())  # fallback: try all
    mapper5 = None
    try:
        import sys, os
        # 1) flat layout (local dev: mapper5.py on sys.path / same dir)
        try:
            import mapper5 as _m5
            mapper5 = _m5
        except ModuleNotFoundError:
            pass
        # 2) HF Space package layout: page_files/categorized/Backend/mapper5.py
        if mapper5 is None:
            try:
                from categorized.Backend import mapper5 as _m5
                mapper5 = _m5
            except ModuleNotFoundError:
                pass
        # 3) add candidate dirs to sys.path and retry flat import
        if mapper5 is None:
            here = os.path.dirname(__file__)
            candidates = [
                here,
                os.path.join(here, ".."),
                os.path.join(here, "..", ".."),
                os.path.join(here, "categorized", "Backend"),
                os.environ.get("MAPPER5_DIR", ""),
                r"C:\Users\999te\InDesDatabaseTool\langchain-env\Lib",
            ]
            for cand in candidates:
                if cand and os.path.isdir(cand):
                    sys.path.insert(0, os.path.abspath(cand))
            import mapper5 as _m5
            mapper5 = _m5
        from sqlalchemy import text
        engine = mapper5.get_db_engine()
    except Exception as e:
        st.caption(f"(plot lookup unavailable: {e})")
        return None

    # Try progressively looser filters: (abbr+prop) -> (name+prop) -> (prop only)
    attempts = []
    if selected_abbr:
        attempts.append(("material_abbreviation = :a AND property_name = :p",
                         {"a": selected_abbr, "p": property_choice}))
    if material_name:
        attempts.append(("material_name = :m AND property_name = :p",
                         {"m": material_name, "p": property_choice}))
    attempts.append(("property_name = :p", {"p": property_choice}))

    for tbl in tables:
        for where, params in attempts:
            try:
                with engine.connect() as conn:
                    try:
                        row = conn.execute(text(
                            f'SELECT image_url, image FROM "{tbl}" '
                            f'WHERE {where} LIMIT 1'), params).fetchone()
                        url, blob = (row[0], row[1]) if row else (None, None)
                    except Exception:
                        # image_url column not present yet -> blob only
                        blob = conn.execute(text(
                            f'SELECT image FROM "{tbl}" '
                            f'WHERE {where} AND image IS NOT NULL LIMIT 1'),
                            params).scalar()
                        url = None
                if url:
                    b = _s3_bytes(url)
                    if b:
                        return b
                if blob:
                    return bytes(blob)   # memoryview -> bytes for st.image
            except Exception:
                continue
    return None
def clear_search():
    """Clear the search box and search term. Called as a callback from other widgets."""
    st.session_state._search_term = None
    st.session_state["_pending_clear_search"] = True  

def switch_tab(index, clear=True):
    st.session_state.current_tab = index
    if clear:
        clear_search()
st.markdown(
    """
    <style>
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stHeader"],
    [data-testid="stSidebar"] {
        display: none !important;
    } 

    [data-testid="stAppViewContainer"] {
        background: #eef2f7 !important;
    }

    .st-emotion-cache-6c7yup [data-testid="stLayoutWrapper"] {
        background: #fff !important;
    
    
    }

    .st-emotion-cache-18kf3ut [data-testid="stLayoutWrapper"] {
        background: #fff !important;
    
    
    }

    .block-container {
        max-width: 100% !important;
        padding: 0.6rem 0.75rem 0.8rem 0.75rem !important;
    }
    
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #e1e7ef !important;
        border-radius: 10px !important;
        background: #ffffff !important;
    }

    .aim-top-strip {
        height: 54px;
        border: 1px solid #e1e7ef;
        border-radius: 10px;
        background: #ffffff;
        margin-bottom: 12px;
    }

    .st-key-top_search_row [data-testid="stHorizontalBlock"] {
        gap: 0 !important;
        align-items: stretch !important;
    }

    .st-key-top_search_input [data-baseweb="input"] {
        height: 46px !important;
        min-height: 46px !important;
        border-radius: 50px 0 0 50px !important;
        border-right: none !important;
        border-color: #d0d8d4 !important;
        background: #ffffff !important;
    }

    .st-key-top_search_input input {
        height: 46px !important;
        color: #0f1f1a !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.92rem !important;
        padding-left: 18px !important;
    }

    .st-key-top_search_input input::placeholder {
        color: #0f1f1a !important;
        opacity: 0.4 !important;
    }

    .st-key-top_search_btn button {
        background: #8ACAFF !important;
        border: 1.5px solid #8ACAFF !important;
        border-left: none !important;
        border-radius: 0 50px 50px 0 !important;
        color: #0f1f1a !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        height: 46px !important;
        padding: 0 28px !important;
        box-shadow: none !important;
    }

    .st-key-top_search_input {
        margin-right: 0 !important;
        padding-right: 0 !important;
    }

    .st-key-top_search_btn {
        margin-left: 0 !important;
        padding-left: 0 !important;
    }

    .aim-logo {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 14px 14px 12px;
        margin: -0.2rem -0.2rem 10px -0.2rem;
        border-bottom: 1px solid #edf2f7;
        border-top-left-radius: 10px;
        border-top-right-radius: 10px;
        background: #bae1fc;
        color: #111827;
        font-size: 0.92rem;
        font-weight: 700;
    }

    .aim-sec {
        color: #94a3b8;
        font-size: 0.64rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin: 8px 0 6px;
    }

    .aim-lbl {
        color: #94a3b8;
        font-size: 0.62rem;
        font-weight: 700;
        letter-spacing: 0.9px;
        text-transform: uppercase;
        margin: 3px 0;
    }

    .aim-breadcrumb {
        color: #94a3b8;
        font-size: 0.7rem;
        letter-spacing: 0.3px;
        margin-top: 4px;
    }

    .aim-breadcrumb span {
        color: #3155d4;
        font-weight: 600;
    }

    .aim-title {
        color: #111827;
        font-size: 2.95rem;
        line-height: 1.03;
        font-weight: 800;
        margin: 4px 0 6px;
    }

    .aim-sub {
        color: #64748b;
        font-size: 0.82rem;
    }

    .aim-sub strong {
        color: #111827;
    }

    .aim-selected {
        color: #64748b;
        font-size: 0.74rem;
        margin-top: 10px;
        line-height: 1.35;
    }

    .aim-selected b {
        color: #111827;
    }

    .stButton > button {
        min-height: 30px;
        border-radius: 999px !important;
        border: 1.35px solid #d8e1eb !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        padding: 0.18rem 0.6rem !important;
        white-space: nowrap !important;
    }

    .stButton > button[kind="secondary"] {
        background: #ffffff !important;
        color: #334155 !important;
    }

    .stButton > button[kind="primary"] {
        background: #111827 !important;
        border-color: #111827 !important;
        color: #ffffff !important;
    }

    div[data-baseweb="select"] > div {
        min-height: 34px;
        border: 1.35px solid #dce4ee !important;
        border-radius: 8px !important;
        background: #ffffff !important;
    }

    div[data-baseweb="select"] * {
        font-size: 0.8rem !important;
        color: #334155 !important;
    }

    [data-testid="stCheckbox"] label p {
        color: #374151 !important;
        font-size: 0.78rem !important;
    }

    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        justify-content: flex-end;
        gap: 10px;
        border-bottom: 1px solid #edf2f7;
    }

    [data-testid="stTabs"] button[data-baseweb="tab"] {
        color: #94a3b8;
        font-size: 0.84rem;
        font-weight: 600;
        padding: 12px 8px 11px;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #111827 !important;
        border-bottom: 2.5px solid #111827 !important;
    }

    div[data-testid="stDataEditor"] {
        border: none !important;
        border-radius: 0 !important;
        overflow: visible !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    div[data-testid="stDataEditor"] [role="columnheader"] {
        background: #f8fafc !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.64rem !important;
        font-weight: 700 !important;
        border-bottom: 1px solid #edf2f7 !important;
    }

    div[data-testid="stDataEditor"] [role="gridcell"] {
        font-size: 0.82rem;
        color: #111827 !important;
        background: #ffffff !important;
        border-bottom: 1px solid #f8fafc !important;
    }

    div[data-testid="stDataEditor"] [role="grid"] {
        background: #ffffff !important;
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
    }

    /* Materials grid must blend with the parent card (no inner boxed layer) */
    div.stElementContainer[class*="st-key-materials_editor_"] div[data-testid="stFullScreenFrame"],
    div.stElementContainer[class*="st-key-materials_editor_"] div[data-testid="stDataFrame"],
    div.stElementContainer[class*="st-key-materials_editor_"] div[data-testid="stDataFrameResizable"],
    div.stElementContainer[class*="st-key-materials_editor_"] div[data-testid="stDataFrameResizable"][style] {
        border: 0 !important;
        border-radius: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }

    div.stElementContainer[class*="st-key-materials_editor_"] div.stDataFrameGlideDataEditor,
    div.stElementContainer[class*="st-key-materials_editor_"] div.stDataFrameGlideDataEditor[style] {
        --gdg-bg-cell: #ffffff !important;
        --gdg-bg-cell-medium: #ffffff !important;
        --gdg-bg-header: #f8fafc !important;
        --gdg-border-color: transparent !important;
        --gdg-horizontal-border-color: transparent !important;
        --gdg-drilldown-border: transparent !important;
        --gdg-text-dark: #111827 !important;
        --gdg-text-medium: rgba(17, 24, 39, 0.82) !important;
        --gdg-text-header: rgba(148, 163, 184, 1) !important;
        --gdg-cell-vertical-padding: 8px !important;
        --gdg-bg-bubble: #dbeafe !important;
        --gdg-text-bubble: #1d4ed8 !important;
        --gdg-bubble-height: 22px !important;
        --gdg-bubble-padding: 10px !important;
    }

    .aim-pg-info {
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 6px;
    }

    .aim-pg-info strong {
        color: #111827;
    }

    .aim-ellipsis {
        color: #94a3b8;
        font-size: 0.85rem;
        text-align: center;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(ttl=60)
def load_data(material_type: str) -> pd.DataFrame:
    return load_material_data(material_type)


@st.cache_data(ttl=60)
def load_all_data() -> pd.DataFrame:
    frames = []
    for material_type in ["Composites", "Polymers", "Fibers"]:
        frame = load_data(material_type)
        frame["_class"] = material_type
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)

@st.cache_data(ttl=60)
def get_all_sections_cached() -> list:
    # get_all_sections() re-queries the DB directly (bypassing load_data's
    # cache) every time it's called. Wrapping it here means the 3 DB round
    # trips only happen once per session instead of on every single rerun
    # (every row click, class pill click, page nav, etc).
    return get_all_sections()


def extract_matrix_fiber(abbr: str):
    if not isinstance(abbr, str):
        return None, None

    text = abbr.lower()

    matrix_map = {
        "epoxy": "Epoxy",
        "cyanate ester": "Cyanate Ester",
        "cynate ester": "Cyanate Ester",
        "polypropylene": "Polypropylene",
        "pp": "Polypropylene",
        "peek": "PEEK",
        "pei": "PEI",
        "nylon": "Nylon",
        "pa6": "PA6",
        "polyester": "Polyester",
        "vinyl ester": "Vinyl Ester",
        "phenolic": "Phenolic",
    }

    fiber_map = {
        "carbon": "Carbon Fiber",
        "glass": "Glass Fiber",
        "e-glass": "E-Glass Fiber",
        "s-glass": "S-Glass Fiber",
        "aramid": "Aramid Fiber",
        "kevlar": "Kevlar Fiber",
        "basalt": "Basalt Fiber",
        "natural": "Natural Fiber",
    }

    matrix = next((value for key, value in matrix_map.items() if key in text), None)
    fiber = next((value for key, value in fiber_map.items() if key in text), None)
    return matrix, fiber


def toggle_class(material_class: str):
    if st.session_state.active_classes == [material_class]:
        st.session_state.active_classes = []  
    else:
        st.session_state.active_classes = [material_class]  
        
    if st.session_state.active_classes != ["Composites"]:
        st.session_state.selected_matrix = "All"
        st.session_state.selected_fiber = "All"
    st.session_state.current_page = 0
    st.session_state.selected_row = None
    st.session_state["last_synced_abbr"] = None
    # Clear the dataframe selection for page 0
    df_key = f"materials_df_0"
    if df_key in st.session_state:
        st.session_state[df_key] = {"selection": {"rows": [], "columns": [], "cells": []}}
    st.session_state["_switch_to_tab"] = 0
    clear_search()



def visible_page_numbers(current_page: int, total_pages: int):
    if total_pages <= 6:
        return list(range(total_pages))

    pages = {0, 1, 2, current_page, total_pages - 1}
    if current_page - 1 > 2:
        pages.add(current_page - 1)
    if current_page + 1 < total_pages - 1:
        pages.add(current_page + 1)
    return sorted(pages)


defaults = {
    "active_classes": [],
    "selected_props": [],
    "selected_matrix": "All",
    "selected_fiber": "All",
    "selected_row": None,
    "current_page": 0,
    "_search_term": None,
    "top_search_input": "",
    "inspect_section": None,
    "inspect_property": None,
    "_reset_prop_checks": False,
    "_scroll_to_results": False, 
    "current_tab": 0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value
if st.session_state.get("_last_page") != "Categorized_Search":
    st.session_state._search_term = None
    st.session_state.top_search_input = ""
    st.session_state._last_page = "Categorized_Search"
if "material_type" in st.session_state:
    incoming_type = st.session_state.pop("material_type")
    if incoming_type in ["Composites", "Polymers", "Fibers"]:
        st.session_state.active_classes = [incoming_type]

if "selected_section" in st.session_state:
    st.session_state.selected_props = [st.session_state.pop("selected_section")]
    st.session_state._reset_prop_checks = True

if "search_term" in st.session_state:
    incoming_term = st.session_state.pop("search_term")
    st.session_state._search_term = incoming_term
    st.session_state.top_search_input = incoming_term

all_data = load_all_data()
if "user_uploaded_data" in st.session_state:
    uploaded = st.session_state["user_uploaded_data"].copy()
    uploaded["_class"] = uploaded["material_class"].map(
        {"Polymer": "Polymers", "Fiber": "Fibers", "Composite": "Composites"}
    )
    all_data = pd.concat([all_data, uploaded], ignore_index=True)

st.session_state["base_data"] = all_data

meta = (
    all_data[["material_abbreviation", "material_name", "_class"]]
    .fillna("")
    .drop_duplicates(subset=["material_abbreviation"])
    .reset_index(drop=True)
)
matrix_fiber = meta["material_abbreviation"].apply(extract_matrix_fiber)
meta["Matrix"] = matrix_fiber.apply(
    lambda pair: pair[0] if isinstance(pair, tuple) and len(pair) >= 2 else None
)
meta["Fiber"] = matrix_fiber.apply(
    lambda pair: pair[1] if isinstance(pair, tuple) and len(pair) >= 2 else None
)
st.sidebar.write(f"meta build: {time.time() - t0:.2f}s")
all_sections = get_all_sections_cached()

if st.session_state._reset_prop_checks:
    selected = set(st.session_state.selected_props)
    for index, section in enumerate(all_sections):
        st.session_state[f"prop_check_{index}"] = section in selected
    st.session_state._reset_prop_checks = False

filtered_meta = meta.copy()
if st.session_state.active_classes:
    filtered_meta = filtered_meta[filtered_meta["_class"].isin(st.session_state.active_classes)]
if st.session_state.selected_matrix != "All":
    filtered_meta = filtered_meta[filtered_meta["Matrix"] == st.session_state.selected_matrix]
if st.session_state.selected_fiber != "All":
    filtered_meta = filtered_meta[filtered_meta["Fiber"] == st.session_state.selected_fiber]

if st.session_state._search_term:
    term = st.session_state._search_term
    try:
        pattern = re.compile(term, re.IGNORECASE)
    except re.error:
        pattern = re.compile(re.escape(term), re.IGNORECASE)

    filtered_meta = filtered_meta[
        filtered_meta["material_abbreviation"].astype(str).str.contains(pattern)
        | filtered_meta["material_name"].astype(str).str.contains(pattern)
    ]

if st.session_state.selected_props:
    valid_abbr = all_data[
        all_data["section"].isin(st.session_state.selected_props) & all_data["value"].notna()
    ]["material_abbreviation"].unique()
    filtered_meta = filtered_meta[filtered_meta["material_abbreviation"].isin(valid_abbr)]

filtered_meta = filtered_meta.reset_index(drop=True)

PAGE_SIZE = 50
total = len(filtered_meta)
total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
st.session_state.current_page = min(st.session_state.current_page, total_pages - 1)

start = st.session_state.current_page * PAGE_SIZE
end = start + PAGE_SIZE
page_meta = filtered_meta.iloc[start:end].reset_index(drop=True)
if st.session_state.pop("_restore_selection_highlight", False):
    st.session_state.pop("_search_just_changed", None)
    if st.session_state.selected_row:
        abbr = st.session_state.selected_row[0]
        match = page_meta[page_meta["material_abbreviation"] == abbr]
        if not match.empty:
            st.session_state["pending_row_select"] = match.index[0]
        else:
            st.session_state.pop("pending_row_select", None)  # ← clear stale index
            st.session_state["_clear_df_selection"] = True    # ← clear wrong highlight
            st.session_state["_search_just_changed"] = True
left_col, right_col = st.columns([1.03, 3.07], gap="small")

with left_col:
    with st.container(border=True):
        logo_html = ""
        logo_path = Path("logo.png")
        if logo_path.exists():
            with logo_path.open("rb") as file_handle:
                logo_b64 = base64.b64encode(file_handle.read()).decode()
            logo_html = (
                f"<img src='data:image/png;base64,{logo_b64}' "
                "style='height:20px;width:20px;object-fit:contain;border-radius:4px;'/>"
            )

        st.markdown(
            f"<div class='aim-logo'>{logo_html} AIM Composites</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='aim-sec'>🧩 Material Class</div>", unsafe_allow_html=True)

        cls_a, cls_b = st.columns(2)
        with cls_a:
            if st.button(
                "Composites",
                key="class_comp",
                use_container_width=True,
                type="primary" if "Composites" in st.session_state.active_classes else "secondary",
            ):
                toggle_class("Composites")
                st.rerun()

        with cls_b:
            if st.button(
                "Polymers",
                key="class_poly",
                use_container_width=True,
                type="primary" if "Polymers" in st.session_state.active_classes else "secondary",
            ):
                toggle_class("Polymers")
                st.rerun()

        if st.button(
            "Fibers",
            key="class_fib",
            use_container_width=True,
            type="primary" if "Fibers" in st.session_state.active_classes else "secondary",
        ):
            toggle_class("Fibers")
            st.rerun()

        if st.session_state.active_classes == ["Composites"]:
            st.markdown("<div class='aim-sec'>🧪 Composition</div>", unsafe_allow_html=True)
            composite_meta = meta[meta["_class"] == "Composites"]
            matrix_options = ["All"] + sorted([item for item in composite_meta["Matrix"].dropna().unique() if item])
            fiber_options = ["All"] + sorted([item for item in composite_meta["Fiber"].dropna().unique() if item])

            st.markdown("<div class='aim-lbl'>🧱 Matrix</div>", unsafe_allow_html=True)
            matrix_value = st.selectbox(
                "Matrix",
                matrix_options,
                index=matrix_options.index(st.session_state.selected_matrix)
                if st.session_state.selected_matrix in matrix_options
                else 0,
                key="matrix_select",
            )

            st.markdown("<div class='aim-lbl'>🧵 Fiber</div>", unsafe_allow_html=True)
            fiber_value = st.selectbox(
                "Fiber",
                fiber_options,
                index=fiber_options.index(st.session_state.selected_fiber)
                if st.session_state.selected_fiber in fiber_options
                else 0,
                key="fiber_select",
                label_visibility="collapsed",
            )

            if matrix_value != st.session_state.selected_matrix:
                st.session_state.selected_matrix = matrix_value
                st.session_state.current_page = 0
                st.rerun()

            if fiber_value != st.session_state.selected_fiber:
                st.session_state.selected_fiber = fiber_value
                st.session_state.current_page = 0
                st.rerun()

        #st.markdown("<div class='aim-sec'>📋 Property Types</div>", unsafe_allow_html=True)
        #selected_props = []
        #with st.container(height=480):
        #    for index, section in enumerate(all_sections):
        #        key = f"prop_check_{index}"
        #        if key not in st.session_state:
        #            st.session_state[key] = section in st.session_state.selected_props

        #        if st.checkbox(section, key=key):
        #            selected_props.append(section)

        #if selected_props != st.session_state.selected_props:
        #    st.session_state.selected_props = selected_props
        #    st.session_state.current_page = 0
        #    st.rerun()

        # if st.button(
        #     "🔎 Inspect",
        #     key="inspect_btn",
        #     use_container_width=True,
        #     type="primary",
        #     disabled=st.session_state.selected_row is None,
        # ):
        #     st.info("Open the Inspect tab on the right panel.")
        #     st.session_state["_switch_to_tab"] = 1
        # if st.session_state.selected_row:
        #     selected_abbr, selected_name = st.session_state.selected_row
        #     st.markdown(
        #         f"<div class='aim-selected'><b>Selected</b><br>{selected_name}<br><span style='font-family:monospace'>{selected_abbr}</span></div>",
        #         unsafe_allow_html=True,
        #     )
                
        #st.markdown("<div class='aim-sec'>📋 Property Types</div>", unsafe_allow_html=True)
        #selected_props = []
        #with st.container(height=480):
        #    for index, section in enumerate(all_sections):
        #        key = f"prop_check_{index}"
        #        if key not in st.session_state:
        #            st.session_state[key] = section in st.session_state.selected_props

        #        if st.checkbox(section, key=key):
        #            selected_props.append(section)

        #if selected_props != st.session_state.selected_props:
        #    st.session_state.selected_props = selected_props
        #    st.session_state.current_page = 0
        #    st.rerun()

    with st.container(border=True):
        if st.button(
            "🔎 Inspect",
            key="inspect_btn",
            use_container_width=True,
            type="primary",
            disabled=st.session_state.selected_row is None,
        ):
            st.session_state["_switch_to_tab"] = 1
        if st.session_state.selected_row:
            selected_abbr, selected_name = st.session_state.selected_row
            st.markdown(
                f"<div class='aim-selected'><b>Selected</b><br>{selected_name}<br><span style='font-family:monospace'>{selected_abbr}</span></div>",
                unsafe_allow_html=True,
            )

if st.session_state.pop("_pending_clear_search", False):
    st.session_state.top_search_input = ""
if st.session_state._search_term != st.session_state.get("_previous_search_term"):
    if st.session_state.top_search_input:
        st.session_state._search_term = st.session_state.top_search_input
    if st.session_state._search_term:
        st.session_state.selected_row = None
        st.session_state["_clear_df_selection"] = True
        switch_tab(0, clear=False)
    st.session_state["_previous_search_term"] = st.session_state._search_term
with right_col:
    
    with st.container(border=True):
        with st.container(key="top_search_row"):
            input_col, btn_col = st.columns([0.82, 0.18], gap="small")
            with input_col:
                search_query = st.text_input(
                    label="Search",
                    placeholder="Search material name or abbreviation...",
                    label_visibility="collapsed",
                    key="top_search_input",
                    on_change=lambda: None,
                )
            with btn_col:
                search_clicked = st.button("Search", key="top_search_btn", width="stretch")
                if st.button("🔄 Refresh", key="refresh_data_btn", width="stretch"):
                    st.cache_data.clear()
                    st.rerun()

    current_input = (search_query or "").strip()
    previous_term = st.session_state._search_term or ""
    
    if search_clicked:
        if current_input:
            st.session_state["_switch_to_tab"] = 0
        st.session_state._search_term = current_input if current_input else None
        st.session_state.current_page = 0
        if current_input:
            st.session_state.selected_row = None                 
        st.session_state["_clear_df_selection"] = True       
        st.rerun()
    elif current_input != previous_term:
        st.session_state._search_term = current_input if current_input else None
        st.session_state.current_page = 0
        if current_input:
            st.session_state.selected_row = None                 
            st.session_state["_clear_df_selection"] = True
        else:
            st.session_state.pop("pending_row_select", None)  # ← clear stale index
            #st.session_state["_clear_df_selection"] = True    # ← clear visual highlight
            st.session_state["_restore_selection_highlight"] = True
        st.session_state["_search_just_changed"] = True
       
        st.rerun()
    with st.container(border=True):
        st.markdown(
            """
            <div style='padding:14px 18px 0;'>
                <!-- <div class='aim-breadcrumb'>INVENTORY / <span>MATERIALS DATABASE</span></div> -->
                <div class='aim-title'>Materials Database</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Clear Selection", key="clear_selection_btn", type="primary"):
            st.session_state.selected_row = None
            st.session_state["last_synced_abbr"] = None
            df_key = f"materials_df_{st.session_state.current_page}"
            if df_key in st.session_state:
                st.session_state[df_key] = {"selection": {"rows": [], "columns": [], "cells": []}}
            st.session_state["_switch_to_tab"] = 0
            st.rerun()

        if st.session_state.get("_switch_to_tab") is not None:
            tab_index = st.session_state["_switch_to_tab"]
            switch_tab(tab_index, clear=(tab_index != 1))
            del st.session_state["_switch_to_tab"]

        if st.session_state.current_tab == 0:
            filter_label = (
                ", ".join(st.session_state.active_classes)
                if st.session_state.active_classes
                else "All Materials"
            )
            shown_start = start + 1 if total > 0 else 0
            shown_end = min(end, total)

            row_left, row_right = st.columns([2.2, 1.2])
            with row_left:
                st.markdown(f"<div class='aim-sub'>{filter_label}</div>", unsafe_allow_html=True)
            with row_right:
                st.markdown(
                    f"<div class='aim-sub' style='text-align:right'>Showing <strong>{shown_start}-{shown_end}</strong> of <strong>{total}</strong> materials</div>",
                    unsafe_allow_html=True,
                )

            selected_abbr = st.session_state.selected_row[0] if st.session_state.selected_row else None

            class_map = {
                "Composites": "🔵 COMPOSITE",
                "Polymers": "🟢 POLYMER",
                "Fibers": "🟠 FIBER",
            }

            matrix_fiber_filtered = (
                "Composites" in st.session_state.active_classes
                and (st.session_state.selected_matrix != "All" or st.session_state.selected_fiber != "All")
            )

            if page_meta.empty and matrix_fiber_filtered:
                st.info("There's no such material with this combination of Matrix and Fiber.")


            if not page_meta.empty:
                table_df = page_meta.copy()

                table_df["Class"] = table_df["_class"].map(class_map)
                table_df["Actions"] = ""
                table_df = table_df[
                    [ "material_name", "material_abbreviation", "Class", "Actions"]
                ].rename(
                    columns={
                        "material_name": "Material Name",
                        "material_abbreviation": "Abbreviation",
                    }
                )
            else:
                table_df = pd.DataFrame(
                    columns=["Select", "Material Name", "Abbreviation", "Class", "Actions"]
                )
                
            df_key = f"materials_df_{st.session_state.current_page}"

            if "pending_row_select" in st.session_state:
                 st.session_state[df_key] = {"selection": {"rows": [st.session_state.pending_row_select], "columns": [], "cells": []}}
                 del st.session_state["pending_row_select"]
            
            if st.session_state.pop("_clear_df_selection", False):
                 st.session_state[df_key] = {"selection": {"rows": [], "columns": [], "cells": []}}
            event = st.dataframe(
                table_df,
                key=df_key,
                width="stretch",
                hide_index=True,
                height=400,
                on_select="rerun",
                selection_mode=["single-cell", "single-row"],
                column_config={
                    "Material Name": st.column_config.TextColumn("MATERIAL NAME", width="large"),
                    "Abbreviation": st.column_config.TextColumn("ABBREVIATION", width="medium"),
                    "Class": st.column_config.TextColumn("CLASS", width="small"),
                    "Actions": st.column_config.TextColumn("ACTIONS", width="small"),
                },
            )
            

            selected_rows = event.selection.rows
            selected_cells = event.selection.cells

            if selected_rows:
                

                row_idx = selected_rows[0]
                if row_idx >= len(page_meta):   
                        st.session_state.selected_row = None
                        st.session_state["_clear_df_selection"] = True
                        st.rerun()

                chosen = page_meta.iloc[row_idx]
                abbr = chosen["material_abbreviation"]
                name = chosen["material_name"]
                
                if not (st.session_state.selected_row and st.session_state.selected_row[0] == abbr):
                     st.session_state.selected_row = (abbr, name)
                     
                     st.rerun()
            else:


                if (
                    st.session_state.selected_row is not None
                    and not selected_cells
                    and not st.session_state.pop("_search_just_changed", False)
                ):
                    st.session_state.selected_row = None
                    

                    st.rerun()
            if selected_cells:
                
                row_idx = selected_cells[0][0]
                if row_idx >= len(page_meta):   
                    st.session_state["_clear_df_selection"] = True
                    st.rerun()

                chosen = page_meta.iloc[row_idx]
                abbr = chosen["material_abbreviation"]
                name = chosen["material_name"]
                if st.session_state.selected_row and st.session_state.selected_row[0] == abbr:
                    st.session_state.selected_row = None
                    st.session_state["_clear_df_selection"] = True
                    
                    st.rerun()
                if st.session_state.selected_row is None or st.session_state.selected_row[0] != abbr:
                    st.session_state.selected_row = (abbr, name)
                    st.session_state["pending_row_select"] = row_idx
                    
                    st.rerun()


            info_col, nav_col = st.columns([2.4, 2.0])
            with info_col:
                st.markdown(
                    f"<div class='aim-pg-info'>Showing <strong>{shown_start}-{shown_end}</strong> of <strong>{total}</strong> materials | Items per page: <strong>{PAGE_SIZE}</strong></div>",
                    unsafe_allow_html=True,
                )

            with nav_col:
                nav_items = []
                nav_items.append(("<<", 0, st.session_state.current_page == 0, False))
                nav_items.append(("<", max(0, st.session_state.current_page - 1), st.session_state.current_page == 0, False))

                visible_pages = visible_page_numbers(st.session_state.current_page, total_pages)
                previous = -1
                for number in visible_pages:
                    if previous >= 0 and number - previous > 1:
                        nav_items.append(("...", None, True, False))
                    nav_items.append(
                        (
                            str(number + 1),
                            number,
                            False,
                            number == st.session_state.current_page,
                        )
                    )
                    previous = number

                nav_items.append(
                    (
                        ">",
                        min(total_pages - 1, st.session_state.current_page + 1),
                        st.session_state.current_page >= total_pages - 1,
                        False,
                    )
                )
                nav_items.append(
                    (
                        ">>",
                        total_pages - 1,
                        st.session_state.current_page >= total_pages - 1,
                        False,
                    )
                )

                nav_columns = st.columns(len(nav_items))
                for idx, (column, item) in enumerate(zip(nav_columns, nav_items)):
                    label, target_page, disabled, active = item
                    with column:
                        if label == "...":
                            st.markdown("<div class='aim-ellipsis'>...</div>", unsafe_allow_html=True)
                        else:
                            if st.button(
                                label,
                                key=f"page_btn_{idx}_{label}_{target_page}",
                                width="stretch",
                                disabled=disabled,
                                type="primary" if active else "secondary",
                            ):
                                old_page = st.session_state.current_page

                                st.session_state.current_page = target_page
                                st.session_state.selected_row = None
                                st.session_state["last_synced_abbr"] = None
                                st.session_state["_clear_df_selection"] = True
                                
                                st.rerun()

        else:
            if not st.session_state.selected_row:
                st.warning("Select a material in All Materials first.")
            else:
                selected_abbr, selected_name = st.session_state.selected_row
                st.markdown(f"**Material:** {selected_name}")
                st.caption(selected_abbr)

                material_df = all_data[
                    (all_data["material_abbreviation"] == selected_abbr)
                    & (all_data["value"].notna())
                    & (all_data["property_name"].notna())
                ]

                section_options = sorted(material_df["section"].dropna().unique().tolist())
                if not section_options:
                    st.warning("No property data found for this material.")
                else:
                    if st.session_state.inspect_section not in section_options:
                        st.session_state.inspect_section = section_options[0]

                    section_choice = st.selectbox(
                        "Type of Property",
                        section_options,
                        index=section_options.index(st.session_state.inspect_section),
                        key="inspect_section_select",
                    )
                    st.session_state.inspect_section = section_choice

                    # properties_df = (
                    #     material_df[material_df["section"] == section_choice][
                    #         ["property_name", "section"]
                    #     ]
                    #     .drop_duplicates()
                    #     .reset_index(drop=True)
                    # )
                    # #st.dataframe(properties_df, use_container_width=True, hide_index=True, height=240)
                    # rows_html = "".join(
                    #     f"<tr><td style='padding:8px 12px;font-size:0.82rem;color:#111827;border-bottom:1px solid #f8fafc;'>{row['property_name']}</td>"
                    #     f"<td style='padding:8px 12px;font-size:0.82rem;color:#64748b;border-bottom:1px solid #f8fafc;'>{row['section']}</td></tr>"
                    #     for _, row in properties_df.iterrows()
                    # )
                    # st.markdown(f"""
                    #     <div style='border:1px solid #edf2f7;border-radius:8px;overflow:hidden;max-height:240px;overflow-y:auto;'>
                    #         <table style='width:100%;border-collapse:collapse;'>
                    #             <thead>
                    #                 <tr style='background:#f8fafc;'>
                    #                     <th style='padding:8px 12px;font-size:0.64rem;font-weight:700;letter-spacing:1px;color:#94a3b8;text-align:left;border-bottom:1px solid #edf2f7;'>PROPERTY NAME</th>
                    #                     <th style='padding:8px 12px;font-size:0.64rem;font-weight:700;letter-spacing:1px;color:#94a3b8;text-align:left;border-bottom:1px solid #edf2f7;'>SECTION</th>
                    #                 </tr>
                    #             </thead>
                    #             <tbody>{rows_html}</tbody>
                    #         </table>
                    #     </div>
                    # """, unsafe_allow_html=True)
                    # property_options = properties_df["property_name"].dropna().tolist()
                    # if property_options:
                    #     if st.session_state.inspect_property not in property_options:
                    #         st.session_state.inspect_property = property_options[0]

                    #     property_choice = st.selectbox(
                    #         "Property",
                    #         property_options,
                    #         index=property_options.index(st.session_state.inspect_property),
                    #         key="inspect_property_select",
                    #     )
                    #     st.session_state.inspect_property = property_choice

                    #     # if st.button("Display", key="inspect_search", type="primary"):
                    #     #     result = all_data[
                    #     #         (all_data["material_abbreviation"] == selected_abbr)
                    #     #         & (all_data["property_name"] == property_choice)
                    #     #         & (all_data["value"].notna())
                    #     #     ]

                    #     #     if result.empty:
                    #     #         st.warning("No data found for this material-property combination")
                    #     #     else:
                    #     #         st.subheader("Property Data")
                    #     #         st.dataframe(result.T, use_container_width=True)

                    #     #         st.subheader("Property Graph")
                    #     #         image_path = Path("images") / f"{selected_abbr}_{property_choice}.png"
                    #     #         if image_path.exists():
                    #     #             image = Image.open(image_path)
                    #     #             st.image(image, use_container_width=True, caption="Stress strain curve")
                    #     #         else:
                    #     #             st.caption("No plot image available for this material-property pair.")

                    #     if st.button("Display", key="inspect_search", type="primary"):
                    #         result = all_data[
                    #             (all_data["material_abbreviation"] == selected_abbr)
                    #             & (all_data["property_name"] == property_choice)
                    #             & (all_data["value"].notna())
                    #         ]

                    #         if result.empty:
                    #             st.warning("No data found for this material-property combination")
                    #         else:
                    #             st.subheader("Property Data")
                    #             st.dataframe(result.T, use_container_width=True)

                    #             st.subheader("Property Graph")
                    #             image_path = Path("images") / f"{selected_abbr}_{property_choice}.png"
                    #             if image_path.exists():
                    #                 image = Image.open(image_path)
                    #                 st.image(image, use_container_width=True, caption="Stress strain curve")
                    #             else:
                    #                 st.caption("No plot image available for this material-property pair.")

                    #             st.session_state._scroll_to_results = True
                    properties_df = (
                        material_df[material_df["section"] == section_choice][
                            ["property_name", "value", "unit", "test_condition", "section"]
                        ]
                        .drop_duplicates()
                        .reset_index(drop=True)
                    )

                    property_event = st.dataframe(
                        properties_df,
                        use_container_width=True,
                        hide_index=True,
                        height=240,
                        on_select="rerun",
                        selection_mode="single-row",
                        key="inspect_properties_table",
                        column_config={
                            "property_name": st.column_config.TextColumn("PROPERTY NAME"),
                            "value": st.column_config.TextColumn("VALUE"),
                            "unit": st.column_config.TextColumn("UNIT"),
                            "test_condition": st.column_config.TextColumn("TEST CONDITION"),
                            "section": st.column_config.TextColumn("SECTION"),
                        },
                    )

                    selected_indices = property_event.selection.rows if property_event and property_event.selection else []

                    selected_prop_row = None
                    if selected_indices:
                        selected_prop_row = properties_df.iloc[selected_indices[0]]
                        property_choice = selected_prop_row["property_name"]
                        st.session_state.inspect_property = property_choice
                    elif st.session_state.inspect_property in properties_df["property_name"].tolist():
                        property_choice = st.session_state.inspect_property
                    else:
                        property_choice = None

                    if st.button("Display", key="inspect_search", type="primary"):
                        if not property_choice:
                            st.warning("Please select a property from the property table!")
                        else:
                            result = all_data[
                                (all_data["material_abbreviation"] == selected_abbr)
                                & (all_data["property_name"] == property_choice)
                                & (all_data["value"].notna())
                            ]
                            # if a specific experiment row was picked, narrow to it
                            # (value + test_condition) so the exact measurement shows
                            if selected_prop_row is not None:
                                result = result[
                                    (result["value"].astype(str) == str(selected_prop_row["value"]))
                                    & (result["test_condition"].astype(str) == str(selected_prop_row["test_condition"]))
                                ]

                            if result.empty:
                                st.warning("No data found for this material-property combination")
                            else:
                                st.subheader("Property Data")
                                st.dataframe(result.T, use_container_width=True)

                                st.subheader("Property Graph")
                                _img_bytes = _load_plot_bytes(
                                    selected_abbr, property_choice, result
                                )
                                if _img_bytes:
                                    st.image(_img_bytes, use_container_width=True,
                                             caption="Stored plot")
                                else:
                                    st.caption("No plot image available for this material-property pair.")

                                st.session_state._scroll_to_results = True

                        if st.session_state._scroll_to_results:
                            scroll_to_here(0, key="inspect_results_scroll")
                            st.session_state._scroll_to_results = False