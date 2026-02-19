import os
import re
import json
import tempfile
import zipfile
from io import BytesIO
import fitz  # PyMuPDF
import cv2
import numpy as np

import streamlit as st
import pandas as pd
import requests
import base64
from typing import Dict, Any, Optional
from collections import defaultdict
import base64

with open("images/Materials_bg_InDeS.png", "rb") as f:
    encoded = base64.b64encode(f.read()).decode()

image_url = f"data:image/png;base64,{encoded}"

API_KEY = ""
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "material_name": {"type": "STRING"},
        "material_abbreviation": {"type": "STRING"},

        "trade_grade": {
            "type": "STRING",
            "description": "Commercial or trade grade name of the material; '' if not provided"
        },

        "manufacturer": {
            "type": "STRING",
            "description": "Company or organization producing the material; '' if not provided"
        },

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
                    "comments": {"type": "STRING"}
                },
                "required": [
                    "section",
                    "property_name",
                    "value",
                    "english",
                    "comments"
                ]
            }
        }
    }
}

def make_abbreviation(name: str) -> str:
    """Create a simple abbreviation from the material name."""
    if not name:
        return "UNKNOWN"
    words = name.split()
    abbr = "".join(w[0] for w in words if w and w[0].isalpha()).upper()
    return abbr or name[:6].upper()

DPI = 300
CAP_RE = re.compile(r"^(Fig\.?\s*\d+|Figure\s*\d+)\b", re.IGNORECASE)

def call_gemini_from_bytes(pdf_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
    """Calls Gemini API with PDF bytes"""
    try:
        encoded_file = base64.b64encode(pdf_bytes).decode("utf-8")
        mime_type = "application/pdf"
    except Exception as e:
        st.error(f"Error encoding PDF: {e}")
        return None

    prompt = (
    "You are an expert materials scientist. From the attached PDF, extract:\n"
    "- material_name (generic material, e.g., isotactic polypropylene)\n"
    "- material_abbreviation\n"
    "- trade_grade (commercial or trade name; write '' if not provided)\n"
    "- manufacturer (company or organization producing the material; write '' if not provided)\n\n"

    "Extract ALL properties across categories (Mechanical, Thermal, Electrical, Physical, "
    "Optical, Rheological, etc.) and return them as 'mechanical_properties' (a single list).\n\n"

    "For each property, you MUST extract:\n"
    "- property_name\n"
    "- value (or range)\n"
    "- unit\n"
    "- english (converted or alternate units, e.g., psi, °F, inches; write '' if not provided)\n"
    "- test_condition\n"
    "- comments (include any notes, footnotes, standards, remarks; write '' if none)\n\n"

    "All fields including english and comments are REQUIRED.\n"
    "Respond ONLY with valid JSON following the schema."
    )


    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": mime_type, "data": encoded_file}}
            ]
        }],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": SCHEMA
        }
    }

    try:
        r = requests.post(API_URL, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        
        candidates = data.get("candidates", [])
        if not candidates:
            return None

        parts = candidates[0].get("content", {}).get("parts", [])
        json_text = None
        for p in parts:
            t = p.get("text", "")
            if t.strip().startswith("{"):
                json_text = t
                break

        return json.loads(json_text) if json_text else None
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        return None

def convert_to_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    """Convert extracted JSON to DataFrame, ensuring abbreviation is not empty."""
    mat_name = data.get("material_name", "") or ""
    mat_abbr = data.get("material_abbreviation", "") or ""
    trade_grade = data.get("trade_grade", "") or ""
    manufacturer = data.get("manufacturer", "") or ""

    if not mat_abbr:
        mat_abbr = make_abbreviation(mat_name)

    rows = []
    for item in data.get("mechanical_properties", []):
        rows.append({
            "material_name": mat_name,
            "material_abbreviation": mat_abbr,
            "trade_grade": trade_grade,
            "manufacturer": manufacturer,
            "section": item.get("section", "") or "Mechanical",
            "property_name": item.get("property_name", "") or "Unknown property",
            "value": item.get("value", "") or "N/A",
            "unit": item.get("unit", "") or "",
            "english": item.get("english", "") or "",
            "test_condition": item.get("test_condition", "") or "",
            "comments": item.get("comments", "") or "",
        })
    return pd.DataFrame(rows)

# --- IMAGE EXTRACTION LOGIC ---
def get_page_image(page):
    pix = page.get_pixmap(matrix=fitz.Matrix(DPI/72, DPI/72))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def is_valid_plot_geometry(binary_crop):
    h, w = binary_crop.shape
    if h < 100 or w < 100: 
        return False
    ink_density = cv2.countNonZero(binary_crop) / (w * h)
    if ink_density > 0.35: 
        return False 
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 4, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 4))
    has_h = cv2.countNonZero(cv2.erode(binary_crop, h_kernel, iterations=1)) > 0
    has_v = cv2.countNonZero(cv2.erode(binary_crop, v_kernel, iterations=1)) > 0
    return has_h or has_v

def merge_boxes(rects):
    if not rects: 
        return []
    rects = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
    merged = []
    for r in rects:
        rx, ry, rw, rh = r
        if not any(rx >= m[0]-15 and ry >= m[1]-15 and rx+rw <= m[0]+m[2]+15 and ry+rh <= m[1]+m[3]+15 for m in merged):
            merged.append(r)
    return merged

def extract_images(pdf_doc):
    """Extract plot images from PDF using improved logic"""
    grouped_data = defaultdict(lambda: {"page": 0, "image_data": []})
    PADDING = 30
    
    for page_num, page in enumerate(pdf_doc, start=1):
        img_bgr = get_page_image(page)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((10, 10), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        page_h, page_w = gray.shape
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 0.03 < (w * h) / (page_w * page_h) < 0.8:
                if is_valid_plot_geometry(binary[y:y+h, x:x+w]):
                    candidates.append((x, y, w, h))

        final_rects = merge_boxes(candidates)
        blocks = page.get_text("blocks")
        
        for (cx, cy, cw, ch) in final_rects:
            best_caption = f"Figure on Page {page_num} (Unlabeled)"
            min_dist = float('inf')
            for b in blocks:
                text = b[4].strip()
                if CAP_RE.match(text):
                    cap_y = b[1] * (DPI/72)
                    dist = cap_y - (cy + ch)
                    if 0 < dist < (page_h * 0.3) and dist < min_dist:
                        best_caption = text.replace('\n', ' ')
                        min_dist = dist
            
            x1, y1 = max(0, cx - PADDING), max(0, cy - PADDING)
            x2, y2 = min(page_w, cx + cw + PADDING), min(page_h, cy + ch + PADDING)
            crop = img_bgr[int(y1):int(y2), int(x1):int(x2)]
            
            # Store image data in memory instead of saving to disk
            _, buffer = cv2.imencode('.png', crop)
            img_bytes = buffer.tobytes()
            
            fname = f"pg{page_num}_{cx}_{cy}.png"
            
            grouped_data[best_caption]["page"] = page_num
            grouped_data[best_caption]["image_data"].append({
                "filename": fname,
                "bytes": img_bytes,
                "array": crop
            })

    results = [{"caption": k, "page": v["page"], "image_data": v["image_data"]} for k, v in grouped_data.items()]
    return results

def create_zip(results, include_json=True):
    """Create a zip file with images and optional JSON"""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        if include_json:
            json_data = [{"caption": r["caption"], "page": r["page"], 
                         "image_count": len(r["image_data"])} for r in results]
            z.writestr("plot_data.json", json.dumps(json_data, indent=4))
        
        for item in results:
            for img_data in item['image_data']:
                z.writestr(img_data['filename'], img_data['bytes'])
    
    buf.seek(0)
    return buf.getvalue()

def input_form():
    PROPERTY_CATEGORIES = {
        "Polymer": [
            "Thermal",
            "Mechanical",
            "Processing",
            "Physical",
            "Descriptive",
        ],
        "Fiber": [
            "Mechanical",
            "Physical",
            "Thermal",
            "Descriptive",
        ],
        "Composite": [
            "Mechanical",
            "Thermal",
            "Processing",
            "Physical",
            "Descriptive",
            "Composition / Reinforcement",
            "Architecture / Structure",
        ],
    }

    PROPERTY_NAMES = {
        "Polymer": {
            "Thermal": [
                "Glass transition temperature (Tg)",
                "Melting temperature (Tm)",
                "Crystallization temperature (Tc)",
                "Degree of crystallinity",
                "Decomposition temperature",
            ],
            "Mechanical": [
                "Tensile modulus",
                "Tensile strength",
                "Elongation at break",
                "Flexural modulus",
                "Impact strength",
            ],
            "Processing": [
                "Melt flow index (MFI)",
                "Processing temperature",
                "Cooling rate",
                "Mold shrinkage",
            ],
            "Physical": [
                "Density",
                "Specific gravity",
            ],
            "Descriptive": [
                "Material grade",
                "Manufacturer",
            ],
        },

        "Fiber": {
            "Mechanical": [
                "Tensile modulus",
                "Tensile strength",
                "Strain to failure",
            ],
            "Physical": [
                "Density",
                "Fiber diameter",
            ],
            "Thermal": [
                "Decomposition temperature",
            ],
            "Descriptive": [
                "Fiber type",
                "Surface treatment",
            ],
        },

        "Composite": {
            "Mechanical": [
                "Longitudinal modulus (E1)",
                "Transverse modulus (E2)",
                "Shear modulus (G12)",
                "Poissons ratio (V12)",
                "Tensile strength (fiber direction)",
                "Interlaminar shear strength",
            ],
            "Thermal": [
                "Glass transition temperature (matrix)",
                "Coefficient of thermal expansion (CTE)",
            ],
            "Processing": [
                "Curing temperature",
                "Curing pressure",
            ],
            "Physical": [
                "Density",
            ],
            "Descriptive": [
                "Laminate type",
            ],
            "Composition / Reinforcement": [
                "Fiber volume fraction",
                "Fiber weight fraction",
                "Fiber type",
                "Matrix type",
            ],
            "Architecture / Structure": [
                "Weave type",
                "Ply orientation",
                "Number of plies",
                "Stacking sequence",
            ],
        },
    }
    st.markdown(f"""
    <style>
        .stApp {{
            background-color: #f2f2f2;
        }}

        h1 {{
            text-align: center;
        }}

        .hero {{
            background-image: linear-gradient(to bottom, rgba(0,0,0,0.5), rgba(0,0,0,0.5)), url('{image_url}');
            background-size: cover;
            background-position: center;
            text-align: center;
            padding: 12rem 12rem 12rem 12rem;
            margin: -6rem -6rem 6rem -6rem;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
        }}
        .hero h3 {{
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            color: white !important;
        }}
        .hero p {{
            color: white !important;
            font-size: 1rem;
            margin: 0;
        }}
    </style>

    <div class="hero">
        <h3>Upload Material Data</h3>
        <p>
            Contribute new material data to the AIM database by uploading experimental results or published PDFs.<br>
            Extracted properties and plots can be reviewed, edited, and mapped to specific materials and properties before final submission.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

    st.title("Materials Property Input Form")

    material_class = st.selectbox(
        "Select Material Class",
        ("Polymer", "Fiber", "Composite"),
        index=None,
        placeholder="Choose material class",
    )

    if material_class:
        property_category = st.selectbox(
            "Select Property Category",
            PROPERTY_CATEGORIES[material_class],
            index=None,
            placeholder="Choose property category",
        )
    else:
        property_category = None

    if material_class and property_category:
        property_name = st.selectbox(
            "Select Property",
            PROPERTY_NAMES[material_class][property_category],
            index=None,
            placeholder="Choose property",
        )
    else:
        property_name = None

    if material_class and property_category and property_name:
        with st.form("user_input"):
            st.subheader("Enter Data")

            material_name = st.text_input("Material Name")
            material_abbr = st.text_input("Material Abbreviation")

            value = st.text_input("Value")
            unit = st.text_input("Unit (SI)")
            english = st.text_input("English Units")
            test_condition = st.text_input("Test Condition")
            comments = st.text_area("Comments")

            submitted = st.form_submit_button("Submit")

            if submitted:
                if not (material_name and value):
                    st.error("Material name and value are required.")
                    
                else:
                    Input_db = pd.DataFrame([{
                        "material_class": material_class,
                        "material_name": material_name,
                        "material_abbreviation": material_abbr,
                        "section": property_category,
                        "property_name": property_name,
                        "value": value,
                        "unit": unit,
                        "english_units": english,
                        "test_condition": test_condition,
                        "comments": comments
                    }])

                    st.success("Property added successfully")
                    st.dataframe(Input_db)
                    
                    if "user_uploaded_data" not in st.session_state:
                        st.session_state["user_uploaded_data"] = Input_db
                        return
                    else:
                        st.session_state["user_uploaded_data"] = pd.concat(
                        [st.session_state["user_uploaded_data"], Input_db],
                        ignore_index=True
                        )
                        
                return

def match_caption_to_property(caption: str, property_name: str) -> bool:
    
    caption_lower = caption.lower()
    prop_lower = property_name.lower()
    
    if prop_lower in caption_lower:
        return True
    
    # Keyword mapping for common property types
    keyword_map = {
        "tensile modulus": ["tensile", "modulus", "young", "elastic"],
        "tensile strength": ["tensile", "strength", "ultimate"],
        "elongation at break": ["elongation", "strain", "break"],
        "glass transition temperature": ["glass transition", "tg", "transition"],
        "melting temperature": ["melting", "tm", "melt"],
        "density": ["density", "specific gravity"],
        "impact strength": ["impact", "izod", "charpy"],
        "flexural modulus": ["flexural", "bending", "flex"],
        "stress": ["stress", "strain"],
        "thermal": ["thermal", "temperature", "heat"],
        "crystallinity": ["crystallinity", "crystalline", "xrd"],
    }
    
    # Check if any keywords from the property are in the caption
    for prop_key, keywords in keyword_map.items():
        if prop_key in prop_lower:
            if any(kw in caption_lower for kw in keywords):
                return True
    
    # Check individual words
    prop_words = set(prop_lower.replace("(", "").replace(")", "").split())
    caption_words = set(caption_lower.replace("(", "").replace(")", "").split())
    
    # If 2+ significant words match, consider it a match
    common_words = prop_words & caption_words
    significant_words = common_words - {"the", "of", "at", "in", "a", "an"}
    
    return len(significant_words) >= 2

def save_matched_images(df: pd.DataFrame, image_results: list, save_dir: str = "images"):
    """
    Match extracted plots to properties and save with proper naming.
    Returns list of successfully saved image paths.
    """
    os.makedirs(save_dir, exist_ok=True)
    saved_images = []
    
    # Get material info
    if df.empty:
        return saved_images
    
    mat_abbr = df.iloc[0]["material_abbreviation"]
    
    # Get all unique properties from the dataframe
    properties = df["property_name"].unique()
    
    # Track which properties have been matched (first match only)
    matched_properties = set()
    
    for img_result in image_results:
        caption = img_result["caption"]
        
        # Try to match this caption to a property
        for prop in properties:
            if prop in matched_properties:
                continue  # Already matched this property
            
            if match_caption_to_property(caption, prop):
                # Found a match! Save the first image in this result
                if img_result["image_data"]:
                    first_img = img_result["image_data"][0]
                    
                    # CRITICAL FIX: Use the EXACT property name from database, not caption
                    # This must match exactly what Page 1 searches for
                    filename = f"{mat_abbr}_{prop}.png"
                    filepath = os.path.join(save_dir, filename)
                    
                    # Save the image
                    cv2.imwrite(filepath, first_img["array"])
                    
                    saved_images.append({
                        "property": prop,
                        "caption": caption,
                        "path": filepath
                    })
                    
                    matched_properties.add(prop)
                    break  # Move to next image result
    
    return saved_images

def save_single_image_with_property(img_array, mat_abbr: str, property_name: str, save_dir: str = "images"):
    """
    Save a single image with the naming convention: {mat_abbr}_{property_name}.png
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Use exact same naming convention as save_matched_images
    filename = f"{mat_abbr}_{property_name}.png"
    filepath = os.path.join(save_dir, filename)
    
    # Save the image
    cv2.imwrite(filepath, img_array)
    
    return filepath

def main():
    #st.set_page_config(page_title="PDF Data & Image Extractor", layout="wide")
    
        
    if 'image_results' not in st.session_state:
        st.session_state.image_results = []
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'current_pdf_name' not in st.session_state:
        st.session_state.current_pdf_name = None
    if 'form_submitted' not in st.session_state:  
        st.session_state.form_submitted = False
    if 'pdf_data_extracted' not in st.session_state:
        st.session_state.pdf_data_extracted = False
    if 'pdf_extracted_df' not in st.session_state:
        st.session_state.pdf_extracted_df = pd.DataFrame()
    if 'saved_image_mapping' not in st.session_state:
        st.session_state.saved_image_mapping = {}

    
    prev_uploaded_count = len(st.session_state.get("user_uploaded_data", pd.DataFrame()))
    input_form()
    curr_uploaded_count = len(st.session_state.get("user_uploaded_data", pd.DataFrame()))
    
    if curr_uploaded_count > prev_uploaded_count:
        st.session_state.form_submitted = True
    
    st.title("PDF Material Data & Plot Extractor")

    uploaded_file = st.file_uploader("Upload PDF (Material Datasheet or Research Paper)", type=["pdf"])
    
    if not uploaded_file:
        
        st.info("Upload a PDF to extract material data and plots")
        st.session_state.pdf_processed = False
        st.session_state.current_pdf_name = None
        st.session_state.image_results = []
        st.session_state.form_submitted = False
        st.session_state.pdf_data_extracted = False       
        st.session_state.pdf_extracted_df = pd.DataFrame()
        st.session_state.saved_image_mapping = {}
        return
        

    paper_id = os.path.splitext(uploaded_file.name)[0].replace(" ", "_")
    
    if st.session_state.current_pdf_name != uploaded_file.name:
        st.session_state.pdf_processed = False
        st.session_state.current_pdf_name = uploaded_file.name
        st.session_state.image_results = []
        st.session_state.form_submitted = False
        st.session_state.saved_image_mapping = {}
    
    if st.session_state.form_submitted:
        st.session_state.form_submitted = False
        st.info("A Form was submitted. But your previous extracted data has been added already. If you want to extract more data/plots " \
        "upload again")
        tab1, tab2 = st.tabs(["Material Data", "Extracted Plots"])
        with tab1:
            st.info("Material data from form has been added to database.")
        with tab2:
            st.info("Plots already extracted")
        return

    tab1, tab2 = st.tabs([" Material Data", " Extracted Plots"])

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with tab1:
            st.subheader("Material Properties Data")

            # Only call Gemini once per PDF
            if not st.session_state.pdf_data_extracted:
                with st.spinner(" Extracting material data..."):
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()

                    data = call_gemini_from_bytes(pdf_bytes, uploaded_file.name)

                    if data:
                        df = convert_to_dataframe(data)
                        if not df.empty:
                            st.session_state.pdf_extracted_df = df
                            st.session_state.pdf_data_extracted = True
                            st.session_state.pdf_extracted_meta = data  # optional: keep raw meta
                        else:
                            st.warning("No data extracted")
                    else:
                        st.error("Failed to extract data from PDF")
            
            # After extraction, or when rerunning, use stored data
            df = st.session_state.pdf_extracted_df
            
            if not df.empty:
                data = st.session_state.get("pdf_extracted_meta", {})
                st.success(f"Extracted {len(df)} properties")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Material", data.get("material_name", "N/A"))
                with col2:
                    st.metric("Abbreviation", data.get("material_abbreviation", "N/A"))

                st.dataframe(df, use_container_width=True, height=400)
                st.subheader("Assign Material Category")

                extracted_material_class = st.selectbox(
                    "Select category for this material",
                    ["Polymer", "Fiber", "Composite"],
                    index=None,
                    placeholder="Required before adding to database"
                )
                
                if st.button("+Add to Database"):
                    if not extracted_material_class:
                        st.error("Please select a material category before adding.")
                    else:
                        df["material_class"] = extracted_material_class
                        df["material_type"] = extracted_material_class

                        # Save matched images BEFORE adding to database
                        if st.session_state.image_results:
                            with st.spinner("Saving matched plot images..."):
                                saved_images = save_matched_images(
                                    df, 
                                    st.session_state.image_results, 
                                    save_dir="images"
                                )
                                
                                if saved_images:
                                    st.success(f" Saved {len(saved_images)} plot image(s)")
                                    with st.expander("View saved images"):
                                        for img_info in saved_images:
                                            st.write(f"✓ **{img_info['property']}** ← {img_info['caption']}")
                                            st.write(f"  Saved to: `{img_info['path']}`")
                                else:
                                    st.info("ℹ No plots matched the extracted properties")

                        # Add to database
                        if "user_uploaded_data" not in st.session_state:
                            st.session_state["user_uploaded_data"] = df
                        else:
                            st.session_state["user_uploaded_data"] = pd.concat(
                                [st.session_state["user_uploaded_data"], df],
                                ignore_index=True
                            )

                        st.success(f"Added to {extracted_material_class} database!")

        with tab2:
            st.subheader("Extracted Plot Images")
            
            if not st.session_state.pdf_processed:
                with st.spinner(" Extracting plots from PDF..."):
                    doc = fitz.open(pdf_path)
                    st.session_state.image_results = extract_images(doc)
                    doc.close()
                    st.session_state.pdf_processed = True
            
            if st.session_state.image_results:
                # Check if we have extracted data to use for property dropdowns
                has_extracted_data = not st.session_state.pdf_extracted_df.empty
                
                if has_extracted_data:
                    # Get material abbreviation and property list from extracted data
                    mat_abbr = st.session_state.pdf_extracted_df.iloc[0]["material_abbreviation"]
                    property_list = st.session_state.pdf_extracted_df["property_name"].unique().tolist()
                    
                    st.info(f" Material: **{mat_abbr}** | {len(property_list)} properties available for mapping")
                else:
                    st.warning(" No extracted material data found. Please extract material data first (Tab 1) to enable property mapping.")
                
                subtab1, subtab2 = st.tabs([" Images", "JSON Preview"])
                
                with subtab1:
                    st.success(f"Extracted {len(st.session_state.image_results)} plots")
                    
                    col_img, col_json, col_all = st.columns(3)
                    
                    with col_img:
                        img_zip = create_zip(st.session_state.image_results, include_json=False)
                        st.download_button(
                            " Download Images Only",
                            data=img_zip,
                            file_name=f"{paper_id}_images.zip",
                            mime="application/zip",
                            use_container_width=True,
                            key="download_images"
                        )
                    
                    with col_json:
                        json_data = [{"caption": r["caption"], "page": r["page"], 
                                     "image_count": len(r["image_data"])} for r in st.session_state.image_results]
                        st.download_button(
                            " Download JSON",
                            data=json.dumps(json_data, indent=4),
                            file_name=f"{paper_id}_metadata.json",
                            mime="application/json",
                            use_container_width=True,
                            key="download_json_top"
                        )
                    
                    with col_all:
                        full_zip = create_zip(st.session_state.image_results, include_json=True)
                        st.download_button(
                            " Download All",
                            data=full_zip,
                            file_name=f"{paper_id}_complete.zip",
                            mime="application/zip",
                            use_container_width=True,
                            key="download_all"
                        )
                    
                    st.divider()
                    
                    # Display saved mappings summary
                    if st.session_state.saved_image_mapping:
                        with st.expander(" Saved Image Mappings", expanded=False):
                            for img_key, mapping_info in st.session_state.saved_image_mapping.items():
                                st.write(f" **{mapping_info['caption']}** → `{mapping_info['property']}`")
                                st.write(f"    Saved as: `{mapping_info['filename']}`")
                        st.divider()
                    
                    results_copy = st.session_state.image_results.copy()
                    
                    for idx in range(len(results_copy)):
                        if idx >= len(st.session_state.image_results):
                            break
                            
                        r = st.session_state.image_results[idx]
                        
                        with st.container(border=True):
                            col_cap, col_btn = st.columns([0.85, 0.15])
                            col_cap.markdown(f"**Page {r['page']}** • {r['caption']}")
                            
                            if col_btn.button("Delete", key=f"del_g_{idx}_{r['page']}"):
                                del st.session_state.image_results[idx]
                                st.rerun()
                            
                            image_data_list = r['image_data']
                            if image_data_list and len(image_data_list) > 0:
                                for p_idx in range(len(image_data_list)):
                                    if p_idx >= len(st.session_state.image_results[idx]['image_data']):
                                        break
                                    
                                    img_data = st.session_state.image_results[idx]['image_data'][p_idx]
                                    
                                    # Create unique key for this specific image
                                    img_unique_key = f"{idx}_{p_idx}_{r['page']}"
                                    
                                    # Display image
                                    st.image(img_data['array'], width=300, channels="BGR")
                                    
                                    # Property mapping UI - only show if we have extracted data
                                    if has_extracted_data:
                                        col_dropdown, col_add_btn, col_remove = st.columns([0.6, 0.2, 0.2])
                                        
                                        with col_dropdown:
                                            selected_property = st.selectbox(
                                                "Select Property",
                                                options=["-- Select --"] + property_list,
                                                key=f"prop_select_{img_unique_key}",
                                                label_visibility="collapsed"
                                            )
                                        
                                        with col_add_btn:
                                            if st.button(" Add", key=f"add_btn_{img_unique_key}"):
                                                if selected_property and selected_property != "-- Select --":
                                                    # Save the image with proper naming
                                                    filepath = save_single_image_with_property(
                                                        img_data['array'],
                                                        mat_abbr,
                                                        selected_property,
                                                        save_dir="images"
                                                    )
                                                    
                                                    # Store the mapping
                                                    st.session_state.saved_image_mapping[img_unique_key] = {
                                                        "property": selected_property,
                                                        "caption": r['caption'],
                                                        "filename": os.path.basename(filepath),
                                                        "path": filepath
                                                    }
                                                    
                                                    st.success(f" Saved as `{mat_abbr}_{selected_property}.png`")
                                                    st.rerun()
                                                else:
                                                    st.warning("Please select a property first")
                                        
                                        with col_remove:
                                            if st.button("Remove", key=f"del_s_{img_unique_key}"):
                                                # Remove from saved mapping if exists
                                                if img_unique_key in st.session_state.saved_image_mapping:
                                                    del st.session_state.saved_image_mapping[img_unique_key]
                                                
                                                del st.session_state.image_results[idx]['image_data'][p_idx]
                                                if len(st.session_state.image_results[idx]['image_data']) == 0:
                                                    del st.session_state.image_results[idx]
                                                st.rerun()
                                        
                                        # Show if this image has been saved
                                        if img_unique_key in st.session_state.saved_image_mapping:
                                            mapping = st.session_state.saved_image_mapping[img_unique_key]
                                            st.info(f"Mapped to: **{mapping['property']}**")
                                    else:
                                        # No extracted data available
                                        col_info, col_remove = st.columns([0.8, 0.2])
                                        with col_info:
                                            st.caption("Extract material data first to enable property mapping")
                                        with col_remove:
                                            if st.button("Remove", key=f"del_s_{img_unique_key}"):
                                                del st.session_state.image_results[idx]['image_data'][p_idx]
                                                if len(st.session_state.image_results[idx]['image_data']) == 0:
                                                    del st.session_state.image_results[idx]
                                                st.rerun()
                                    
                                    st.divider()
                
                with subtab2:
                    st.subheader("Metadata Preview")
                    json_data = [{"caption": r["caption"], "page": r["page"], 
                                 "image_count": len(r["image_data"]),
                                 "images": [img["filename"] for img in r["image_data"]]} 
                                for r in st.session_state.image_results]
                    
                    st.download_button(
                        " Download JSON",
                        data=json.dumps(json_data, indent=4),
                        file_name=f"{paper_id}_metadata.json",
                        mime="application/json",
                        key="download_json_bottom"
                    )
                    
                    st.json(json_data)
            else:
                st.warning("No plots found in PDF")

if __name__ == "__main__":
    main()
