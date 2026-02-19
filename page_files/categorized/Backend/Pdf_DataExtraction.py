import streamlit as st
import pandas as pd
from PIL import Image
import requests
import base64
import json
import os
from typing import Dict, Any, Optional




# Backend PDF extraction Logic
API_KEY = ""  
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "material_name": {"type": "STRING"},
        "material_abbreviation": {"type": "STRING"},
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
                "required": ["section", "property_name", "value", "english", "comments"]
            }
        }
    }
}

# === GEMINI CALL FUNCTION ===
def call_gemini_from_bytes(pdf_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
    """Calls Gemini API with PDF bytes"""
    try:
        encoded_file = base64.b64encode(pdf_bytes).decode("utf-8")
        mime_type = "application/pdf"
    except Exception as e:
        st.error(f"Error encoding PDF: {e}")
        return None

    prompt = (
         "Extract all experimental data from this research paper. "
         "For each measurement, extract: "
         "- experiment_name, measured_value, unit, uncertainty, method, conditions. "
         "Return as JSON."
        # "You are an expert materials scientist. From the attached PDF, extract the material name, "
        # "abbreviation, and ALL properties across categories (Mechanical, Thermal, Electrical, Physical, "
        # "Optical, Rheological, etc.). Return them as 'mechanical_properties' (a single list). "
        # "For each property, you MUST extract:\n"
        # "- property_name\n- value (or range)\n- unit\n"
        # "- english (converted or alternate units, e.g., psi, °F, inches; write '' if not provided)\n"
        # "- test_condition\n- comments (include any notes, footnotes, standards, remarks; write '' if none)\n"
        # "All fields including english and comments are REQUIRED. Respond ONLY with valid JSON following the schema."
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": mime_type, "data": encoded_file}}
                ]
            }
        ],
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
    """Convert extracted JSON to DataFrame"""
    rows = []
    for item in data.get("mechanical_properties", []):
        rows.append({
            "material_name": data.get("material_name", ""),
            "material_abbreviation": data.get("material_abbreviation", ""),
            "section": item.get("section", ""),
            "property_name": item.get("property_name", ""),
            "value": item.get("value", ""),
            "unit": item.get("unit", ""),
            "english": item.get("english", ""),
            "test_condition": item.get("test_condition", ""),
            "comments": item.get("comments", "")
        })
    return pd.DataFrame(rows)



#using sentence transformers and semantic search techniques
import sqlite3
import pandas as pd
import os
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================
# CONFIGURATION
# ==========================
DB_PATH = "output_materials.db"
EXCEL_PATH = "5.1__actual.xlsx"
OUTPUT_EXCEL = "5.1__filled.xlsx"
GEMINI_KEY = "AIzaSyBJ_2gJmwwT7gMNWHo2Lgh5dNYOmGDQZWE"

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"


# ==========================
# GEMINI YES/NO MATCH CHECK
# ==========================
def gemini_same_property(excel_prop, db_prop):
    prompt = f"""
You are an expert materials scientist. Determine if BOTH property names refer
to the SAME mechanical property.

Excel property: "{excel_prop}"
Database property: "{db_prop}"

Rules:
- Compare meaning, not formatting.
- Ignore units, values, and numbers.
- If either refers to conditions, test setup, or non-property info, return NO.
- Return ONLY YES or NO.
"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    response = requests.post(
        GEMINI_URL,
        params={"key": GEMINI_KEY},
        json=payload,
        timeout=60
    ).json()

    try:
        ans = response["candidates"][0]["content"]["parts"][0]["text"].strip().upper()
    except:
        return False

    return ans == "YES"


# ==========================
# SEMANTIC MATCHER (fallback)
# ==========================
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_match(excel_prop, df_section):
    if df_section.empty:
        return None

    # compute embeddings
    db_props = df_section["property_name"].tolist()
    db_vecs = embed_model.encode(db_props, convert_to_numpy=True)
    q_vec = embed_model.encode([excel_prop], convert_to_numpy=True)

    sims = cosine_similarity(q_vec, db_vecs)[0]

    df_section = df_section.copy()
    df_section["sim"] = sims
    df_section = df_section.sort_values("sim", ascending=False)

    # Take top-5 candidates for Gemini check
    top5 = df_section.head(5)

    for _, row in top5.iterrows():
        cand = row["property_name"]
        if gemini_same_property(excel_prop, cand):
            return row

    return None


# ==========================
# MAIN PIPELINE
# ==========================
conn = sqlite3.connect(DB_PATH)

# Get material tables
tables = pd.read_sql_query(
    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';",
    conn
)["name"].tolist()

print(f"Detected tables: {tables}")

# Load Excel template once
df_excel_template = pd.read_excel(EXCEL_PATH)
cols = df_excel_template.columns.tolist()

section_col = next((c for c in cols if "section" in c.lower()), None)
prop_col = next((c for c in cols if "property" in c.lower()), cols[0])

print(f"Detected section column: {section_col}")
print(f"Detected property column: {prop_col}")

with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:

    for table in tables:
        print(f"\nProcessing table: {table}")

        # Load DB table
        df_db = pd.read_sql_query(f"""
            SELECT section, property_name, value, unit, english, comments
            FROM '{table}'
        """, conn)

        df_excel = df_excel_template.copy()
        df_excel["Matched Property"] = ""
        df_excel["Value"] = ""
        df_excel["Unit"] = ""
        df_excel["English"] = ""
        df_excel["Comments"] = ""

        # Process each Excel property
        for i, row in df_excel.iterrows():
            excel_prop = str(row[prop_col]).strip()
            excel_section = str(row.get(section_col, "")).strip().lower()


            if section_col:
                df_sec = df_db[df_db["section"].str.lower() == excel_section]
            else:
                df_sec = df_db

            # ==========================
            # 1️ EXACT MATCH
            # ==========================
            exact = df_sec[df_sec["property_name"].str.lower() == excel_prop.lower()]

            if not exact.empty:
                r = exact.iloc[0]
                df_excel.at[i, "Matched Property"] = r["property_name"]
                df_excel.at[i, "Value"] = r["value"]
                df_excel.at[i, "Unit"] = r["unit"]
                df_excel.at[i, "English"] = r["english"]
                df_excel.at[i, "Comments"] = r["comments"]
                continue  # done

            # ==========================
            # 2️ SEMANTIC + GEMINI MATCH
            # ==========================
            best = semantic_match(excel_prop, df_sec)

            if best is not None:
                df_excel.at[i, "Matched Property"] = best["property_name"]
                df_excel.at[i, "Value"] = best["value"]
                df_excel.at[i, "Unit"] = best["unit"]
                df_excel.at[i, "English"] = best["english"]
                df_excel.at[i, "Comments"] = best["comments"]
            else:
                df_excel.at[i, "Matched Property"] = ""

        # Write one sheet per material
        df_excel.to_excel(writer, sheet_name=table[:31], index=False)

print(f"\nDONE → Final filled Excel: {OUTPUT_EXCEL}")
conn.close()
