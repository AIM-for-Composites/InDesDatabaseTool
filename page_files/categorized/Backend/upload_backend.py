import os
import re
import json
import zipfile
from io import BytesIO
from typing import Dict, Any, Optional
from collections import defaultdict

import cv2
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import requests
import streamlit as st
import base64

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-2.5-flash-preview-09-2025:generateContent?key="
    f"{API_KEY}"
    if API_KEY
    else None
)

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "material_name": {"type": "STRING"},
        "material_abbreviation": {"type": "STRING"},
        "trade_grade": {
            "type": "STRING",
            "description": "Commercial or trade grade name of the material; '' if not provided",
        },
        "manufacturer": {
            "type": "STRING",
            "description": "Company or organization producing the material; '' if not provided",
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

DPI = 300
CAP_RE = re.compile(r"^(Fig\.?\s*\d+|Figure\s*\d+)\b", re.IGNORECASE)


def make_abbreviation(name: str) -> str:
    if not name:
        return "UNKNOWN"
    words = name.split()
    abbr = "".join(w[0] for w in words if w and w[0].isalpha()).upper()
    return abbr or name[:6].upper()


def call_gemini_from_bytes(pdf_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
    if not API_KEY or not API_URL:
        st.error("Missing Gemini API key. Set GEMINI_API_KEY in environment variables.")
        return None

    try:
        encoded_file = base64.b64encode(pdf_bytes).decode("utf-8")
        mime_type = "application/pdf"
    except Exception as exc:
        st.error(f"Error encoding PDF: {exc}")
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
        "- english (converted or alternate units, e.g., psi, Â°F, inches; write '' if not provided)\n"
        "- test_condition\n"
        "- comments (include any notes, footnotes, standards, remarks; write '' if none)\n\n"
        "All fields including english and comments are REQUIRED.\n"
        "Respond ONLY with valid JSON following the schema."
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": mime_type, "data": encoded_file}},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": SCHEMA,
        },
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()

        candidates = data.get("candidates", [])
        if not candidates:
            return None

        parts = candidates[0].get("content", {}).get("parts", [])
        json_text = None
        for part in parts:
            text = part.get("text", "")
            if text.strip().startswith("{"):
                json_text = text
                break

        return json.loads(json_text) if json_text else None
    except Exception as exc:
        st.error(f"Gemini API Error: {exc}")
        return None


def convert_to_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    mat_name = data.get("material_name", "") or ""
    mat_abbr = data.get("material_abbreviation", "") or ""
    trade_grade = data.get("trade_grade", "") or ""
    manufacturer = data.get("manufacturer", "") or ""

    if not mat_abbr:
        mat_abbr = make_abbreviation(mat_name)

    rows = []
    for item in data.get("mechanical_properties", []):
        rows.append(
            {
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
            }
        )
    return pd.DataFrame(rows)


def get_page_image(page):
    pix = page.get_pixmap(matrix=fitz.Matrix(DPI / 72, DPI / 72))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def is_valid_plot_geometry(binary_crop):
    height, width = binary_crop.shape
    if height < 100 or width < 100:
        return False
    ink_density = cv2.countNonZero(binary_crop) / (width * height)
    if ink_density > 0.35:
        return False
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 4, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 4))
    has_h = cv2.countNonZero(cv2.erode(binary_crop, h_kernel, iterations=1)) > 0
    has_v = cv2.countNonZero(cv2.erode(binary_crop, v_kernel, iterations=1)) > 0
    return has_h or has_v


def merge_boxes(rects):
    if not rects:
        return []
    rects = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
    merged = []
    for rect in rects:
        rx, ry, rw, rh = rect
        if not any(
            rx >= m[0] - 15
            and ry >= m[1] - 15
            and rx + rw <= m[0] + m[2] + 15
            and ry + rh <= m[1] + m[3] + 15
            for m in merged
        ):
            merged.append(rect)
    return merged


def extract_images(pdf_doc):
    grouped_data = defaultdict(lambda: {"page": 0, "image_data": []})
    padding = 30

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
                if is_valid_plot_geometry(binary[y : y + h, x : x + w]):
                    candidates.append((x, y, w, h))

        final_rects = merge_boxes(candidates)
        blocks = page.get_text("blocks")

        for (cx, cy, cw, ch) in final_rects:
            best_caption = f"Figure on Page {page_num} (Unlabeled)"
            min_dist = float("inf")
            for block in blocks:
                text = block[4].strip()
                if CAP_RE.match(text):
                    cap_y = block[1] * (DPI / 72)
                    dist = cap_y - (cy + ch)
                    if 0 < dist < (page_h * 0.3) and dist < min_dist:
                        best_caption = text.replace("\n", " ")
                        min_dist = dist

            x1, y1 = max(0, cx - padding), max(0, cy - padding)
            x2, y2 = min(page_w, cx + cw + padding), min(page_h, cy + ch + padding)
            crop = img_bgr[int(y1) : int(y2), int(x1) : int(x2)]

            _, buffer = cv2.imencode(".png", crop)
            img_bytes = buffer.tobytes()
            fname = f"pg{page_num}_{cx}_{cy}.png"

            grouped_data[best_caption]["page"] = page_num
            grouped_data[best_caption]["image_data"].append(
                {"filename": fname, "bytes": img_bytes, "array": crop}
            )

    return [
        {"caption": key, "page": value["page"], "image_data": value["image_data"]}
        for key, value in grouped_data.items()
    ]


def create_zip(results, include_json=True):
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        if include_json:
            json_data = [
                {"caption": item["caption"], "page": item["page"], "image_count": len(item["image_data"])}
                for item in results
            ]
            zf.writestr("plot_data.json", json.dumps(json_data, indent=4))

        for item in results:
            for img_data in item["image_data"]:
                zf.writestr(img_data["filename"], img_data["bytes"])

    buf.seek(0)
    return buf.getvalue()


def match_caption_to_property(caption: str, property_name: str) -> bool:
    caption_lower = caption.lower()
    prop_lower = property_name.lower()

    if prop_lower in caption_lower:
        return True

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

    for prop_key, keywords in keyword_map.items():
        if prop_key in prop_lower and any(kw in caption_lower for kw in keywords):
            return True

    prop_words = set(prop_lower.replace("(", "").replace(")", "").split())
    caption_words = set(caption_lower.replace("(", "").replace(")", "").split())

    common_words = prop_words & caption_words
    significant_words = common_words - {"the", "of", "at", "in", "a", "an"}

    return len(significant_words) >= 2


def save_matched_images(df: pd.DataFrame, image_results: list, save_dir: str = "images"):
    os.makedirs(save_dir, exist_ok=True)
    saved_images = []

    if df.empty:
        return saved_images

    mat_abbr = df.iloc[0]["material_abbreviation"]
    properties = df["property_name"].unique()
    matched_properties = set()

    for img_result in image_results:
        caption = img_result["caption"]

        for prop in properties:
            if prop in matched_properties:
                continue
            if match_caption_to_property(caption, prop):
                if img_result["image_data"]:
                    first_img = img_result["image_data"][0]
                    filename = f"{mat_abbr}_{prop}.png"
                    filepath = os.path.join(save_dir, filename)
                    cv2.imwrite(filepath, first_img["array"])
                    saved_images.append({"property": prop, "caption": caption, "path": filepath})
                    matched_properties.add(prop)
                    break

    return saved_images


def save_single_image_with_property(
    img_array, mat_abbr: str, property_name: str, save_dir: str = "images"
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{mat_abbr}_{property_name}.png"
    filepath = os.path.join(save_dir, filename)
    cv2.imwrite(filepath, img_array)
    return filepath
