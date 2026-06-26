"""
plot_property_mapper.py
-----------------------
Maps extracted plot images to material properties stored in PostgreSQL.

Strategy:
  1. Fetch all properties for the material from the DB (Polymers / Fibers / Composites_materials)
  2. For each plot: send image + caption + extracted JSON data to Gemini
  3. Gemini returns the best-matching property_name + confidence reasoning
  4. Caller can confirm/override and persist the match

DB schema (per table: Polymers, Fibers, Composites_materials):
    material_name, material_abbreviation, section,
    property_name, value, unit, english, test_condition, comments
"""

from __future__ import annotations

import base64
import json
import os
import re
from io import BytesIO
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Gemini config  (re-uses the same key / model you already use)
# ---------------------------------------------------------------------------
_GEMINI_KEY = os.getenv(
    "GEMINI_API_KEY",
    "",   # fallback – prefer env var
)
_GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"
_GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/"
    f"models/{_GEMINI_MODEL}:generateContent?key={_GEMINI_KEY}"
)

# ---------------------------------------------------------------------------
# Table routing  (mirrors data_loader.py)
# ---------------------------------------------------------------------------
TABLE_MAP = {
    "Polymer":   "Polymers",
    "Fiber":     "Fibers",
    "Composite": "Composites_materials",
}

# ---------------------------------------------------------------------------
# DB helpers  (thin wrappers – import your existing fetch_all from db.py)
# ---------------------------------------------------------------------------

def fetch_properties_for_material(
    material_abbr: str,
    material_class: str,
    fetch_all_fn,               # pass db.fetch_all so we don't re-import
) -> list[dict]:
    """
    Return all property rows for a given material abbreviation.
    Falls back to all rows in the table if nothing matches the abbreviation.
    """
    table = TABLE_MAP.get(material_class)
    if not table:
        raise ValueError(f"Unknown material_class: {material_class!r}")

    query = f"""
        SELECT
            material_name,
            material_abbreviation,
            section,
            property_name,
            value,
            unit,
            english,
            test_condition,
            comments
        FROM "{table}"
        WHERE LOWER(material_abbreviation) = LOWER(:abbr)
        ORDER BY section, property_name
    """
    rows = fetch_all_fn(query, {"abbr": material_abbr})

    # fallback: try by name fragment
    if not rows:
        query2 = f"""
            SELECT
                material_name, material_abbreviation, section,
                property_name, value, unit, english, test_condition, comments
            FROM "{table}"
            ORDER BY section, property_name
        """
        rows = fetch_all_fn(query2)

    return rows


def save_plot_image_mapping(
    material_abbr: str,
    property_name: str,
    section: str,
    image_array: np.ndarray,    # BGR numpy array from cv2
    save_dir: str = "images",
) -> str:
    """
    Save the plot image to disk as  <save_dir>/<abbr>_<safe_property>.png
    Returns the file path.
    """
    os.makedirs(save_dir, exist_ok=True)
    safe_prop = re.sub(r"[^\w\s-]", "", property_name).strip().replace(" ", "_")
    filename = f"{material_abbr}_{safe_prop}.png"
    filepath = os.path.join(save_dir, filename)
    cv2.imwrite(filepath, image_array)
    return filepath


# ---------------------------------------------------------------------------
# Core: Gemini image + data → property match
# ---------------------------------------------------------------------------

def _encode_image_bgr(bgr: np.ndarray) -> tuple[str, str]:
    """Convert BGR numpy array → base64 PNG string."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64, "image/png"


def map_plot_to_property(
    image_bgr: np.ndarray,
    caption: str,
    extracted_json: dict[str, Any],   # full Gemini data-extraction output
    db_properties: list[dict],        # rows from fetch_properties_for_material
    gemini_api_key: str | None = None,
) -> dict:
    """
    Ask Gemini which DB property best matches this plot.

    Returns
    -------
    {
        "property_name": str,          # best match from DB
        "section":       str,
        "confidence":    "high"|"medium"|"low",
        "reasoning":     str,
        "db_row":        dict | None,  # full matching DB row
        "all_candidates": list[dict],  # top-3 ranked by Gemini
    }
    """
    key = gemini_api_key or _GEMINI_KEY
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{_GEMINI_MODEL}:generateContent?key={key}"
    )

    # Build a compact list of DB properties for the prompt
    prop_list_text = "\n".join(
        f"  - [{row['section']}] {row['property_name']}  "
        f"(value={row['value']}, unit={row['unit']})"
        for row in db_properties[:80]   # cap at 80 to stay within context
    )

    # Compact extracted JSON (just material + property names + values)
    extracted_summary = {
        "material_name": extracted_json.get("material_name", ""),
        "material_abbreviation": extracted_json.get("material_abbreviation", ""),
        "properties": [
            {
                "section": p.get("section"),
                "property_name": p.get("property_name"),
                "value": p.get("value"),
                "unit": p.get("unit"),
            }
            for p in extracted_json.get("mechanical_properties", [])[:40]
        ],
    }

    prompt = f"""You are an expert materials scientist.

TASK: Identify which property from the DATABASE LIST best matches the provided plot image.

PLOT CAPTION:
"{caption}"

EXTRACTED TEXT DATA FROM THE SAME PDF (JSON summary):
{json.dumps(extracted_summary, indent=2)}

DATABASE PROPERTIES FOR THIS MATERIAL:
{prop_list_text}

INSTRUCTIONS:
1. Examine the plot image carefully (axes labels, units, curve shapes, legend text).
2. Use the caption AND the extracted JSON data as additional context.
3. Select the TOP 3 best-matching property names from the DATABASE PROPERTIES list above.
4. For each candidate give a confidence: high / medium / low.
5. Return ONLY valid JSON — no markdown, no explanation outside the JSON.

REQUIRED JSON FORMAT:
{{
  "best_match": {{
    "property_name": "<exact name from DB list>",
    "section": "<section from DB list>",
    "confidence": "high|medium|low",
    "reasoning": "<1-2 sentence explanation>"
  }},
  "candidates": [
    {{"rank": 1, "property_name": "...", "section": "...", "confidence": "..."}},
    {{"rank": 2, "property_name": "...", "section": "...", "confidence": "..."}},
    {{"rank": 3, "property_name": "...", "section": "...", "confidence": "..."}}
  ]
}}
"""

    img_b64, img_mime = _encode_image_bgr(image_bgr)

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": img_mime, "data": img_b64}},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json()

        parts = raw.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        json_text = ""
        for p in parts:
            t = p.get("text", "")
            if t.strip().startswith("{"):
                json_text = t
                break

        if not json_text:
            return _empty_result("Gemini returned no JSON")

        result = json.loads(json_text)

    except Exception as exc:
        return _empty_result(str(exc))

    # Attach full DB row to best_match
    best = result.get("best_match", {})
    matched_prop = best.get("property_name", "")
    db_row = next(
        (r for r in db_properties if r["property_name"] == matched_prop),
        None,
    )
    best["db_row"] = db_row

    return {
        "property_name": matched_prop,
        "section":       best.get("section", ""),
        "confidence":    best.get("confidence", "low"),
        "reasoning":     best.get("reasoning", ""),
        "db_row":        db_row,
        "all_candidates": result.get("candidates", []),
    }


def _empty_result(error: str) -> dict:
    return {
        "property_name": "",
        "section":       "",
        "confidence":    "low",
        "reasoning":     f"Error: {error}",
        "db_row":        None,
        "all_candidates": [],
    }


# ---------------------------------------------------------------------------
# Batch mapper  (call once per PDF after extraction)
# ---------------------------------------------------------------------------

def batch_map_plots(
    image_results: list[dict],      # from extract_images() in upload_backend.py
    extracted_json: dict,           # from call_gemini_from_bytes()
    db_properties: list[dict],      # from fetch_properties_for_material()
    gemini_api_key: str | None = None,
    progress_callback=None,         # optional fn(current, total, caption)
) -> list[dict]:
    """
    Map every extracted plot to a DB property.

    Returns a list parallel to image_results:
    [
      {
        "caption":        str,
        "page":           int,
        "image_data":     [...],    # original image_data list
        "mapping_result": { ...map_plot_to_property output... }
      },
      ...
    ]
    """
    total = len(image_results)
    out = []

    for i, item in enumerate(image_results):
        caption = item.get("caption", "")
        page    = item.get("page", 0)

        if progress_callback:
            progress_callback(i, total, caption)

        # Use first sub-image for mapping (usually there's only one per caption)
        img_list = item.get("image_data", [])
        if not img_list:
            out.append({**item, "mapping_result": _empty_result("No image data")})
            continue

        bgr = img_list[0].get("array")
        if bgr is None:
            out.append({**item, "mapping_result": _empty_result("Missing array")})
            continue

        result = map_plot_to_property(
            image_bgr=bgr,
            caption=caption,
            extracted_json=extracted_json,
            db_properties=db_properties,
            gemini_api_key=gemini_api_key,
        )

        out.append({
            "caption":        caption,
            "page":           page,
            "image_data":     img_list,
            "mapping_result": result,
        })

    return out

def save_plot_image_to_db(
    material_abbr: str,
    property_name: str,
    image_bgr,
    material_class: str,
    execute_query_fn,
) -> bool:
    """Save plot image as BYTEA into the matching property row in PostgreSQL."""

    table_map = {
        "Polymer":   "Polymers",
        "Fiber":     "Fibers",
        "Composite": "Composites_materials",
    }
    table = table_map.get(material_class)
    if not table:
        return False

    _, buffer = cv2.imencode(".png", image_bgr)
    image_bytes = buffer.tobytes()

    query = f"""
        UPDATE "{table}"
        SET image = :image
        WHERE LOWER(material_abbreviation) = LOWER(:abbr)
        AND LOWER(property_name) = LOWER(:prop)
    """
    rows_updated = execute_query_fn(
        query,
        {
            "image": image_bytes,
            "abbr":  material_abbr,
            "prop":  property_name,
        }
    )
    return rows_updated > 0
