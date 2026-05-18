import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
import json
import tempfile
import re
import os

import numpy as np
import cv2
from typing import Tuple



# ----------------------------
# Gemini init (more deterministic)
# ----------------------------
def init_gemini(api_key: str):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    return model


def robust_json_load(text: str):
    """
    Handles cases where Gemini returns JSON + extra text.
    """
    if not text:
        raise ValueError("Empty model output")

    cleaned = text.strip().replace("```json", "```").replace("```", "").strip()

    # Find first JSON start
    s1 = cleaned.find("[")
    s2 = cleaned.find("{")
    if s1 == -1 and s2 == -1:
        raise ValueError("No JSON found in model output")

    start = s1 if (s1 != -1 and (s2 == -1 or s1 < s2)) else s2
    dec = json.JSONDecoder()
    parsed, _ = dec.raw_decode(cleaned[start:])
    return parsed


def get_plot_data_from_llm(model, pdf_path: str):
    sample_file = genai.upload_file(path=pdf_path)

    prompt = prompt = """
Analyze this PDF and identify ONLY true data plots:
(Line, Scatter, Bar, Histogram, Heatmap, Box, Violin).
EXCLUDE: flowcharts, block diagrams, photos, icons, gauges/dials, 
graphical abstracts, tables, and pure text regions.

DETECTION RULES:
1) INCLUDE ALL SUBPLOTS: If a figure has panels like (a), (b), (c), 
   return ONE bounding box covering the ENTIRE figure group including 
   all panels, axes, and legends.
2) CROP AREA: box_2d MUST include:
   - The full plot area (all data points, bars, lines)
   - ALL axis lines and tick marks
   - ALL axis labels and tick labels  
   - The legend if present
   - A small margin above the top of the plot
   box_2d MUST EXCLUDE figure caption text below the figure.
3) IMPORTANT: Make the bounding box GENEROUS — it is better to include 
   too much whitespace than to cut off any part of the plot.
4) DO NOT return tables or paragraphs as images.
5) Skip graphical abstracts and decorative figures.
6) Extract the FULL caption text into "caption".

COORDINATES:
- box_2d is [ymin, xmin, ymax, xmax] in 0..1000 normalized page coordinates.
- ymin should be ABOVE the top of the plot area.
- ymax should be BELOW the bottom axis but ABOVE the caption text.
- xmin should be LEFT of the y-axis labels.
- xmax should be RIGHT of the rightmost data point or legend.

Return ONLY a raw JSON array like:
[
  {
    "caption":"...", 
    "page": 1, 
    "box_2d":[ymin,xmin,ymax,xmax], 
    "figure_kind":"plot|nonplot", 
    "plot_type":"scatter|line|bar|hist|heatmap|box|violin|other"
  }
]
IMPORTANT: Only include items where figure_kind="plot".
"""

    # Try to force JSON + reduce randomness
    generation_config = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_output_tokens": 4096,
    }

    # Some environments accept response_mime_type; if yours errors, remove it.
    response = model.generate_content(
        [sample_file, prompt],
        generation_config=generation_config
    )

    data = robust_json_load(response.text)

    # Hard filter if model still returns nonplots
    filtered = []
    for it in data if isinstance(data, list) else []:
        if not isinstance(it, dict):
            continue
        if (it.get("figure_kind") or "").strip().lower() == "plot":
            filtered.append(it)
    return filtered


# ----------------------------
# Caption trimming using PDF text blocks
# ----------------------------
CAP_RE = re.compile(r"^(fig\.?\s*\d+|figure\s*\d+)\b", re.IGNORECASE)


def trim_caption_region(page: fitz.Page, crop_rect: fitz.Rect) -> fitz.Rect:
    """
    If crop includes caption text (Fig./Figure...) near the bottom,
    shrink ymax to exclude it.
    """
    blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text,block_no,block_type)
    if not blocks:
        return crop_rect

    # Look for caption blocks that intersect crop and are below plot area
    candidate_tops = []
    for b in blocks:
        x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
        if not text:
            continue
        t = text.strip().replace("\n", " ")
        if not CAP_RE.search(t):
            continue

        block_rect = fitz.Rect(x0, y0, x1, y1)
        if block_rect.intersects(crop_rect):
            # likely caption inside crop, use its top boundary
            candidate_tops.append(y0)

    if not candidate_tops:
        return crop_rect

    cap_top = min(candidate_tops)
    # Reduce crop bottom just above caption top
    new = fitz.Rect(crop_rect.x0, crop_rect.y0, crop_rect.x1, min(crop_rect.y1, cap_top - 2))
    return new if new.y1 > new.y0 + 10 else crop_rect


# ----------------------------
# Plot-likeness filter (reject text/diagrams)
# ----------------------------
def plot_likeness_score(bgr: np.ndarray) -> float:
    """
    Heuristic: plots usually have strong horizontal/vertical lines (axes, grids),
    and moderate edge density. Text blocks often have many tiny components and no axes.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h < 40 or w < 40:
        return 0.0

    # edges
    edges = cv2.Canny(gray, 60, 160)
    edge_density = float(np.mean(edges > 0))

    # detect long straight lines (axes/grid)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=int(min(h, w) * 0.35), maxLineGap=10)

   
    hv_lines = 0
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dx > 3 * dy or dy > 3 * dx:
                hv_lines += 1

    # connected components for "textiness"
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    # count small-ish components (typical for dense text)
    small = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 10 <= area <= 120:
            small += 1
    small_density = small / (h * w / 10000.0 + 1e-6)

    # Score: prefer hv_lines + moderate edges; penalize extreme textiness
    score = (0.55 * min(1.0, hv_lines / 8.0)) + (0.35 * min(1.0, edge_density / 0.12)) - (0.20 * min(1.0, small_density / 25.0))
    return float(max(0.0, min(1.0, score)))


def looks_like_plot(bgr: np.ndarray, threshold: float = 0.35) -> Tuple[bool, float]:
    s = plot_likeness_score(bgr)
    return (s >= threshold), s


# ----------------------------
# Extraction with better padding
# ----------------------------
def extract_plots(pdf_path: str, plot_data: list, pad: int, score_thresh: float):
    doc = fitz.open(pdf_path)
    results = []

    for idx, item in enumerate(plot_data):
        page_num = int(item.get("page", 1)) - 1
        if page_num < 0 or page_num >= len(doc):
            continue

        page = doc[page_num]
        rect = page.rect
        box = item.get("box_2d", None)
        if not box or len(box) != 4:
            continue

        # normalized 0..1000 → absolute page coords
        ymin = (box[0] * rect.height / 1000.0)
        xmin = (box[1] * rect.width / 1000.0)
        ymax = (box[2] * rect.height / 1000.0)
        xmax = (box[3] * rect.width / 1000.0)

        # Asymmetric padding: keep labels, avoid caption blow-up
        pad_left = pad * 1.4
        pad_right = pad * 0.9
        pad_top = pad * 0.9
        pad_bottom = pad * 0.7  # smaller bottom pad to avoid captions

        fitz_rect = fitz.Rect(
            max(0, xmin - pad_left),
            max(0, ymin - pad_top),
            min(rect.width, xmax + pad_right),
            min(rect.height, ymax + pad_bottom)
        )

        # Trim caption if accidentally included
        fitz_rect = trim_caption_region(page, fitz_rect)

        # Render
        pix = page.get_pixmap(clip=fitz_rect, dpi=300)
        img_path = f"plot_res_{idx}.png"
        pix.save(img_path)

        # Post-filter: reject non-plots
        bgr = cv2.imread(img_path)
        ok, score = looks_like_plot(bgr, threshold=score_thresh)

        if not ok:
            # remove saved non-plot image
            try:
                os.remove(img_path)
            except Exception:
                pass
            continue

        results.append({
            "caption": item.get("caption", f"Figure {idx}"),
            "page": item.get("page", 1),
            "path": img_path,
            "plot_score": round(score, 3),
            "plot_type": item.get("plot_type", "unknown")
        })

    doc.close()
    return results


# ----------------------------
# Streamlit UI
# ----------------------------
def main():
    st.set_page_config(layout="wide", page_title="Scientific Figure Extractor")
    st.title("Scientific Figure & Subplot Extractor")

    if "raw_data" not in st.session_state:
        st.session_state.raw_data = None
    if "temp_pdf" not in st.session_state:
        st.session_state.temp_pdf = None

    with st.sidebar:
        api_key = st.text_input("Gemini API Key", type="password")
        st.divider()
        st.subheader("Crop & Filter")
        crop_pad = st.slider("Padding (px-like on page coords)", 0, 80, 22)
        score_thresh = st.slider("Reject non-plots (higher = stricter)", 0.10, 0.80, 0.20, 0.01)

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file and api_key:
        if st.button("Extract All Data Figures"):
            with st.spinner("AI is scanning for plots and sub-panels..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    st.session_state.temp_pdf = tmp.name

                model = init_gemini(api_key)
                st.session_state.raw_data = get_plot_data_from_llm(model, st.session_state.temp_pdf)
                st.success(f"Detected {len(st.session_state.raw_data)} candidate plot figures (LLM).")

    # Results Display
    if st.session_state.raw_data and st.session_state.temp_pdf:
        st.divider()
        processed_figures = extract_plots(
            st.session_state.temp_pdf,
            st.session_state.raw_data,
            pad=crop_pad,
            score_thresh=score_thresh
        )

        st.success(f"Kept {len(processed_figures)} figures after non-plot filtering.")

        for fig in processed_figures:
            with st.container(border=True):
                st.markdown(f"**{fig['caption']}**")
                st.caption(f"Page {fig['page']} | plot_score={fig['plot_score']} | plot_type={fig['plot_type']}")
                st.image(fig["path"], use_container_width=True)


if __name__ == "__main__":
    main()