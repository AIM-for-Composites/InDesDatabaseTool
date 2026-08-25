"""
Plot extraction
Plot Extraction - Detection + Missed-Plot Recovery + Crop Verification
==========================================================================
NO DATABASE. Just gives you back the extracted plots.

  1. DETECT   - OpenCV geometry heuristic + PyMuPDF embedded-image list,
                merged, paired with parent/subplot captions.
  2. RECOVER  - per page, ask Gemini "list every distinct figure/plot on
                this page with its bounding box" and compare against what
                Step 1 already found (via IoU overlap). Anything Gemini
                sees that Step 1 missed gets cropped from the full page
                render and added as a recovered figure. This is the
                "did we miss any plots" check.
  3. VERIFY   - every crop (original AND recovered) gets a Gemini call
                judging crop QUALITY: complete/not cut off, not merged
                with something else, no leaked neighboring content. This
                info is attached to each returned image dict as
                "verification" — nothing is filtered out or stored
                anywhere; you decide what to do with it.

GEMINI_API_KEY comes from os.getenv. If it's missing, steps 2 and 3 are
skipped (not silently faked) — you still get Step 1's plots back, just
without the recall check or quality verdicts.

Usage:
    from plot_extraction_verified import extract_and_verify_plots, create_plot_zip

    with open("paper.pdf", "rb") as f:
        pdf_bytes = f.read()

    plot_results, coverage_report = extract_and_verify_plots(pdf_bytes)
    # plot_results: [{"caption","page","image_data":[...]}], each image
    #   dict has "bytes","array","subplot_label","subplot_caption","source",
    #   and "verification" (Gemini's crop-quality verdict, or None if skipped)

    zip_bytes = create_plot_zip(plot_results)  # images + metadata JSON
"""
#individual llm with gpt, claude check results are fine
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import zipfile
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import fitz
import numpy as np
import requests
from dotenv import load_dotenv

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("opencv-python not installed")

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()


def get_gemini_api_key() -> str:
    """Reads GEMINI_API_KEY from the environment via os.getenv (populated
    by load_dotenv() above if a .env file is present). Returns "" if it
    isn't set — callers check for that rather than crashing, so the app
    can still run detection-only (no verification/recovery) without a key."""
    return os.getenv("GEMINI_API_KEY", "")


GEMINI_API_KEY = get_gemini_api_key()

_GEMINI_V3_FALLBACK_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.0-pro",
    "gemini-3.0-flash",
]
_MODEL_VERSION_RE = re.compile(r"gemini-(\d+(?:\.\d+)?)")


def _gemini_model_version(model_name: str) -> float:
    m = _MODEL_VERSION_RE.search(model_name)
    return float(m.group(1)) if m else 0.0


def fetch_gemini_models_v3_plus(api_key: str) -> List[str]:
    """Live-fetch the account's Gemini models, keep only 3.0+, sorted highest first.
    Falls back to a static list if the key is missing or the call fails —
    this is what powers the sidebar model dropdown."""
    if not api_key:
        return _GEMINI_V3_FALLBACK_MODELS
    try:
        resp = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
            timeout=10,
        )
        if not resp.ok:
            return _GEMINI_V3_FALLBACK_MODELS
        models = resp.json().get("models", [])
        candidates = [
            m["name"].replace("models/", "") for m in models
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]
        v3_plus = sorted(
            [n for n in candidates if _gemini_model_version(n) >= 3.0],
            key=_gemini_model_version, reverse=True,
        )
        return v3_plus or _GEMINI_V3_FALLBACK_MODELS
    except Exception as e:
        log.warning(f"Gemini model list fetch failed ({e}) - using fallback.")
        return _GEMINI_V3_FALLBACK_MODELS


def get_best_gemini_v3_model(api_key: str) -> str:
    models = fetch_gemini_models_v3_plus(api_key)
    return models[0] if models else _GEMINI_V3_FALLBACK_MODELS[0]


GEMINI_MODEL = get_best_gemini_v3_model(GEMINI_API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# GPT / CLAUDE CONFIG — optional, for side-by-side verification comparison
# ─────────────────────────────────────────────────────────────────────────────

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

GPT_API_URL        = "https://api.openai.com/v1/chat/completions"
CLAUDE_API_URL     = "https://api.anthropic.com/v1/messages"
CLAUDE_API_VERSION = "2023-06-01"

GPT_FALLBACK_MODELS    = ["gpt-4o", "gpt-4o-mini"]
CLAUDE_FALLBACK_MODELS = ["claude-sonnet-4-6", "claude-opus-4-8", "claude-haiku-4-5-20251001"]

PLOT_DPI = 300
PADDING = 30

_CAPTION_RE = re.compile(r"^(Fig\.?\s*\d+|Figure\s*\d+)\b", re.IGNORECASE)
# Lowercase only: real subplot panel labels are conventionally lowercase
# "(a)", "(b)"... Uppercase parenthetical single letters like "(B)", "(O)"
# are almost always garbled legend-marker glyphs (filled square/triangle/
# circle symbols that lost their font mapping during PDF text extraction),
# not real panel labels — matching those uppercase caused false-positive
# subplot splitting on ordinary single-panel figures whose legend uses
# symbol markers.
_SUBPLOT_MARKER_RE = re.compile(r"^\(?([a-h])\)?\.?$")
_PANEL_SPLIT_RE = re.compile(r"\(([a-h])\)")


def _plot_page_image(page):
    pix = page.get_pixmap(matrix=fitz.Matrix(PLOT_DPI / 72, PLOT_DPI / 72))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _is_valid_plot_geometry(binary_crop):
    h, w = binary_crop.shape
    if h < 100 or w < 100:
        return False
    ink_density = cv2.countNonZero(binary_crop) / float(w * h)
    if ink_density > 0.35:
        return False
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 4), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 4)))
    has_h = cv2.countNonZero(cv2.erode(binary_crop, h_kernel, iterations=1)) > 0
    has_v = cv2.countNonZero(cv2.erode(binary_crop, v_kernel, iterations=1)) > 0
    return has_h or has_v


def _merge_plot_boxes(rects):
    if not rects:
        return []
    rects = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
    merged = []
    for r in rects:
        rx, ry, rw, rh = r
        if not any(
            rx >= m[0] - 15 and ry >= m[1] - 15 and
            rx + rw <= m[0] + m[2] + 15 and
            ry + rh <= m[1] + m[3] + 15
            for m in merged
        ):
            merged.append(r)
    return merged


def _extract_embedded_image_boxes(page):
    scale = PLOT_DPI / 72
    boxes = []
    try:
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                continue
            for rect in rects:
                x, y = int(rect.x0 * scale), int(rect.y0 * scale)
                w, h = int(rect.width * scale), int(rect.height * scale)
                if w < 60 or h < 60:
                    continue
                boxes.append((x, y, w, h))
    except Exception as e:
        log.warning(f"Embedded image box extraction failed: {e}")
    return boxes


def _find_parent_caption(blocks, cx, cy, cw, ch, page_h):
    """
    Nearest 'Fig./Figure N' caption below the plot region — but ONLY among
    captions that horizontally overlap the plot's column (or are wide
    enough to plausibly span the full page width, which is common for
    captions that sit below both columns). Without this, a two-column
    layout can pick up the neighboring column's caption just because it's
    vertically closer than the plot's own caption — this was the cause of
    Fig. 4's plot getting grouped under Fig. 2's caption when both sat at
    the same page height in different columns.
    """
    best_caption = ""
    min_dist = float("inf")
    plot_x1, plot_x2 = cx, cx + cw
    plot_width = max(1, cw)
    for b in blocks:
        if len(b) < 5:
            continue
        text = (b[4] or "").strip()
        if _CAPTION_RE.match(text):
            cap_x1 = b[0] * (PLOT_DPI / 72)
            cap_x2 = b[2] * (PLOT_DPI / 72)
            cap_y  = b[1] * (PLOT_DPI / 72)
            overlap = max(0, min(plot_x2, cap_x2) - max(plot_x1, cap_x1))
            caption_width = max(1, cap_x2 - cap_x1)
            spans_wide = caption_width > plot_width * 2.2  # likely a full-width caption block
            if overlap <= 0 and not spans_wide:
                continue  # caption sits in a different column — not this plot's caption
            dist = cap_y - (cy + ch)
            if -20 < dist < (page_h * 0.4) and abs(dist) < abs(min_dist):
                best_caption = text.replace("\n", " ")
                min_dist = dist
    return best_caption


def _sort_reading_order(boxes):
    if not boxes:
        return []
    boxes_sorted = sorted(boxes, key=lambda b: b[1])
    rows = []
    current_row = [boxes_sorted[0]]
    row_y, row_h = boxes_sorted[0][1], boxes_sorted[0][3]
    for b in boxes_sorted[1:]:
        if b[1] < row_y + row_h * 0.6:
            current_row.append(b)
        else:
            rows.append(current_row)
            current_row = [b]
            row_y, row_h = b[1], b[3]
    rows.append(current_row)
    ordered = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda b: b[0]))
    return ordered


def _find_subplot_label(blocks, box):
    x, y, w, h = box
    search_pad = 40
    for b in blocks:
        if len(b) < 5:
            continue
        text = (b[4] or "").strip()
        m = _SUBPLOT_MARKER_RE.match(text)
        if not m:
            continue
        bx = b[0] * (PLOT_DPI / 72)
        by = b[1] * (PLOT_DPI / 72)
        if (x - search_pad) <= bx <= (x + w) and (y - search_pad) <= by <= (y + h * 0.5):
            return f"({m.group(1).lower()})"
    return None


def _split_parent_caption_into_panels(caption_text):
    """
    Split '(a) text (b) text (c) text' into {'a': 'text', 'b': 'text', ...}.
    Requires the matched letters to form a strict, contiguous run starting
    at 'a' (a,b / a,b,c / ...) — real multi-panel captions always label
    panels this way. Any other pattern (e.g. a lone 'e', or 'b' and 'e'
    with no 'a'/'c'/'d' between them) is almost certainly not real subplot
    lettering and is rejected rather than guessed at.
    """
    markers = list(_PANEL_SPLIT_RE.finditer(caption_text))
    if len(markers) < 2:
        return {}
    letters = [m.group(1) for m in markers]
    if letters[0] != "a":
        return {}
    for i in range(1, len(letters)):
        if ord(letters[i]) != ord(letters[i - 1]) + 1:
            return {}
    panels = {}
    for i, m in enumerate(markers):
        letter = m.group(1)
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(caption_text)
        panels[letter] = caption_text[start:end].strip(" .,:;")
    return panels


MISSED_PLOT_CHECK_PROMPT = (
    "This is a full page from a scientific PDF, rendered as an image. Identify EVERY "
    "distinct figure, plot, chart, graph, or diagram visible on this page. Ignore plain "
    "tables of numbers and body-text paragraphs - only visual figures.\n\n"
    "For each one, give its approximate bounding box on the page and a short description. "
    "Bounding box format: [y_min, x_min, y_max, x_max], each an integer 0-1000 representing "
    "position as a fraction of page height/width (0 = top/left edge, 1000 = bottom/right edge).\n\n"
    "Respond ONLY with a JSON array, no markdown, no explanation:\n"
    "[{\"box_2d\": [y_min, x_min, y_max, x_max], \"description\": \"short description\"}]\n"
    "If there are no figures on this page, respond with exactly: []"
)

IOU_OVERLAP_THRESHOLD = 0.30


def _box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a[0], box_a[1], box_a[0] + box_a[2], box_a[1] + box_a[3]
    bx1, by1, bx2, by2 = box_b[0], box_b[1], box_b[0] + box_b[2], box_b[1] + box_b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _gemini_box_to_pixels(box_2d, page_w, page_h):
    y_min, x_min, y_max, x_max = box_2d
    x = int(x_min / 1000 * page_w)
    y = int(y_min / 1000 * page_h)
    w = int((x_max - x_min) / 1000 * page_w)
    h = int((y_max - y_min) / 1000 * page_h)
    return (max(0, x), max(0, y), max(1, w), max(1, h))


def find_all_figures_on_page_gemini(page_png_bytes):
    if not GEMINI_API_KEY:
        return []
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    encoded = base64.b64encode(page_png_bytes).decode("utf-8")
    payload = {
        "contents": [{
            "parts": [
                {"text": MISSED_PLOT_CHECK_PROMPT},
                {"inlineData": {"mimeType": "image/png", "data": encoded}},
            ]
        }],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    try:
        resp = requests.post(url, json=payload, timeout=90)
        if not resp.ok:
            log.warning(f"Gemini page scan HTTP {resp.status_code}: {resp.text[:300]}")
            return []
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return []
        parts = candidates[0].get("content", {}).get("parts", [])
        raw = "".join(p.get("text", "") for p in parts).strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        entries = json.loads(cleaned)
        return entries if isinstance(entries, list) else []
    except Exception as e:
        log.warning(f"Gemini page scan failed: {e}")
        return []


CROP_VERIFICATION_PROMPT = (
    "You are checking whether this image is a PROPERLY CROPPED scientific figure/plot "
    "extracted automatically from a PDF page. You are judging the CROP QUALITY - the "
    "framing and boundaries - not trying to read data values off the plot.\n\n"
    "Check specifically:\n"
    "1. Is this a complete, single plot/graph - not a photo, not a block of text, and "
    "not two or more unrelated figures merged together into one crop?\n"
    "2. Are the axes, axis labels, tick numbers, and legend (if present) FULLY visible "
    "and not cut off at any edge of the crop?\n"
    "3. Does the crop include excess unrelated content - page margin whitespace far "
    "beyond the plot, stray paragraph text, or part of a neighboring figure/table "
    "bleeding into the frame?\n\n"
    "Respond ONLY with this JSON object, no markdown, no explanation:\n"
    "{\n"
    '  "is_valid_plot": true/false,\n'
    '  "is_complete_not_cutoff": true/false,\n'
    '  "has_excess_content": true/false,\n'
    '  "recommended_action": "keep" | "recrop" | "discard",\n'
    '  "issues": ["short description of each problem found, empty list if none"],\n'
    '  "confidence": "high" | "medium" | "low"\n'
    "}"
)


def verify_crop_with_gemini(image_bytes):
    if not GEMINI_API_KEY:
        return {
            "verified": False, "recommended_action": "unverified",
            "is_valid_plot": None, "is_complete_not_cutoff": None,
            "has_excess_content": None, "issues": ["GEMINI_API_KEY not set"],
            "confidence": "none", "raw_response": "",
        }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "contents": [{
            "parts": [
                {"text": CROP_VERIFICATION_PROMPT},
                {"inlineData": {"mimeType": "image/png", "data": encoded}},
            ]
        }],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    try:
        resp = requests.post(url, json=payload, timeout=60)
        if not resp.ok:
            log.warning(f"Gemini crop verification HTTP {resp.status_code}: {resp.text[:300]}")
            return {
                "verified": False, "recommended_action": "unverified",
                "is_valid_plot": None, "is_complete_not_cutoff": None,
                "has_excess_content": None,
                "issues": [f"HTTP {resp.status_code}"], "confidence": "none",
                "raw_response": resp.text[:500],
            }
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("no candidates in Gemini response")
        parts = candidates[0].get("content", {}).get("parts", [])
        raw = "".join(p.get("text", "") for p in parts).strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        parsed = json.loads(cleaned)
        parsed["verified"] = True
        parsed["raw_response"] = raw
        parsed.setdefault("recommended_action", "discard")
        return parsed
    except Exception as e:
        log.warning(f"Gemini crop verification failed: {e}")
        return {
            "verified": False, "recommended_action": "unverified",
            "is_valid_plot": None, "is_complete_not_cutoff": None,
            "has_excess_content": None,
            "issues": [f"error: {e}"], "confidence": "none", "raw_response": "",
        }


_UNVERIFIED_TEMPLATE = {
    "verified": False, "recommended_action": "unverified",
    "is_valid_plot": None, "is_complete_not_cutoff": None,
    "has_excess_content": None, "confidence": "none", "raw_response": "",
}


def verify_crop_with_gpt(image_bytes: bytes, model: str = "gpt-4o") -> Dict[str, Any]:
    """Same crop-quality check as verify_crop_with_gemini, via GPT vision — lets
    you compare verdicts across models rather than trusting Gemini alone."""
    if not OPENAI_API_KEY:
        return {**_UNVERIFIED_TEMPLATE, "issues": ["OPENAI_API_KEY not set"]}
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": CROP_VERIFICATION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
            ],
        }],
    }
    try:
        resp = requests.post(GPT_API_URL, headers=headers, json=payload, timeout=60)
        if not resp.ok:
            return {**_UNVERIFIED_TEMPLATE, "issues": [f"HTTP {resp.status_code}"],
                    "raw_response": resp.text[:500]}
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        parsed = json.loads(cleaned)
        parsed["verified"] = True
        parsed["raw_response"] = raw
        parsed.setdefault("recommended_action", "discard")
        return parsed
    except Exception as e:
        log.warning(f"GPT crop verification failed: {e}")
        return {**_UNVERIFIED_TEMPLATE, "issues": [f"error: {e}"]}


def verify_crop_with_claude(image_bytes: bytes, model: str = "claude-sonnet-4-6") -> Dict[str, Any]:
    """Same crop-quality check via Claude vision."""
    if not ANTHROPIC_API_KEY:
        return {**_UNVERIFIED_TEMPLATE, "issues": ["ANTHROPIC_API_KEY not set"]}
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": CLAUDE_API_VERSION,
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 512,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": encoded}},
                {"type": "text", "text": CROP_VERIFICATION_PROMPT},
            ],
        }],
    }
    try:
        resp = requests.post(CLAUDE_API_URL, headers=headers, json=payload, timeout=60)
        if not resp.ok:
            return {**_UNVERIFIED_TEMPLATE, "issues": [f"HTTP {resp.status_code}"],
                    "raw_response": resp.text[:500]}
        content_blocks = resp.json().get("content", [])
        raw = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text").strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        parsed = json.loads(cleaned)
        parsed["verified"] = True
        parsed["raw_response"] = raw
        parsed.setdefault("recommended_action", "discard")
        return parsed
    except Exception as e:
        log.warning(f"Claude crop verification failed: {e}")
        return {**_UNVERIFIED_TEMPLATE, "issues": [f"error: {e}"]}


_VERIFY_ENGINES = {
    "gemini": lambda img_bytes, model: verify_crop_with_gemini(img_bytes),
    "gpt":    verify_crop_with_gpt,
    "claude": verify_crop_with_claude,
}


def _majority_action(verdicts: Dict[str, Dict[str, Any]]) -> str:
    """Combine multiple engines' recommended_action into one decision:
    majority vote among engines that actually returned a verdict; 'discard'
    wins ties (safer default than accidentally keeping a bad crop)."""
    actions = [v["recommended_action"] for v in verdicts.values() if v.get("verified")]
    if not actions:
        return "unverified"
    counts: Dict[str, int] = {}
    for a in actions:
        counts[a] = counts.get(a, 0) + 1
    best = max(counts.items(), key=lambda kv: (kv[1], kv[0] == "discard"))
    return best[0]


def verify_crop_multi(image_bytes: bytes, engines: List[str], models: Dict[str, str]) -> Dict[str, Any]:
    """
    Runs the crop-quality check across every engine in `engines` (subset of
    "gemini"/"gpt"/"claude"), each with its own model from `models`.
    Returns {"by_engine": {engine: verdict, ...}, "majority_action": str,
    "verified": bool} — "verified" is True if at least one engine actually
    returned a verdict (vs. all being skipped for missing keys).
    """
    by_engine: Dict[str, Dict[str, Any]] = {}
    for engine in engines:
        fn = _VERIFY_ENGINES.get(engine)
        if fn is None:
            continue
        by_engine[engine] = fn(image_bytes, models.get(engine, ""))
    majority = _majority_action(by_engine)
    any_verified = any(v.get("verified") for v in by_engine.values())
    return {"by_engine": by_engine, "majority_action": majority, "verified": any_verified}


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI-BASED RE-CROP FOR MERGED PLOTS
# For crops the verification step already flagged "recrop" (typically:
# two+ distinct charts merged into one image because a tight subplot grid's
# gutter got bridged by the contour-detection dilation step). Rather than
# guessing the split point from pixel density — tested and found unreliable,
# since real single charts routinely have large legitimately-blank regions
# that look identical to a real gutter by that measure — this asks Gemini
# to look at the specific flagged image and report where the actual chart
# boundaries are, since it can visually tell "there are two charts here"
# with far more reliability than a density heuristic ever could.
# ─────────────────────────────────────────────────────────────────────────────

RESPLIT_PROMPT = (
    "This image may contain TWO OR MORE separate, distinct charts/plots that "
    "were accidentally cropped together into one image (e.g. two adjacent "
    "subplots from a grid, merged because their gutter was too narrow).\n\n"
    "Look carefully. If this image genuinely contains only ONE chart, respond "
    "with exactly: {\"is_merged\": false, \"charts\": []}\n\n"
    "If it contains multiple distinct charts, respond with the bounding box of "
    "EACH individual chart (not the whole image — each chart's own axes/plot "
    "area), in normalized 0-1000 coordinates relative to this image "
    "([y_min, x_min, y_max, x_max], 0=top/left, 1000=bottom/right):\n"
    "{\"is_merged\": true, \"charts\": [{\"box_2d\": [y_min, x_min, y_max, x_max]}, ...]}\n\n"
    "Respond ONLY with the JSON object, no markdown, no explanation."
)


def resplit_merged_crop(image_bytes: bytes) -> Optional[List[bytes]]:
    """
    Sends a single flagged crop to Gemini asking whether it's actually
    multiple charts merged together, and if so, where each one is.
    Returns None if Gemini says it's one chart (or the call fails — fail
    closed, don't destroy the original crop on an API error), or a list of
    2+ PNG-encoded sub-crop byte strings if a genuine split was found.
    """
    if not GEMINI_API_KEY or not CV2_AVAILABLE:
        return None

    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    h, w = img_bgr.shape[:2]

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "contents": [{
            "parts": [
                {"text": RESPLIT_PROMPT},
                {"inlineData": {"mimeType": "image/png", "data": encoded}},
            ]
        }],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    try:
        resp = requests.post(url, json=payload, timeout=60)
        if not resp.ok:
            log.warning(f"Re-split check HTTP {resp.status_code}: {resp.text[:300]}")
            return None
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts", [])
        raw = "".join(p.get("text", "") for p in parts).strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        parsed = json.loads(cleaned)

        if not parsed.get("is_merged") or len(parsed.get("charts", [])) < 2:
            return None

        sub_crops: List[bytes] = []
        for chart in parsed["charts"]:
            box_2d = chart.get("box_2d")
            if not box_2d or len(box_2d) != 4:
                continue
            y_min, x_min, y_max, x_max = box_2d
            x1, y1 = int(x_min / 1000 * w), int(y_min / 1000 * h)
            x2, y2 = int(x_max / 1000 * w), int(y_max / 1000 * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 30 or y2 - y1 < 30:
                continue
            sub = img_bgr[y1:y2, x1:x2]
            ok, buf = cv2.imencode(".png", sub)
            if ok:
                sub_crops.append(buf.tobytes())

        return sub_crops if len(sub_crops) >= 2 else None
    except Exception as e:
        log.warning(f"Re-split check failed: {e}")
        return None


def extract_plot_images(pdf_bytes, check_missed_plots=True):
    if not CV2_AVAILABLE:
        log.warning("Plot extraction skipped - opencv-python not installed.")
        return [], {"pages": [], "total_detected": 0, "total_recovered": 0, "gemini_check_ran": False}

    page_groups = defaultdict(list)
    page_images = {}
    page_blocks = {}
    coverage_pages = []
    gemini_check_ran = False

    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf_doc:
        for page_num, page in enumerate(pdf_doc, start=1):
            img_bgr = _plot_page_image(page)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((10, 10), np.uint8)
            dilated = cv2.dilate(binary, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            page_h, page_w = gray.shape
            geometry_candidates = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if 0.03 < (w * h) / float(page_w * page_h) < 0.8:
                    if _is_valid_plot_geometry(binary[y:y + h, x:x + w]):
                        # Split back apart if dilation bridged 2+ adjacent
                        # subplots (tight NxM grid) into one contour.
                        geometry_candidates.append((x, y, w, h))

            embedded_candidates = [
                (x, y, w, h) for (x, y, w, h) in _extract_embedded_image_boxes(page)
                if 0.01 < (w * h) / float(page_w * page_h) < 0.9
            ]

            detected_rects = _merge_plot_boxes(geometry_candidates + embedded_candidates)
            blocks = page.get_text("blocks")
            page_images[page_num] = img_bgr
            page_blocks[page_num] = blocks

            gemini_reported_count = 0
            recovered_count = 0
            recovered_boxes = []

            if check_missed_plots and GEMINI_API_KEY:
                gemini_check_ran = True
                ok, page_buf = cv2.imencode(".png", img_bgr)
                if ok:
                    gemini_figures = find_all_figures_on_page_gemini(page_buf.tobytes())
                    gemini_reported_count = len(gemini_figures)
                    for fig in gemini_figures:
                        box_2d = fig.get("box_2d")
                        if not box_2d or len(box_2d) != 4:
                            continue
                        g_box = _gemini_box_to_pixels(box_2d, page_w, page_h)
                        # Compare against BOTH locally-detected boxes AND
                        # recovered boxes already accepted this page — Gemini
                        # can list the same physical figure more than once
                        # (e.g. once per subplot with imprecise/overlapping
                        # coordinate estimates) with each individual box's
                        # IoU against the *original* detected set staying
                        # below threshold even though two recovered boxes
                        # are themselves near-duplicates of each other.
                        already_covered = detected_rects + recovered_boxes
                        best_iou = max(
                            (_box_iou(g_box, d_box) for d_box in already_covered),
                            default=0.0,
                        )
                        if best_iou < IOU_OVERLAP_THRESHOLD:
                            recovered_boxes.append(g_box)
                            recovered_count += 1
                            log.info(f"Page {page_num}: recovering figure Gemini found but "
                                     f"OpenCV/embedded missed - {fig.get('description', '')!r}")

            # Final pass: merge/dedupe the combined set (local + recovered)
            # together, since containment/near-duplicate boxes can still
            # slip through the two checks above when a recovered box fully
            # contains (rather than closely overlaps) a locally-detected one,
            # or vice versa.
            source_by_box_key: Dict[Tuple[int, int, int, int], str] = {}
            for box in detected_rects:
                source_by_box_key[box] = "opencv_or_embedded"
            for box in recovered_boxes:
                source_by_box_key.setdefault(box, "gemini_recovery")
            deduped_page_boxes = _merge_plot_boxes(list(source_by_box_key.keys()))

            coverage_pages.append({
                "page": page_num,
                "detected": len(detected_rects),
                "gemini_reported": gemini_reported_count,
                "recovered": recovered_count,
            })

            all_boxes_with_source = [
                (box, source_by_box_key.get(box, "opencv_or_embedded"))
                for box in deduped_page_boxes
            ]

            for box, source in all_boxes_with_source:
                cx, cy, cw, ch = box
                parent_caption = _find_parent_caption(blocks, cx, cy, cw, ch, page_h)
                if not parent_caption:
                    parent_caption = f"Figure on Page {page_num} (Unlabeled)"
                page_groups[page_num].append((box, parent_caption, source))

    grouped_data = defaultdict(lambda: {"page": 0, "image_data": []})

    for page_num, entries in page_groups.items():
        blocks = page_blocks[page_num]
        img_bgr = page_images[page_num]
        page_h, page_w = img_bgr.shape[:2]

        by_caption = defaultdict(list)
        for box, caption, source in entries:
            by_caption[caption].append((box, source))

        for caption, box_source_pairs in by_caption.items():
            ordered = _sort_reading_order([b for b, _ in box_source_pairs])
            source_by_box = {b: s for b, s in box_source_pairs}
            panel_text_by_letter = _split_parent_caption_into_panels(caption)
            letters_in_order = sorted(panel_text_by_letter.keys())
            # Position-based fallback (no explicit per-region '(a)' text found)
            # is only trustworthy when the number of detected regions exactly
            # matches the number of caption panels. If detection produced too
            # many regions (e.g. one subplot's grid lines got split into two
            # boxes, or a near-duplicate slipped through dedup) or too few,
            # blindly zipping by position silently mis-assigns or duplicates
            # letters — safer to leave those extras unlabeled so it's visibly
            # obvious something needs a manual look, rather than confidently
            # wrong.
            counts_match = len(ordered) == len(letters_in_order)
            if panel_text_by_letter and not counts_match:
                log.warning(
                    f"Caption panel count mismatch for {caption!r}: "
                    f"{len(ordered)} region(s) detected vs {len(letters_in_order)} "
                    f"panel letter(s) in caption — skipping position-based subplot "
                    f"assignment for regions with no explicit per-region marker."
                )

            for i, box in enumerate(ordered):
                cx, cy, cw, ch = box
                source = source_by_box.get(box, "opencv_or_embedded")

                label = _find_subplot_label(blocks, box)
                subplot_caption = None
                if label:
                    letter = label.strip("()").lower()
                    subplot_caption = panel_text_by_letter.get(letter)
                elif panel_text_by_letter and counts_match:
                    letter = letters_in_order[i]
                    label = f"({letter})"
                    subplot_caption = panel_text_by_letter[letter]

                x1, y1 = max(0, cx - PADDING), max(0, cy - PADDING)
                x2, y2 = min(page_w, cx + cw + PADDING), min(page_h, cy + ch + PADDING)
                crop = img_bgr[int(y1):int(y2), int(x1):int(x2)]
                if crop.size == 0:
                    continue
                ok, buffer = cv2.imencode(".png", crop)
                if not ok:
                    continue

                fname = f"pg{page_num}_{cx}_{cy}.png"
                grouped_data[caption]["page"] = page_num
                grouped_data[caption]["image_data"].append({
                    "filename": fname,
                    "bytes": buffer.tobytes(),
                    "array": crop,
                    "subplot_label": label,
                    "subplot_caption": subplot_caption,
                    "source": source,
                })

    plot_results = [
        {"caption": k, "page": v["page"], "image_data": v["image_data"]}
        for k, v in grouped_data.items()
    ]

    total_detected = sum(p["detected"] for p in coverage_pages)
    total_recovered = sum(p["recovered"] for p in coverage_pages)
    coverage_report = {
        "pages": coverage_pages,
        "total_detected": total_detected,
        "total_recovered": total_recovered,
        "gemini_check_ran": gemini_check_ran,
    }
    return plot_results, coverage_report


def create_plot_zip(results, include_json=True):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        if include_json:
            json_data = [
                {
                    "caption": r["caption"], "page": r["page"], "image_count": len(r["image_data"]),
                    "images": [
                        {"filename": img["filename"], "subplot_label": img.get("subplot_label"),
                         "subplot_caption": img.get("subplot_caption"), "source": img.get("source"),
                         "verification": img.get("verification")}
                        for img in r["image_data"]
                    ],
                }
                for r in results
            ]
            z.writestr("plot_data.json", json.dumps(json_data, indent=2))
        for item in results:
            for img_data in item["image_data"]:
                z.writestr(img_data["filename"], img_data["bytes"])
    buf.seek(0)
    return buf.getvalue()


def extract_and_verify_plots(
    pdf_bytes: bytes,
    check_missed_plots: bool = True,
    verify_crops: bool = True,
    verify_engines: Optional[List[str]] = None,
    verify_models: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Runs detection + missed-plot recovery (extract_plot_images), then — if
    verify_crops is True — attaches a "verification" dict to every returned
    image entry, built by running the crop-quality check across every
    engine in verify_engines (subset of "gemini"/"gpt"/"claude", default
    ["gemini"] only). Nothing is filtered, discarded, or stored anywhere;
    every crop that was detected comes back to you, so you can decide what
    to do with each one — the Streamlit UI filters on majority_action, but
    that's a display choice, not something baked in here.

    Each image_data entry gains a "verification" key:
        {"by_engine": {"gemini": {...}, "gpt": {...}, "claude": {...}},
         "majority_action": "keep"|"recrop"|"discard"|"unverified",
         "verified": bool}
    verification is None if verify_crops=False.
    """
    verify_engines = verify_engines or ["gemini"]
    verify_models = verify_models or {"gemini": GEMINI_MODEL, "gpt": "gpt-4o", "claude": "claude-sonnet-4-6"}

    plot_results, coverage_report = extract_plot_images(pdf_bytes, check_missed_plots=check_missed_plots)

    if verify_crops:
        for group in plot_results:
            for img in group["image_data"]:
                img["verification"] = verify_crop_multi(img["bytes"], verify_engines, verify_models)
    else:
        for group in plot_results:
            for img in group["image_data"]:
                img["verification"] = None

    return plot_results, coverage_report


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

def _resize_for_display(img_bgr: np.ndarray, target_width: int = 260) -> np.ndarray:
    """Resize a crop to a consistent display width so the grid looks uniform
    regardless of how large or small the original detected region was —
    detected plots vary a lot in native size, and letting Streamlit render
    each at its raw resolution makes the grid ragged and some thumbnails
    huge. Keeps aspect ratio, only ever shrinks (never upscales blurry)."""
    h, w = img_bgr.shape[:2]
    if w <= target_width:
        return img_bgr
    scale = target_width / w
    new_h = max(1, int(h * scale))
    return cv2.resize(img_bgr, (target_width, new_h), interpolation=cv2.INTER_AREA)


def _run_streamlit() -> None:
    import streamlit as st
    global GEMINI_MODEL

    st.set_page_config(page_title="Plot Extraction + Verification", page_icon="", layout="wide")
    st.title(" Plot Extraction — Detection + Missed-Plot Check + Crop Verification")
    st.caption("No database — this just extracts, checks recall, and shows you the plots.")

    with st.sidebar:
        st.header("⚙️ Settings")

        st.subheader("Gemini")
        st.markdown(f"**GEMINI_API_KEY set:** {'' if GEMINI_API_KEY else ''}")
        gemini_models = fetch_gemini_models_v3_plus(GEMINI_API_KEY)
        selected_gemini_model = st.selectbox(
            "Gemini model (3.0+)", gemini_models,
            index=gemini_models.index(GEMINI_MODEL) if GEMINI_MODEL in gemini_models else 0,
            disabled=not GEMINI_API_KEY,
        )

        st.divider()
        st.subheader("Compare with other LLMs")
        st.caption("Runs the same crop-quality check with each engine you enable, "
                  "so you can compare verdicts side by side.")
        use_gpt = st.checkbox(" Also verify with GPT", value=False, disabled=not OPENAI_API_KEY,
                              help="OPENAI_API_KEY not set" if not OPENAI_API_KEY else None)
        gpt_model = st.selectbox("GPT model", GPT_FALLBACK_MODELS, disabled=not use_gpt) if use_gpt else GPT_FALLBACK_MODELS[0]
        use_claude = st.checkbox(" Also verify with Claude", value=False, disabled=not ANTHROPIC_API_KEY,
                                 help="ANTHROPIC_API_KEY not set" if not ANTHROPIC_API_KEY else None)
        claude_model = st.selectbox("Claude model", CLAUDE_FALLBACK_MODELS, disabled=not use_claude) if use_claude else CLAUDE_FALLBACK_MODELS[0]

        st.divider()
        st.markdown(f"**OpenCV available:** {'' if CV2_AVAILABLE else ' not installed'}")
        if not GEMINI_API_KEY:
            st.warning("No GEMINI_API_KEY found — detection will still run; the "
                      "missed-plot check and crop verification will be skipped.")
        st.divider()

        check_missed = st.checkbox(" Check for missed plots", value=True, disabled=not GEMINI_API_KEY)
        verify_crops = st.checkbox(" Verify crop quality", value=True, disabled=not GEMINI_API_KEY)
        st.divider()
        show_discarded = st.checkbox("Show discarded crops too", value=False,
                                     help="Off by default — crops the verifier(s) recommend "
                                          "discarding (logos, headers, non-plot content) are "
                                          "hidden from the results below.")
        thumb_width = st.slider("Thumbnail width (px)", 150, 500, 260, step=10)

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if not uploaded:
        st.info("Upload a PDF to get started.")
        return

    pdf_bytes = uploaded.getvalue()
    stem = uploaded.name.rsplit(".", 1)[0]

    if not CV2_AVAILABLE:
        st.error("opencv-python is not installed — plot extraction is unavailable.")
        return

    if st.button(" Extract Plots", type="primary", use_container_width=True):
        GEMINI_MODEL = selected_gemini_model
        verify_engines = ["gemini"] + (["gpt"] if use_gpt else []) + (["claude"] if use_claude else [])
        verify_models = {"gemini": selected_gemini_model, "gpt": gpt_model, "claude": claude_model}

        with st.spinner(f"Detecting figures, checking for missed plots, verifying crops "
                        f"({', '.join(verify_engines)})…"):
            plot_results, coverage_report = extract_and_verify_plots(
                pdf_bytes, check_missed_plots=check_missed, verify_crops=verify_crops,
                verify_engines=verify_engines, verify_models=verify_models,
            )
        st.session_state["plot_results"] = plot_results
        st.session_state["coverage_report"] = coverage_report
        st.session_state["stem"] = stem
        st.session_state["verify_engines"] = verify_engines
        st.session_state["extracted"] = True

    if not st.session_state.get("extracted", False):
        return

    plot_results   = st.session_state["plot_results"]
    coverage_report = st.session_state["coverage_report"]
    stem            = st.session_state["stem"]
    verify_engines  = st.session_state.get("verify_engines", ["gemini"])

    def _action_of(img: Dict[str, Any]) -> str:
        v = img.get("verification")
        if not v:
            return "unverified"
        return v.get("majority_action", "unverified")

    total_images = sum(len(g["image_data"]) for g in plot_results)
    discarded_count = sum(
        1 for g in plot_results for img in g["image_data"] if _action_of(img) == "discard"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Figure groups", len(plot_results))
    c2.metric("Total images", total_images)
    c3.metric("Detected (local)", coverage_report.get("total_detected", 0))
    c4.metric("Recovered (Gemini)", coverage_report.get("total_recovered", 0))
    c5.metric("Discarded (hidden)", discarded_count if not show_discarded else 0)

    with st.expander(" Coverage report (missed-plot check)", expanded=False):
        if not coverage_report.get("gemini_check_ran"):
            st.warning("Skipped — no GEMINI_API_KEY or check disabled, so only local "
                      "detection ran; no recall check was possible.")
        else:
            for p in coverage_report["pages"]:
                flag = "  recovered figure(s) local detection missed" if p["recovered"] else ""
                st.write(f"Page {p['page']}: detected={p['detected']} "
                        f"gemini_reported={p['gemini_reported']} recovered={p['recovered']}{flag}")

    st.divider()

    if not plot_results:
        st.warning("No figures detected in this PDF.")
    else:
        verify_models_current = {"gemini": GEMINI_MODEL, "gpt": GPT_FALLBACK_MODELS[0], "claude": CLAUDE_FALLBACK_MODELS[0]}
        any_shown = False
        for group_idx, group in enumerate(plot_results):
            visible_indices = [
                i for i, img in enumerate(group["image_data"])
                if show_discarded or _action_of(img) != "discard"
            ]
            if not visible_indices:
                continue
            any_shown = True
            with st.container(border=True):
                st.markdown(f"**Page {group['page']}** — {group['caption']}")
                n_cols = min(len(visible_indices), 4) or 1
                cols = st.columns(n_cols)
                for pos, img_idx in enumerate(visible_indices):
                    img = group["image_data"][img_idx]
                    with cols[pos % n_cols]:
                        display_img = _resize_for_display(img["array"], thumb_width)
                        st.image(display_img, channels="BGR", width=thumb_width)
                        label = img.get("subplot_label") or ""
                        cap = img.get("subplot_caption") or ""
                        if label or cap:
                            st.caption(f"{label} {cap}".strip())

                        v = img.get("verification")
                        any_engine_flagged_recrop = False
                        if v and v.get("verified"):
                            majority = v.get("majority_action", "unverified")
                            badge = {"keep": "✅", "recrop": "⚠️", "discard": "❌"}.get(majority, "❔")
                            label_txt = f"{badge} {majority}"
                            if len(verify_engines) > 1:
                                label_txt += f" ({len(v['by_engine'])} engines)"
                            st.caption(label_txt)
                            any_engine_flagged_recrop = any(
                                verdict.get("recommended_action") == "recrop"
                                for verdict in v["by_engine"].values()
                            )
                            if len(verify_engines) > 1:
                                for engine, verdict in v["by_engine"].items():
                                    ebadge = {"keep": "✅", "recrop": "⚠️", "discard": "❌"}.get(
                                        verdict.get("recommended_action"), "❔")
                                    st.caption(f"  {ebadge} {engine}: {verdict.get('recommended_action')} "
                                              f"({verdict.get('confidence', '?')})")
                                if any_engine_flagged_recrop and majority != "recrop":
                                    st.caption(" at least one engine flagged this for recrop "
                                              "even though majority said otherwise")
                            else:
                                only = next(iter(v["by_engine"].values()), {})
                                if only.get("issues"):
                                    st.caption(f"Issues: {', '.join(only['issues'])}")
                        else:
                            st.caption("❔ unverified")
                        st.caption(f"source: {img.get('source', '')}")

                        # Re-split: available on any crop, but especially
                        # relevant when the crop looks like it might contain
                        # multiple merged charts — Gemini looks at THIS
                        # specific image and decides, rather than guessing
                        # from pixel density.
                        if GEMINI_API_KEY:
                            btn_label = " Re-split (looks merged)" if any_engine_flagged_recrop else " Check for merged charts"
                            if st.button(btn_label, key=f"resplit_{group_idx}_{img_idx}",
                                        use_container_width=True):
                                with st.spinner("Asking Gemini whether this is multiple merged charts…"):
                                    sub_crops = resplit_merged_crop(img["bytes"])
                                if not sub_crops:
                                    st.info("Gemini says this is a single chart — nothing to split.")
                                else:
                                    new_entries = []
                                    for k, sub_bytes in enumerate(sub_crops):
                                        nparr = np.frombuffer(sub_bytes, np.uint8)
                                        sub_arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                        new_img = {
                                            "filename": img["filename"].replace(".png", f"_split{k}.png"),
                                            "bytes": sub_bytes,
                                            "array": sub_arr,
                                            "subplot_label": None,
                                            "subplot_caption": None,
                                            "source": "gemini_resplit",
                                        }
                                        if verify_crops:
                                            new_img["verification"] = verify_crop_multi(
                                                sub_bytes, verify_engines, verify_models_current
                                            )
                                        else:
                                            new_img["verification"] = None
                                        new_entries.append(new_img)
                                    st.session_state["plot_results"][group_idx]["image_data"][img_idx:img_idx + 1] = new_entries
                                    st.success(f"Split into {len(new_entries)} chart(s).")
                                    st.rerun()

        if not any_shown:
            st.info("All detected crops were flagged for discard — enable "
                    "**Show discarded crops too** in the sidebar to review them.")

    st.divider()
    zip_bytes = create_plot_zip(plot_results)
    st.download_button(
        " Download all plots (ZIP + metadata JSON, includes discarded)",
        data=zip_bytes,
        file_name=f"{stem}_plots.zip",
        mime="application/zip",
        use_container_width=True,
    )


if __name__ == "__main__":
    _in_streamlit = False
    try:
        import streamlit.runtime.scriptrunner as _sr
        if _sr.get_script_run_ctx() is not None:
            _in_streamlit = True
    except Exception:
        pass

    if _in_streamlit:
        _run_streamlit()
    else:
        print("This script now runs as a Streamlit app.\nUsage:\n  streamlit run plot_extraction_verified.py")
        print(f"\nGemini model (3.0+): {GEMINI_MODEL}  (GEMINI_API_KEY set: {bool(GEMINI_API_KEY)})")