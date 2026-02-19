import os
import re
import json
import math
import tempfile
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import streamlit as st

# -------------------
# Config
# -------------------
DPI = 300
OUT_DIR = "outputs"

KEEP_ONLY_STRESS_STRAIN = False

CAP_RE = re.compile(r"^(Fig\.?\s*\d+|Figure\s*\d+)\b", re.IGNORECASE)
SS_KW  = re.compile(
    r"(stress\s*[-–]?\s*strain|stress|strain|tensile|MPa|GPa|kN|yield|elongation)",
    re.IGNORECASE
)

# -------------------
# Render helpers
# -------------------
def render_page(page, dpi=DPI):
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img, mat

def pdf_to_px_bbox(bbox_pdf, mat):
    x0, y0, x1, y1 = bbox_pdf
    sx, sy = mat.a, mat.d
    return (int(float(x0) * sx), int(float(y0) * sy), int(float(x1) * sx), int(float(y1) * sy))

def safe_crop_px(pil_img, box):
    if not isinstance(box, (tuple, list)):
        return None
    if len(box) == 1 and isinstance(box[0], (tuple, list)) and len(box[0]) == 4:
        box = box[0]
    if len(box) != 4:
        return None

    x0, y0, x1, y1 = box
    if any(isinstance(v, (tuple, list)) for v in (x0, y0, x1, y1)):
        return None

    try:
        x0 = int(x0)
        y0 = int(y0)
        x1 = int(x1)
        y1 = int(y1)
    except (TypeError, ValueError):
        return None

    if x1 < x0: 
        x0, x1 = x1, x0
    if y1 < y0: 
        y0, y1 = y1, y0

    W, H = pil_img.size
    x0 = max(0, min(W, x0))
    x1 = max(0, min(W, x1))
    y0 = max(0, min(H, y0))
    y1 = max(0, min(H, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return pil_img.crop((x0, y0, x1, y1))

# -------------------
# Captions
# -------------------
def find_caption_blocks(page):
    caps = []
    blocks = page.get_text("blocks")
    for b in blocks:
        x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
        t = " ".join(str(text).strip().split())
        if CAP_RE.match(t):
            caps.append({"bbox": (x0, y0, x1, y1), "text": t})
    return caps

# -------------------
# Dedupe: dHash
# -------------------
def dhash64(pil_img):
    gray = pil_img.convert("L").resize((9, 8), Image.LANCZOS)
    pixels = list(gray.getdata())
    bits = 0
    for r in range(8):
        for c in range(8):
            left = pixels[r * 9 + c]
            right = pixels[r * 9 + c + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits

# -------------------
# Rejectors
# -------------------
def has_colorbar_like_strip(pil_img):
    img = np.array(pil_img)
    if img.ndim != 3:
        return False
    H, W, _ = img.shape
    if W < 250 or H < 150:
        return False
    strip_w = max(18, int(0.07 * W))
    strip = img[:, W-strip_w:W, :]
    q = (strip // 24).reshape(-1, 3)
    uniq = np.unique(q, axis=0)
    return len(uniq) > 70

def texture_score(pil_img):
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())

def is_mostly_legend(pil_img):
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    bw = cv2.medianBlur(bw, 3)
    H, W = bw.shape
    fill = float(np.count_nonzero(bw)) / float(H * W)
    return (0.03 < fill < 0.18) and (min(H, W) < 260)

# -------------------
# Plot detection
# -------------------
def detect_axes_lines(pil_img):
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    H, W = gray.shape
    min_len = int(0.28 * min(H, W))

    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180,
        threshold=90,
        minLineLength=min_len,
        maxLineGap=14
    )
    if lines is None:
        return None, None

    horizontals, verticals = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = abs(x2-x1), abs(y2-y1)
        length = math.hypot(dx, dy)
        if dy < 18 and dx > 0.35 * W:
            horizontals.append((length, (x1, y1, x2, y2)))
        if dx < 18 and dy > 0.35 * H:
            verticals.append((length, (x1, y1, x2, y2)))

    if not horizontals or not verticals:
        return None, None

    horizontals.sort(key=lambda t: t[0], reverse=True)
    verticals.sort(key=lambda t: t[0], reverse=True)
    return horizontals[0][1], verticals[0][1]

def axis_intersection_ok(x_axis, y_axis, W, H):
    xa_y = int(round((x_axis[1] + x_axis[3]) / 2))
    ya_x = int(round((y_axis[0] + y_axis[2]) / 2))
    if not (0 <= xa_y < H and 0 <= ya_x < W):
        return False
    if ya_x > int(0.95 * W) or xa_y < int(0.05 * H):
        return False
    return True

def tick_text_presence_score(pil_img, x_axis, y_axis):
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    bw = cv2.medianBlur(bw, 3)

    H, W = gray.shape
    xa_y = int(round((x_axis[1] + x_axis[3]) / 2))
    ya_x = int(round((y_axis[0] + y_axis[2]) / 2))

    y0a = max(0, xa_y - 40)
    y1a = min(H, xa_y + 110)
    x_roi = bw[y0a:y1a, 0:W]

    x0b = max(0, ya_x - 180)
    x1b = min(W, ya_x + 50)
    y_roi = bw[0:H, x0b:x1b]

    def count_small_components(mask):
        num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cnt = 0
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if 4 <= w <= 150 and 4 <= h <= 150 and 20 <= area <= 5000:
                cnt += 1
        return cnt

    return count_small_components(x_roi) + count_small_components(y_roi)

def is_real_plot(pil_img):
    if has_colorbar_like_strip(pil_img):
        return False
    if is_mostly_legend(pil_img):
        return False

    x_axis, y_axis = detect_axes_lines(pil_img)
    if x_axis is None or y_axis is None:
        return False

    arr = np.array(pil_img)
    H, W = arr.shape[0], arr.shape[1]
    if not axis_intersection_ok(x_axis, y_axis, W, H):
        return False

    if texture_score(pil_img) > 2200:
        return False

    score = tick_text_presence_score(pil_img, x_axis, y_axis)
    return score >= 18

# -------------------
# Candidate boxes in a region
# -------------------
def connected_components_boxes(pil_img):
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray < 245).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    boxes = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        boxes.append((int(area), (int(x), int(y), int(x + w), int(y + h))))
    boxes.sort(key=lambda t: t[0], reverse=True)
    return boxes

def expand_box(box, W, H, left=0.10, right=0.06, top=0.06, bottom=0.18):
    x0, y0, x1, y1 = box
    bw = x1 - x0
    bh = y1 - y0
    ex0 = max(0, int(x0 - left * bw))
    ex1 = min(W, int(x1 + right * bw))
    ey0 = max(0, int(y0 - top * bh))
    ey1 = min(H, int(y1 + bottom * bh))
    return (ex0, ey0, ex1, ey1)

# -------------------
# Crop plot from caption
# -------------------
def crop_plot_from_caption(page_img, cap_bbox_pdf, mat):
    cap_px = pdf_to_px_bbox(cap_bbox_pdf, mat)
    cap_y0 = cap_px[1]
    cap_y1 = cap_px[3]

    W, H = page_img.size
    search_top = max(0, cap_y0 - int(0.95 * H))
    search_bot = min(H, cap_y1 + int(0.20 * H))
    region = safe_crop_px(page_img, (0, search_top, W, search_bot))
    if region is None:
        return None

    comps = connected_components_boxes(region)
    best = None
    best_area = -1

    for area, box in comps[:35]:
        x0, y0, x1, y1 = box
        bw = x1 - x0
        bh = y1 - y0
        if bw < 220 or bh < 180:
            continue

        exp = expand_box(box, region.size[0], region.size[1])
        cand = safe_crop_px(region, exp)
        if cand is None:
            continue

        if not is_real_plot(cand):
            continue

        if area > best_area:
            best_area = area
            best = cand

    return best

# -------------------
# Streamlit UI
# -------------------
def run_extraction(pdf_path, paper_id="uploaded_paper"):
    out_paper = os.path.join(OUT_DIR, paper_id)
    out_imgs = os.path.join(out_paper, "plots_with_axes")
    os.makedirs(out_imgs, exist_ok=True)

    doc = fitz.open(pdf_path)
    results = []
    seen = set()
    saved = 0

    for p in range(len(doc)):
        page = doc[p]
        caps = find_caption_blocks(page)
        if not caps:
            continue

        page_img, mat = render_page(page, dpi=DPI)

        for cap in caps:
            cap_text = cap["text"]

            if KEEP_ONLY_STRESS_STRAIN and not SS_KW.search(cap_text):
                continue

            fig = crop_plot_from_caption(page_img, cap["bbox"], mat)
            if fig is None:
                continue

            if fig.size[0] > 8 and fig.size[1] > 8:
                fig = fig.crop((2, 2, fig.size[0]-2, fig.size[1]-2))

            try:
                h = dhash64(fig)
            except Exception:
                continue

            if h in seen:
                continue
            seen.add(h)

            img_name = f"p{p+1:02d}_{saved:04d}.png"
            img_path = os.path.join(out_imgs, img_name)
            fig.save(img_path)

            results.append({
                "page": p + 1,
                "caption": cap_text,
                "image": img_path
            })
            saved += 1

    out_json = os.path.join(out_paper, "plots_with_axes.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results, out_json

def main():
    st.set_page_config(page_title="Research Paper Plot Extractor", layout="wide")
    st.title(" Plot Extractor (Upload PDF)")

    uploaded = st.file_uploader("Upload a research paper PDF", type=["pdf"])
    if not uploaded:
        st.info("Upload a PDF to extract plots.")
        return

    paper_id = os.path.splitext(uploaded.name)[0].replace(" ", "_")

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, uploaded.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded.read())

        with st.spinner("Extracting plots..."):
            results, out_json = run_extraction(pdf_path, paper_id=paper_id)

        st.success(f"Extracted {len(results)} plots.")

        # Show images + captions
        for r in results:
            st.markdown(f"**Page {r['page']}** — {r['caption']}")
            st.image(r["image"], use_container_width=True)
            st.divider()

        # JSON viewer + download
        st.subheader("JSON Output")
        st.json(results)

        with open(out_json, "rb") as f:
            st.download_button(
                "Download JSON",
                data=f,
                file_name=os.path.basename(out_json),
                mime="application/json"
            )

if __name__ == "__main__":
    main()
