r"""
figures.py — single source of truth for everything image-related in the AIM
Composites pipeline (mirrors extraction.py's role for text).

Figures in a PDF — stress–strain curves, modulus-vs-temperature plots, property
tables scanned as images — are invisible to the text pass. This module makes
them first-class data with provenance, in three stages::

    pdf_bytes
       -> harvest_figures()      # PyMuPDF: embedded rasters + vector clusters -> PNGs
       -> classify_figures()     # ONE Gemini vision call per PDF: figure_kind + material
       -> mine_figure()          # one vision call per property_plot / table_image
       -> figure_properties_to_rows()  # -> PropertyRow(origin='figure', status='figure_estimate')

Non-negotiable principle
------------------------
A number read off a graph is an ESTIMATE, not a grounded fact. Figure-derived
rows are visibly second-class: ``origin='figure'``, their own status
``figure_estimate`` (never ``ok`` unless a human promotes them), a
``figure_id`` pointing at the PNG on disk, ``source_quote`` = the figure
caption, ``page`` = the figure page. They are never exported to the app
unless promoted (the Space's data_loader publish gate is ``status='ok'``).

Explicitly out of scope: full curve digitization / point-by-point tracing
(WebPlotDigitizer territory). Salient scalars only — labeled points, peaks,
plateaus, endpoints, legend-stated values, table cells.

Cost model: ≤ 1 classify call per PDF (0 when no figures), ≤ 1 mining call per
plot/table figure, hard cap ``max_figures`` per PDF. Every call goes through
extraction's retry/backoff. ``temperature=0``; ``FIGURE_PROMPT_VERSION`` is
stamped on every figure row (extraction.PROMPT_VERSION is untouched).
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

try:  # PyMuPDF
    import fitz  # type: ignore
except Exception:  # pragma: no cover - import guard
    fitz = None  # type: ignore

import extraction
from extraction import (
    GEMINI_MODEL,
    GEMINI_URL_TEMPLATE,
    Extraction,
    Material,
    Property,
    PropertyRow,
    SECTION_ENUM,
    _empty_value,
    _fill_numeric,
    _norm_section,
    canonicalize,
    gemini_request,
    plausibility_problem,
    to_rows,
)

log = logging.getLogger("figures")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FIGURE_PROMPT_VERSION = "1.0"

DEFAULT_FIGURES_DIR = Path("crawl_out/figures")
DEFAULT_MAX_FIGURES = 12

# Rendering: every stored PNG is a render of the figure's (slightly expanded)
# page region — this includes axis labels/legends drawn as vector text over
# an embedded raster, which the raw embedded image lacks. Longest side is
# capped so the PNG doubles as the Gemini payload (no separate downscale).
RENDER_MAX_SIDE_PX = 1600
RENDER_MIN_DPI = 72
RENDER_MARGIN_FRAC = 0.04     # expand bbox by 4% of its shorter side (min 6 pt)
RENDER_MARGIN_MIN_PT = 6.0

# Junk filters (raster route)
MIN_IMAGE_SIDE_PX = 120
MIN_IMAGE_AREA_PX = 15_000
MIN_PLACED_AREA_FRAC = 0.012  # placed smaller than 1.2% of the page = logo/icon
MAX_PLACED_AREA_FRAC = 0.90   # near-full-page = background/scan, not a figure
REPEATED_ON_PAGES = 3         # same image bytes on >= this many pages = header/footer art

# Junk filters (vector route)
MIN_CLUSTER_W_PT = 120.0
MIN_CLUSTER_H_PT = 80.0
MIN_CLUSTER_AREA_FRAC = 0.03
MAX_CLUSTER_AREA_FRAC = 0.90
HEADER_FOOTER_BAND_FRAC = 0.07   # clusters entirely inside the top/bottom 7% = decoration
CLUSTER_GEOM_ROUND_PT = 5.0      # geometry key rounding for the repeated-on-pages test
CLUSTER_OVERLAP_SKIP = 0.60      # cluster mostly covered by a harvested raster = same figure
# A vector cluster that is dense in LIVE TEXT and holds only a handful of
# drawing objects (a few ruled lines) is a text table, not a figure — the
# text pass already extracts it, grounded, so mining it would only spend a
# vision call to produce figure_estimate duplicates of ok rows. Measured on the
# corpus: datasheet tables 5-12 chars/kpt² with 1-5 objects; vector plots
# <2 chars/kpt² with 26-500+ objects. Set the density to 0 to disable.
VECTOR_TEXT_DENSITY_SKIP = 4.0   # chars per 1000 pt²
VECTOR_TEXT_TABLE_MAX_DRAWINGS = 20

CAPTION_RE = re.compile(r"^\s*(Fig(?:ure)?\.?|FIG\.?)\s*\d+", re.IGNORECASE)
CAPTION_MAX_CHARS = 600
CAPTION_GAP_BELOW_PT = 180.0
CAPTION_GAP_ABOVE_PT = 120.0
CAPTION_GAP_BESIDE_PT = 110.0   # margin captions (Springer style): beside the figure

FIGURE_KINDS = [
    "property_plot", "table_image", "micrograph", "photo", "schematic",
    "chemical_structure", "other",
]
MINEABLE_KINDS = ("property_plot", "table_image")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Figure:
    figure_id: str              # sha256(PNG bytes)[:16]
    source_pdf: str
    source_sha1: str
    page: int                   # 1-based
    bbox: tuple[float, float, float, float]   # PDF points, unexpanded figure region
    caption: str                # verbatim caption block text ('' if none)
    image_path: str
    image_sha256: str
    width_px: int
    height_px: int
    route: str                  # "raster" | "vector"
    # filled later
    figure_kind: str = ""
    material_guess: str = ""
    material_key: str = ""
    mining_status: str = ""     # "" | not_mined | skipped_kind | mined | mining_failed | classify_failed
    n_values: int = 0
    model: str = GEMINI_MODEL
    figure_prompt_version: str = FIGURE_PROMPT_VERSION
    # not persisted: raw PNG bytes for the vision calls
    png_bytes: bytes = dataclasses.field(default=b"", repr=False, compare=False)

    @property
    def label(self) -> str:
        """Short human label used in flag_reason: 'Figure 3' if the caption
        names it, else 'figure p7#2'."""
        m = re.match(r"\s*(Fig(?:ure)?\.?|FIG\.?)\s*(\d+)", self.caption or "", re.I)
        if m:
            return f"Figure {m.group(2)}"
        return f"figure p{self.page}"


@dataclasses.dataclass
class HarvestStats:
    """Filter counts, so a run report can prove the junk filters worked."""
    raster_candidates: int = 0
    vector_candidates: int = 0
    skipped_small: int = 0
    skipped_placed_tiny: int = 0
    skipped_fullpage: int = 0
    skipped_repeated: int = 0
    skipped_header_footer: int = 0
    skipped_overlap_raster: int = 0
    skipped_text_table: int = 0
    skipped_dup_sha: int = 0
    skipped_render_error: int = 0
    kept: int = 0
    capped: int = 0
    files_written: int = 0
    files_existing: int = 0

    def add(self, other: "HarvestStats") -> None:
        for f in dataclasses.fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))


@dataclasses.dataclass
class VisionStats:
    classify_calls: int = 0
    mining_calls: int = 0
    failed_calls: int = 0

    @property
    def total(self) -> int:
        return self.classify_calls + self.mining_calls

    def add(self, other: "VisionStats") -> None:
        self.classify_calls += other.classify_calls
        self.mining_calls += other.mining_calls
        self.failed_calls += other.failed_calls


@dataclasses.dataclass
class MinedFigure:
    """One figure's structured readout, before it becomes rows."""
    figure: Figure
    x_label: str = ""
    x_unit: str = ""
    y_label: str = ""
    y_unit: str = ""
    # (material_guess, series_name, Property) triples
    values: list[tuple[str, str, Property]] = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------
# Task 1 — harvest
# ---------------------------------------------------------------------------


def _pixmap_png(page: "fitz.Page", rect: "fitz.Rect") -> tuple[bytes, int, int]:
    """Render `rect` (already clipped to the page) so its longest side is
    <= RENDER_MAX_SIDE_PX; return (png_bytes, width_px, height_px)."""
    longest = max(rect.width, rect.height, 1.0)
    # PyMuPDF rounds the pixmap size up, so aim a hair under the cap.
    scale = (RENDER_MAX_SIDE_PX - 1) / longest
    scale = max(scale, RENDER_MIN_DPI / 72.0)
    # never upscale a region beyond what ~300 dpi would give
    scale = min(scale, 300.0 / 72.0)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=rect, alpha=False)
    if max(pix.width, pix.height) > RENDER_MAX_SIDE_PX:   # tiny regions at 300 dpi cap
        pix.shrink(1)
    return pix.tobytes("png"), pix.width, pix.height


def _expand(rect: "fitz.Rect", page_rect: "fitz.Rect") -> "fitz.Rect":
    m = max(RENDER_MARGIN_MIN_PT, RENDER_MARGIN_FRAC * min(rect.width, rect.height))
    return fitz.Rect(rect.x0 - m, rect.y0 - m, rect.x1 + m, rect.y1 + m) & page_rect


def _caption_blocks(page: "fitz.Page") -> list[tuple["fitz.Rect", str]]:
    out: list[tuple[fitz.Rect, str]] = []
    for b in page.get_text("blocks"):
        if len(b) < 5 or b[-1] != 0:      # block_type 0 = text
            continue
        text = b[4]
        if CAPTION_RE.match(text):
            clean = re.sub(r"\s+", " ", text).strip()[:CAPTION_MAX_CHARS]
            out.append((fitz.Rect(b[0], b[1], b[2], b[3]), clean))
    return out


def _h_overlap(a: "fitz.Rect", b: "fitz.Rect") -> float:
    return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))


def _v_overlap(a: "fitz.Rect", b: "fitz.Rect") -> float:
    return max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))


def _pair_caption(rect: "fitz.Rect", caps: list[tuple["fitz.Rect", str]]) -> str:
    """Nearest caption below the region (journal convention), else beside it
    (margin captions), else above, else the page's only caption. '' if none."""
    if not caps:
        return ""
    best: Optional[tuple[float, str]] = None
    for crect, text in caps:                       # below
        gap = crect.y0 - rect.y1
        if -5.0 <= gap <= CAPTION_GAP_BELOW_PT and _h_overlap(rect, crect) > 0:
            if best is None or gap < best[0]:
                best = (gap, text)
    if best:
        return best[1]
    for crect, text in caps:                       # beside (left or right margin)
        if _v_overlap(rect, crect) <= 0:
            continue
        gap = max(rect.x0 - crect.x1, crect.x0 - rect.x1)   # horizontal gap
        if 0.0 <= gap <= CAPTION_GAP_BESIDE_PT:
            if best is None or gap < best[0]:
                best = (gap, text)
    if best:
        return best[1]
    for crect, text in caps:                       # above
        gap = rect.y0 - crect.y1
        if -5.0 <= gap <= CAPTION_GAP_ABOVE_PT and _h_overlap(rect, crect) > 0:
            if best is None or gap < best[0]:
                best = (gap, text)
    if best:
        return best[1]
    if len(caps) == 1:
        return caps[0][1]
    return ""


def _frac_covered(inner: "fitz.Rect", outer: "fitz.Rect") -> float:
    """Fraction of `inner`'s area covered by `outer`."""
    if inner.is_empty:
        return 0.0
    i = inner & outer
    if i.is_empty:
        return 0.0
    return (i.width * i.height) / (inner.width * inner.height)


def harvest_figures(
    pdf_bytes: bytes,
    source_pdf: str,
    source_sha1: str,
    out_dir: Path | str = DEFAULT_FIGURES_DIR,
    max_figures: int = DEFAULT_MAX_FIGURES,
    stats: Optional[HarvestStats] = None,
) -> list[Figure]:
    """Harvest a PDF's figures as PNGs with page + bbox + caption provenance.

    Two routes: embedded raster images (``page.get_images``) and vector
    figure regions (``page.cluster_drawings``). Junk (logos, header/footer
    art, tiny/near-full-page images, repeated art) is filtered; everything is
    deduped by sha256 of the PNG; caption-matched figures are preferred when
    the ``max_figures`` cap bites. PNGs go to ``<out_dir>/<sha1>/p<page>_<n>.png``
    and are not rewritten if already present (re-runs write 0 new files).
    """
    st = stats if stats is not None else HarvestStats()
    if fitz is None:  # pragma: no cover
        log.warning("PyMuPDF not available; figure harvest skipped")
        return []
    out_dir = Path(out_dir)
    # MuPDF prints structure-tree warnings ("No common ancestor ...") to stderr
    # on some journal PDFs; they are harmless and drown the run log.
    prev_display = None
    try:
        prev_display = fitz.TOOLS.mupdf_display_errors(False)
    except Exception:
        pass
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        log.warning("harvest: cannot open %s: %s", source_pdf, exc)
        _restore_display(prev_display)
        return []

    try:
        n_pages = doc.page_count
        # ---- pass 1: count how many pages each embedded image / cluster geometry appears on
        img_pages: dict[str, set[int]] = {}
        geom_pages: dict[tuple[int, int], set[int]] = {}
        img_digest: dict[tuple[int, int], str] = {}   # (page, xref) -> raw-bytes digest
        for pno in range(n_pages):
            page = doc[pno]
            for im in page.get_images(full=True):
                xref = im[0]
                try:
                    raw = doc.extract_image(xref)
                    digest = hashlib.sha256(raw["image"]).hexdigest()
                except Exception:
                    continue
                img_digest[(pno, xref)] = digest
                img_pages.setdefault(digest, set()).add(pno)
            try:
                for c in page.cluster_drawings():
                    key = (int(round(c.width / CLUSTER_GEOM_ROUND_PT)),
                           int(round(c.height / CLUSTER_GEOM_ROUND_PT)))
                    geom_pages.setdefault(key, set()).add(pno)
            except Exception:
                pass

        # ---- pass 2: collect candidates
        candidates: list[dict[str, Any]] = []   # dicts with rect/page/route/caption/prio
        for pno in range(n_pages):
            page = doc[pno]
            prect = page.rect
            page_area = max(prect.width * prect.height, 1.0)
            caps = _caption_blocks(page)
            raster_rects: list[fitz.Rect] = []

            # raster route
            for im in page.get_images(full=True):
                xref = im[0]
                st.raster_candidates += 1
                digest = img_digest.get((pno, xref))
                if digest is None:
                    continue
                w, h = int(im[2] or 0), int(im[3] or 0)
                if w < MIN_IMAGE_SIDE_PX or h < MIN_IMAGE_SIDE_PX or w * h < MIN_IMAGE_AREA_PX:
                    st.skipped_small += 1
                    continue
                if len(img_pages.get(digest, ())) >= REPEATED_ON_PAGES:
                    st.skipped_repeated += 1
                    continue
                try:
                    rects = page.get_image_rects(xref)
                except Exception:
                    rects = []
                if not rects:
                    continue
                for r in rects:
                    frac = (r.width * r.height) / page_area
                    if frac < MIN_PLACED_AREA_FRAC:
                        st.skipped_placed_tiny += 1
                        continue
                    if frac > MAX_PLACED_AREA_FRAC:
                        st.skipped_fullpage += 1
                        continue
                    raster_rects.append(r)
                    candidates.append(dict(page=pno, rect=r, route="raster",
                                           caption=_pair_caption(r, caps)))

            # vector route
            try:
                clusters = page.cluster_drawings()
            except Exception:
                clusters = []
            drawing_rects: list[fitz.Rect] = []
            if clusters and VECTOR_TEXT_DENSITY_SKIP > 0:
                try:
                    drawing_rects = [fitz.Rect(d["rect"]) for d in page.get_drawings()]
                except Exception:
                    drawing_rects = []
            for c in clusters:
                st.vector_candidates += 1
                if c.width < MIN_CLUSTER_W_PT or c.height < MIN_CLUSTER_H_PT:
                    st.skipped_small += 1
                    continue
                frac = (c.width * c.height) / page_area
                if frac < MIN_CLUSTER_AREA_FRAC:
                    st.skipped_placed_tiny += 1
                    continue
                if frac > MAX_CLUSTER_AREA_FRAC:
                    st.skipped_fullpage += 1
                    continue
                band = HEADER_FOOTER_BAND_FRAC * prect.height
                if c.y1 <= prect.y0 + band or c.y0 >= prect.y1 - band:
                    st.skipped_header_footer += 1
                    continue
                key = (int(round(c.width / CLUSTER_GEOM_ROUND_PT)),
                       int(round(c.height / CLUSTER_GEOM_ROUND_PT)))
                if len(geom_pages.get(key, ())) >= REPEATED_ON_PAGES:
                    st.skipped_repeated += 1
                    continue
                if any(_frac_covered(c, rr) >= CLUSTER_OVERLAP_SKIP for rr in raster_rects):
                    st.skipped_overlap_raster += 1
                    continue
                if VECTOR_TEXT_DENSITY_SKIP > 0:
                    try:
                        n_chars = len(page.get_text("text", clip=c).strip())
                    except Exception:
                        n_chars = 0
                    density = n_chars / max(c.width * c.height / 1000.0, 1e-6)
                    n_draw = sum(1 for dr in drawing_rects if dr.intersects(c))
                    if density > VECTOR_TEXT_DENSITY_SKIP and n_draw <= VECTOR_TEXT_TABLE_MAX_DRAWINGS:
                        st.skipped_text_table += 1
                        continue
                candidates.append(dict(page=pno, rect=c, route="vector",
                                       caption=_pair_caption(c, caps)))

        # ---- priority: captioned first (page order), then raster, then vector
        def prio(c: dict[str, Any]) -> tuple[int, int, int, float]:
            return (0 if c["caption"] else 1, 0 if c["route"] == "raster" else 1,
                    c["page"], c["rect"].y0)
        candidates.sort(key=prio)

        # ---- render, dedup by PNG sha, cap
        figures: list[Figure] = []
        seen_sha: set[str] = set()
        per_page_n: dict[int, int] = {}
        for c in candidates:
            if len(figures) >= max_figures:
                st.capped += 1
                continue
            page = doc[c["page"]]
            rect = c["rect"]
            try:
                png, wpx, hpx = _pixmap_png(page, _expand(rect, page.rect))
            except Exception as exc:
                log.debug("render failed p%d: %s", c["page"] + 1, exc)
                st.skipped_render_error += 1
                continue
            sha = hashlib.sha256(png).hexdigest()
            if sha in seen_sha:
                st.skipped_dup_sha += 1
                continue
            seen_sha.add(sha)
            n = per_page_n.get(c["page"], 0) + 1
            per_page_n[c["page"]] = n
            dest_dir = out_dir / source_sha1
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"p{c['page'] + 1}_{n}.png"
            if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == sha:
                st.files_existing += 1
            else:
                dest.write_bytes(png)
                st.files_written += 1
            figures.append(Figure(
                figure_id=sha[:16], source_pdf=source_pdf, source_sha1=source_sha1,
                page=c["page"] + 1,
                bbox=(round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2)),
                caption=c["caption"], image_path=str(dest), image_sha256=sha,
                width_px=wpx, height_px=hpx, route=c["route"], png_bytes=png,
            ))
            st.kept += 1
        return figures
    finally:
        doc.close()
        _restore_display(prev_display)


def _restore_display(prev) -> None:
    if prev is None:
        return
    try:
        fitz.TOOLS.mupdf_display_errors(bool(prev))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Task 2 — classify (one batched vision call per PDF)
# ---------------------------------------------------------------------------

CLASSIFY_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "figures": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "index": {"type": "INTEGER"},
                    "figure_kind": {"type": "STRING", "enum": FIGURE_KINDS},
                    "material_guess": {"type": "STRING"},
                },
                "required": ["index", "figure_kind", "material_guess"],
            },
        }
    },
    "required": ["figures"],
}

CLASSIFY_PROMPT = (
    "You are an expert materials scientist. You will be shown {n} figures "
    "harvested from one PDF about polymer / fiber / composite materials, each "
    "preceded by a line '[figure i] page P — caption: ...'.\n\n"
    "For EVERY figure, return one entry with its `index` (the i shown), a "
    "`figure_kind` from: property_plot (a chart/graph with numeric axes — "
    "stress–strain curves, modulus vs temperature, bar charts of properties), "
    "table_image (a table of values rendered as an image), micrograph "
    "(SEM/optical/CT), photo (specimens, equipment, parts), schematic "
    "(diagrams, process sketches, setups), chemical_structure, or other.\n"
    "And a `material_guess`: the material the figure is about. Prefer one of "
    "these names already extracted from the document's text — {materials} — "
    "copied exactly; if none applies, a short generic name; '' if unclear.\n\n"
    "Return exactly {n} entries, one per index. Respond ONLY with JSON "
    "following the schema."
)


def _inline_png_part(png: bytes) -> dict[str, Any]:
    return {"inlineData": {"mimeType": "image/png",
                           "data": base64.b64encode(png).decode("ascii")}}


def _parse_json_response(resp) -> Optional[dict[str, Any]]:
    return extraction._parse_extraction_json(resp)


def classify_figures(
    figures: list[Figure],
    text_materials: list[str],
    api_key: str,
    *,
    model: str = GEMINI_MODEL,
    stats: Optional[VisionStats] = None,
) -> list[Figure]:
    """One Gemini vision call classifies all of a PDF's figures.

    Fills ``figure_kind`` and ``material_guess`` in place. Zero figures =>
    zero calls. On failure every figure gets ``mining_status='classify_failed'``
    and ``figure_kind='other'`` (nothing raises; the caller counts it).
    """
    vs = stats if stats is not None else VisionStats()
    if not figures:
        return figures
    mats = ", ".join(f"'{m}'" for m in dict.fromkeys(text_materials) if m) or "(none)"
    parts: list[dict[str, Any]] = [
        {"text": CLASSIFY_PROMPT.format(n=len(figures), materials=mats)}
    ]
    for i, f in enumerate(figures):
        cap = f.caption or "(no caption)"
        parts.append({"text": f"[figure {i}] page {f.page} — caption: {cap}"})
        parts.append(_inline_png_part(f.png_bytes))
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": CLASSIFY_SCHEMA,
        },
    }
    url = GEMINI_URL_TEMPLATE.format(model=model, key=api_key)
    vs.classify_calls += 1
    try:
        resp = gemini_request(url, payload)
        raw = _parse_json_response(resp) if resp is not None else None
    except Exception as exc:
        log.warning("figure classification failed: %s", exc)
        raw = None
    if not raw:
        vs.failed_calls += 1
        for f in figures:
            f.figure_kind = f.figure_kind or "other"
            f.mining_status = "classify_failed"
        return figures
    by_index: dict[int, dict[str, Any]] = {}
    for e in raw.get("figures") or []:
        try:
            by_index[int(e.get("index"))] = e
        except (TypeError, ValueError):
            continue
    for i, f in enumerate(figures):
        e = by_index.get(i, {})
        kind = (e.get("figure_kind") or "other").strip()
        f.figure_kind = kind if kind in FIGURE_KINDS else "other"
        f.material_guess = (e.get("material_guess") or "").strip()
        f.model = model
        f.mining_status = "not_mined" if f.figure_kind in MINEABLE_KINDS else "skipped_kind"
    return figures


# ---------------------------------------------------------------------------
# Task 3 — mine plots / table images (one vision call per figure)
# ---------------------------------------------------------------------------

# Same property shape as EXTRACTION_SCHEMA's items, minus source_quote/page
# (we set those from the figure) — so a mined value drops straight into
# extraction.Property.
_FIGURE_PROPERTY_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "section": {"type": "STRING", "enum": SECTION_ENUM},
        "property_name": {"type": "STRING"},
        "value_raw": {"type": "STRING"},
        "value_num": {"type": "NUMBER", "nullable": True},
        "value_min": {"type": "NUMBER", "nullable": True},
        "value_max": {"type": "NUMBER", "nullable": True},
        "qualifier": {"type": "STRING"},
        "unit": {"type": "STRING"},
        "test_condition": {"type": "STRING"},
        "comments": {"type": "STRING"},
    },
    "required": ["section", "property_name", "value_raw", "unit"],
}

MINING_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "figure_kind": {"type": "STRING", "enum": FIGURE_KINDS},
        "x_label": {"type": "STRING"},
        "x_unit": {"type": "STRING"},
        "y_label": {"type": "STRING"},
        "y_unit": {"type": "STRING"},
        "series": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "series_name": {"type": "STRING"},
                    "material_guess": {"type": "STRING"},
                    "values": {"type": "ARRAY", "items": _FIGURE_PROPERTY_SCHEMA},
                },
                "required": ["series_name", "material_guess", "values"],
            },
        },
    },
    "required": ["figure_kind", "series"],
}

MINING_PROMPT = (
    "You are an expert materials scientist reading ONE figure from a PDF about "
    "polymer / fiber / composite materials.\n"
    "Figure caption: {caption}\n"
    "Materials named in the document text: {materials}\n"
    "Prior classification: {kind}\n\n"
    "Read the figure and return a structured readout:\n"
    "- For a plot: x_label, x_unit, y_label, y_unit exactly as printed.\n"
    "- `series`: one entry per curve / bar group / table row-group. "
    "`series_name` = the legend or row label as printed; `material_guess` = "
    "the material it represents (prefer one of the document's material names, "
    "copied exactly; '' if unclear).\n"
    "- `values`: ONLY salient scalars a reader would state from this figure — "
    "labeled points, peaks / plateaus (e.g. ultimate tensile strength = the peak "
    "of a stress–strain curve), endpoints (e.g. strain at break), values stated "
    "in the legend/annotations, or table cells. Do NOT trace curves point by "
    "point. Do NOT invent values that are not readable. Read axis units "
    "carefully; if you must estimate from gridlines, use qualifier '~'.\n"
    "Each value: section (one of {sections}), property_name (e.g. 'Tensile "
    "strength', 'Strain at break', 'Storage modulus'), value_raw (the number as "
    "read, e.g. '~610' or '824'), value_num, unit (the axis/table unit), "
    "test_condition (e.g. '23 °C' if shown), comments (how it was read, e.g. "
    "'peak of PPS curve').\n"
    "Respond ONLY with JSON following the schema."
)


def mine_figure(
    fig: Figure,
    text_materials: list[str],
    api_key: str,
    *,
    model: str = GEMINI_MODEL,
    stats: Optional[VisionStats] = None,
) -> Optional[MinedFigure]:
    """One vision call: structured readout of a property_plot / table_image.

    Returns None (and sets ``fig.mining_status='mining_failed'``) on failure;
    sets ``'mined'`` and ``fig.n_values`` on success.
    """
    vs = stats if stats is not None else VisionStats()
    mats = ", ".join(f"'{m}'" for m in dict.fromkeys(text_materials) if m) or "(none)"
    prompt = MINING_PROMPT.format(
        caption=fig.caption or "(no caption)", materials=mats,
        kind=fig.figure_kind or "unknown", sections=", ".join(SECTION_ENUM),
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}, _inline_png_part(fig.png_bytes)]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": MINING_SCHEMA,
        },
    }
    url = GEMINI_URL_TEMPLATE.format(model=model, key=api_key)
    vs.mining_calls += 1
    try:
        resp = gemini_request(url, payload)
        raw = _parse_json_response(resp) if resp is not None else None
    except Exception as exc:
        log.warning("figure mining failed for %s p%d: %s", fig.source_pdf, fig.page, exc)
        raw = None
    if not raw:
        vs.failed_calls += 1
        fig.mining_status = "mining_failed"
        return None
    mined = MinedFigure(
        figure=fig,
        x_label=(raw.get("x_label") or "").strip(), x_unit=(raw.get("x_unit") or "").strip(),
        y_label=(raw.get("y_label") or "").strip(), y_unit=(raw.get("y_unit") or "").strip(),
    )
    for s in raw.get("series") or []:
        sname = (s.get("series_name") or "").strip()
        mguess = (s.get("material_guess") or "").strip()
        for pj in s.get("values") or []:
            value_raw = (pj.get("value_raw") or "").strip()
            prop = Property(
                section=_norm_section(pj.get("section")),
                property_name=(pj.get("property_name") or "").strip() or "Unknown property",
                value_raw=value_raw,
                unit=(pj.get("unit") or "").strip(),
                value_num=extraction._as_float(pj.get("value_num")),
                value_min=extraction._as_float(pj.get("value_min")),
                value_max=extraction._as_float(pj.get("value_max")),
                qualifier=(pj.get("qualifier") or "").strip(),
                test_condition=(pj.get("test_condition") or "").strip(),
                comments=(pj.get("comments") or "").strip(),
                source_quote=fig.caption,
                page=fig.page,
            )
            _fill_numeric(prop)
            mined.values.append((mguess, sname, prop))
    fig.mining_status = "mined"
    fig.model = model
    fig.n_values = len(mined.values)
    return mined


# ---------------------------------------------------------------------------
# Status + rows
# ---------------------------------------------------------------------------


def _figure_status(prop: Property, fig: Figure, series_name: str) -> None:
    """Set prop.status / flag_reason for a figure-derived value.

    Precedence: empty_value -> unit_review -> out_of_range -> figure_estimate.
    Never 'ok'; never verify_against_text (by definition not in the text).
    """
    unit_canonical, value_si, unit_problem = canonicalize(prop)
    prop.unit_canonical = unit_canonical
    prop.value_si = value_si
    origin = f"read from {fig.label} ({fig.figure_kind or 'figure'}"
    origin += f", series '{series_name}')" if series_name else ")"
    if _empty_value(prop):
        prop.status, prop.flag_reason = "empty_value", f"empty_value; {origin}"
        return
    if unit_problem:
        prop.status, prop.flag_reason = "unit_review", f"{unit_problem}; {origin}"
        return
    rng = plausibility_problem(prop)
    if rng:
        prop.status, prop.flag_reason = "out_of_range", f"{rng}; {origin}"
        return
    prop.status, prop.flag_reason = "figure_estimate", origin


def _match_text_material(guess: str, text_materials: list[Material]) -> Optional[Material]:
    g = re.sub(r"\s+", " ", (guess or "").strip().lower())
    if not g:
        return None
    for m in text_materials:               # exact (case-insensitive) first
        for cand in (m.material_name, m.material_abbreviation, m.trade_grade):
            if cand and cand.strip().lower() == g:
                return m
    for m in text_materials:               # then containment either way
        for cand in (m.material_name, m.material_abbreviation, m.trade_grade):
            c = (cand or "").strip().lower()
            if c and len(c) >= 3 and (c in g or g in c):
                return m
    return None


def figure_properties_to_rows(
    mined: list[MinedFigure],
    text_materials: list[Material],
    source_pdf: str,
    source_sha1: str,
) -> list[PropertyRow]:
    """Turn mined readouts into PropertyRows with origin='figure'.

    Each value is attached to the text-pass material it names (so it lands in
    the right table with the right material_key); unmatched guesses become
    their own material named after the guess (class via the deterministic
    keyword fallback). ``model`` / ``prompt_version`` are the figure ones.
    """
    # group by (material identity) -> Material with figure properties
    buckets: dict[str, Material] = {}
    fig_of_prop: dict[int, Figure] = {}
    for mf in mined:
        fig = mf.figure
        for mguess, sname, prop in mf.values:
            base = _match_text_material(mguess or fig.material_guess, text_materials)
            if base is not None:
                key = f"text::{id(base)}"
                mat = buckets.get(key)
                if mat is None:
                    mat = dataclasses.replace(base, properties=[])
                    buckets[key] = mat
            else:
                name = (mguess or fig.material_guess or "").strip() or "Unknown material (figure)"
                key = f"fig::{name.lower()}"
                mat = buckets.get(key)
                if mat is None:
                    mat = Material(material_name=name, properties=[])
                    buckets[key] = mat
            _figure_status(prop, fig, sname)
            mat.properties.append(prop)
            fig_of_prop[id(prop)] = fig

    ext = Extraction(materials=list(buckets.values()), model=GEMINI_MODEL,
                     prompt_version=FIGURE_PROMPT_VERSION)
    rows = to_rows(ext, source_pdf, source_sha1)
    # to_rows iterates materials x properties in order; walk the same order to
    # attach figure ids.
    props_in_order = [p for m in ext.materials for p in m.properties]
    out: list[PropertyRow] = []
    for row, prop in zip(rows, props_in_order):
        fig = fig_of_prop[id(prop)]
        row.origin = "figure"
        row.figure_id = fig.figure_id
        row.model = fig.model or GEMINI_MODEL
        row.prompt_version = fig.figure_prompt_version
        row.source_quote = fig.caption or f"[{fig.label}, no caption]"
        row.page = fig.page
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Convenience: the whole figure stage for one PDF
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FigureStageResult:
    figures: list[Figure]
    rows: list[PropertyRow]
    harvest: HarvestStats
    vision: VisionStats
    mined_figures: int = 0
    error: Optional[str] = None


def run_figure_stage(
    pdf_bytes: bytes,
    source_pdf: str,
    source_sha1: str,
    text_materials: list[Material],
    api_key: str,
    *,
    out_dir: Path | str = DEFAULT_FIGURES_DIR,
    max_figures: int = DEFAULT_MAX_FIGURES,
    mine: bool = True,
    model: str = GEMINI_MODEL,
) -> FigureStageResult:
    """harvest -> classify -> mine -> rows. Never raises; errors are recorded
    in ``.error`` so a figure-stage failure cannot kill a PDF's text rows."""
    hs, vs = HarvestStats(), VisionStats()
    res = FigureStageResult(figures=[], rows=[], harvest=hs, vision=vs)
    try:
        figs = harvest_figures(pdf_bytes, source_pdf, source_sha1, out_dir, max_figures, stats=hs)
        res.figures = figs
        if not figs:
            return res
        names = [n for m in text_materials for n in (m.material_name, m.material_abbreviation) if n]
        classify_figures(figs, names, api_key, model=model, stats=vs)
        if not mine:
            return res
        mined: list[MinedFigure] = []
        for f in figs:
            if f.figure_kind in MINEABLE_KINDS and f.mining_status != "classify_failed":
                mf = mine_figure(f, names, api_key, model=model, stats=vs)
                if mf is not None:
                    mined.append(mf)
        res.mined_figures = len(mined)
        res.rows = figure_properties_to_rows(mined, text_materials, source_pdf, source_sha1)
        # material_key on the figure record = the key of the material its rows landed on
        keyed = {r.figure_id: r.material_key for r in res.rows}
        for f in figs:
            f.material_key = keyed.get(f.figure_id, "")
    except Exception as exc:  # belt and braces: the caller must survive this
        log.exception("figure stage failed for %s", source_pdf)
        res.error = f"figure_stage_error:{type(exc).__name__}:{exc}"
    return res
