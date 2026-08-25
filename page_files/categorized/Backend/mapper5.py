"""
Extraction mapping plots with properties
mapper.py — Plot ⇄ Property mapping + persistence
==========================================================================
Combines what used to be two files (Code 3 "mapping" + Code 4 "storage")
into one module. It sits on top of:



WHAT IT DOES
------------
1. MAP    — for each extracted property ROW (a Code 1 dataframe: Consensus /
            Source-Verified / per-model) find the figure/subplot in the
            Code 2 plot CROPS that depicts it, fusing four signals in
            descending confidence:
              (1) explicit "Fig. N"/"Fig. N(b)" citation in the row's
                  source_text/comments  ← the only grounding-strength signal
              (2) page co-location (same page, or ±1/±2 with less weight)
              (3) SciBERT semantic similarity (caption ↔ property), batched
                  into one property×caption matrix, reusing Code 1's embedder
              (4) token overlap (cheap complement / embedding fallback)
            Every produced link carries its score and which signals fired;
            a row with no confident figure is left unmapped, not forced.

2. STORE  — save each mapped crop's PNG to <out_dir>/<pdf_stem>/plots/,
            write the property rows into a SQLite DB, and attach the image
            file path(s) to each row (a `plot_image_path` column for the
            best crop + a `property_images` link table for the full set).
            Paths are stored RELATIVE to out_dir so the DB + plots/ zip is
            portable. Re-running for the same pdf_stem is idempotent.

PUBLIC API
----------
    map_plots_to_properties(df, plot_results, ...)  -> (links_df, df_augmented)
    build_figure_property_index(links_df)           -> figure → props df
    run_full_pipeline(pdf_bytes, source_table=...)   -> map, from a PDF
    store_properties_with_plots(df_aug, links_df, plot_results, out_dir, stem)
    run_full_pipeline_to_db(pdf_bytes, out_dir, stem, source_table=...)
    query_properties_with_plots(db_path, where_sql="", params=())
    bundle_zip(out_dir, pdf_stem, db_filename)

CLI:  streamlit run mapper.py     (needs 2.py + image2.py on the import path)
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import sqlite3
import zipfile
import typing
from typing import Any
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2

log = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")

EMBED_MODEL_NAME = "allenai/scibert_scivocab_uncased"  # must match Code 1 (2.py)

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE CONNECTION (RDS)
# Credentials are read from environment / .env — NEVER hard-code secrets here.
# Put these lines in a `.env` file next to this script and fill them in:
#     DB_HOST=aimindesdatabase.cfomsym2qoqo.us-east-2.rds.amazonaws.com
#     DB_PORT=5432                # 5432 = PostgreSQL, 3306 = MySQL
#     DB_NAME=<your database name>
#     DB_USER=<your username>
#     DB_PASSWORD=<your password>
#     DB_KIND=postgresql          # or: mysql
#     # optional full override (wins over the parts above):
#     # DB_URL=postgresql+psycopg2://user:pass@host:5432/dbname
# Requires:  pip install sqlalchemy python-dotenv
#            + PostgreSQL: pip install psycopg2-binary   (MySQL: pip install pymysql)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

DB_HOST     = os.getenv("DB_HOST", "aimindesdatabase.cfomsym2qoqo.us-east-2.rds.amazonaws.com")
DB_PORT     = os.getenv("DB_PORT", "5432")          # <-- 5432 Postgres · 3306 MySQL
DB_NAME     = os.getenv("DB_NAME", "")              # <-- ENTER (or set in .env)
DB_USER     = os.getenv("DB_USER", "")              # <-- ENTER (or set in .env)
DB_PASSWORD = os.getenv("DB_PASSWORD", "")          # <-- ENTER (or set in .env)
DB_KIND     = os.getenv("DB_KIND", "postgresql")    # "postgresql" or "mysql"
DB_URL      = os.getenv("DB_URL", "")               # full SQLAlchemy URL (optional override)

_db_engine = None


def db_configured() -> bool:
    """True once enough credentials are present to attempt a connection."""
    return bool(DB_URL or (DB_NAME and DB_USER and DB_PASSWORD))


def get_db_engine():
    """Lazily build (and cache) the SQLAlchemy engine for the RDS instance."""
    global _db_engine
    if _db_engine is not None:
        return _db_engine
    from sqlalchemy import create_engine
    if DB_URL:
        url = DB_URL
    elif DB_KIND.lower().startswith("mysql"):
        url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    else:
        url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    _db_engine = create_engine(url, pool_pre_ping=True)
    return _db_engine


def db_healthcheck() -> Any:
    """Return 1 if the DB is reachable; raises on failure (surface the error)."""
    from sqlalchemy import text
    with get_db_engine().connect() as c:
        return c.execute(text("SELECT 1")).scalar()


def push_properties_to_db(store: Dict[str, Any], table: str = "material_properties",
                          replace_stem: bool = True) -> Dict[str, Any]:
    """
    Push the stored property rows (store['df_store'], which carries
    plot_image_path) to the RDS table. Adds pdf_stem + source_table columns so
    rows are attributable, and (idempotently) deletes any prior rows for this
    pdf_stem before inserting so re-running a paper doesn't duplicate.

    NOTE: plot_image_path values are LOCAL relative paths. For a shared/cloud
    database, migrate images to S3 (store the URL) or a BYTEA/BLOB column —
    local paths won't resolve for other clients.
    """
    from sqlalchemy import text
    df = store.get("df_store")
    if df is None or df.empty:
        return {"pushed": 0, "table": table}
    df = df.copy()
    df["pdf_stem"] = store.get("pdf_stem", "")
    if "source_table" not in df.columns:
        df["source_table"] = store.get("source_table", "")

    engine = get_db_engine()
    if replace_stem:
        try:
            with engine.begin() as conn:
                conn.execute(text(f"DELETE FROM {table} WHERE pdf_stem = :s"),
                             {"s": store.get("pdf_stem", "")})
        except Exception as e:
            log.info(f"push_properties_to_db: no prior rows to clear ({e})")
    df.to_sql(table, engine, if_exists="append", index=False)
    log.info(f"push_properties_to_db: wrote {len(df)} row(s) to '{table}'")
    return {"pushed": len(df), "table": table}

# ─────────────────────────────────────────────────────────────────────────────
# S3 IMAGE STORAGE
# Credentials from environment / .env — NEVER hard-code. boto3 auto-reads
# AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION.
# Requires: pip install boto3
# ─────────────────────────────────────────────────────────────────────────────
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_PREFIX = os.getenv("S3_PREFIX", "plots").strip("/")
S3_REGION = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION", "us-east-2"))
S3_PUBLIC = os.getenv("S3_PUBLIC", "false").lower() in ("1", "true", "yes")

_s3_client = None


def s3_configured() -> bool:
    return bool(S3_BUCKET)


def get_s3_client():
    global _s3_client
    if _s3_client is None:
        import boto3
        _s3_client = boto3.client("s3", region_name=S3_REGION)
    return _s3_client

def s3_key_for(pdf_stem: str, filename: str) -> str:
    """One 'folder' per paper: <prefix>/<pdf_stem>/<filename>."""
    parts = [p for p in (S3_PREFIX, _safe(pdf_stem), filename) if p]
    return "/".join(parts)


def s3_url_for(key: str) -> str:
    if S3_PUBLIC:
        return f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{key}"
    return f"s3://{S3_BUCKET}/{key}"          # private bucket → store the URI


def upload_bytes_to_s3(data: bytes, key: str, content_type: str = "image/png") -> str:
    get_s3_client().put_object(Bucket=S3_BUCKET, Key=key,
                               Body=data, ContentType=content_type)
    return s3_url_for(key)


def s3_presigned_url(key: str, expires: int = 3600) -> str:
    """Temporary GET URL for viewing/downloading a private-bucket object."""
    return get_s3_client().generate_presigned_url(
        "get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=expires)


# ─────────────────────────────────────────────────────────────────────────────
# TUNABLES  (module constants so the paper can cite exact values)
# ─────────────────────────────────────────────────────────────────────────────

SAME_PAGE_SCORE = 1.00
ADJ_PAGE_SCORE  = 0.40   # |Δpage| == 1
NEAR_PAGE_SCORE = 0.15   # |Δpage| == 2
PAGE_WINDOW     = 2

W_PAGE  = 0.35           # fallback (non-citation) weights — sum to 1.0
W_EMBED = 0.45
W_TOKEN = 0.20
W_PAGE_NOEMBED  = 0.40   # redistribution when the embedder is unavailable
W_TOKEN_NOEMBED = 0.60

CITATION_BASE_SCORE    = 0.90
CITATION_SUBPLOT_BONUS = 0.05
CITATION_PAGE_BONUS    = 0.05

MAP_MIN_SCORE = 0.45     # fallback links below this are dropped (citation kept)
TOP_K         = 1

_PROP_ID_FIELDS = [
    "material_name", "material_abbreviation", "manufacturer",
    "section", "property_name", "value", "unit", "test_condition",
    "source_page", "chunk_type", "verified_by", "match_layer",
]

# Property fields stored as explicit, queryable DB columns (whichever exist)
_PROP_COLUMNS = [
    "material_name", "material_abbreviation", "manufacturer",
    "section", "property_name", "value", "unit", "english",
    "test_condition", "comments", "source_text", "source_page", "chunk_type",
    "verified_by", "match_layer", "doi_url",
]

# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower().strip())


def _tokenize(s: str) -> List[str]:
    s = re.sub(r"[^a-z0-9]+", " ", str(s).lower().strip())
    return [t for t in s.split() if len(t) >= 2]


def _token_overlap(a: str, b: str) -> float:
    sa, sb = set(_tokenize(a)), set(_tokenize(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _safe(s: Any) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")
    return out or "x"



def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _canonical_path(info: Dict[str, Any]) -> str:
    """Prefer the S3 URL; fall back to the local relative path."""
    return info.get("s3_url") or info.get("rel", "")


def _rowdict(row: pd.Series) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    for k, v in row.items():
        try:
            if pd.isna(v):
                v = ""
        except (TypeError, ValueError):
            pass
        d[k] = v
    return d


_embed_model: Optional[Any] = None
_embed_tried: bool = False


def _get_shared_embed_model() -> Optional[Any]:
    """Reuse Code 1's (2.py) already-loaded SciBERT if importable; else load
    our own copy; else None (→ token-only matching)."""
    global _embed_model, _embed_tried
    if _embed_model is not None or _embed_tried:
        return _embed_model
    _embed_tried = True
    try:
        import importlib
        _embed_model = importlib.import_module("Pdf_DataExtraction")._get_embed_model()  # Code 1 = 2.py
        log.info("mapper: reusing Code 1's SciBERT embedding model.")
        return _embed_model
    except Exception as e:
        log.info(f"mapper: Code 1 embedder unavailable ({e}); loading own copy.")
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        log.info("mapper: loaded standalone SciBERT embedding model.")
    except Exception as e:
        log.warning(f"mapper: no embedding model ({e}) — page + token-overlap only.")
        _embed_model = None
    return _embed_model


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE-REFERENCE PARSING
# ─────────────────────────────────────────────────────────────────────────────

_CAPTION_FIGNUM_RE = re.compile(r"\bfig(?:ure)?s?\b\.?\s*(\d+)", re.IGNORECASE)
_FIG_ANCHOR_RE = re.compile(r"\bfig(?:ure)?s?\b\.?", re.IGNORECASE)
_FIG_ITEM_RE = re.compile(
    r"\s*(\d+)\s*(?:\(\s*([a-h])\s*\)|([a-h])(?![a-z0-9]))?", re.IGNORECASE
)
_FIG_SEP_RE = re.compile(r"^\s*(?:,|and|&|-|\u2013|\u2014|to)\s*", re.IGNORECASE)


def _caption_figure_number(caption: str) -> Optional[int]:
    m = _CAPTION_FIGNUM_RE.search(caption or "")
    return int(m.group(1)) if m else None


def _parse_figure_refs(text: str) -> List[Tuple[int, Optional[str]]]:
    """Every (figure_number, optional_subplot_letter) the text cites.
    Handles 'Fig. 3', 'Figure 3(b)', 'Figs. 3 and 4', 'Figs 3, 4', 'Fig. 3-5'."""
    if not text:
        return []
    refs: List[Tuple[int, Optional[str]]] = []
    for anchor in _FIG_ANCHOR_RE.finditer(text):
        cursor = anchor.end()
        for _ in range(8):
            m = _FIG_ITEM_RE.match(text, cursor)
            if not m:
                break
            num = int(m.group(1))
            letter = m.group(2) or m.group(3)
            refs.append((num, letter.lower() if letter else None))
            cursor = m.end()
            sep = _FIG_SEP_RE.match(text, cursor)
            if not sep:
                break
            cursor = sep.end()
    seen: set = set()
    out: List[Tuple[int, Optional[str]]] = []
    for r in refs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _page_int(page_label: Any) -> Optional[int]:
    """Code 1 stores 'Page 3' / 'Unknown'; Code 2 stores an int."""
    if page_label is None:
        return None
    m = re.search(r"\d+", str(page_label))
    return int(m.group()) if m else None


# ─────────────────────────────────────────────────────────────────────────────
# PLOT CATALOG
# ─────────────────────────────────────────────────────────────────────────────

def _index_plots(plot_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    for gi, g in enumerate(plot_results or []):
        caption = g.get("caption", "") or ""
        page = g.get("page")
        fig_num = _caption_figure_number(caption)
        images = g.get("image_data", []) or []
        subplot_caps = " ".join(str(im.get("subplot_caption") or "") for im in images)
        groups.append({
            "group_idx": gi,
            "figure_id": f"p{page}_fig{fig_num if fig_num is not None else 'x' + str(gi)}",
            "caption": caption,
            "page": page,
            "figure_number": fig_num,
            "n_images": len(images),
            "subplot_labels": [im.get("subplot_label") for im in images],
            "text_for_embed": f"{caption} {subplot_caps}".strip(),
        })
    return groups


def _subplot_image_index(group: Dict[str, Any], letter: str) -> Optional[int]:
    for idx, lab in enumerate(group.get("subplot_labels", [])):
        if lab and lab.strip("()").lower() == letter:
            return idx
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SIMILARITY + SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _embed_matrix(prop_texts: List[str], group_texts: List[str]) -> Optional[np.ndarray]:
    model = _get_shared_embed_model()
    if model is None or not prop_texts or not group_texts:
        return None
    try:
        pv = model.encode(prop_texts, normalize_embeddings=True,
                          batch_size=128, show_progress_bar=False)
        gv = model.encode(group_texts, normalize_embeddings=True,
                          batch_size=128, show_progress_bar=False)
        return np.asarray(pv) @ np.asarray(gv).T  # [n_props, n_groups] cosine
    except Exception as e:
        log.warning(f"mapper: embedding matrix failed ({e}) — token-only.")
        return None


def _page_score(pp: Optional[int], pg: Optional[int]) -> float:
    if pp is None or pg is None:
        return 0.0
    d = abs(pp - pg)
    if d == 0:
        return SAME_PAGE_SCORE
    if d == 1:
        return ADJ_PAGE_SCORE
    if d <= PAGE_WINDOW:
        return NEAR_PAGE_SCORE
    return 0.0


def _fallback_score(pscore: float, tscore: float,
                    esim: Optional[float]) -> Tuple[float, List[str]]:
    signals: List[str] = []
    if esim is None:
        score = W_PAGE_NOEMBED * pscore + W_TOKEN_NOEMBED * tscore
    else:
        e = max(0.0, float(esim))
        score = W_PAGE * pscore + W_EMBED * e + W_TOKEN * tscore
        if e > 0:
            signals.append("embedding")
    if pscore > 0:
        signals.append("page")
    if tscore > 0:
        signals.append("tokens")
    return score, signals


# ─────────────────────────────────────────────────────────────────────────────
# CORE MAPPING
# ─────────────────────────────────────────────────────────────────────────────

def map_plots_to_properties(
    df: pd.DataFrame,
    plot_results: List[Dict[str, Any]],
    top_k: int = TOP_K,
    min_score: float = MAP_MIN_SCORE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Map each property row in `df` (any Code 1 output) to figure crop(s) in
    `plot_results` (Code 2 output).

    Returns:
      links_df    — one row per accepted (property, figure) link.
      df_augmented— `df` (index reset) plus mapped_figure_* / map_score /
                    map_signals columns for the single best link per row.
    """
    empty_links = pd.DataFrame()
    if df is None or df.empty or not plot_results:
        aug = df.reset_index(drop=True).copy() if df is not None else pd.DataFrame()
        for col in ("mapped_figure_id", "mapped_figure_caption", "mapped_figure_page",
                    "mapped_figure_number", "mapped_subplot", "map_score", "map_signals"):
            if not aug.empty:
                aug[col] = ""
        return empty_links, aug

    dfx = df.reset_index(drop=True).copy()
    groups = _index_plots(plot_results)

    num_index: Dict[int, List[int]] = {}
    for g in groups:
        if g["figure_number"] is not None:
            num_index.setdefault(g["figure_number"], []).append(g["group_idx"])

    prop_texts = [
        " ".join(str(r.get(f, "")) for f in
                 ("property_name", "material_name", "section", "test_condition"))
        for _, r in dfx.iterrows()
    ]
    group_texts = [g["text_for_embed"] for g in groups]
    sim = _embed_matrix(prop_texts, group_texts)

    id_fields = [f for f in _PROP_ID_FIELDS if f in dfx.columns]

    link_rows: List[Dict[str, Any]] = []
    best_by_row: Dict[int, Dict[str, Any]] = {}

    for i, row in dfx.iterrows():
        prop_page = _page_int(row.get("source_page"))
        prop_name = str(row.get("property_name", ""))
        material  = str(row.get("material_name", ""))
        cite_text = f"{row.get('source_text', '')} {row.get('comments', '')}"
        refs      = _parse_figure_refs(cite_text)

        candidates: List[Dict[str, Any]] = []

        # Signal 1: explicit figure citation (near-decisive)
        for num, letter in refs:
            for gi in num_index.get(num, []):
                g = groups[gi]
                score = CITATION_BASE_SCORE
                signals = ["figure_citation"]
                img_idx: Optional[int] = None
                subplot_lab: Optional[str] = None
                if letter is not None:
                    idx = _subplot_image_index(g, letter)
                    if idx is not None:
                        img_idx = idx
                        subplot_lab = f"({letter})"
                        score += CITATION_SUBPLOT_BONUS
                        signals.append("subplot")
                if prop_page is not None and g["page"] == prop_page:
                    score += CITATION_PAGE_BONUS
                    signals.append("page_confirm")
                candidates.append({
                    "group": g, "image_index": img_idx, "subplot_label": subplot_lab,
                    "score": min(score, 1.0), "signals": signals,
                })

        # Signals 2-4: page + semantic + token fallback
        if not candidates:
            for gi, g in enumerate(groups):
                pscore = _page_score(prop_page, g["page"])
                tscore = _token_overlap(f"{prop_name} {material}", g["caption"])
                esim = float(sim[i, gi]) if sim is not None else None
                score, signals = _fallback_score(pscore, tscore, esim)
                if score < min_score:
                    continue
                img_idx = 0 if g["n_images"] == 1 else None
                subplot_lab = (g["subplot_labels"][0] if g["n_images"] == 1 else None)
                candidates.append({
                    "group": g, "image_index": img_idx, "subplot_label": subplot_lab,
                    "score": round(score, 4), "signals": signals,
                })

        if not candidates:
            continue

        candidates.sort(key=lambda c: -c["score"])
        for rank, cand in enumerate(candidates[:max(1, top_k)]):
            g = cand["group"]
            base = {f: row.get(f, "") for f in id_fields}
            link = {
                **base,
                "prop_row": int(i),
                "group_idx": g["group_idx"],
                "figure_id": g["figure_id"],
                "figure_caption": g["caption"],
                "figure_page": g["page"],
                "figure_number": g["figure_number"],
                "matched_image_index": cand["image_index"],
                "matched_subplot": cand["subplot_label"],
                "match_score": round(float(cand["score"]), 4),
                "match_signals": ", ".join(cand["signals"]),
                "match_rank": rank + 1,
            }
            link_rows.append(link)
            if rank == 0:
                best_by_row[int(i)] = link

    links_df = pd.DataFrame(link_rows)

    aug = dfx.copy()
    aug["mapped_figure_id"]      = [best_by_row.get(i, {}).get("figure_id", "") for i in range(len(aug))]
    aug["mapped_figure_caption"] = [best_by_row.get(i, {}).get("figure_caption", "") for i in range(len(aug))]
    aug["mapped_figure_page"]    = [best_by_row.get(i, {}).get("figure_page", "") for i in range(len(aug))]
    aug["mapped_figure_number"]  = [best_by_row.get(i, {}).get("figure_number", "") for i in range(len(aug))]
    aug["mapped_subplot"]        = [best_by_row.get(i, {}).get("matched_subplot", "") for i in range(len(aug))]
    aug["map_score"]             = [best_by_row.get(i, {}).get("match_score", "") for i in range(len(aug))]
    aug["map_signals"]           = [best_by_row.get(i, {}).get("match_signals", "") for i in range(len(aug))]

    n_mapped = len(best_by_row)
    n_cited  = int(links_df["match_signals"].str.contains("figure_citation").sum()) if not links_df.empty else 0
    log.info(f"mapper: {n_mapped}/{len(dfx)} rows mapped ({n_cited} via citation, "
             f"rest page+semantic); {len(links_df)} total link(s).")
    return links_df, aug


def build_figure_property_index(links_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse links to one row per figure, listing the properties on it."""
    if links_df is None or links_df.empty:
        return pd.DataFrame()
    out_rows: List[Dict[str, Any]] = []
    for fig_id, grp in links_df.groupby("figure_id"):
        props = []
        for _, r in grp.iterrows():
            unit = str(r.get("unit", "")).strip()
            val = str(r.get("value", "")).strip()
            sub = f" {r['matched_subplot']}" if r.get("matched_subplot") else ""
            props.append(
                f"{r.get('property_name', '')}={val}{(' ' + unit) if unit else ''}"
                f" [{r.get('material_name', '')}]{sub}"
            )
        first = grp.iloc[0]
        out_rows.append({
            "figure_id": fig_id,
            "figure_number": first.get("figure_number", ""),
            "figure_page": first.get("figure_page", ""),
            "figure_caption": first.get("figure_caption", ""),
            "n_properties": len(grp),
            "mean_match_score": round(float(grp["match_score"].mean()), 4),
            "properties": " ; ".join(props),
        })
    return pd.DataFrame(out_rows).sort_values(
        ["figure_page", "figure_number"], na_position="last"
    ).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAP FROM A PDF  (Code 1 → Code 2 → map)
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    pdf_bytes: bytes,
    source_table: str = "consensus",
    doi_override: str = "",
    top_k: int = TOP_K,
    min_score: float = MAP_MIN_SCORE,
    verify_engines: Optional[List[str]] = None,
    check_missed_plots: bool = True,
    verify_crops: bool = True,
) -> Dict[str, Any]:
    """Run Code 1 (2.py) extraction, Code 2 (image2.py) plots, then map.
    `source_table` ∈ {consensus, verified, consensus_verified, gemini, gpt, claude}."""
    import importlib
    run_pipeline = importlib.import_module("Pdf_DataExtraction").run_pipeline  # Code 1 = 2.py
    from Pdf_ImageExtraction import extract_and_verify_plots               # Code 2 = image2.py

    (df_consensus, df_verified, df_consensus_verified,
     df_gemini, df_gpt, df_claude,
     _chunks, api_errors, meta) = run_pipeline(pdf_bytes, doi_override=doi_override)

    table_map = {
        "consensus":          df_consensus,
        "verified":           df_verified,
        "consensus_verified": df_consensus_verified,
        "gemini":             df_gemini,
        "gpt":                df_gpt,
        "claude":             df_claude,
    }
    source_df = table_map.get(source_table, df_consensus)

    plot_results, coverage_report = extract_and_verify_plots(
        pdf_bytes,
        check_missed_plots=check_missed_plots,
        verify_crops=verify_crops,
        verify_engines=verify_engines or ["gemini"],
    )

    links_df, df_aug = map_plots_to_properties(
        source_df, plot_results, top_k=top_k, min_score=min_score
    )
    figure_index = build_figure_property_index(links_df)

    return {
        "links_df": links_df,
        "df_augmented": df_aug,
        "figure_index": figure_index,
        "plot_results": plot_results,
        "coverage_report": coverage_report,
        "source_df": source_df,
        "source_table": source_table,
        "meta": meta,
        "api_errors": api_errors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SAVE CROPS TO DISK
# ─────────────────────────────────────────────────────────────────────────────

def save_plot_images(
    plot_results: List[Dict[str, Any]],
    out_dir: str,
    pdf_stem: str,
    group_idxs: Optional[set] = None,
    upload_s3: Optional[bool] = None,
    write_local: bool = True,
) -> Tuple[Dict[Tuple[int, int], Dict[str, Any]], List[Dict[str, Any]]]:
    """Write each figure crop's PNG bytes to <out_dir>/<pdf_stem>/plots/.
    Returns (path_map keyed by (group_idx, image_index), groups catalog).
    Paths in path_map are POSIX-style relative to out_dir."""
    groups = _index_plots(plot_results)
    rel_root = f"{pdf_stem}/plots"
    abs_root = os.path.join(out_dir, pdf_stem, "plots")
    os.makedirs(abs_root, exist_ok=True)
    if upload_s3 is None:
        upload_s3 = s3_configured()

    path_map: Dict[Tuple[int, int], Dict[str, Any]] = {}
    used_names: set = set()

    for g in groups:
        gi = g["group_idx"]
        if group_idxs is not None and gi not in group_idxs:
            continue
        for idx, img in enumerate(plot_results[gi].get("image_data", []) or []):
            data = img.get("bytes")
            if not data:
                continue
            sub = img.get("subplot_label")
            suffix = _safe(sub.strip("()")) if sub else f"img{idx}"
            base = f"{_safe(g['figure_id'])}_{suffix}"
            name = f"{base}.png"
            k = 1
            while name in used_names:
                name = f"{base}_{k}.png"
                k += 1
            used_names.add(name)

            entry = {
                "rel": f"{rel_root}/{name}",
                "subplot": sub,
                "source": img.get("source"),
                "verification": img.get("verification"),
            }
            if write_local:
                abs_path = os.path.join(abs_root, name)
                with open(abs_path, "wb") as fh:
                    fh.write(data)
                entry["abs"] = abs_path
            if upload_s3:
                try:
                    key = s3_key_for(pdf_stem, name)
                    entry["s3_key"] = key
                    entry["s3_url"] = upload_bytes_to_s3(data, key)
                except Exception as e:
                    log.warning(f"mapper: S3 upload failed for {name} ({e}); using local path.")
            path_map[(gi, idx)] = entry
    log.info(f"mapper: saved {len(path_map)} plot crop(s) to {abs_root}")
    return path_map, groups


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS figures (
    pdf_stem       TEXT,
    figure_id      TEXT,
    page           INTEGER,
    figure_number  INTEGER,
    caption        TEXT,
    PRIMARY KEY (pdf_stem, figure_id)
);
CREATE TABLE IF NOT EXISTS figure_images (
    image_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    pdf_stem             TEXT,
    figure_id            TEXT,
    subplot_label        TEXT,
    source               TEXT,
    verification_action  TEXT,
    file_path            TEXT
);
CREATE TABLE IF NOT EXISTS properties (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    pdf_stem              TEXT,
    source_table          TEXT,
    material_name         TEXT,
    material_abbreviation TEXT,
    manufacturer          TEXT,
    section               TEXT,
    property_name         TEXT,
    value                 TEXT,
    unit                  TEXT,
    english               TEXT,
    test_condition        TEXT,
    comments              TEXT,
    source_text           TEXT,
    source_page           TEXT,
    chunk_type            TEXT,
    verified_by           TEXT,
    match_layer           TEXT,
    doi_url               TEXT,
    mapped_figure_id      TEXT,
    mapped_figure_number  TEXT,
    mapped_figure_page    TEXT,
    mapped_subplot        TEXT,
    map_score             REAL,
    map_signals           TEXT,
    plot_image_path       TEXT,
    plot_image_paths      TEXT,
    row_json              TEXT
);
CREATE TABLE IF NOT EXISTS property_images (
    property_id  INTEGER,
    image_id     INTEGER,
    relation     TEXT
);
CREATE INDEX IF NOT EXISTS ix_props_stem  ON properties(pdf_stem);
CREATE INDEX IF NOT EXISTS ix_props_fig   ON properties(mapped_figure_id);
CREATE INDEX IF NOT EXISTS ix_figimg_stem ON figure_images(pdf_stem, figure_id);
"""


def _insert(cur: sqlite3.Cursor, table: str, dct: Dict[str, Any]) -> int:
    cols = list(dct.keys())
    placeholders = ",".join(["?"] * len(cols))
    cur.execute(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})",
                [dct[c] for c in cols])
    return int(cur.lastrowid)


def store_properties_with_plots(
    df_augmented: pd.DataFrame,
    links_df: pd.DataFrame,
    plot_results: List[Dict[str, Any]],
    out_dir: str,
    pdf_stem: str,
    source_table: str = "consensus",
    db_filename: Optional[str] = None,
    save_all_group_images: bool = False,
) -> Dict[str, Any]:
    """Persist `df_augmented` into a SQLite DB under `out_dir`, save mapped
    crops to <out_dir>/<pdf_stem>/plots/, and attach the file path(s) to each
    row. Idempotent per pdf_stem. Returns a summary dict."""
    os.makedirs(out_dir, exist_ok=True)
    df = df_augmented.reset_index(drop=True).copy()

    mapped_gis = set(int(g) for g in links_df["group_idx"].tolist()) if not links_df.empty else set()
    # None = save every detected crop. If nothing mapped (empty links), fall back
    # to saving all crops so images still persist instead of writing zero.
    group_idxs = None if (save_all_group_images or not mapped_gis) else mapped_gis

    path_map, groups = save_plot_images(plot_results, out_dir, pdf_stem, group_idxs=group_idxs)
    group_by_idx = {g["group_idx"]: g for g in groups}

    best_link: Dict[int, pd.Series] = {}
    if not links_df.empty:
        for _, l in links_df[links_df["match_rank"] == 1].iterrows():
            best_link[int(l["prop_row"])] = l

    db_filename = db_filename or f"{pdf_stem}.db"
    db_path = os.path.join(out_dir, db_filename)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript(_SCHEMA)

    # idempotency
    cur.execute("SELECT id FROM properties WHERE pdf_stem = ?", (pdf_stem,))
    old_prop_ids = [r[0] for r in cur.fetchall()]
    if old_prop_ids:
        cur.executemany("DELETE FROM property_images WHERE property_id = ?",
                        [(pid,) for pid in old_prop_ids])
    cur.execute("DELETE FROM properties    WHERE pdf_stem = ?", (pdf_stem,))
    cur.execute("DELETE FROM figure_images WHERE pdf_stem = ?", (pdf_stem,))
    cur.execute("DELETE FROM figures       WHERE pdf_stem = ?", (pdf_stem,))

    # figures + figure_images
    img_id_map: Dict[Tuple[int, int], int] = {}
    seen_fig: set = set()
    for (gi, idx), info in path_map.items():
        g = group_by_idx[gi]
        if gi not in seen_fig:
            seen_fig.add(gi)
            cur.execute(
                "INSERT OR REPLACE INTO figures (pdf_stem, figure_id, page, figure_number, caption) "
                "VALUES (?,?,?,?,?)",
                (pdf_stem, g["figure_id"], g["page"], g["figure_number"], g["caption"]),
            )
        verdict = info.get("verification")
        action = verdict.get("majority_action", "") if isinstance(verdict, dict) else ""
        img_id_map[(gi, idx)] = _insert(cur, "figure_images", {
            "pdf_stem": pdf_stem,
            "figure_id": g["figure_id"],
            "subplot_label": info.get("subplot") or "",
            "source": info.get("source") or "",
            "verification_action": action,
            "file_path": _canonical_path(info),
        })

    # properties + property_images
    n_props = 0
    n_linked = 0
    store_paths: List[str] = []
    for prow, row in df.iterrows():
        prow = int(prow)
        link = best_link.get(prow)

        primary_rel = ""
        all_rels: List[str] = []
        primary_images: List[Tuple[int, str]] = []

        if link is not None:
            gi = int(link["group_idx"])
            mii = link.get("matched_image_index")
            has_specific = mii is not None and not (isinstance(mii, float) and pd.isna(mii))
            if has_specific and (gi, int(mii)) in path_map:
                key = (gi, int(mii))
                primary_rel = _canonical_path(path_map[key])
                all_rels = [primary_rel]
                primary_images = [(img_id_map[key], "primary")]
            else:
                group_imgs = sorted((k for k in path_map if k[0] == gi), key=lambda k: k[1])
                all_rels = [_canonical_path(path_map[k]) for k in group_imgs]
                primary_rel = all_rels[0] if all_rels else ""
                primary_images = [(img_id_map[k], "group") for k in group_imgs]

        record = {"pdf_stem": pdf_stem, "source_table": source_table}
        for col in _PROP_COLUMNS:
            record[col] = "" if col not in df.columns else ("" if pd.isna(row.get(col)) else row.get(col))
        record.update({
            "mapped_figure_id":     row.get("mapped_figure_id", "")           if "mapped_figure_id" in df.columns else "",
            "mapped_figure_number": str(row.get("mapped_figure_number", ""))  if "mapped_figure_number" in df.columns else "",
            "mapped_figure_page":   str(row.get("mapped_figure_page", ""))    if "mapped_figure_page" in df.columns else "",
            "mapped_subplot":       row.get("mapped_subplot", "")             if "mapped_subplot" in df.columns else "",
            "map_score":            _to_float(row.get("map_score")) if "map_score" in df.columns else None,
            "map_signals":          row.get("map_signals", "")               if "map_signals" in df.columns else "",
            "plot_image_path":      primary_rel,
            "plot_image_paths":     " ; ".join(all_rels),
            "row_json":             json.dumps(_rowdict(row), default=str, ensure_ascii=False),
        })

        prop_id = _insert(cur, "properties", record)
        n_props += 1
        store_paths.append(primary_rel)
        for image_id, relation in primary_images:
            cur.execute("INSERT INTO property_images (property_id, image_id, relation) VALUES (?,?,?)",
                        (prop_id, image_id, relation))
            n_linked += 1

    conn.commit()
    conn.close()

    df_store = df.copy()
    df_store["plot_image_path"] = store_paths
    n_with_image = sum(1 for p in store_paths if p)
    log.info(f"mapper: stored {n_props} propert(ies) → {db_path}; {n_with_image} with a "
             f"plot, {len(img_id_map)} image file(s), {n_linked} property↔image link(s).")

    return {
        "db_path": db_path,
        "db_filename": db_filename,
        "image_dir": os.path.join(out_dir, pdf_stem, "plots"),
        "out_dir": out_dir,
        "pdf_stem": pdf_stem,
        "n_properties": n_props,
        "n_properties_with_plot": n_with_image,
        "n_images_saved": len(img_id_map),
        "n_property_image_links": n_linked,
        "df_store": df_store,
    }


# ─────────────────────────────────────────────────────────────────────────────
# READ-BACK + BUNDLE
# ─────────────────────────────────────────────────────────────────────────────

def query_properties_with_plots(db_path: str, where_sql: str = "",
                                params: tuple = ()) -> pd.DataFrame:
    """Read stored property rows back (each carries plot_image_path)."""
    conn = sqlite3.connect(db_path)
    try:
        q = "SELECT * FROM properties"
        if where_sql:
            q += f" WHERE {where_sql}"
        return pd.read_sql_query(q, conn, params=params)
    finally:
        conn.close()


def fetch_property_image_paths(db_path: str, property_id: int) -> List[Dict[str, Any]]:
    """All image files linked to one stored property."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT fi.file_path, fi.subplot_label, fi.verification_action, pi.relation "
            "FROM property_images pi JOIN figure_images fi ON fi.image_id = pi.image_id "
            "WHERE pi.property_id = ?",
            (property_id,),
        )
        return [
            {"file_path": r[0], "subplot_label": r[1],
             "verification_action": r[2], "relation": r[3]}
            for r in cur.fetchall()
        ]
    finally:
        conn.close()


def bundle_zip(out_dir: str, pdf_stem: str, db_filename: str) -> bytes:
    """Zip the DB + pdf_stem/plots/ into one portable archive (paths relative
    to out_dir, so the DB's stored image paths resolve after unzip)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        db_path = os.path.join(out_dir, db_filename)
        if os.path.exists(db_path):
            z.write(db_path, arcname=db_filename)
        plots_dir = os.path.join(out_dir, pdf_stem, "plots")
        for root, _dirs, files in os.walk(plots_dir):
            for fn in files:
                fp = os.path.join(root, fn)
                z.write(fp, arcname=os.path.relpath(fp, out_dir))
    buf.seek(0)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE → DB
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline_to_db(
    pdf_bytes: bytes,
    out_dir: str,
    pdf_stem: str,
    source_table: str = "consensus",
    doi_override: str = "",
    top_k: int = TOP_K,
    min_score: float = MAP_MIN_SCORE,
    verify_engines: Optional[List[str]] = None,
    save_all_group_images: bool = False,
) -> Dict[str, Any]:
    """Code 1 (extract) → Code 2 (plots) → map → store. Returns the map
    result dict plus a 'store' key with the persistence summary."""
    res = run_full_pipeline(
        pdf_bytes, source_table=source_table, doi_override=doi_override,
        top_k=top_k, min_score=min_score, verify_engines=verify_engines,
    )
    res["store"] = store_properties_with_plots(
        res["df_augmented"], res["links_df"], res["plot_results"],
        out_dir=out_dir, pdf_stem=pdf_stem, source_table=source_table,
        save_all_group_images=save_all_group_images,
    )
    return res


# ─────────────────────────────────────────────────────────────────────────────
# NON-PLOT GATE + EXTRACTION CACHE  (glue only — never mutates 2.py/image2.py output)
# ─────────────────────────────────────────────────────────────────────────────

# Captions that name a photo / micrograph / spectrum / logo → not a data plot.
_NONPLOT_CAPTION_RE = re.compile(
    r"\b(sem|tem|afm|micrograph|micrographs|photograph|photographs|photo|"
    r"optical image|optical micrograph|schematic|logo)\b",
    re.IGNORECASE,
)


def select_real_plots(plot_results, use_caption=True, use_verifier=True,
                      drop_unlabeled=False):
    """Return a FILTERED COPY of image2.py's plot_results, keeping only real
    data plots. This is pure selection — the upstream outputs are untouched,
    so results stay identical to what 2.py / image2.py produced.

    Drops:
      - figure groups whose caption names a photo/micrograph/logo (caption gate)
      - individual crops the verifier's aggregated verdict marked "discard"
        (keeps every "keep" and "recrop" — recrop is a real plot, not a non-plot)
      - optionally, unlabeled detections with no 'Fig. N' caption
    """
    kept = []
    for g in plot_results or []:
        cap = g.get("caption", "") or ""
        if use_caption and _NONPLOT_CAPTION_RE.search(cap):
            continue
        if drop_unlabeled and _caption_figure_number(cap) is None:
            continue
        imgs = []
        for im in g.get("image_data", []) or []:
            if use_verifier:
                # Drop ONLY on the verifier's aggregated "discard" verdict — the same
                # label shown in the mirror view. This removes logos / photos / SEM /
                # schematics (majority_action == "discard") while KEEPING every "keep"
                # and "recrop" crop. "recrop" is a real plot with merged panels, not a
                # non-plot, so it must survive; dropping on a single dissenting engine
                # was wrongly pruning genuine multi-panel figures (e.g. the stacked DMA
                # and 2x2 enthalpy grids).
                v = im.get("verification") or {}
                if isinstance(v, dict) and v.get("majority_action") == "discard":
                    continue
            imgs.append(im)
        if imgs:
            kept.append({**g, "image_data": imgs})
    return kept


CACHE_DIR = os.getenv("MAPPER_CACHE_DIR", ".mapper_cache")


def _pdf_hash(pdf_bytes: bytes) -> str:
    import hashlib
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


def _extract(pdf_bytes: bytes, doi_override: str = "",
             verify_engines=None) -> Dict[str, Any]:
    """Run Code 1 (2.py) + Code 2 (image2.py) ONCE. Returns a bundle with all
    six property tables + plot_results + coverage + meta + errors."""
    import importlib
    run_pipeline = importlib.import_module("Pdf_DataExtraction").run_pipeline
    from Pdf_ImageExtraction import extract_and_verify_plots
    (dfc, dfv, dfcv, dfg, dfp, dfcl, _chunks, errors, meta) = run_pipeline(
        pdf_bytes, doi_override=doi_override)
    plot_results, coverage = extract_and_verify_plots(
        pdf_bytes, verify_engines=verify_engines or ["gemini"])
    return {
        "table_map": {"consensus": dfc, "verified": dfv, "consensus_verified": dfcv,
                      "gemini": dfg, "gpt": dfp, "claude": dfcl},
        "plot_results": plot_results,
        "coverage": coverage,
        "meta": meta,
        "api_errors": errors,
    }


def get_or_extract(pdf_bytes: bytes, doi_override: str = "", verify_engines=None,
                   use_disk_cache: bool = True, force: bool = False):
    """Extraction cached by the PDF's CONTENT HASH — never by source_table.
    On a cache hit, 2.py and image2.py are NOT re-run, so switching tables /
    pruning / re-mapping is free. Returns (pdf_hash, bundle)."""
    import pickle
    h = _pdf_hash(pdf_bytes)
    path = os.path.join(CACHE_DIR, f"{h}.pkl")
    if use_disk_cache and not force and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                bundle = pickle.load(f)
            bundle["_cache"] = "disk"
            return h, bundle
        except Exception as e:
            log.warning(f"mapper: cache read failed ({e}); re-extracting.")
    bundle = _extract(pdf_bytes, doi_override=doi_override, verify_engines=verify_engines)
    bundle["_cache"] = "fresh"
    if use_disk_cache:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(bundle, f)
        except Exception as e:
            log.warning(f"mapper: cache write failed ({e}).")
    return h, bundle


def _apply_links(source_df, links_df):
    """Rebuild mapped_* columns on the property dataframe from the CURRENT
    (possibly hand-pruned) links table. Best link per property = lowest
    match_rank; a property with no remaining link is left unmapped."""
    aug = source_df.reset_index(drop=True).copy()
    n = len(aug)
    best = {}
    if links_df is not None and not links_df.empty:
        ld = links_df.sort_values(["prop_row", "match_rank"], ascending=[True, True])
        for _, l in ld.iterrows():
            best.setdefault(int(l["prop_row"]), l)

    def pick(field):
        return [best[i][field] if i in best else "" for i in range(n)]

    aug["mapped_figure_id"]      = pick("figure_id")
    aug["mapped_figure_caption"] = pick("figure_caption")
    aug["mapped_figure_page"]    = pick("figure_page")
    aug["mapped_figure_number"]  = pick("figure_number")
    aug["mapped_subplot"]        = pick("matched_subplot")
    aug["map_score"]             = pick("match_score")
    aug["map_signals"]           = pick("match_signals")
    return aug


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI  (extract-once → prune → map → prune links → store)
# ─────────────────────────────────────────────────────────────────────────────

def _run_streamlit() -> None:
    import streamlit as st

    st.set_page_config(page_title="mapper — Plot ⇄ Property", page_icon="🔗", layout="wide")
    st.title("🔗 mapper — Plot ⇄ Property mapping + database")
    st.caption("Runs 2.py + image2.py once (cached), auto-drops non-plots, lets you "
               "prune what's left, then attaches plots to properties on Map.")

    ss = st.session_state

    # ---- mutation helpers ---------------------------------------------------
    def _rebuild_aug():
        src = ss["bundle"]["table_map"][ss["source_table"]]
        ss["df_aug"] = _apply_links(src, ss["links_df"])

    def _drop_group_links(gi):
        ld = ss.get("links_df")
        if ld is not None and not ld.empty:
            ss["links_df"] = ld[ld["group_idx"] != gi].reset_index(drop=True)

    def _remove_image(gi, idx):
        try:
            del ss["plot_curated"][gi]["image_data"][idx]
        except Exception:
            pass
        _drop_group_links(gi)
        if ss.get("mapped"):
            _rebuild_aug()
            ss["map_stale"] = True
        st.rerun()

    def _remove_figure(gi):
        ss["plot_curated"][gi]["image_data"] = []
        _drop_group_links(gi)
        if ss.get("mapped"):
            _rebuild_aug()
            ss["map_stale"] = True
        st.rerun()

    def _remove_link(link_id):
        ld = ss["links_df"]
        ss["links_df"] = ld[ld["link_id"] != link_id].reset_index(drop=True)
        _rebuild_aug()
        st.rerun()

    def _apply_filter():
        ss["plot_curated"] = select_real_plots(ss["bundle"]["plot_results"], **ss["gate"])
        ss["mapped"] = False
        ss["store"] = None
        ss["map_stale"] = False
        ss.pop("links_df", None)
        ss.pop("df_aug", None)

    # ---- sidebar ------------------------------------------------------------
    with st.sidebar:
        st.header("Settings")
        st.subheader("Extraction")
        use_gpt = st.checkbox("Also verify crops with GPT", value=False)
        use_claude = st.checkbox("Also verify crops with Claude", value=False)
        doi_manual = st.text_input("DOI override (optional)", placeholder="10.xxxx/...")
        use_disk_cache = st.checkbox("Use on-disk extraction cache", value=True,
                                     help="Skip 2.py + image2.py entirely when this exact "
                                          "PDF was extracted before.")
        st.divider()
        st.subheader("Auto figure filter")
        gate_caption = st.checkbox("Drop photo/micrograph/logo captions", value=True)
        gate_verifier = st.checkbox("Drop crops the verifier rejected", value=True)
        gate_unlabeled = st.checkbox("Drop unlabeled detections (no 'Fig. N')", value=False)
        st.divider()
        st.subheader("Mapping")
        source_table = st.selectbox(
            "Property table",
            ["consensus", "verified", "consensus_verified", "gemini", "gpt", "claude"],
        )
        top_k = st.slider("Candidate figures per property", 1, 3, TOP_K)
        min_score = st.slider("Min fallback map score", 0.0, 1.0, MAP_MIN_SCORE, 0.05,
                              help="Non-citation links below this are dropped. "
                                   "Explicit 'Fig. N' citations always kept.")
        st.divider()
        st.subheader("Persistence")
        out_dir = st.text_input("Output directory", value="./aim_efrc_db")
        save_all = st.checkbox("Save ALL detected crops", value=False)

    ss["gate"] = {"use_caption": gate_caption, "use_verifier": gate_verifier,
                  "drop_unlabeled": gate_unlabeled}

    # ---- upload + hash-guarded extraction -----------------------------------
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if not uploaded:
        st.info("Upload a PDF to begin.")
        return

    pdf_bytes = uploaded.getvalue()
    h = _pdf_hash(pdf_bytes)
    need_extract = ss.get("pdf_hash") != h

    ec1, ec2 = st.columns([3, 1])
    do_extract = force = False
    if need_extract:
        do_extract = ec1.button(" Extract ", type="primary",
                                use_container_width=True)
    else:
        ec1.success("✓ This paper is loaded — switch tables / prune / map freely; "
                    "2.py and image2.py will NOT re-run.")
    force = ec2.button("↻ Force re-extract", use_container_width=True)

    if do_extract or force:
        engines = ["gemini"] + (["gpt"] if use_gpt else []) + (["claude"] if use_claude else [])
        with st.spinner("Running 2.py extraction/consensus + image2.py plot detection…"):
            _h, bundle = get_or_extract(pdf_bytes, doi_override=doi_manual,
                                        verify_engines=engines,
                                        use_disk_cache=use_disk_cache, force=force)
        ss["pdf_hash"] = h
        ss["bundle"] = bundle
        ss["stem"] = _safe(uploaded.name.rsplit(".", 1)[0])
        ss["extracted"] = True
        _apply_filter()

    if not ss.get("extracted") or ss.get("pdf_hash") != h:
        st.info("Click **① Extract** — instant if this PDF was processed before.")
        return

    bundle = ss["bundle"]
    stem = ss["stem"]
    if bundle.get("_cache") == "disk":
        st.caption(" Loaded from on-disk cache — 2.py and image2.py were not re-run.")

    # ---- three top-level views ---------------------------------------------
    view_props, view_plots, work = st.tabs(
        [" Extraction (2.py)", " Plots (image2.py)", " Map & curate"])

    with view_props:
        if bundle.get("api_errors"):
            with st.expander(f" {len(bundle['api_errors'])} API error(s)"):
                for e in bundle["api_errors"]:
                    st.code(e)
        meta = bundle.get("meta", {})
        st.caption(f"DOI: {meta.get('doi', '—')} · tables={meta.get('chunks_tables', '?')} · "
                   f"text={meta.get('chunks_text', '?')} · batches={meta.get('batches', '?')}")
        for label, df in bundle["table_map"].items():
            with st.expander(f"{label} ({len(df)})", expanded=(label == "consensus")):
                st.dataframe(df.drop(columns=["doi_url"], errors="ignore"),
                             use_container_width=True, hide_index=True)

    with view_plots:
        cov = bundle.get("coverage", {})
        raw_n = sum(1 for g in bundle["plot_results"] if g.get("image_data"))
        kept_groups = [g for g in ss["plot_curated"] if g.get("image_data")]
        st.caption(f"image2.py output — keep-only. Detected {cov.get('total_detected', '?')}, "
                   f"recovered {cov.get('total_recovered', '?')}; showing {len(kept_groups)} plot "
                   f"group(s) of {raw_n} after dropping non-plots/discards.")
        if not kept_groups:
            st.info("No real plots left after filtering. Loosen the auto-filter toggles in the "
                    "sidebar if this removed too much.")
        for g in kept_groups:
            with st.container(border=True):
                st.markdown(f"**Page {g.get('page')}** — {g.get('caption', '')}")
                icols = st.columns(min(len(g["image_data"]), 4) or 1)
                for i, img in enumerate(g["image_data"]):
                    with icols[i % len(icols)]:
                        try:
                            st.image(img["array"], channels="BGR", width=190)
                        except Exception:
                            st.caption("(image unavailable)")
                        v = img.get("verification") or {}
                        act = v.get("majority_action") if isinstance(v, dict) else ""
                        cap = f"{img.get('subplot_label') or ''} · {act}".strip(" ·")
                        if cap:
                            st.caption(cap)

    # ---- workflow: prune → map → prune links → store ------------------------
    with work:
        plot_curated = ss["plot_curated"]
        live_groups = [(gi, g) for gi, g in enumerate(plot_curated) if g.get("image_data")]
        raw_count = sum(1 for g in bundle["plot_results"] if g.get("image_data"))

        fc1, fc2 = st.columns([3, 1])
        fc1.caption(f"Auto-filter kept **{len(live_groups)}** plot group(s) of {raw_count} "
                    f"detected (caption={ss['gate']['use_caption']}, "
                    f"verifier={ss['gate']['use_verifier']}, "
                    f"unlabeled_dropped={ss['gate']['drop_unlabeled']}).")
        if fc2.button("↻ Re-apply auto-filter", use_container_width=True):
            _apply_filter()
            st.rerun()

        st.subheader("② Review figures — remove anything that isn't a data plot")
        st.caption("x removes one image · 🗑 removes the whole figure. Non-plots the "
                   "auto-filter missed (or wrongly kept) can be pruned here by hand.")
        with st.expander("Show figures to review", expanded=not ss.get("mapped")):
            if not live_groups:
                st.info("No plot figures left.")
            for gi, group in live_groups:
                with st.container(border=True):
                    hc1, hc2 = st.columns([6, 1])
                    hc1.markdown(f"**Page {group.get('page')}** — {group.get('caption', '')}")
                    if hc2.button(" Remove figure", key=f"rmfig_{gi}"):
                        _remove_figure(gi)
                    images = group.get("image_data", [])
                    icols = st.columns(min(len(images), 4) or 1)
                    for idx, img in enumerate(images):
                        with icols[idx % len(icols)]:
                            try:
                                st.image(img["array"], channels="BGR", width=180)
                            except Exception:
                                st.caption("(image unavailable)")
                            if img.get("subplot_label"):
                                st.caption(img["subplot_label"])
                            if st.button("x remove", key=f"rmimg_{gi}_{idx}"):
                                _remove_image(gi, idx)

        map_label = " Map — attach plots to properties" + (" (re-map)" if ss.get("mapped") else "")
        if st.button(map_label, type="primary", use_container_width=True):
            source_df = bundle["table_map"][source_table]
            with st.spinner("Mapping property rows to figures…"):
                links_df, df_aug = map_plots_to_properties(
                    source_df, ss["plot_curated"], top_k=top_k, min_score=min_score)
                if not links_df.empty:
                    links_df = links_df.reset_index(drop=True)
                    links_df["link_id"] = range(len(links_df))
            ss["links_df"] = links_df
            ss["df_aug"] = df_aug
            ss["source_table"] = source_table
            ss["mapped"] = True
            ss["map_stale"] = False
            ss["store"] = None

        if not ss.get("mapped"):
            st.info("Prune figures above, then click **③ Map**.")
            return

        if ss.get("map_stale"):
            st.warning("Figures changed since mapping — affected links were removed. "
                       "Click **③ Map (re-map)** to rescore against the current figures.")

        links_df = ss["links_df"]
        df_aug = ss["df_aug"]
        store = ss.get("store")

        n_rows = len(df_aug)
        n_mapped = int((df_aug["mapped_figure_id"] != "").sum()) if "mapped_figure_id" in df_aug.columns else 0
        n_cited = int(links_df["match_signals"].str.contains("figure_citation").sum()) if not links_df.empty else 0

        mc = st.columns(6 if store else 5)
        mc[0].metric("Property rows", n_rows)
        mc[1].metric("Rows mapped", n_mapped)
        mc[2].metric("Via citation", n_cited)
        mc[3].metric("Live figures", len(live_groups))
        mc[4].metric("Links", len(links_df))
        if store:
            mc[5].metric("Images saved", store["n_images_saved"])
            st.success(f"Stored to `{store['db_path']}` · crops under `{store['image_dir']}`")

        def _images_for_prop(prow: int):
            """Crop array(s) matched to property row `prow`, pulled from the
            live (possibly hand-pruned) links + curated plots in session.
            Returns a list of (bgr_array, caption)."""
            ld = ss.get("links_df")
            pc = ss.get("plot_curated", []) or []
            out = []
            if ld is None or ld.empty:
                return out
            rows = ld[(ld["prop_row"] == prow) & (ld["match_rank"] == 1)]
            for _, l in rows.iterrows():
                gi = int(l["group_idx"])
                if gi >= len(pc):
                    continue
                imgs = pc[gi].get("image_data", []) or []
                mii = l.get("matched_image_index")
                if mii is not None and not (isinstance(mii, float) and pd.isna(mii)):
                    idxs = [int(mii)]
                else:
                    idxs = list(range(len(imgs)))   # whole figure if no specific subplot
                for ix in idxs:
                    if 0 <= ix < len(imgs):
                        sub = imgs[ix].get("subplot_label") or ""
                        cap = f"{l.get('figure_id', '')} {sub}".strip()
                        out.append((imgs[ix].get("array"), cap))
            return out

        names = [" Figures & attached rows (remove links)",
                 f" All links ({len(links_df)})",
                 f" Augmented ({n_rows})",
                 " Store / Export"]
        if store:
            names.insert(3, f" Stored ({store['n_properties']})")
        names.append(" Properties + image")
        wt = st.tabs(names)
        prop_img_tab = wt[-1]

        with prop_img_tab:
            st.caption("Search a property row and see its matched plot rendered inline — "
                       "not just the path. Reflects your current link edits.")
            pc1, pc2 = st.columns([3, 1])
            q = pc1.text_input("Filter by property or material name",
                               key="propimg_q",
                               placeholder="e.g. half-life, PBSA").strip().lower()
            only_mapped = pc2.checkbox("Only matched rows", value=False, key="propimg_only")
            shown = 0
            for prow, row in df_aug.iterrows():
                pname = str(row.get("property_name", ""))
                mat = str(row.get("material_name", ""))
                if q and q not in pname.lower() and q not in mat.lower():
                    continue
                imgs = _images_for_prop(int(prow))
                if only_mapped and not imgs:
                    continue
                shown += 1
                with st.container(border=True):
                    left, right = st.columns([2, 1])
                    with left:
                        st.markdown(f"**{pname}** — {row.get('value', '')} {row.get('unit', '')}")
                        st.caption(f"{mat} · {row.get('section', '')} · {row.get('source_page', '')}")
                        fid = str(row.get("mapped_figure_id", ""))
                        if fid:
                            st.caption(f"figure {fid} {row.get('mapped_subplot', '') or ''} · "
                                       f"score {row.get('map_score', '')} · "
                                       f"{row.get('map_signals', '')}")
                    with right:
                        if imgs:
                            for arr, cap in imgs:
                                try:
                                    st.image(arr, channels="BGR", width=230)
                                except Exception:
                                    st.caption("(image unavailable)")
                                if cap:
                                    st.caption(cap)
                        else:
                            st.caption("— no plot matched —")
            if shown == 0:
                st.info("No property rows match your filter.")

        with wt[0]:
            st.caption("Each figure with its attached property rows. x removes a wrong "
                       "property↔figure link.")
            if links_df.empty:
                st.info("No links remain.")
            else:
                for gi, group in live_groups:
                    grp = links_df[links_df["group_idx"] == gi]
                    if grp.empty:
                        continue
                    with st.container(border=True):
                        st.markdown(f"**Page {group.get('page')}** — {group.get('caption', '')}")
                        images = group.get("image_data", [])
                        if images:
                            icols = st.columns(min(len(images), 4) or 1)
                            for pos, img in enumerate(images):
                                with icols[pos % len(icols)]:
                                    try:
                                        st.image(img["array"], channels="BGR", width=170)
                                    except Exception:
                                        pass
                        hdr = st.columns([3, 1.3, 1, 2, 1.2, 1, 0.7])
                        for hh, tt in zip(hdr, ["property", "value", "unit", "material",
                                               "subplot", "score", ""]):
                            hh.caption(tt)
                        for _, l in grp.iterrows():
                            c = st.columns([3, 1.3, 1, 2, 1.2, 1, 0.7])
                            c[0].write(str(l.get("property_name", "")))
                            c[1].write(str(l.get("value", "")))
                            c[2].write(str(l.get("unit", "")))
                            c[3].write(str(l.get("material_name", "")))
                            c[4].write(str(l.get("matched_subplot") or ""))
                            c[5].write(str(l.get("match_score", "")))
                            if c[6].button("x", key=f"rmlink_{int(l['link_id'])}"):
                                _remove_link(int(l["link_id"]))

        with wt[1]:
            st.caption("Every remaining (property, figure) link with its score and signals.")
            if links_df.empty:
                st.info("No links remain.")
            else:
                st.dataframe(links_df.drop(columns=["link_id"], errors="ignore"),
                             use_container_width=True, hide_index=True)

        with wt[2]:
            st.caption(f"The `{ss['source_table']}` table plus mapped_figure_* / map_score / "
                       f"map_signals, reflecting your link edits.")
            st.dataframe(df_aug.drop(columns=["doi_url"], errors="ignore"),
                         use_container_width=True, hide_index=True)

        export_tab = wt[4] if store else wt[3]
        if store:
            with wt[3]:
                st.caption("Rows as written to the DB, each with its attached plot_image_path.")
                dfs = store["df_store"]
                show = [c for c in ("property_name", "value", "unit", "material_name",
                                    "section", "mapped_figure_id", "mapped_subplot",
                                    "map_score", "plot_image_path") if c in dfs.columns]
                st.dataframe(dfs[show] if show else dfs, use_container_width=True, hide_index=True)

        with export_tab:
            st.markdown("**Store the curated result to the database**")
            st.caption("Writes the pruned figures + edited links; each stored row points at "
                       "its saved crop on disk. Re-running for the same PDF overwrites cleanly.")
            if st.button(" Store to database", type="primary", use_container_width=True):
                with st.spinner("Saving crops and writing SQLite…"):
                    ss["store"] = store_properties_with_plots(
                        ss["df_aug"], ss["links_df"], ss["plot_curated"],
                        out_dir=out_dir, pdf_stem=stem, source_table=ss["source_table"],
                        save_all_group_images=save_all)
                st.rerun()

            st.divider()
            st.markdown("**☁️ Push to RDS database**")
            if not db_configured():
                st.caption("Set DB_NAME / DB_USER / DB_PASSWORD in your `.env` "
                           "(host is already configured) to enable the RDS push.")
            else:
                import category_push
                category = st.selectbox(
                    "Category table (routes by material type)",
                    category_push.CATEGORY_TABLES,
                    help="Rows conform to that table's 33-col schema; columns it "
                         "can't hold go to <table>_extras; matched crops are copied "
                         "to rds_plots/<category>/<stem>/.")
                embed_img = st.checkbox("Embed matched plot into the table's image column",
                                        value=True)
                cA, cB = st.columns(2)
                if cA.button(" Test connection", use_container_width=True):
                    try:
                        db_healthcheck()
                        st.success(f"Connected to {DB_HOST} ({DB_KIND}).")
                    except Exception as e:
                        st.error(f"Connection failed: {e}")
                push_ready = bool(store) and store.get("n_properties", 0) > 0
                if cB.button(f" Push to {category}", type="primary",
                             use_container_width=True, disabled=not push_ready):
                    try:
                        with st.spinner(f"Conforming + writing rows to '{category}'…"):
                            res = category_push.push_by_category(
                                store, category, get_db_engine(), embed_image=embed_img)
                        st.success(
                            f"Pushed {res['pushed']} row(s) to '{res['table']}', "
                            f"{res['extras']} to '{res['extras_table']}', "
                            f"{res['plots_copied']} crop(s) → {res['plots_dir']}.")
                    except Exception as e:
                        st.error(f"Push failed: {e}")
                if not push_ready:
                    st.caption("Click ** Store to database** first — the RDS push "
                               "uploads those stored rows.")

            st.divider()
            st.markdown("**Downloads**")
            d1, d2, d3 = st.columns(3)
            d1.download_button(
                " Links CSV",
                links_df.drop(columns=["link_id"], errors="ignore").to_csv(index=False).encode()
                if not links_df.empty else b"",
                f"{stem}_links.csv", "text/csv", use_container_width=True, disabled=links_df.empty)
            d2.download_button(
                " Augmented CSV",
                df_aug.drop(columns=["doi_url"], errors="ignore").to_csv(index=False).encode(),
                f"{stem}_properties_mapped.csv", "text/csv", use_container_width=True)
            if store:
                d3.download_button(
                    " DB + plots (ZIP)",
                    data=bundle_zip(store["out_dir"], store["pdf_stem"], store["db_filename"]),
                    file_name=f"{stem}_db_bundle.zip", mime="application/zip",
                    use_container_width=True)
            else:
                d3.caption("Store first to enable the DB bundle download.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

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
        print(
            "\nmapper.py — Plot ⇄ Property mapping + persistence\n"
            "  streamlit run mapper.py\n\n"
            "Requires on the import path (same directory):\n"
            "  2.py       (Code 1 — extraction + consensus)\n"
            "  image2.py  (Code 2 — plot extraction)\n\n"
            "Library use (map + store outputs you already have in memory):\n"
            "  from mapper import map_plots_to_properties, store_properties_with_plots\n"
            "  links_df, df_aug = map_plots_to_properties(df_consensus, plot_results)\n"
            "  store_properties_with_plots(df_aug, links_df, plot_results,\n"
            "                              out_dir='./aim_efrc_db', pdf_stem='paper1')\n"
        )
        
    