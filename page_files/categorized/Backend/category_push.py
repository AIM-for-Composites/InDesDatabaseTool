"""
category_push.py — route extracted property rows into the correct RDS category
table (Composites_materials / Fibers / Polymers), conforming to that table's
real 33-column schema, and preserve everything that doesn't fit.

Why this exists
---------------
The three category tables share ONE fixed schema that does NOT contain most of
what the pipeline emits (no source_text, no mapped_figure_*, no plot_image_path,
no pdf_stem, ...). A raw df.to_sql(..., append) therefore fails with
'column "source_text" does not exist'. This module:

  * MAPS the pipeline columns onto the target schema (rename + parse), so the
    INSERT matches the table  ................................. (fixes the error)
  * ROUTES to whichever of the 3 tables you pick  ............ (composites/etc.)
  * writes the leftover rich columns to a companion "<table>_extras" table so
    nothing is lost  ......................................... (the "other tab")
  * copies each matched crop to a separate plots directory AND (optionally)
    embeds the PNG bytes into the table's `image` column  .... (plots elsewhere)

Idempotent per paper: keyed on source_pdf (= pdf_stem).
"""

from __future__ import annotations

import os
import re
import shutil
import logging
import datetime as _dt
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

CATEGORY_TABLES = ["Composites_materials", "Fibers", "Polymers"]

# the real 33 columns shared by all three tables (from information_schema)
TARGET_COLUMNS = [
    "material_name", "material_abbreviation", "section", "property_name",
    "value", "unit", "english", "test_condition", "comments", "image", "image_url",
    "material_key", "material_class", "trade_grade", "manufacturer", "matrix",
    "fiber", "fiber_volume_fraction", "value_raw", "value_num", "value_min",
    "value_max", "qualifier", "unit_canonical", "value_si", "source_pdf",
    "source_sha1", "page", "source_quote", "status", "flag_reason", "model",
    "prompt_version", "extracted_at",
]

# columns the category tables can't hold — preserved in <table>_extras
EXTRAS_COLUMNS = [
    "source_pdf", "material_name", "property_name", "value", "unit", "section",
    "pdf_stem", "source_table", "chunk_type", "source_text", "source_page",
    "confirmed_by_gpt", "confirmed_by_claude", "verified_by", "match_layer",
    "doi_url", "mapped_figure_id", "mapped_figure_caption", "mapped_figure_number",
    "mapped_figure_page", "mapped_subplot", "map_score", "map_signals",
    "plot_image_path", "plot_local_path",
]

_CLASS_OF = {"Composites_materials": "composite", "Fibers": "fiber", "Polymers": "polymer"}
_TRUE = {"true", "1", "yes", "t", "y"}

log = logging.getLogger(__name__)

# --- S3 image storage --------------------------------------------------------
# Same env config as mapper5 - one .env drives both. boto3 auto-reads
# AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION.
# Requires: pip install boto3
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_PREFIX = os.getenv("S3_PREFIX", "plots").strip("/")
S3_REGION = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION", "us-east-2"))
S3_PUBLIC = os.getenv("S3_PUBLIC", "false").lower() in ("1", "true", "yes")

_s3_client = None


def _s3_ready() -> bool:
    return bool(S3_BUCKET)


def _s3():
    global _s3_client
    if _s3_client is None:
        import boto3
        _s3_client = boto3.client("s3", region_name=S3_REGION)
    return _s3_client


def _s3_key(category_table: str, stem: str, filename: str) -> str:
    parts = [p for p in (S3_PREFIX, category_table, stem, filename) if p]
    return "/".join(parts)


def _s3_url(key: str) -> str:
    if S3_PUBLIC:
        return f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{key}"
    return f"s3://{S3_BUCKET}/{key}"


def _s3_put(data: bytes, key: str, content_type: str = "image/png") -> str:
    _s3().put_object(Bucket=S3_BUCKET, Key=key, Body=data, ContentType=content_type)
    return _s3_url(key)


def _s3_get(key: str) -> bytes:
    return _s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()


def _is_url(p: str) -> bool:
    return p.startswith("s3://") or p.startswith("http://") or p.startswith("https://")


def _s3_key_from_url(url: str):
    if url.startswith("s3://"):
        rest = url[len("s3://"):]
        return rest.split("/", 1)[1] if "/" in rest else None
    m = re.match(r"https?://[^/]+/(.+)$", url)
    return m.group(1) if m else None


def _s(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(v)
    # Postgres text/varchar columns cannot contain NUL (0x00). PDF text
    # extraction occasionally emits embedded nulls (and other C0 control
    # chars) inside quotes/comments — strip them so the INSERT doesn't fail
    # with "A string literal cannot contain NUL (0x00) characters".
    if "\x00" in s or any(ord(c) < 32 and c not in "\t\n\r" for c in s):
        s = s.replace("\x00", "")
        s = "".join(c for c in s if ord(c) >= 32 or c in "\t\n\r")
    return s


def _page_int(v: Any) -> Optional[int]:
    m = re.search(r"\d+", _s(v))
    return int(m.group()) if m else None


def _parse_value(v: Any) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """(value_num, value_min, value_max). '5-30'->(None,5,30); '293 000'->(293000,..);
    '0.496'->(0.496,None,None); '-40'->(-40,None,None)."""
    s = _s(v).strip()
    if not s:
        return (None, None, None)
    s2 = re.sub(r"(?<=\d)[ ,](?=\d)", "", s)           # 293 000 / 1,234 -> plain
    # range "A-B" / "A to B": both endpoints are positive; the dash is a separator
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|\u2013|\u2014|to)\s*(\d+(?:\.\d+)?)", s2)
    if m:
        return (None, float(m.group(1)), float(m.group(2)))
    m2 = re.search(r"-?\d+(?:\.\d+)?", s2)              # single (possibly negative) value
    if m2:
        return (float(m2.group()), None, None)
    return (None, None, None)


def _model_str(row: pd.Series) -> str:
    if _s(row.get("verified_by")):
        return _s(row.get("verified_by"))
    who = []
    if _s(row.get("confirmed_by_gpt")).lower() in _TRUE:
        who.append("gpt")
    if _s(row.get("confirmed_by_claude")).lower() in _TRUE:
        who.append("claude")
    return "+".join(who)


def conform_to_category(
    df_store: pd.DataFrame,
    category_table: str,
    out_dir: str,
    stem: str,
    plots_root: Optional[str] = None,
    embed_image: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Return (main_df on TARGET_COLUMNS, extras_df on EXTRAS_COLUMNS, n_plots).
    Copies each matched crop into plots_root/<category>/<stem>/ and, if
    embed_image, embeds its PNG bytes into the `image` column."""
    if category_table not in CATEGORY_TABLES:
        raise ValueError(f"category_table must be one of {CATEGORY_TABLES}")
    df = df_store.reset_index(drop=True).copy()
    material_class = _CLASS_OF[category_table]
    now = _dt.datetime.now().isoformat(timespec="seconds")

    plots_root = plots_root or os.path.join(out_dir, "rds_plots")
    dest_dir = os.path.join(plots_root, category_table, stem)

    # source_sha1 scopes the DB dedup index to THIS paper. It was previously ""
    # for every row, which collapsed the unique key across all papers. A stable
    # hash of the stem keeps re-pushes of the same paper consistent.
    import hashlib
    source_sha1 = hashlib.sha1((stem or "").encode("utf-8")).hexdigest()

    main_rows: List[Dict[str, Any]] = []
    extras_rows: List[Dict[str, Any]] = []
    n_plots = 0
    copied: set = set()

    for _, row in df.iterrows():
        num, vmin, vmax = _parse_value(row.get("value"))

        # resolve -> S3 (preferred) or local copy -> optionally embed
        img_bytes = None
        stored_path = ""              # primary path extras.plot_image_path records
        local_path = ""               # local relative copy (always kept when possible)
        rel = _s(row.get("plot_image_path"))
        if rel and _is_url(rel):
            # mapper already uploaded this crop to S3 - reuse the URL as-is
            stored_path = rel
            if rel not in copied:
                copied.add(rel)
                n_plots += 1
            if embed_image and _s3_ready():
                key = _s3_key_from_url(rel)
                if key:
                    try:
                        img_bytes = _s3_get(key)
                        # also drop a local copy alongside so a local path exists too
                        try:
                            fn = os.path.basename(key)
                            os.makedirs(dest_dir, exist_ok=True)
                            with open(os.path.join(dest_dir, fn), "wb") as _fh:
                                _fh.write(img_bytes)
                            local_path = os.path.join(category_table, stem, fn)
                        except Exception as e:
                            log.warning(f"category_push: local copy from S3 failed for {rel} ({e}).")
                    except Exception as e:
                        log.warning(f"category_push: S3 fetch failed for {rel} ({e})")
        elif rel:
            abs_src = rel if os.path.isabs(rel) else os.path.join(out_dir, rel)
            if os.path.exists(abs_src):
                fn = os.path.basename(abs_src)
                with open(abs_src, "rb") as fh:
                    data = fh.read()
                if embed_image:
                    img_bytes = data
                local_rel = ""
                s3_url = ""
                # ALWAYS keep a local copy under plots_root/<category>/<stem>/
                try:
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.copyfile(abs_src, os.path.join(dest_dir, fn))
                    local_rel = os.path.join(category_table, stem, fn)
                except Exception as e:
                    log.warning(f"category_push: local copy failed for {fn} ({e}).")
                # AND upload to S3 when configured
                if _s3_ready():
                    try:
                        s3_url = _s3_put(data, _s3_key(category_table, stem, fn))
                    except Exception as e:
                        log.warning(f"category_push: S3 upload failed for {fn} ({e}); keeping local path.")
                # primary path recorded in extras: prefer the S3 URL, else local
                stored_path = s3_url or local_rel
                # keep both resolvable references on the row
                local_path = local_rel
                if fn not in copied:
                    copied.add(fn)
                    n_plots += 1

        main_rows.append({
            "material_name":         _s(row.get("material_name")),
            "material_abbreviation": _s(row.get("material_abbreviation")),
            "section":               _s(row.get("section")),
            "property_name":         _s(row.get("property_name")),
            "value":                 _s(row.get("value")),
            "unit":                  _s(row.get("unit")),
            "english":               _s(row.get("english")),
            "test_condition":        _s(row.get("test_condition")),
            "comments":              _s(row.get("comments")),
            "image":                 img_bytes,
            "image_url":             stored_path,
            "material_key":          _s(row.get("material_name")).lower().replace(" ", "_"),
            "material_class":        material_class,
            "trade_grade":           "",
            "manufacturer":          _s(row.get("manufacturer")),
            "matrix":                "",
            "fiber":                 "",
            "fiber_volume_fraction": "",
            "value_raw":             _s(row.get("value")),
            "value_num":             num,
            "value_min":             vmin,
            "value_max":             vmax,
            "qualifier":             "",
            "unit_canonical":        _s(row.get("unit")),
            "value_si":              None,
            "source_pdf":            _s(row.get("pdf_stem")) or stem,
            "source_sha1":           source_sha1,
            "page":                  _page_int(row.get("source_page")),
            "source_quote":          _s(row.get("source_text")),
            "status":                "extracted",
            "flag_reason":           "",
            "model":                 _model_str(row),
            "prompt_version":        "",
            "extracted_at":          now,
        })

        extras_rows.append({
            "source_pdf":            _s(row.get("pdf_stem")) or stem,
            "material_name":         _s(row.get("material_name")),
            "property_name":         _s(row.get("property_name")),
            "value":                 _s(row.get("value")),
            "unit":                  _s(row.get("unit")),
            "section":               _s(row.get("section")),
            "pdf_stem":              _s(row.get("pdf_stem")) or stem,
            "source_table":          _s(row.get("source_table")),
            "chunk_type":            _s(row.get("chunk_type")),
            "source_text":           _s(row.get("source_text")),
            "source_page":           _s(row.get("source_page")),
            "confirmed_by_gpt":      _s(row.get("confirmed_by_gpt")),
            "confirmed_by_claude":   _s(row.get("confirmed_by_claude")),
            "verified_by":           _model_str(row),
            "match_layer":           _s(row.get("match_layer")),
            "doi_url":               _s(row.get("doi") or row.get("doi_url")),
            "mapped_figure_id":      _s(row.get("mapped_figure_id")),
            "mapped_figure_caption": _s(row.get("mapped_figure_caption")),
            "mapped_figure_number":  _s(row.get("mapped_figure_number")),
            "mapped_figure_page":    _s(row.get("mapped_figure_page")),
            "mapped_subplot":        _s(row.get("mapped_subplot")),
            "map_score":             _s(row.get("map_score")),
            "map_signals":           _s(row.get("map_signals")),
            "plot_image_path":       stored_path,
            "plot_local_path":       local_path,
        })

    main_df = pd.DataFrame(main_rows, columns=TARGET_COLUMNS)
    extras_df = pd.DataFrame(extras_rows, columns=EXTRAS_COLUMNS)
    return main_df, extras_df, n_plots


def push_by_category(
    store: Dict[str, Any],
    category_table: str,
    engine,
    embed_image: bool = True,
    plots_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Conform store['df_store'] to `category_table`, write the main rows there
    and the leftover columns to '<category_table>_extras', idempotent on
    source_pdf. `engine` is a SQLAlchemy engine (e.g. mapper.get_db_engine())."""
    from sqlalchemy import text
    df_store = store.get("df_store")
    if df_store is None or df_store.empty:
        return {"pushed": 0, "table": category_table}
    out_dir = store.get("out_dir", ".")
    stem = store.get("pdf_stem", "")

    main_df, extras_df, n_plots = conform_to_category(
        df_store, category_table, out_dir, stem,
        plots_root=plots_root, embed_image=embed_image)

    # keep-first dedup on the exact columns of ix_..._pipeline_dedup, so two
    # identical measurements in one paper collapse to one row instead of
    # tripping the unique constraint mid-batch.
    _DEDUP_KEYS = ["source_sha1", "material_key", "section",
                   "property_name", "test_condition", "value_raw"]
    keys = [k for k in _DEDUP_KEYS if k in main_df.columns]
    if keys:
        before = len(main_df)
        main_df = main_df.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)
        dropped = before - len(main_df)
        if dropped:
            log.info(f"category_push: dropped {dropped} in-batch duplicate row(s) on {keys}")
        # keep extras aligned to the surviving main rows (1:1 by position)
        extras_df = extras_df.loc[main_df.index].reset_index(drop=True) \
            if len(extras_df) == before else extras_df

    extras_table = f"{category_table}_extras"
    # idempotency: clear this paper first (tables may not exist yet -> ignore)
    with engine.begin() as conn:
        for tbl, col in ((category_table, "source_pdf"), (extras_table, "source_pdf")):
            try:
                conn.execute(text(f'DELETE FROM "{tbl}" WHERE {col} = :s'), {"s": stem})
            except Exception:
                pass

    pushed = _insert_idempotent(engine, category_table, main_df)
    extras_df.to_sql(extras_table, engine, if_exists="append", index=False)
    if _s3_ready():
        plots_dir = _s3_url(_s3_key(category_table, stem, "")).rstrip("/")
    else:
        plots_dir = os.path.join(plots_root or os.path.join(out_dir, "rds_plots"),
                                 category_table, stem)
    return {"pushed": pushed, "table": category_table,
            "extras_table": extras_table, "extras": len(extras_df),
            "plots_copied": n_plots, "plots_dir": plots_dir}


def _insert_idempotent(engine, table: str, df: pd.DataFrame) -> int:
    """Append rows one INSERT ... ON CONFLICT DO NOTHING at a time, so a row that
    still collides with the table's unique dedup index is skipped instead of
    rolling back the whole batch. Returns the number of rows actually inserted."""
    from sqlalchemy import text
    if df is None or df.empty:
        return 0
    cols = list(df.columns)
    collist = ", ".join(f'"{c}"' for c in cols)
    params = ", ".join(f":{c}" for c in cols)
    sql = text(f'INSERT INTO "{table}" ({collist}) VALUES ({params}) ON CONFLICT DO NOTHING')
    inserted = 0
    with engine.begin() as conn:
        for _, r in df.iterrows():
            payload = {c: (None if pd.isna(v) else v) for c, v in r.items()}
            res = conn.execute(sql, payload)
            inserted += (res.rowcount or 0)
    return inserted