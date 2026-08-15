# page_files/Recently_Added.py — "What's new in the database"
#
# Drop this file into the Space repo at page_files/Recently_Added.py and add
# one line to app.py (see README_ADD_TO_SPACE.md next to this file).
#
# Reads the `sources` table the ingest pipeline maintains (one row per
# document) plus per-document property counts. Degrades gracefully if the
# pipeline hasn't run yet or the DB is unreachable.

import pandas as pd
import streamlit as st

from db import fetch_all

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { display: none !important; }
    .block-container { max-width: 1000px !important; padding-top: 1rem !important; }
    .ra-title { font-size: 2.2rem; font-weight: 800; color: #111827; margin: 4px 0 2px; }
    .ra-sub { color: #64748b; font-size: .9rem; margin-bottom: 18px; }
    .ra-card { border: 1px solid #e1e7ef; border-radius: 12px; background: #fff;
               padding: 14px 18px; margin-bottom: 10px; }
    .ra-name { font-weight: 700; color: #111827; font-size: .95rem; }
    .ra-meta { color: #64748b; font-size: .78rem; margin-top: 2px; }
    .ra-badge { display: inline-block; font-size: .66rem; font-weight: 800;
                letter-spacing: .5px; padding: 2px 9px; border-radius: 6px;
                background: #e0edff; color: #1d4ed8; margin-left: 8px; }
    .ra-count { float: right; font-weight: 800; color: #111827; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='ra-title'>🆕 Recently added</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='ra-sub'>Documents the automated pipeline ingested most recently — "
    "every value extracted from them is traceable to its PDF page.</div>",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300)  # refresh at most every 5 minutes
def load_recent(limit: int = 25) -> pd.DataFrame:
    # One row per ingested document, newest first.
    docs = fetch_all(
        """
        SELECT pdf_filename, pdf_sha1, ingested_at, material_class, material_abbreviation
        FROM sources
        ORDER BY ingested_at DESC
        LIMIT :lim
        """,
        {"lim": limit},
    )
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)

    # Count published (status='ok') properties per document across the 3 tables.
    counts: dict[str, int] = {}
    for table in ("Polymers", "Fibers", "Composites_materials"):
        rows = fetch_all(
            f"""
            SELECT source_sha1, count(*) AS n
            FROM "{table}"
            WHERE source_sha1 IS NOT NULL AND COALESCE(status,'ok') = 'ok'
            GROUP BY source_sha1
            """
        )
        for r in rows:
            counts[r["source_sha1"]] = counts.get(r["source_sha1"], 0) + r["n"]
    df["properties"] = df["pdf_sha1"].map(counts).fillna(0).astype(int)
    return df


try:
    recent = load_recent()
except Exception:
    recent = pd.DataFrame()

if recent.empty:
    st.info(
        "No pipeline-ingested documents yet. Once the automated ingest runs, "
        "new datasheets and papers will appear here."
    )
else:
    total_docs = len(recent)
    total_props = int(recent["properties"].sum())
    c1, c2 = st.columns(2)
    c1.metric("Documents added", total_docs)
    c2.metric("Verified properties from them", total_props)

    class_badge = {
        "Composite": "COMPOSITE", "Polymer": "POLYMER", "Fiber": "FIBER",
        "scanned_no_text": "SKIPPED (SCAN)",
    }
    for _, row in recent.iterrows():
        name = str(row["pdf_filename"] or "").replace("_", " ").removesuffix(".pdf")
        badge = class_badge.get(str(row["material_class"] or ""), str(row["material_class"] or "—"))
        when = str(row["ingested_at"] or "")[:16]
        props = int(row["properties"])
        st.markdown(
            f"""
            <div class='ra-card'>
              <span class='ra-count'>{props} properties</span>
              <div class='ra-name'>📄 {name}<span class='ra-badge'>{badge}</span></div>
              <div class='ra-meta'>added {when} UTC</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
