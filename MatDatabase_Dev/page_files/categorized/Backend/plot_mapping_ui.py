"""
plot_mapping_ui.py
------------------
Drop-in replacement for the "Extracted Plots" tab (tab2) in Upload_Data.py.

HOW TO INTEGRATE
────────────────
1. Copy plot_property_mapper.py next to upload_backend.py.
2. In Upload_Data.py, add at the top:

       from plot_mapping_ui import render_plot_mapping_tab

3. Replace the entire `with tab2:` block with:

       with tab2:
           render_plot_mapping_tab(pdf_path, paper_id)

That's it.  The function reads everything it needs from st.session_state
(which your existing tab1 code already populates).
"""

import json
import os

import cv2
import numpy as np
import streamlit as st

from plot_property_mapper import (
    batch_map_plots,
    fetch_properties_for_material,
    save_plot_image_mapping,
)

# ── tiny helper ───────────────────────────────────────────────────────────────

def _confidence_badge(conf: str) -> str:
    colors = {"high": "#16a34a", "medium": "#d97706", "low": "#dc2626"}
    c = colors.get(conf.lower(), "#6b7280")
    return (
        f"<span style='background:{c};color:#fff;padding:2px 10px;"
        f"border-radius:99px;font-size:0.78rem;font-weight:700'>{conf.upper()}</span>"
    )


# ── main render function ───────────────────────────────────────────────────────

def render_plot_mapping_tab(pdf_path: str, paper_id: str):
    """Render the full Extracted Plots + Property Mapping tab."""

    st.subheader("Extracted Plot Images & Property Mapping")

    # ── session-state keys ────────────────────────────────────────────────────
    for key, default in [
        ("pdf_processed",       False),
        ("image_results",       []),
        ("mapped_results",      []),
        ("mapping_done",        False),
        ("saved_image_mapping", {}),
        ("pdf_extracted_df",    __import__("pandas").DataFrame()),
        ("pdf_extracted_meta",  {}),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── 1. Extract plots if not done yet ─────────────────────────────────────
    if not st.session_state.pdf_processed:
        with st.spinner("Extracting plots from PDF…"):
            import fitz
            from upload_backend import extract_images

            doc = fitz.open(pdf_path)
            st.session_state.image_results = extract_images(doc)
            doc.close()
            st.session_state.pdf_processed  = True
            st.session_state.mapping_done   = False   # reset mapping on new PDF

    image_results = st.session_state.image_results

    if not image_results:
        st.warning("No plots found in this PDF.")
        return

    # ── 2. Info bar ───────────────────────────────────────────────────────────
    has_data = not st.session_state.pdf_extracted_df.empty
    material_class = st.session_state.get("selected_material_class")   # set below

    if has_data:
        df = st.session_state.pdf_extracted_df
        mat_abbr = df.iloc[0]["material_abbreviation"]
        st.info(
            f"**{len(image_results)} plots** extracted  |  "
            f"Material: **{mat_abbr}**  |  "
            f"{len(df['property_name'].unique())} DB properties available"
        )
    else:
        st.warning(
            "Extract material data in the **Material Data** tab first "
            "to enable AI property mapping."
        )

    st.divider()

    # ── 3. Download buttons (always visible) ──────────────────────────────────
    from upload_backend import create_zip

    col_img, col_json, col_all = st.columns(3)
    with col_img:
        img_zip = create_zip(image_results, include_json=False)
        st.download_button(
            "⬇ Download Images",
            data=img_zip,
            file_name=f"{paper_id}_images.zip",
            mime="application/zip",
            use_container_width=True,
            key="dl_images",
        )
    with col_json:
        json_data = [
            {"caption": r["caption"], "page": r["page"],
             "image_count": len(r["image_data"])}
            for r in image_results
        ]
        st.download_button(
            "⬇ Download JSON",
            data=json.dumps(json_data, indent=4),
            file_name=f"{paper_id}_metadata.json",
            mime="application/json",
            use_container_width=True,
            key="dl_json",
        )
    with col_all:
        full_zip = create_zip(image_results, include_json=True)
        st.download_button(
            "⬇ Download All",
            data=full_zip,
            file_name=f"{paper_id}_complete.zip",
            mime="application/zip",
            use_container_width=True,
            key="dl_all",
        )

    st.divider()

    # ── 4. AI mapping panel (only when data is extracted) ────────────────────
    if has_data:
        from db import fetch_all  # your existing db module

        df          = st.session_state.pdf_extracted_df
        mat_abbr    = df.iloc[0]["material_abbreviation"]
        extracted_json = st.session_state.get("pdf_extracted_meta", {})

        # Material class selector (needed to route to the right table)
        material_class = st.selectbox(
            "Material class (for DB lookup)",
            ["Polymer", "Fiber", "Composite"],
            index=0,
            key="selected_material_class",
            help="Determines which PostgreSQL table to fetch properties from.",
        )

        run_mapping = st.button(
            "🤖  Run AI Property Mapping",
            type="primary",
            disabled=st.session_state.mapping_done,
            help="Sends each plot + caption + extracted JSON to Gemini for matching.",
        )

        if run_mapping:
            # Fetch DB properties
            with st.spinner("Fetching properties from PostgreSQL…"):
                try:
                    db_properties = fetch_properties_for_material(
                        mat_abbr, material_class, fetch_all
                    )
                except Exception as exc:
                    st.error(f"DB error: {exc}")
                    db_properties = []

            if not db_properties:
                st.warning(
                    f"No properties found for **{mat_abbr}** in the "
                    f"**{material_class}** table.  Mapping will use all properties."
                )

            # Run batch mapping with progress bar
            progress_bar = st.progress(0, text="Mapping plots…")

            def _update(i, total, caption):
                pct = int((i / max(total, 1)) * 100)
                progress_bar.progress(
                    pct,
                    text=f"Mapping {i+1}/{total}: {caption[:60]}…",
                )

            with st.spinner("AI is analysing plots…"):
                mapped = batch_map_plots(
                    image_results=image_results,
                    extracted_json=extracted_json,
                    db_properties=db_properties,
                    progress_callback=_update,
                )

            progress_bar.progress(100, text="Done!")
            st.session_state.mapped_results = mapped
            st.session_state.mapping_done   = True
            st.success(f"✅  Mapped {len(mapped)} plots")
            st.rerun()

        if st.session_state.mapping_done and st.session_state.mapped_results:
            st.caption("Mapping complete. Review & confirm each match below.")

    st.divider()

    # ── 5. Plot cards ─────────────────────────────────────────────────────────
    use_mapped = (
        has_data
        and st.session_state.mapping_done
        and bool(st.session_state.mapped_results)
    )

    display_list = (
        st.session_state.mapped_results if use_mapped else image_results
    )

    for idx, item in enumerate(display_list):
        caption     = item.get("caption", f"Figure {idx+1}")
        page        = item.get("page", "?")
        img_list    = item.get("image_data", [])
        mapping     = item.get("mapping_result") if use_mapped else None

        with st.container(border=True):

            # — header row —
            col_cap, col_del = st.columns([0.88, 0.12])
            col_cap.markdown(f"**Page {page}** — {caption}")
            if col_del.button("🗑 Delete", key=f"del_group_{idx}"):
                if use_mapped:
                    st.session_state.mapped_results.pop(idx)
                else:
                    st.session_state.image_results.pop(idx)
                st.rerun()

            # — AI mapping result banner —
            if mapping:
                prop_name  = mapping.get("property_name", "")
                section    = mapping.get("section", "")
                confidence = mapping.get("confidence", "low")
                reasoning  = mapping.get("reasoning", "")
                db_row     = mapping.get("db_row")
                candidates = mapping.get("all_candidates", [])

                badge = _confidence_badge(confidence)
                if prop_name:
                    st.markdown(
                        f"🔗 **AI Match:** `{section}` › **{prop_name}** &nbsp; {badge}",
                        unsafe_allow_html=True,
                    )
                    if reasoning:
                        st.caption(f"💬 {reasoning}")

                    # DB row details
                    if db_row:
                        with st.expander("📋 Matched DB row", expanded=False):
                            col_v, col_u, col_c = st.columns(3)
                            col_v.metric("Value",    db_row.get("value",        "—"))
                            col_u.metric("Unit",     db_row.get("unit",         "—"))
                            col_c.metric("Condition", db_row.get("test_condition", "—"))
                            if db_row.get("comments"):
                                st.caption(f"Comments: {db_row['comments']}")
                            if db_row.get("english"):
                                st.caption(f"English units: {db_row['english']}")

                    # Alternative candidates
                    if candidates:
                        with st.expander("🔄 All candidates", expanded=False):
                            for c in candidates:
                                rank = c.get("rank", "?")
                                cn   = c.get("confidence", "low")
                                st.markdown(
                                    f"{rank}. `{c.get('section','?')}` › "
                                    f"**{c.get('property_name','?')}** "
                                    f"&nbsp; {_confidence_badge(cn)}",
                                    unsafe_allow_html=True,
                                )
                else:
                    st.warning("⚠️ AI could not match this plot to any DB property.")

            # — sub-images —
            for p_idx, img_data in enumerate(img_list):
                bgr = img_data.get("array")
                if bgr is None:
                    continue

                img_key = f"{idx}_{p_idx}_{page}"

                # Show the plot
                st.image(bgr, channels="BGR", width=420)

                # — mapping controls —
                if has_data:
                    df          = st.session_state.pdf_extracted_df
                    mat_abbr    = df.iloc[0]["material_abbreviation"]
                    property_list = df["property_name"].unique().tolist()

                    # Pre-select the AI suggestion if available
                    ai_suggestion = mapping.get("property_name", "") if mapping else ""
                    default_idx   = 0
                    options       = ["— Select property —"] + property_list
                    if ai_suggestion in property_list:
                        default_idx = property_list.index(ai_suggestion) + 1

                    col_sel, col_sec, col_save, col_rem = st.columns(
                        [0.42, 0.18, 0.20, 0.20]
                    )

                    with col_sel:
                        selected = st.selectbox(
                            "Property",
                            options=options,
                            index=default_idx,
                            key=f"prop_sel_{img_key}",
                            label_visibility="collapsed",
                        )

                    with col_sec:
                        section_override = st.text_input(
                            "Section",
                            value=mapping.get("section", "") if mapping else "",
                            key=f"sec_{img_key}",
                            label_visibility="collapsed",
                            placeholder="Section",
                        )

                    with col_save:
                        if st.button("💾 Save", key=f"save_{img_key}"):
                            if selected and selected != "— Select property —":
                                filepath = save_plot_image_mapping(
                                    mat_abbr,
                                    selected,
                                    section_override,
                                    bgr,
                                    save_dir="images",
                                )
                                st.session_state.saved_image_mapping[img_key] = {
                                    "property":  selected,
                                    "section":   section_override,
                                    "caption":   caption,
                                    "filename":  os.path.basename(filepath),
                                    "path":      filepath,
                                }
                                st.success(f"Saved → `{os.path.basename(filepath)}`")
                                st.rerun()
                            else:
                                st.warning("Select a property first.")

                    with col_rem:
                        if st.button("✕ Remove", key=f"rem_{img_key}"):
                            img_list.pop(p_idx)
                            if not img_list:
                                if use_mapped:
                                    st.session_state.mapped_results.pop(idx)
                                else:
                                    st.session_state.image_results.pop(idx)
                            st.rerun()

                    # Saved badge
                    if img_key in st.session_state.saved_image_mapping:
                        m = st.session_state.saved_image_mapping[img_key]
                        st.info(f"✅ Saved as **{m['property']}** → `{m['filename']}`")

                else:
                    # No data extracted yet — just allow removal
                    col_msg, col_rem = st.columns([0.80, 0.20])
                    col_msg.caption(
                        "Extract material data in the **Material Data** tab to enable mapping."
                    )
                    if col_rem.button("✕ Remove", key=f"rem_nodata_{img_key}"):
                        img_list.pop(p_idx)
                        if not img_list:
                            st.session_state.image_results.pop(idx)
                        st.rerun()

                st.divider()

    # ── 6. Saved-mappings summary ─────────────────────────────────────────────
    if st.session_state.saved_image_mapping:
        with st.expander(
            f"📁 Saved mappings ({len(st.session_state.saved_image_mapping)})",
            expanded=False,
        ):
            for key, info in st.session_state.saved_image_mapping.items():
                st.markdown(
                    f"**{info['property']}** &nbsp;›&nbsp; `{info['filename']}`  \n"
                    f"<small>Caption: {info['caption']}</small>",
                    unsafe_allow_html=True,
                )
