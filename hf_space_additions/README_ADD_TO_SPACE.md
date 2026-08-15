# Add the "Recently added" page to the HF Space (2-minute change)

Shows visitors which papers/datasheets the pipeline just ingested, with
verified-property counts. Reads the `sources` table the pipeline maintains —
no other backend changes needed.

## Steps (via the Hugging Face web UI — no git required)

1. Open https://huggingface.co/spaces/aim4composites/MaterialsDatabase/tree/main
2. Go into `page_files/` → **Add file → Upload file** → upload `Recently_Added.py`
   (from this folder) → Commit.
3. Open `app.py` → **Edit** → add one line to the pages list:

```python
pages = {
    "": [
        st.Page("page_files/Home.py", title="Home"),
        st.Page("page_files/Categorized_Search.py", title="Categorized Search"),
        st.Page("page_files/Recently_Added.py", title="Recently Added"),   # <-- add this
        st.Page("page_files/Upload_Data.py", title="Upload Data"),
        st.Page("page_files/Contact_Team.py", title="Contact Team"),
    ]
}
```

4. Commit — the Space rebuilds itself (~2–4 min) and the new tab appears.

## Optional 2: sort the main materials table by date added (newest first)

Gives the All Materials table an **ADDED** column, sorts newest-first by
default, and tags fresh rows (≤14 days) with 🆕. Legacy rows show "—" and sink
to the bottom. Two files to touch **in the Space repo** (edit in the HF web
editor; every commit is revertible via the file history):

**A. `data_loader.py`** — replace it with the `data_loader.py` in this folder
(only change: the SELECT and column list now include `extracted_at`).
If the file has changed recently, just add `extracted_at` to both the
`EMPTY_MATERIAL_COLUMNS` list and the `SELECT` instead.

**B. `page_files/Categorized_Search.py`** — two find-and-replace edits:

FIND (the `meta = (...)` block near the top, after `st.session_state["base_data"] = all_data`):

```python
meta = (
    all_data[["material_abbreviation", "material_name", "_class"]]
    .fillna("")
    .drop_duplicates(subset=["material_abbreviation"])
    .reset_index(drop=True)
)
```

REPLACE WITH:

```python
if "extracted_at" not in all_data.columns:
    all_data["extracted_at"] = None
_msrc = all_data[["material_abbreviation", "material_name", "_class", "extracted_at"]].copy()
_msrc["_added"] = pd.to_datetime(_msrc["extracted_at"], errors="coerce", utc=True)
_msrc = _msrc.sort_values("_added", ascending=False, na_position="last")
meta = (
    _msrc[["material_abbreviation", "material_name", "_class", "_added"]]
    .fillna({"material_abbreviation": "", "material_name": "", "_class": ""})
    .drop_duplicates(subset=["material_abbreviation"])
    .reset_index(drop=True)
)
```

FIND (inside the `if not page_meta.empty:` branch):

```python
                table_df["Class"] = table_df["_class"].map(class_map)
                table_df["Actions"] = ""
                table_df = table_df[
                    [ "material_name", "material_abbreviation", "Class", "Actions"]
                ].rename(
```

REPLACE WITH:

```python
                table_df["Class"] = table_df["_class"].map(class_map)
                _added = pd.to_datetime(table_df.get("_added"), errors="coerce", utc=True)
                table_df["Added"] = _added.dt.strftime("%Y-%m-%d")
                _fresh = (pd.Timestamp.now(tz="UTC") - _added).dt.days <= 14
                table_df.loc[_fresh.fillna(False), "Added"] = "🆕 " + table_df["Added"]
                table_df["Added"] = table_df["Added"].fillna("—")
                table_df["Actions"] = ""
                table_df = table_df[
                    [ "material_name", "material_abbreviation", "Class", "Added", "Actions"]
                ].rename(
```

FIND (the `column_config={` block of the main table):

```python
                    "Class": st.column_config.TextColumn("CLASS", width="small"),
                    "Actions": st.column_config.TextColumn("ACTIONS", width="small"),
```

REPLACE WITH:

```python
                    "Class": st.column_config.TextColumn("CLASS", width="small"),
                    "Added": st.column_config.TextColumn("ADDED", width="small"),
                    "Actions": st.column_config.TextColumn("ACTIONS", width="small"),
```

Commit → the Space rebuilds → newest materials appear at the top of page 1
with 🆕 dates.

## Notes

- Safe if the pipeline hasn't run yet: the page shows a friendly "nothing yet"
  message instead of erroring.
- Counts only `status='ok'` (verified) properties — quarantined rows are
  excluded, matching the pipeline's publish rules.
- Cached for 5 minutes per load, so it adds no meaningful DB load.
- Optional same-visit cleanup (coordinate with Abhijit): in `data_loader.py`,
  add `WHERE COALESCE(status,'ok') = 'ok'` to the SELECT in
  `load_material_data()` so flagged rows never appear in search results either.
