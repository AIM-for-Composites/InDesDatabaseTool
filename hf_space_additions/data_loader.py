from db import execute_query, fetch_all
import pandas as pd

EMPTY_MATERIAL_COLUMNS = [
    "material_name",
    "material_abbreviation",
    "section",
    "property_name",
    "value",
    "unit",
    "english",
    "test_condition",
    "comments",
    "extracted_at",
]

def load_material_data(material_type: str) -> pd.DataFrame:
    table_map = {
        "Polymers": "Polymers",
        "Fibers": "Fibers",
        "Composites": "Composites_materials",
    }

    table = table_map.get(material_type)
    if not table:
        return pd.DataFrame(columns=EMPTY_MATERIAL_COLUMNS)

    # Publish gate: only rows the pipeline has verified (status='ok') reach the
    # search page. Rows the ingester quarantined (unverified / unit_review /
    # out_of_range / empty_value, and figure-derived estimates) stay in the
    # tables for the review queue but are never shown here until promoted.
    # Legacy rows have status NULL and read as 'ok' via COALESCE.
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
            comments,
            extracted_at
        FROM "{table}"
        WHERE COALESCE(status, 'ok') = 'ok'
    """

    try:
        rows = fetch_all(query)
    except Exception:
        # Most likely cause: the DB was never migrated (no `status` column).
        # Degrade to the unfiltered SELECT rather than blanking the whole
        # search page — but say so, loudly, in the Space logs.
        import logging
        logging.getLogger("data_loader").warning(
            "status-filtered SELECT on %s failed; falling back to UNFILTERED "
            "rows. Run `python pg_migrate.py --apply` so quarantined rows are "
            "hidden from search.", table, exc_info=True,
        )
        try:
            rows = fetch_all(query.replace("WHERE COALESCE(status, 'ok') = 'ok'", ""))
        except Exception:
            return pd.DataFrame(columns=EMPTY_MATERIAL_COLUMNS)
    return pd.DataFrame(rows, columns=EMPTY_MATERIAL_COLUMNS)

def get_all_sections():
    all_data = pd.concat([
        load_material_data("Polymers"),
        load_material_data("Fibers"),
        load_material_data("Composites"),
    ], ignore_index=True)

    if all_data.empty or "section" not in all_data.columns:
        return []
    return sorted(all_data["section"].dropna().unique().tolist())


def insert_material_rows(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0

    table_map = {
        "Polymer": "Polymers",
        "Fiber": "Fibers",
        "Composite": "Composites_materials",
    }

    insert_template = """
        INSERT INTO "{table}" (
            material_name,
            material_abbreviation,
            section,
            property_name,
            value,
            unit,
            english,
            test_condition,
            comments
        ) VALUES (
            :material_name,
            :material_abbreviation,
            :section,
            :property_name,
            :value,
            :unit,
            :english,
            :test_condition,
            :comments
        )
    """

    inserted = 0
    for _, row in df.iterrows():
        table = table_map.get(row.get("material_class"))
        if not table:
            continue

        params = {
            "material_name": row.get("material_name", ""),
            "material_abbreviation": row.get("material_abbreviation", ""),
            "section": row.get("section", ""),
            "property_name": row.get("property_name", ""),
            "value": row.get("value", ""),
            "unit": row.get("unit", ""),
            "english": row.get("english", ""),
            "test_condition": row.get("test_condition", ""),
            "comments": row.get("comments", ""),
        }

        try:
            inserted += execute_query(insert_template.format(table=table), params)
        except Exception:
            return inserted

    return inserted
