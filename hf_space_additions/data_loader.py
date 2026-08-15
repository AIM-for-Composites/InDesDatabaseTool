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
    """

    try:
        rows = fetch_all(query)
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
