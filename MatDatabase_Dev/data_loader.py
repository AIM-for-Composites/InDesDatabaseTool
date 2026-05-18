from db import fetch_all
import pandas as pd

def load_material_data(material_type: str) -> pd.DataFrame:
    table_map = {
        "Polymers": "Polymers",
        "Fibers": "Fibers",
        "Composites": "Composites_materials",  
    }

    table = table_map[material_type]
    if not table:
        return pd.DataFrame()

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
            comments
        FROM "{table}"
    """

    rows = fetch_all(query)
    return pd.DataFrame(rows)

def get_all_sections():
    all_data = pd.concat([
        load_material_data("Polymers"),
        load_material_data("Fibers"),
        load_material_data("Composites"),
    ], ignore_index=True)

    return sorted(all_data["section"].dropna().unique().tolist())

def insert_material_rows(df):
    """Insert DataFrame rows into the correct material table based on material_class."""
    from db import execute_query
    
    inserted = 0
    for _, row in df.iterrows():
        material_class = row.get("material_class", "")
        table_map = {
            "Polymer":   "Polymers",
            "Fiber":     "Fibers",
            "Composite": "Composites_materials",
        }
        table = table_map.get(material_class)
        if not table:
            continue

        query = f"""
            INSERT INTO "{table}"
                (material_name, material_abbreviation, section,
                 property_name, value, unit, english, test_condition, comments)
            VALUES
                (:material_name, :material_abbreviation, :section,
                 :property_name, :value, :unit, :english, :test_condition, :comments)
        """
        params = {
            "material_name":         str(row.get("material_name", "")),
            "material_abbreviation": str(row.get("material_abbreviation", "")),
            "section":               str(row.get("section", "")),
            "property_name":         str(row.get("property_name", "")),
            "value":                 str(row.get("value", "")),
            "unit":                  str(row.get("unit", "")),
            "english":               str(row.get("english", "")),
            "test_condition":        str(row.get("test_condition", "")),
            "comments":              str(row.get("comments", "")),
        }
        inserted += execute_query(query, params)

    return inserted

def load_property_image(material_abbr: str, property_name: str, material_class: str):
    """Fetch image bytes from DB for a given property."""
    from db import fetch_one

    table_map = {
        "Polymer":   "Polymers",
        "Fiber":     "Fibers",
        "Composite": "Composites_materials",
    }

    table = table_map.get(material_class)
    if not table:
        return None

    row = fetch_one(
        f"""
        SELECT image FROM "{table}"
        WHERE LOWER(material_abbreviation) = LOWER(:abbr)
        AND LOWER(property_name) = LOWER(:prop)
        AND image IS NOT NULL
        LIMIT 1
        """,
        {"abbr": material_abbr, "prop": property_name},
    )

    return row["image"] if row else None