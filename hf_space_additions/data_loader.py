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

# Publish gate (see load_material_data). Kept as one constant so the
# unmigrated-DB fallback's replace() can never drift from the SQL.
_STATUS_FILTER = "WHERE COALESCE(status, 'ok') = 'ok'"

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
        {_STATUS_FILTER}
    """

    try:
        rows = fetch_all(query)
    except Exception as exc:
        # Degrade to the unfiltered SELECT ONLY when the failure is the one
        # schema case this filter can cause — the DB was never migrated and
        # has no `status` column (Postgres SQLSTATE 42703 "undefined column").
        # Any other error (connection reset, timeout, pool exhausted) keeps
        # the previous behavior — an empty frame — because the caller wraps
        # this in st.cache_data with no TTL, and one transient hiccup must
        # not publish quarantined rows for the life of the process.
        if _is_undefined_column(exc, "status"):
            import logging
            logging.getLogger("data_loader").warning(
                "%s has no `status` column; falling back to the UNFILTERED "
                "SELECT. Run `python pg_migrate.py --apply` so quarantined "
                "rows are hidden from search.", table,
            )
            try:
                rows = fetch_all(query.replace(_STATUS_FILTER, ""))
            except Exception:
                return pd.DataFrame(columns=EMPTY_MATERIAL_COLUMNS)
        else:
            return pd.DataFrame(columns=EMPTY_MATERIAL_COLUMNS)
    return pd.DataFrame(rows, columns=EMPTY_MATERIAL_COLUMNS)


def _is_undefined_column(exc: Exception, column: str) -> bool:
    """True iff `exc` is Postgres 'column ... does not exist' for `column`."""
    sqlstate = getattr(exc, "sqlstate", None) or getattr(
        getattr(exc, "diag", None), "sqlstate", None) or getattr(exc, "pgcode", None)
    msg = str(exc).lower()
    if sqlstate == "42703":
        return column in msg
    return ("does not exist" in msg or "undefined column" in msg) and column in msg

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
