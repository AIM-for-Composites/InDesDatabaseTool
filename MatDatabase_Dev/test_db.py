from db import fetch_all

rows = fetch_all("SELECT 1 AS test_value")
print(rows)