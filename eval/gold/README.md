# Gold set for the extraction eval harness

Each gold case is a pair:

- `<name>.pdf` — the source PDF (a real datasheet/paper)
- `<name>.json` — the hand-labeled expected extraction

Run the harness from the project root:

```bash
export GEMINI_API_KEY=...
python -m eval                      # score every gold case against live extraction
python -m eval --baseline prev.json # print the aggregate delta vs a prior report
python -m eval --selfcheck          # offline check of the scoring logic (no API)
```

`python -m eval` runs `extraction.extract_from_pdf` + `verify_against_text` on
each PDF and scores:

- **material presence** — precision/recall of expected materials found
- **(material, property) presence** — precision/recall of expected properties
- **value-within-tolerance** — fraction of matched properties whose value is
  within `tolerance_pct` of gold, **compared in SI** so a ksi/MPa or GPa/MPa
  unit difference is not counted as a miss

## Gold JSON schema

```jsonc
{
  "pdf": "<file>.pdf",
  "notes": "where the values came from / what this case tests",
  "materials": [
    {
      "material_name": "string",
      "aliases": ["substrings that should match the predicted name"],
      "material_class": "Polymer | Fiber | Composite",
      "properties": [
        {
          "property_name": "string",
          "aliases": ["substrings that should match the predicted property_name"],
          "value_num": 0.0,        // single value (omit for a range)
          "value_min": 0.0,        // range low (optional)
          "value_max": 0.0,        // range high (optional)
          "unit": "MPa",           // unit of value_* as written here
          "tolerance_pct": 5       // allowed deviation when value-matching
        }
      ]
    }
  ]
}
```

## Current cases

| case | what it exercises |
|------|-------------------|
| `tc920_pc_abs` | **3 materials in one datasheet** (fiberglass UD tape, carbon UD tape, neat resin) — multi-material separation + no cross-contamination; ksi/Msi vs MPa/GPa; Tg range 70–75 °C |
| `tc910_pa6` | single composite (carbon/PA6); ksi/Msi → MPa/GPa normalization |

## Adding a case

1. Drop the PDF in this folder.
2. Open it, read the values, and write `<name>.json` (label only values you can
   see verbatim — the harness rewards grounded, correct values, not volume).
3. Re-run `python -m eval`. Aim for 3–5 cases spanning polymer / fiber /
   composite and at least one multi-material and one range-valued sheet.
