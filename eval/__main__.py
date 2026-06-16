"""Run the extraction eval harness.

Usage:
    export GEMINI_API_KEY=...
    python -m eval                       # score every gold/*.json against live extraction
    python -m eval --report eval_report.json
    python -m eval --baseline prev.json  # also print delta vs a prior report
    python -m eval --selfcheck           # offline: prove the scoring logic (no API)

For each gold PDF it runs extraction.extract_from_pdf + verify_against_text, then
scores material/property presence and value-within-tolerance. Writes a JSON
report and prints a summary. Run it after each P0 change to prove the change
helps (or holds) versus the baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import extraction as E
from eval import scoring

GOLD_DIR = Path(__file__).parent / "gold"


def _load_gold() -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(GOLD_DIR.glob("*.json"))]


def run_live(report_path: Path) -> dict[str, Any]:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set; cannot run live eval.",
              file=sys.stderr)
        print("Use `python -m eval --selfcheck` to validate the harness offline.",
              file=sys.stderr)
        return {}

    reports: list[dict[str, Any]] = []
    for gold in _load_gold():
        pdf_path = GOLD_DIR / gold["pdf"]
        if not pdf_path.exists():
            print(f"  missing gold PDF: {pdf_path}", file=sys.stderr)
            continue
        pdf_bytes = pdf_path.read_bytes()
        ext = E.extract_from_pdf(pdf_bytes, pdf_path.name, api_key)
        E.verify_against_text(ext, E.pdf_page_texts(pdf_bytes))
        rep = scoring.score_extraction(ext, gold)
        reports.append(rep)
        print(f"  {rep['pdf']}: mat R/P={rep['material_recall']}/{rep['material_precision']} "
              f"prop R/P={rep['prop_recall']}/{rep['prop_precision']} "
              f"value_acc={rep['value_accuracy']}")

    out = {"per_pdf": reports, "aggregate": scoring.aggregate(reports)}
    report_path.write_text(json.dumps(out, indent=2))
    print(f"\nAggregate: {json.dumps(out['aggregate'], indent=2)}")
    print(f"Report written to {report_path}")
    return out


def selfcheck() -> int:
    """Offline proof that the scoring logic is correct (no API needed)."""
    gold_list = _load_gold()
    if not gold_list:
        print("no gold files found", file=sys.stderr)
        return 1
    gold = next((g for g in gold_list if g["pdf"] == "tc920_pc_abs.pdf"), gold_list[0])

    # Build a near-perfect prediction straight from the gold, but deliberately
    # express some values in *different units* (ksi/Msi) to prove SI comparison,
    # and drop one property to prove recall < 1.
    def mk_props(gm, unit_swap=False, drop_last=False):
        props = []
        glist = gm["properties"][:-1] if drop_last else gm["properties"]
        for gp in glist:
            unit = gp.get("unit", "")
            vnum = gp.get("value_num")
            if unit_swap and unit == "MPa" and vnum is not None:
                unit, vnum = "ksi", vnum / 6.894757  # same physical value
            p = E.Property(section="Mechanical", property_name=gp["property_name"],
                           value_raw=str(vnum), unit=unit,
                           value_num=vnum, value_min=gp.get("value_min"),
                           value_max=gp.get("value_max"))
            _, p.value_si, _ = E.canonicalize(p)
            props.append(p)
        return props

    mats = [
        E.Material(material_name=gm["material_name"], material_class=gm["material_class"],
                   properties=mk_props(gm, unit_swap=(i == 1), drop_last=(i == 0)))
        for i, gm in enumerate(gold["materials"])
    ]
    pred = E.Extraction(materials=mats)
    rep = scoring.score_extraction(pred, gold)
    print(json.dumps(rep, indent=2))

    ok = True
    # all 3 materials should be matched
    if rep["material_recall"] != 1.0:
        ok = False; print("FAIL material_recall", rep["material_recall"])
    # one property dropped from material 0 -> prop_recall < 1
    if not (0.0 < rep["prop_recall"] < 1.0):
        ok = False; print("FAIL prop_recall not in (0,1)", rep["prop_recall"])
    # ksi-swapped values must still match in SI -> value_accuracy should be 1.0
    if rep["value_accuracy"] != 1.0:
        ok = False; print("FAIL value_accuracy (SI unit-robust compare)", rep["value_accuracy"])
    print("SELFCHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", type=Path, default=Path("eval_report.json"))
    ap.add_argument("--baseline", type=Path, default=None,
                    help="Prior report JSON to diff the aggregate against")
    ap.add_argument("--selfcheck", action="store_true",
                    help="Validate the scoring logic offline (no API)")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()

    out = run_live(args.report)
    if not out:
        return 2
    if args.baseline and args.baseline.exists():
        base = json.loads(args.baseline.read_text()).get("aggregate", {})
        print("\nDelta vs baseline:")
        for k, v in out["aggregate"].items():
            if isinstance(v, (int, float)) and k in base:
                d = round(v - base[k], 4)
                print(f"  {k}: {base[k]} -> {v} ({'+' if d >= 0 else ''}{d})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
