"""Run the extraction eval harness.

Usage:
    export GEMINI_API_KEY=...
    python -m eval                       # score every gold/*.json against live extraction
    python -m eval --report eval_report.json
    python -m eval --baseline prev.json  # also print delta vs a prior report
    python -m eval --selfcheck           # offline: prove the scoring logic (no API)
    python -m eval --gold-check          # offline: every gold value/alias is in its PDF text

For each gold PDF it runs extraction.extract_from_pdf + verify_against_text, then
scores material/property presence and value-within-tolerance. Writes a JSON
report and prints a summary. Run it after each P0 change to prove the change
helps (or holds) versus the baseline.

Evidence retention: every live run also dumps each case's full Extraction to
``eval/last_run/<case>.extraction.json`` (gitignored) and records a
``pred_summary`` (predicted material names + property names) in the report, so a
surprising score — e.g. a 0.0 on a case whose gold is verifiably in the PDF —
can be diagnosed without re-spending the API call.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any

import extraction as E
from eval import scoring

GOLD_DIR = Path(__file__).parent / "gold"
LAST_RUN_DIR = Path(__file__).parent / "last_run"


def _load_gold() -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(GOLD_DIR.glob("*.json"))]


def _pred_summary(ext: E.Extraction) -> dict[str, Any]:
    """Compact view of what the model returned, for the report."""
    return {
        "doc_status": ext.doc_status,
        "materials": [
            {
                "material_name": m.material_name,
                "material_class": m.material_class,
                "trade_grade": m.trade_grade,
                "n_properties": len(m.properties),
                "property_names": sorted({p.property_name for p in m.properties})[:60],
            }
            for m in ext.materials
        ],
    }


def run_live(report_path: Path) -> dict[str, Any]:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set; cannot run live eval.",
              file=sys.stderr)
        print("Use `python -m eval --selfcheck` to validate the harness offline.",
              file=sys.stderr)
        return {}

    LAST_RUN_DIR.mkdir(exist_ok=True)
    reports: list[dict[str, Any]] = []
    for gold in _load_gold():
        pdf_path = GOLD_DIR / gold["pdf"]
        if not pdf_path.exists():
            print(f"  missing gold PDF: {pdf_path}", file=sys.stderr)
            continue
        pdf_bytes = pdf_path.read_bytes()
        ext = E.extract_from_pdf(pdf_bytes, pdf_path.name, api_key)
        E.verify_against_text(ext, E.pdf_page_texts(pdf_bytes))
        # keep the evidence
        (LAST_RUN_DIR / f"{pdf_path.stem}.extraction.json").write_text(
            json.dumps(dataclasses.asdict(ext), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        rep = scoring.score_extraction(ext, gold)
        rep["pred_summary"] = _pred_summary(ext)
        reports.append(rep)
        print(f"  {rep['pdf']}: mat R/P={rep['material_recall']}/{rep['material_precision']} "
              f"prop R/P={rep['prop_recall']}/{rep['prop_precision']} "
              f"value_acc={rep['value_accuracy']}")

    out = {"per_pdf": reports, "aggregate": scoring.aggregate(reports)}
    report_path.write_text(json.dumps(out, indent=2))
    print(f"\nAggregate: {json.dumps(out['aggregate'], indent=2)}")
    print(f"Report written to {report_path}; raw extractions in {LAST_RUN_DIR}/")
    return out


def gold_check() -> int:
    """Offline: prove each gold case is *answerable* from its PDF's text.

    For every gold material, at least one name/alias must occur in the PDF; for
    every gold property, at least one alias must occur AND the gold number
    (value_num, or value_min/max) must occur on digit boundaries. Catches gold
    authoring mistakes (typos, values read from a different table/condition)
    without an API key. A case that passes here but scores 0.0 live is an
    extraction-side problem — look at eval/last_run/<case>.extraction.json.
    """
    import re
    from extraction import _grounded, _normalize_text

    problems = 0
    for gold in _load_gold():
        pdf_path = GOLD_DIR / gold["pdf"]
        if not pdf_path.exists():
            print(f"  {gold['pdf']}: MISSING PDF"); problems += 1; continue
        pages = [_normalize_text(t) for t in E.pdf_page_texts(pdf_path.read_bytes())]
        text = " ".join(pages)
        case_ok = True
        for gm in gold.get("materials", []):
            names = [gm.get("material_name", "")] + list(gm.get("aliases", []))
            if not any(_normalize_text(n) and _normalize_text(n) in text for n in names):
                print(f"  {gold['pdf']}: material {gm.get('material_name')!r}: "
                      f"no name/alias found in PDF text"); case_ok = False
            for gp in gm.get("properties", []):
                keys = [gp.get("property_name", "")] + list(gp.get("aliases", []))
                if not any(_normalize_text(k) and _normalize_text(k) in text for k in keys):
                    print(f"  {gold['pdf']}: property {gp.get('property_name')!r}: "
                          f"no name/alias found in PDF text"); case_ok = False
                # Try the ways a datasheet prints the number: 58 / 58.0 / 58.00,
                # and for a range the whole 'lo-hi' token as well as endpoints.
                def forms(n: float) -> list[str]:
                    return list(dict.fromkeys([f"{n:g}", f"{n:.1f}", f"{n:.2f}"]))
                candidates: list[str] = []
                if gp.get("value_num") is not None:
                    candidates += forms(gp["value_num"])
                lo, hi = gp.get("value_min"), gp.get("value_max")
                if lo is not None and hi is not None:
                    candidates += [f"{a}-{b}" for a in forms(lo) for b in forms(hi)]
                    candidates += forms(lo) + forms(hi)
                elif lo is not None:
                    candidates += forms(lo)
                elif hi is not None:
                    candidates += forms(hi)
                if candidates and not any(_grounded(c, pages) for c in candidates):
                    print(f"  {gold['pdf']}: property {gp.get('property_name')!r}: "
                          f"value {candidates[0]} not found in PDF text"); case_ok = False
        print(f"  {gold['pdf']}: {'OK' if case_ok else 'PROBLEMS'}")
        problems += 0 if case_ok else 1
    print("GOLD-CHECK", "PASS" if problems == 0 else f"FAIL ({problems} case(s))")
    return 0 if problems == 0 else 1


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
    ap.add_argument("--gold-check", action="store_true",
                    help="Offline: verify every gold value/alias is present in its PDF text")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()
    if args.gold_check:
        return gold_check()

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
