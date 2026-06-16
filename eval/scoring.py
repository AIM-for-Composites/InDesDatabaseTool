"""Pure scoring functions for the extraction eval harness (Task 9).

Scores a predicted :class:`extraction.Extraction` against a gold spec on:

* material presence (precision / recall)
* (material, property) presence (precision / recall)
* value-within-tolerance, compared in SI so a ksi/MPa or GPa/MPa unit
  difference doesn't count as a value miss.

Kept import-light and free of any network/API use so it can be unit-tested
offline (see ``test_scoring`` in ``python -m eval --selfcheck``).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Optional

import extraction as E


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"[^a-z0-9/ ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _material_matches(pred: E.Material, gold: dict[str, Any]) -> bool:
    pname = _norm(pred.material_name)
    names = [gold.get("material_name", "")] + list(gold.get("aliases", []))
    for cand in names:
        c = _norm(cand)
        if c and (c in pname or pname in c):
            return True
    return False


def _property_matches(pred: E.Property, gold_prop: dict[str, Any]) -> bool:
    pname = _norm(pred.property_name)
    keys = [gold_prop.get("property_name", "")] + list(gold_prop.get("aliases", []))
    return any(_norm(k) and _norm(k) in pname for k in keys)


def _gold_si(gold_prop: dict[str, Any]) -> Optional[float]:
    """Representative SI value of a gold property, via extraction's own unit logic."""
    p = E.Property(
        section="",
        property_name=gold_prop.get("property_name", ""),
        value_raw="",
        unit=gold_prop.get("unit", ""),
        value_num=gold_prop.get("value_num"),
        value_min=gold_prop.get("value_min"),
        value_max=gold_prop.get("value_max"),
    )
    _, value_si, _ = E.canonicalize(p)
    if value_si is not None:
        return value_si
    return E._representative_value(p)


def _value_within_tol(pred: E.Property, gold_prop: dict[str, Any]) -> Optional[bool]:
    """True/False if comparable, None if the gold has no numeric target."""
    gold_si = _gold_si(gold_prop)
    if gold_si is None:
        return None
    if pred.value_si is not None:
        pred_val = pred.value_si
    else:
        pred_val = E._representative_value(pred)
    if pred_val is None:
        return False
    tol = float(gold_prop.get("tolerance_pct", 5)) / 100.0
    denom = abs(gold_si) if gold_si else 1.0
    return abs(pred_val - gold_si) <= tol * denom + 1e-9


def score_extraction(pred: E.Extraction, gold: dict[str, Any]) -> dict[str, Any]:
    """Score one predicted Extraction against one gold spec; return a metrics dict."""
    gold_materials = gold.get("materials", [])
    pred_materials = pred.materials

    # --- material presence ---
    matched_gold_mat = 0
    gold_to_pred: dict[int, list[E.Material]] = {}
    for gi, gm in enumerate(gold_materials):
        hits = [pm for pm in pred_materials if _material_matches(pm, gm)]
        gold_to_pred[gi] = hits
        if hits:
            matched_gold_mat += 1
    matched_pred_mat = sum(
        1 for pm in pred_materials if any(_material_matches(pm, gm) for gm in gold_materials)
    )

    # --- (material, property) presence + value accuracy ---
    gold_prop_total = 0
    gold_prop_found = 0
    value_total = 0
    value_correct = 0
    misses: list[str] = []
    for gi, gm in enumerate(gold_materials):
        cand_preds = gold_to_pred[gi]
        cand_props = [p for pm in cand_preds for p in pm.properties]
        for gp in gm.get("properties", []):
            gold_prop_total += 1
            matches = [pp for pp in cand_props if _property_matches(pp, gp)]
            if not matches:
                misses.append(f"{gm.get('material_name','?')} :: {gp.get('property_name','?')} (absent)")
                continue
            gold_prop_found += 1
            # pick the closest-by-value match for value scoring
            within = [_value_within_tol(pp, gp) for pp in matches]
            verdicts = [v for v in within if v is not None]
            if verdicts:
                value_total += 1
                if any(verdicts):
                    value_correct += 1
                else:
                    misses.append(
                        f"{gm.get('material_name','?')} :: {gp.get('property_name','?')} (value off)"
                    )

    # predicted-property precision: a predicted prop is a TP if it matches some
    # gold prop under a material it was matched to.
    pred_prop_total = sum(len(pm.properties) for pm in pred_materials)
    pred_prop_tp = 0
    for pm in pred_materials:
        matched_golds = [gm for gm in gold_materials if _material_matches(pm, gm)]
        for pp in pm.properties:
            if any(_property_matches(pp, gp) for gm in matched_golds for gp in gm.get("properties", [])):
                pred_prop_tp += 1

    def ratio(n: int, d: int) -> float:
        return round(n / d, 4) if d else 0.0

    return {
        "pdf": gold.get("pdf"),
        "materials_gold": len(gold_materials),
        "materials_pred": len(pred_materials),
        "material_recall": ratio(matched_gold_mat, len(gold_materials)),
        "material_precision": ratio(matched_pred_mat, len(pred_materials)),
        "prop_recall": ratio(gold_prop_found, gold_prop_total),
        "prop_precision": ratio(pred_prop_tp, pred_prop_total),
        "value_accuracy": ratio(value_correct, value_total),
        "value_checked": value_total,
        "misses": misses,
    }


def aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Micro-averaged-ish summary across per-PDF reports."""
    def avg(key: str) -> float:
        vals = [r[key] for r in reports if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "pdfs": len(reports),
        "material_recall": avg("material_recall"),
        "material_precision": avg("material_precision"),
        "prop_recall": avg("prop_recall"),
        "prop_precision": avg("prop_precision"),
        "value_accuracy": avg("value_accuracy"),
    }
