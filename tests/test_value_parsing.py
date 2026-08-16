"""Regression tests for extraction.parse_value_raw / _fill_numeric.

Every case here was a real failure found in the 2026-08 adversarial review
(or a pre-existing behavior we must not regress). Inputs are exactly what a
PDF prints — typographic minus (U+2212), en/em dashes, thousands separators,
scientific notation, ± in three spellings.
"""

from __future__ import annotations

import pytest

from extraction import Property, _fill_numeric, parse_value_raw


@pytest.mark.parametrize(
    "raw, expected",
    [
        # --- U+2212 typographic minus (was: sign flipped / range lost) ---
        ("−18.0", (-18.0, None, None, "")),
        ("20−30", (None, 20.0, 30.0, "")),
        ("−40 to −10", (None, -40.0, -10.0, "")),
        ("−5 ± 1", (-5.0, -6.0, -4.0, "±")),
        # --- ± with thousands separators / scientific notation (was: mangled) ---
        ("1,200 ± 100", (1200.0, 1100.0, 1300.0, "±")),
        ("1e-3 ± 2e-4", (1e-3, 8e-4, 1.2e-3, "±")),
        ("5.0 +/- 0.2", (5.0, 4.8, 5.2, "±")),
        ("5.0 +- 0.2", (5.0, 4.8, 5.2, "±")),
        # --- pre-existing behaviors that must hold ---
        ("-40 to -10", (None, -40.0, -10.0, "")),
        ("1,000-2,000", (None, 1000.0, 2000.0, "")),
        ("2.5e3 - 3.5e3", (None, 2500.0, 3500.0, "")),
        ("2.5e-3 - 3.5e-3", (None, 2.5e-3, 3.5e-3, "")),
        (">= 1,500", (1500.0, None, None, ">=")),
        ("-18.0", (-18.0, None, None, "")),
        ("~3.5", (3.5, None, None, "~")),
        ("100–120", (None, 100.0, 120.0, "")),   # en dash
        ("100—120", (None, 100.0, 120.0, "")),   # em dash
        ("≤ -18.0", (-18.0, None, None, "<=")),
        ("3.2", (3.2, None, None, "")),
        ("1e-5", (1e-5, None, None, "")),
        ("12,345.6", (12345.6, None, None, "")),
        ("0.5 ... 0.9", (None, 0.5, 0.9, "")),
        ("20 - -5", (None, -5.0, 20.0, "")),          # reversed range is sorted
        ("150", (150.0, None, None, "")),
        ("n/a", (None, None, None, "")),
        ("", (None, None, None, "")),
    ],
)
def test_parse_value_raw(raw, expected):
    num, lo, hi, qual = parse_value_raw(raw)
    e_num, e_lo, e_hi, e_qual = expected
    assert qual == e_qual
    for got, want in ((num, e_num), (lo, e_lo), (hi, e_hi)):
        if want is None:
            assert got is None
        else:
            assert got == pytest.approx(want, rel=1e-9)


def test_fill_numeric_backfills_from_value_raw():
    """When the model omits value_num, _fill_numeric must use the fixed parser."""
    p = Property(section="Thermal", property_name="Glass transition", value_raw="−60")
    _fill_numeric(p)
    assert p.value_num == -60.0
    assert p.value_min is None and p.value_max is None


def test_fill_numeric_keeps_model_numbers():
    p = Property(section="Mechanical", property_name="Tensile strength",
                 value_raw="1,200 ± 100", value_num=1200.0, qualifier="±")
    _fill_numeric(p)
    assert (p.value_num, p.value_min, p.value_max) == (1200.0, None, None)
