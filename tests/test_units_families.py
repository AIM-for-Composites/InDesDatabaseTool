"""Regression tests for property-family matching, unit canonicalization and
plausibility (extraction._match_family / canonicalize / plausibility_problem).

Each case was a verified failure in the 2026-08 adversarial review, or a
pre-existing behavior we must not regress.
"""

from __future__ import annotations

import pytest

from extraction import (
    Property,
    _fill_numeric,
    _match_family,
    _preprocess_unit,
    canonicalize,
    plausibility_problem,
)


def _prop(name: str, raw: str, unit: str) -> Property:
    p = Property(section="", property_name=name, value_raw=raw, unit=unit)
    _fill_numeric(p)
    return p


def _fam(name: str) -> str | None:
    f = _match_family(_prop(name, "1", ""))
    return f.name if f else None


# --------------------------------------------------------------------------
# family matching
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, family",
    [
        # short tokens must match on token boundaries (was: 'tm' in 'ASTM',
        # 'tg' in 'outgassing')
        ("Dielectric constant (ASTM D150)", None),
        ("Water absorption (ASTM D570)", None),
        ("Hardness (ASTM D2240)", None),
        ("Outgassing (TML)", None),
        ("Tg", "glass_transition"),
        ("Tg (DSC)", "glass_transition"),
        ("Glass transition temperature", "glass_transition"),
        ("Melting point (Tm)", "melting"),
        ("Melting temperature", "melting"),
        ("HDT", "hdt"),
        ("HDT @ 1.8 MPa", "hdt"),
        ("CTE", "cte"),
        ("CLTE (flow)", "cte"),
        # bare 'strength' no longer swallows non-pressure strengths
        ("Dielectric strength", "dielectric_strength"),
        ("Impact strength (Izod, notched)", "impact_strength"),
        ("Notched Izod", "impact_strength"),
        ("Charpy impact", "impact_strength"),
        ("Tear strength", "tear_strength"),
        ("Shear strength", "shear_strength"),
        ("Interlaminar shear strength", "shear_strength"),
        ("ILSS", "shear_strength"),
        ("Tensile strength", "tensile_strength"),
        ("Compressive strength", "compressive_strength"),
        ("Flexural strength", "flexural_strength"),
        # modulus family
        ("Tensile modulus", "tensile_modulus"),
        ("Young's modulus", "tensile_modulus"),
        ("Youngs modulus", "tensile_modulus"),
        ("Modulus of elasticity", "tensile_modulus"),
        ("Flexural modulus", "flexural_modulus"),
        ("Storage modulus", "storage_modulus"),
        # density vs specific gravity
        ("Specific gravity", "specific_gravity"),
        ("Density", "density"),
        # elongation
        ("Elongation at break", "elongation"),
        ("Strain at break", "elongation"),
        ("Strain to failure", "elongation"),
        # unknown property: no family
        ("Poisson ratio", None),
    ],
)
def test_family_matching(name, family):
    assert _fam(name) == family


# --------------------------------------------------------------------------
# canonicalize: value_si + unit problems
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, raw, unit, exp_uc, exp_si, exp_prob_prefix",
    [
        # N/mm2 — the standard European MPa spelling (was: unparseable)
        ("Tensile strength", "450", "N/mm2", "MPa", 450e6, None),
        ("Tensile strength", "450", "N/mm^2", "MPa", 450e6, None),
        ("Tensile modulus", "3500", "N/mm2", "GPa", 3.5e9, None),
        ("Flexural modulus", "3.5", "kN/mm2", "GPa", 3.5e9, None),
        ("Tensile modulus", "3.5", "GPa", "GPa", 3.5e9, None),
        # ksi / Msi custom units
        ("Tensile strength", "250", "ksi", "MPa", 1.7237e9, None),
        ("Tensile modulus", "30", "Msi", "GPa", 2.0684e11, None),
        # temperature paths
        ("Glass transition temperature", "143", "°C", "°C", 416.15, None),
        ("Melting temperature", "343", "C", "°C", 616.15, None),
        ("Tg", "350", "K", "°C", 350.0, None),
        ("Tg", "650", "F", "°C", 616.483, None),
        ("Tg", "143", "%", "°C", None, "unit_review:bad_temp_unit"),
        # CTE: printed unit is now honored (was: si_factor applied blindly)
        ("CTE", "2.3e-5", "1/K", "ppm/°C", 2.3e-5, None),
        ("CTE", "2.3E-05", "/°C", "ppm/°C", 2.3e-5, None),
        ("Coefficient of thermal expansion", "23", "ppm/°C", "ppm/°C", 2.3e-5, None),
        ("CTE", "23", "µm/m/°C", "ppm/°C", 2.3e-5, None),
        ("CTE", "23", "µm/(m·K)", "ppm/°C", 2.3e-5, None),
        ("CTE", "23", "10^-6/K", "ppm/°C", 2.3e-5, None),
        ("CTE", "23", "x10-6/°C", "ppm/°C", 2.3e-5, None),
        ("CTE", "23", "10-6 /K", "ppm/°C", 2.3e-5, None),
        ("CTE", "23", "", "ppm/°C", 2.3e-5, None),          # bare = ppm/°C implied
        ("CTE", "13", "ppm/°F", "ppm/°C", 2.34e-5, None),   # F->C basis: x1.8
        ("CTE", "23", "MPa", "ppm/°C", None, "unit_review:unexpected_unit"),
        # elongation: bare number / ratio is a strain FRACTION (was: 100x off)
        ("Elongation at break", "2.4", "%", "%", 0.024, None),
        ("Elongation at break", "0.024", "", "%", 0.024, None),
        ("Elongation at break", "0.024", "mm/mm", "%", 0.024, None),
        ("Elongation at break", "2.4", "mm", "%", None, "unit_review:unexpected_unit"),
        # specific gravity is dimensionless (was: unit_review:missing_unit)
        ("Specific gravity", "1.38", "", "", 1.38, None),
        ("Specific gravity", "1.38", "g/cm3", "", 1.38, None),
        # density still needs a unit
        ("Density", "1.38", "g/cm3", "g/cm³", 1380.0, None),
        ("Density", "1380", "kg/m3", "g/cm³", 1380.0, None),
        ("Density", "1.38", "", "g/cm³", None, "unit_review:missing_unit"),
        # passthrough families: unit passed through, no SI, no problem
        ("Dielectric strength", "20", "kV/mm", "kV/mm", None, None),
        ("Impact strength (Izod, notched)", "50", "kJ/m2", "kJ/m2", None, None),
        ("Tear strength", "50", "N/mm", "N/mm", None, None),
        # unknown family: passthrough
        ("Poisson ratio", "0.3", "", "", None, None),
        # a property that used to false-hit 'tm' now passes through cleanly
        ("Dielectric constant (ASTM D150)", "3.2", "", "", None, None),
    ],
)
def test_canonicalize(name, raw, unit, exp_uc, exp_si, exp_prob_prefix):
    uc, si, prob = canonicalize(_prop(name, raw, unit))
    assert uc == exp_uc
    if exp_si is None:
        assert si is None
    else:
        assert si == pytest.approx(exp_si, rel=1e-3)
    if exp_prob_prefix is None:
        assert prob is None
    else:
        assert prob is not None and prob.startswith(exp_prob_prefix)


# --------------------------------------------------------------------------
# plausibility after conversion
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, raw, unit, flagged",
    [
        ("Tensile modulus", "1200", "MPa", False),   # 1.2 GPa, in range
        ("Tensile modulus", "3.5", "GPa", False),
        ("Tensile modulus", "5000", "GPa", True),
        ("Glass transition temperature", "143", "°C", False),
        ("Glass transition temperature", "900", "°C", True),
        ("Density", "1.38", "g/cm3", False),
        ("Density", "50", "g/cm3", True),
        ("Specific gravity", "1.38", "", False),
        ("Elongation at break", "0.024", "", False),        # 2.4 %, in range
        ("Elongation at break", "2.4", "%", False),
        ("CTE", "2.3e-5", "1/K", False),                    # 23 ppm/°C, in range
        ("CTE", "23", "ppm/°C", False),
        # ASTM false-hit used to yield out_of_range on a dielectric constant
        ("Dielectric constant (ASTM D150)", "3.2", "", False),
        # passthrough families never range-check
        ("Impact strength", "50", "kJ/m2", False),
        ("Dielectric strength", "20", "kV/mm", False),
    ],
)
def test_plausibility(name, raw, unit, flagged):
    assert (plausibility_problem(_prop(name, raw, unit)) is not None) == flagged


# --------------------------------------------------------------------------
# unit preprocessing
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "unit, expected",
    [
        ("N/mm2", "N/mm**2"),
        ("kN/mm2", "kN/mm**2"),
        ("g/cm3", "g/cm**3"),
        ("kg/m3", "kg/m**3"),
        ("lb/in3", "lb/in**3"),
        ("g/cc", "g/cm**3"),
        ("N/mm^2", "N/mm**2"),
        ("g/cm³", "g/cm**3"),
        ("kJ/m²", "kJ/m**2"),
        ("MPa", "MPa"),
        ("m2", "m**2"),
        ("10^-6/K", "10**-6/K"),  # not a power-suffix; untouched apart from ^
    ],
)
def test_preprocess_unit(unit, expected):
    assert _preprocess_unit(unit) == expected
