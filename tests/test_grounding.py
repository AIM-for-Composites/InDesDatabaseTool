"""Regression tests for text grounding (extraction._grounded / verify_against_text)."""

from __future__ import annotations

from extraction import (
    Extraction,
    Material,
    Property,
    _grounded,
    _normalize_text,
    verify_against_text,
)


def _pages(*texts: str) -> list[str]:
    return [_normalize_text(t) for t in texts]


# --------------------------------------------------------------------------
# _grounded: digit boundaries for numeric needles
# --------------------------------------------------------------------------

def test_short_number_does_not_match_inside_standard_number():
    # was: '3' matched the '3' in 'ISO 527-3' -> hallucinated modulus 'verified'
    assert not _grounded("3", _pages("Tensile test per ISO 527-3 at 23 °C"))


def test_number_does_not_match_inside_longer_number():
    assert not _grounded("1.2", _pages("Modulus 11.25 GPa"))
    assert not _grounded("25", _pages("Value 125 MPa and 250 MPa"))


def test_number_matches_on_boundaries():
    assert _grounded("3", _pages("Modulus: 3 GPa"))
    assert _grounded("3.5", _pages("modulus of 3.5 GPa"))
    assert _grounded("776", _pages("Tensile strength 776 MPa"))
    assert _grounded("1200", _pages("(1200)"))
    assert _grounded("23", _pages("tested at 23 °C"))


def test_qualifier_is_stripped_but_sign_kept():
    assert _grounded("<= -18.0", _pages("Tg: -18.0 °C"))
    assert _grounded("~3.5", _pages("approx. 3.5 GPa"))
    # the sign is part of the number: a negative needle must not ground on the
    # positive value
    assert not _grounded("-18.0", _pages("Tg: 18.0 °C"))
    assert _grounded("-18.0", _pages("Tg: −18.0 °C"))   # U+2212 in the PDF


def test_range_and_dash_variants_ground():
    # PDF prints an en dash; needle uses ASCII hyphen after normalization
    assert _grounded("70-75", _pages("Tg 70–75 °C"))
    assert _grounded("70–75", _pages("Tg 70-75 °C"))
    # spacing around the dash must not matter either way
    assert _grounded("70-75", _pages("Tg 70 – 75 °C"))
    assert _grounded("70 - 75", _pages("Tg 70–75 °C"))


def test_text_needle_uses_substring():
    assert _grounded("tensile strength (0°)", _pages("Table 2. Tensile Strength (0°) 776 MPa"))
    assert not _grounded("", _pages("anything"))


# --------------------------------------------------------------------------
# verify_against_text: status + soft reasons
# --------------------------------------------------------------------------

def _ext(prop: Property) -> Extraction:
    return Extraction(materials=[Material(material_name="PEEK", material_class="Polymer",
                                          properties=[prop])])


def test_grounded_on_cited_page_is_clean_ok():
    p = Property(section="Mechanical", property_name="Tensile strength", value_raw="100",
                 unit="MPa", page=2, value_num=100.0)
    verify_against_text(_ext(p), ["intro", "tensile strength 100 MPa"])
    assert p.status == "ok"
    assert p.flag_reason == ""


def test_off_page_fallback_stays_ok_but_records_soft_reason():
    p = Property(section="Mechanical", property_name="Tensile strength", value_raw="100",
                 unit="MPa", page=1, value_num=100.0)
    verify_against_text(_ext(p), ["intro", "tensile strength 100 MPa"])
    assert p.status == "ok"
    assert "grounded_off_page" in p.flag_reason


def test_hallucinated_short_value_is_unverified():
    # was: status ok because '3' occurs inside 'ISO 527-3'
    p = Property(section="Mechanical", property_name="Tensile modulus", value_raw="3",
                 unit="GPa", page=2, value_num=3.0)
    verify_against_text(_ext(p), ["ISO 527-3 procedure", "no numbers here"])
    assert p.status == "unverified"
    assert "value_not_in_pdf_text" in p.flag_reason


def test_quote_fallback_records_soft_reason():
    p = Property(section="Mechanical", property_name="Tensile strength", value_raw="1.2e3",
                 unit="MPa", page=1, value_num=1200.0,
                 source_quote="tensile strength of 1200 MPa was measured")
    verify_against_text(_ext(p), ["The tensile strength of 1200 MPa was measured."])
    assert p.status == "ok"
    assert "grounded_via_quote" in p.flag_reason


def test_precedence_unit_review_after_grounding():
    p = Property(section="Thermal", property_name="Tg", value_raw="143", unit="%",
                 page=1, value_num=143.0)
    verify_against_text(_ext(p), ["Tg 143 %"])
    assert p.status == "unit_review"


def test_no_page_texts_skips_grounding():
    p = Property(section="Mechanical", property_name="Tensile strength", value_raw="100",
                 unit="MPa", page=1, value_num=100.0)
    verify_against_text(_ext(p), [])
    assert p.status == "ok"
