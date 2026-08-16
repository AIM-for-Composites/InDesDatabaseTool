"""Regressions from the 2026-08-15 adversarial review of the bugfix batch.

Every case here reproduces a finding the review verified against the code
(18/18 confirmed). Grouped by subsystem; see the commit message for the map.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import batch_ingest as bi
import extraction as E
import pdf_crawler as C
from extraction import (
    Property, _fill_numeric, _grounded, _match_family, _normalize_text as N,
    _preprocess_unit, canonicalize, plausibility_problem, verify_against_text,
)
from migrate import backfill_material_key_grade, ensure_sources_sha1_unique, migrate


def _p(name, raw, unit):
    p = Property(section="", property_name=name, value_raw=raw, unit=unit)
    _fill_numeric(p)
    return p


def _fam(name):
    f = _match_family(_p(name, "1", ""))
    return f.name if f else None


# ---------------------------------------------------------------------------
# [1] elongation / CTE bare-unit magnitude heuristic; '%' in value_raw wins
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, raw, unit, si", [
    ("Elongation at break", "2.4 %", "", 0.024),     # was 2.4 (240 %) after batch B
    ("Elongation at break", "2.4", "-", 0.024),
    ("Elongation at break", "2.4", "", 0.024),
    ("Strain to failure", "1.5", "", 0.015),
    ("Elongation at break", "0.8", "", 0.008),
    ("Elongation at break", "0.024", "", 0.024),     # strain fraction
    ("Elongation at break", "0.024", "mm/mm", 0.024),
    ("Elongation at break", "2.4", "%", 0.024),
    ("CTE", "2.3E-05", "", 2.3e-5),                  # 1/K printed bare
    ("CTE", "23", "", 2.3e-5),                       # ppm printed bare
])
def test_bare_unit_magnitude(name, raw, unit, si):
    _, got, prob = canonicalize(_p(name, raw, unit))
    assert prob is None
    assert got == pytest.approx(si, rel=1e-6)


# ---------------------------------------------------------------------------
# [2] superscripts before NFKC; power rule must skip scientific-notation 'e'
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("unit, expected", [
    ("10³ psi", "10**3 psi"), ("10⁶ psi", "10**6 psi"),
    ("1e3 psi", "1e3 psi"), ("10E3 psi", "10E3 psi"),
    ("kJ/m²", "kJ/m**2"), ("g/cm³", "g/cm**3"), ("N/mm2", "N/mm**2"),
    ("W/m·K", "W/m*K"), ("cm3/10min", "cm**3/10min"), ("g/10min", "g/10min"),
    ("MJ/m3", "MJ/m**3"), ("V/mil", "V/mil"), ("lb/ft3", "lb/ft**3"),
])
def test_preprocess_unit_powers(unit, expected):
    assert _preprocess_unit(unit) == expected


def test_ksi_msi_superscript_spellings_convert():
    _, si, prob = canonicalize(_p("Tensile strength", "65", "10³ psi"))
    assert prob is None and si == pytest.approx(448.16e6, rel=1e-3)      # was 46 MPa
    _, si, prob = canonicalize(_p("Tensile modulus", "20", "10⁶ psi"))
    assert prob is None and si == pytest.approx(137.9e9, rel=1e-3)      # was 0.0146 GPa
    _, si, prob = canonicalize(_p("Tensile strength", "65", "1e3 psi"))
    assert prob is None and si == pytest.approx(448.16e6, rel=1e-3)      # was dim_mismatch


# ---------------------------------------------------------------------------
# [3] CAI / short-beam / bearing / open-hole strengths are MPa families again
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, family", [
    ("Compression after impact strength", "cai_strength"),
    ("CAI strength", "cai_strength"),
    ("CAI", "cai_strength"),
    ("Short beam strength (SBS)", "other_strength"),
    ("Short-beam strength", "other_strength"),
    ("Bearing strength", "other_strength"),
    ("Open hole tension strength", "other_strength"),
    ("Filled-hole compression strength", "compressive_strength"),
    ("Notched tensile strength", "tensile_strength"),
    # shields still win
    ("Impact strength", "impact_strength"),
    ("Peel strength", "peel_strength"),
    ("Weld strength", "peel_strength"),
    ("Dielectric strength", "dielectric_strength"),
])
def test_strength_families(name, family):
    assert _fam(name) == family
    if family not in ("impact_strength", "peel_strength", "dielectric_strength"):
        _, si, prob = canonicalize(_p(name, "250", "MPa"))
        assert prob is None and si == pytest.approx(250e6)


# ---------------------------------------------------------------------------
# [4] trailing digit is not a token boundary breaker
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, family", [
    ("Tg2", "glass_transition"), ("Tg1", "glass_transition"),
    ("CTE1", "cte"), ("CTE2", "cte"), ("HDT1.8", "hdt"), ("Tm2", "melting"),
    ("Tg (DMA)", "glass_transition"),
    ("ASTM D638", None), ("outgassing", None), ("Dielectric constant (ASTM D150)", None),
])
def test_short_token_trailing_digit(name, family):
    assert _fam(name) == family


# ---------------------------------------------------------------------------
# [5] unit-key folding: dashes, º, X⁻¹ suffixes, SG spellings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("unit, si", [
    ("10⁻⁶ K⁻¹", 2.3e-5), ("10^-6 K^-1", 2.3e-5), ("10⁻⁶ °C⁻¹", 2.3e-5),
    ("µm·m⁻¹·K⁻¹", 2.3e-5), ("ppm/ºC", 2.3e-5), ("µm/m/ºC", 2.3e-5),
    ("10–6/K", 2.3e-5), ("·10⁻⁶/K", 2.3e-5), ("µm/(m·K)", 2.3e-5),
    ("10^-6 in/in/°F", 4.14e-5),
])
def test_cte_unit_spellings(unit, si):
    _, got, prob = canonicalize(_p("CTE", "23", unit))
    assert prob is None and got == pytest.approx(si, rel=1e-6)


@pytest.mark.parametrize("unit, si", [
    ("g/cm^3", 1.2), ("g/cm**3", 1.2), ("kg/m3", 0.0012), ("kg/dm3", 1.2),
    ("—", 1.2), ("–", 1.2), ("kg/l", 1.2),
])
def test_specific_gravity_spellings(unit, si):
    _, got, prob = canonicalize(_p("Specific gravity", "1.2", unit))
    assert prob is None and got == pytest.approx(si)


# ---------------------------------------------------------------------------
# [6]-[9] grounding: no haystack tightening; digitless / comma / period / sup
# ---------------------------------------------------------------------------

def test_placeholder_dash_cell_and_spaced_minus_ground():
    h = [N("Flexural Modulus ksi GPa – 2986 20.6 3065 21.1 – 2250 15.5 2000 13.8")]
    assert _grounded("2250", h) and _grounded("21.1", h)                    # was False
    h2 = [N("Panel A 2817 2828 2818 -0.46 -0.07 -0.42")]
    assert _grounded("-0.46", h2)                                            # was False
    assert not _grounded("0.46", h2)                                         # sign matters


def test_range_needles_ground_either_spacing():
    assert _grounded("70-75", [N("Tg 70 – 75 °C")])
    assert _grounded("70-75", [N("70–75")])
    assert _grounded("70 - 75", [N("70–75")])
    assert not _grounded("3", [N("per ISO 527-3 at 23 C")])                # still rejected


def test_digitless_needles_never_ground():
    for v in ("–", "—", "−", "-", ".", "e", "/", "±"):
        assert not _grounded(v, [N("fiber-reinforced. e / ± . -")])


def test_typographic_dash_placeholders_are_empty_value():
    for v in ("–", "—", "−", "-", "…"):
        p = Property(section="m", property_name="Tensile strength", value_raw=v, unit="MPa", page=1)
        verify_against_text(E.Extraction(materials=[E.Material(material_name="X", properties=[p])]),
                            ["fiber-reinforced laminate. In-plane shear 4.5 GPa."])
        assert p.status == "empty_value", v


def test_comma_is_a_number_character():
    assert not _grounded("200", [N("1,200 mpa")])
    assert not _grounded("1", [N("1,200 mpa")])
    assert not _grounded("32", [N("density 1,32 g/cm3")])
    assert _grounded("1,200", [N("1,200 mpa")])
    assert _grounded("3", [N("values 3, 4 and 5")])          # list comma is a boundary


def test_sentence_end_and_superscript_footnotes():
    assert _grounded("0.3", [N("poisson's ratio was 0.3.")])
    assert _grounded("776", [N("tensile strength 776¹ mpa")])
    assert _grounded("45", [N("0⁰/45⁰/90⁰")])
    assert not _grounded("1.2", [N("11.25")])


# ---------------------------------------------------------------------------
# [10] cited page out of range records the soft reason
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("page, reason", [(2, ""), (1, "grounded_off_page"),
                                          (99, "grounded_off_page"), (None, "")])
def test_out_of_range_page_soft_reason(page, reason):
    p = Property(section="M", property_name="Tensile Modulus", value_raw="3.5", unit="GPa",
                 page=page, value_num=3.5)
    verify_against_text(E.Extraction(materials=[E.Material(material_name="PEEK", properties=[p])]),
                        ["Intro page.", "Tensile modulus 3.5 GPa per ISO 527."])
    assert p.status == "ok" and p.flag_reason == reason


# ---------------------------------------------------------------------------
# [12] atomic sources rebuild; [13] material_key backfill
# ---------------------------------------------------------------------------

def _legacy_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    for t in ("Polymers", "Fibers", "Composites_materials"):
        conn.execute(f"CREATE TABLE {t} (id INTEGER PRIMARY KEY AUTOINCREMENT, material_name TEXT,"
                     " material_abbreviation TEXT, section TEXT, property_name TEXT, value TEXT,"
                     " unit TEXT, english TEXT, test_condition TEXT, comments TEXT)")
    conn.execute("CREATE TABLE sources (id INTEGER PRIMARY KEY AUTOINCREMENT, pdf_filename TEXT UNIQUE,"
                 " pdf_sha1 TEXT, ingested_at TEXT, material_class TEXT, material_abbreviation TEXT)")
    conn.execute("INSERT INTO sources (pdf_filename, pdf_sha1) VALUES ('a.pdf','sha_A')")
    conn.commit(); conn.close()


def test_sources_rebuild_is_atomic_and_self_healing(tmp_path):
    p = tmp_path / "t.sqlite"; _legacy_db(p)
    conn = sqlite3.connect(str(p))
    # simulate a crash that left a stale half-built sources__new behind
    conn.execute("CREATE TABLE sources__new (id INTEGER PRIMARY KEY, junk TEXT)")
    conn.commit()
    assert ensure_sources_sha1_unique(conn) is True          # used to raise "already exists"
    assert conn.execute("SELECT pdf_filename, pdf_sha1 FROM sources").fetchall() == [("a.pdf", "sha_A")]
    assert conn.execute("SELECT name FROM sqlite_master WHERE name='sources__new'").fetchone() is None
    assert not conn.in_transaction
    conn.close()


def test_material_key_backfill(tmp_path):
    p = tmp_path / "t.sqlite"; _legacy_db(p)
    migrate(p, backup=False)                                  # adds columns
    conn = sqlite3.connect(str(p))
    # July-era rows: key without grade, trade_grade populated
    for grade in ("150G", "450G"):
        conn.execute("INSERT INTO Polymers (material_name, material_key, trade_grade, source_sha1, "
                     "section, property_name, value_raw) VALUES ('PEEK','peek',?, 'sha_A','Physical','Density','1.30')", (grade,))
    # legacy (no source_sha1) and already-keyed rows must be untouched
    conn.execute("INSERT INTO Polymers (material_name, material_key, trade_grade) VALUES ('PPS','pps','X')")
    conn.execute("INSERT INTO Polymers (material_name, material_key, trade_grade, source_sha1) VALUES ('PA','pa|z','Z','s')")
    conn.commit()
    assert backfill_material_key_grade(conn, "Polymers") == 2
    assert backfill_material_key_grade(conn, "Polymers") == 0     # idempotent
    keys = sorted(r[0] for r in conn.execute("SELECT material_key FROM Polymers"))
    assert keys == ["pa|z", "peek|150g", "peek|450g", "pps"]
    # and it matches what material_key() produces, so re-ingest dedups
    assert E.material_key(E.Material(material_name="PEEK", trade_grade="150G")) == "peek|150g"
    conn.close()


# ---------------------------------------------------------------------------
# [14] per-run attempted set; [15] relevance gate inside search lanes
# ---------------------------------------------------------------------------

def test_duplicate_candidate_in_one_run_does_not_burn_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(C.THROTTLE, "wait", lambda url: None)
    calls = {"n": 0}
    class _H:
        status_code = 503; headers = {}
    def fake_head(url, timeout=10, allow_redirects=True):
        calls["n"] += 1; return _H()
    monkeypatch.setattr(C.SESSION, "head", fake_head)
    st = C.CrawlerState(tmp_path / "state.json")
    cand = C.Candidate(title="t", pdf_url="http://x/p.pdf", source="openalex")
    for _ in range(5):
        C.download_pdf(cand, tmp_path, st)
    assert calls["n"] == 1                       # HEAD once per run
    assert st.failed == {"http://x/p.pdf": 1}    # was: 3 -> blacklisted in-run
    assert "http://x/p.pdf" not in st.seen_urls


def test_arxiv_overfetch_yields_relevant_candidates(monkeypatch):
    def entry(i, relevant):
        title = ("Carbon fiber PEEK thermoplastic composite tensile modulus" if relevant
                 else "Galaxy rotation curves and dark matter halos")
        return (f'<entry><title>{title} {i}</title><summary>{title}</summary>'
                f'<published>2024-01-01</published>'
                f'<link title="pdf" href="http://arxiv.org/pdf/{i}.pdf"/></entry>')
    feed = ('<feed xmlns="http://www.w3.org/2005/Atom">'
            + "".join(entry(i, i >= 10) for i in range(30)) + "</feed>")
    class _R:
        text = feed
    monkeypatch.setattr(C, "http_get", lambda url, **kw: _R())
    got = list(C.search_arxiv("q", 10, C.MIN_SCORE))
    assert len(got) == 10
    assert all(C.relevance_score(c.title + " " + c.abstract) >= C.MIN_SCORE for c in got)  # was: 10 irrelevant


# ---------------------------------------------------------------------------
# [17] .gitignore patterns actually ignore
# ---------------------------------------------------------------------------

def test_gitignore_patterns_are_effective():
    import subprocess
    root = Path(__file__).resolve().parent.parent
    paths = ["eval/last_run/x.json", "pg_backup_x/a.csv", "run_report.json",
             "foo.sqlite", "foo.sqlite.bak", "review_queue.csv"]
    r = subprocess.run(["git", "check-ignore", "--no-index", *paths],
                       cwd=root, capture_output=True, text=True)
    ignored = set(r.stdout.split())
    assert ignored == set(paths), f"not ignored: {set(paths) - ignored}"


# ---------------------------------------------------------------------------
# [18] data_loader degrades only on a genuinely missing status column
# ---------------------------------------------------------------------------

def test_data_loader_fallback_only_on_undefined_column(tmp_path, monkeypatch):
    import importlib, sys, types
    # pandas is a Space dependency, not a pipeline one; a minimal stand-in is
    # enough to exercise the SQL/exception logic when it isn't installed.
    if "pandas" not in sys.modules:
        try:
            import pandas  # noqa: F401
        except ImportError:
            fake_pd = types.ModuleType("pandas")
            class _DF:
                def __init__(self, rows=None, columns=None):
                    self.rows = list(rows or []); self.columns = list(columns or [])
                @property
                def empty(self):
                    return not self.rows
                def __getitem__(self, col):
                    i = self.columns.index(col)
                    class _S(list):
                        def __eq__(s, other):
                            return _S(x == other for x in s)
                        def any(s):
                            return any(s)
                    return _S(r[i] for r in self.rows)
            fake_pd.DataFrame = _DF
            monkeypatch.setitem(sys.modules, "pandas", fake_pd)
    calls: list[str] = []
    state = {"mode": "transient"}
    db = types.ModuleType("db")
    def fetch_all(q):
        calls.append(q)
        if state["mode"] == "transient" and len(calls) == 1:
            raise ConnectionResetError("reset by peer")
        if state["mode"] == "nocol" and "status" in q:
            raise Exception('column "status" does not exist')
        return [("m", "a", "s", "p", "v", "u", "e", "t", "c", None)] + (
            [("QUAR", "a", "s", "p", "v", "u", "e", "t", "c", None)] if "status" not in q else [])
    db.fetch_all = fetch_all; db.execute_query = lambda *a, **k: 0
    monkeypatch.setitem(sys.modules, "db", db)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hf_space_additions"))
    try:
        dl = importlib.import_module("data_loader"); importlib.reload(dl)
        # transient error -> empty frame, NO unfiltered second query
        df = dl.load_material_data("Polymers")
        assert df.empty and len(calls) == 1
        # genuinely missing column -> unfiltered fallback (with the QUAR row)
        calls.clear(); state["mode"] = "nocol"
        df = dl.load_material_data("Polymers")
        assert len(calls) == 2 and "status" not in calls[1]
        assert (df["material_name"] == "QUAR").any()
    finally:
        sys.path.pop(0)
        sys.modules.pop("data_loader", None)
