"""Figure-mining phase tests (figures.py + batch_ingest --figures wiring).

Everything runs OFFLINE: the Gemini transport is a scripted fake that answers
the text extraction, the batched classify call and the per-figure mining call
with canned JSON. Harvesting runs on the real corpus PDFs when they are
present (crawl_out/pdfs is gitignored) and on a synthetic PDF otherwise, so
the suite passes on a fresh clone too.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

fitz = pytest.importorskip("fitz")

import batch_ingest as bi
import extraction as E
import figures as F

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _corpus(pattern: str) -> Path | None:
    hits = sorted(glob.glob(str(ROOT / "crawl_out" / "pdfs" / pattern)))
    return Path(hits[0]) if hits else None


def _synthetic_pdf(tmp_path: Path) -> Path:
    """A 3-page PDF: p1 text-only + tiny logo, p2 a big raster 'plot' with a
    caption below, p3 a vector-drawn chart with a margin caption + a repeated
    logo, so every route/filter has something to bite on."""
    doc = fitz.open()
    # a fake raster "plot": 600x400 RGB pixmap with a diagonal line
    pm = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 600, 400), False)
    pm.clear_with(255)
    for i in range(400):
        pm.set_pixel(min(599, int(i * 1.4)), 399 - i, (200, 40, 40))
    plot_png = pm.tobytes("png")
    logo = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40), False)
    logo.clear_with(20)
    logo_png = logo.tobytes("png")

    p1 = doc.new_page()
    p1.insert_text((72, 72), "Average tensile strength (608 MPa) of the PPS composite. "
                              "The PEKK composite (824 MPa) is stronger. " * 10, fontsize=9)
    p1.insert_image(fitz.Rect(500, 20, 540, 60), stream=logo_png)

    p2 = doc.new_page()
    p2.insert_image(fitz.Rect(72, 80, 472, 346), stream=plot_png)
    p2.insert_text((72, 370), "Figure 1. Stress-strain curve of PEEK laminate", fontsize=10)
    p2.insert_image(fitz.Rect(500, 20, 540, 60), stream=logo_png)

    p3 = doc.new_page()
    sh = p3.new_shape()
    for i in range(0, 300, 6):            # a bar chart: many drawing objects
        sh.draw_rect(fitz.Rect(150 + i, 400 - i / 2, 154 + i, 400))
        sh.finish(color=(0, 0, 1), fill=(0.3, 0.3, 0.9))
    sh.draw_line((150, 400), (460, 400)); sh.finish(color=(0, 0, 0))
    sh.draw_line((150, 400), (150, 200)); sh.finish(color=(0, 0, 0))
    sh.commit()
    p3.insert_text((40, 260), "Fig. 2 Modulus\nvs temperature", fontsize=9)   # margin caption
    p3.insert_image(fitz.Rect(500, 20, 540, 60), stream=logo_png)

    out = tmp_path / "synthetic.pdf"
    doc.save(str(out)); doc.close()
    return out


class FakeGemini:
    """Scripted stand-in for extraction._request_with_retry.

    Decides what to answer from the request payload: the text-extraction
    schema, the figure classify schema, or the mining schema. Records calls
    so tests can assert the cost model (1 classify per PDF, 1 mine per
    mineable figure). `fail` makes every vision call return a 500.
    """

    def __init__(self, text_json: dict, kinds: list[str] | None = None,
                 mining_json: dict | None = None, fail_vision: bool = False,
                 fail_vision_kind: str = "http"):
        self.text_json = text_json
        self.kinds = kinds
        self.mining_json = mining_json or {"figure_kind": "property_plot", "series": []}
        self.fail_vision = fail_vision
        self.fail_vision_kind = fail_vision_kind
        self.calls: list[str] = []

    def __call__(self, method, url, *, what="Gemini", timeout=None, _sleep=None, **kw):
        import requests
        payload = kw.get("json") or {}
        schema = payload.get("generationConfig", {}).get("responseSchema", {})
        props = schema.get("properties", {})
        if "materials" in props:
            kind = "text"; body = self.text_json
        elif "figures" in props:
            kind = "classify"
            n = sum(1 for p in payload["contents"][0]["parts"] if "inlineData" in p)
            ks = self.kinds or ["property_plot"] * n
            body = {"figures": [{"index": i, "figure_kind": ks[i % len(ks)],
                                 "material_guess": "PPS composite" if i % 2 == 0 else "PEKK composite"}
                                for i in range(n)]}
        else:
            kind = "mine"; body = self.mining_json
        self.calls.append(kind)
        if kind != "text" and self.fail_vision:
            if self.fail_vision_kind == "conn":
                raise requests.ConnectionError("network down")
            resp = _Resp(500, text="boom")
            resp.raise_for_status()
        return _Resp(200, {"candidates": [{"content": {"parts": [{"text": json.dumps(body)}]}}]})


class _Resp:
    def __init__(self, status, body=None, text=""):
        self.status_code = status; self._body = body or {}; self.text = text; self.headers = {}
    def json(self): return self._body
    def raise_for_status(self):
        import requests
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code), response=self)


TEXT_JSON = {"materials": [
    {"material_name": "PPS composite", "material_abbreviation": "PPS", "material_class": "Composite",
     "trade_grade": "", "manufacturer": "", "matrix": "PPS", "fiber": "carbon", "fiber_volume_fraction": "",
     "properties": [{"section": "Mechanical", "property_name": "Tensile strength", "value_raw": "608",
                     "unit": "MPa", "source_quote": "Average tensile strength (608 MPa)", "page": 7}]},
    {"material_name": "PEKK composite", "material_abbreviation": "PEKK", "material_class": "Composite",
     "trade_grade": "", "manufacturer": "", "matrix": "PEKK", "fiber": "carbon", "fiber_volume_fraction": "",
     "properties": [{"section": "Mechanical", "property_name": "Tensile strength", "value_raw": "824",
                     "unit": "MPa", "source_quote": "PEKK composite (824 MPa", "page": 7}]},
]}

MINING_JSON = {
    "figure_kind": "property_plot", "x_label": "Strain", "x_unit": "", "y_label": "Stress", "y_unit": "MPa",
    "series": [
        {"series_name": "PPS", "material_guess": "PPS composite", "values": [
            {"section": "Mechanical", "property_name": "Tensile strength", "value_raw": "~610",
             "value_num": 610, "unit": "MPa", "test_condition": "", "comments": "peak of PPS curve"},
            {"section": "Mechanical", "property_name": "Strain at break", "value_raw": "~0.027",
             "value_num": 0.027, "unit": "", "comments": "endpoint of PPS curve"},
            {"section": "Mechanical", "property_name": "Tensile strength", "value_raw": "",
             "unit": "MPa", "comments": "empty on purpose"},
        ]},
        {"series_name": "PEKK", "material_guess": "PEKK composite", "values": [
            {"section": "Mechanical", "property_name": "Tensile strength", "value_raw": "~830",
             "value_num": 830, "unit": "MPa", "comments": "peak of PEKK curve"},
            {"section": "Mechanical", "property_name": "Tensile modulus", "value_raw": "~24000",
             "value_num": 24000, "unit": "GPa", "comments": "absurd on purpose -> out_of_range"},
            {"section": "Mechanical", "property_name": "Tensile modulus", "value_raw": "~24",
             "value_num": 24, "unit": "bananas", "comments": "bad unit -> unit_review"},
        ]},
    ],
}


@pytest.fixture
def fake_gemini(monkeypatch):
    def _make(**kw):
        fg = FakeGemini(TEXT_JSON, mining_json=MINING_JSON, **kw)
        monkeypatch.setattr(E, "_request_with_retry", fg)
        return fg
    return _make


# ---------------------------------------------------------------------------
# Task 1 — harvest (real corpus PDF when available, synthetic otherwise)
# ---------------------------------------------------------------------------

def test_harvest_synthetic_pdf_routes_captions_and_junk(tmp_path):
    pdf = _synthetic_pdf(tmp_path)
    b = pdf.read_bytes(); sha1 = hashlib.sha1(b).hexdigest()
    st = F.HarvestStats()
    figs = F.harvest_figures(b, pdf.name, sha1, tmp_path / "figs", 12, stats=st)
    routes = {f.route for f in figs}
    assert "raster" in routes and "vector" in routes
    assert st.skipped_small >= 3            # the 40x40 logo on every page
    by_page = {f.page: f for f in figs}
    assert by_page[2].caption.startswith("Figure 1.")
    assert by_page[3].caption.startswith("Fig. 2")           # margin caption (beside)
    for f in figs:
        p = Path(f.image_path)
        assert p.exists() and p.parent.name == sha1 and p.name.startswith(f"p{f.page}_")
        assert f.figure_id == f.image_sha256[:16] and len(f.figure_id) == 16
        assert f.width_px <= F.RENDER_MAX_SIDE_PX and f.height_px <= F.RENDER_MAX_SIDE_PX
    # re-run: 0 new files, same ids
    st2 = F.HarvestStats()
    figs2 = F.harvest_figures(b, pdf.name, sha1, tmp_path / "figs", 12, stats=st2)
    assert st2.files_written == 0 and st2.files_existing == len(figs)
    assert [f.figure_id for f in figs2] == [f.figure_id for f in figs]


def test_harvest_cap_prefers_captioned(tmp_path):
    pdf = _synthetic_pdf(tmp_path)
    b = pdf.read_bytes(); sha1 = hashlib.sha1(b).hexdigest()
    st = F.HarvestStats()
    figs = F.harvest_figures(b, pdf.name, sha1, tmp_path / "figs", 1, stats=st)
    assert len(figs) == 1 and figs[0].caption and st.capped >= 1


@pytest.mark.skipif(_corpus("openalex_thermoforming*pekk*") is None, reason="corpus not present")
def test_harvest_real_paper_pekk(tmp_path):
    pdf = _corpus("openalex_thermoforming*pekk*")
    b = pdf.read_bytes(); sha1 = hashlib.sha1(b).hexdigest()
    st = F.HarvestStats()
    figs = F.harvest_figures(b, pdf.name, sha1, tmp_path / "figs", 12, stats=st)
    assert len(figs) >= 6
    assert all(f.caption for f in figs)                          # every figure captioned
    fig6 = [f for f in figs if f.caption.startswith("Figure 6")]
    assert fig6 and fig6[0].page == 7 and fig6[0].route == "raster"
    assert fig6[0].label == "Figure 6"
    # junk proof: the 88x88 logo / 16x16 icon on p1 were filtered
    assert st.skipped_small >= 2
    assert not any(f.width_px < 200 for f in figs)


@pytest.mark.skipif(_corpus("datasheet_toray_cetex_tc1225*") is None, reason="corpus not present")
def test_harvest_text_table_datasheet_yields_nothing(tmp_path):
    pdf = _corpus("datasheet_toray_cetex_tc1225*")
    b = pdf.read_bytes(); sha1 = hashlib.sha1(b).hexdigest()
    st = F.HarvestStats()
    figs = F.harvest_figures(b, pdf.name, sha1, tmp_path / "figs", 12, stats=st)
    assert figs == []
    assert st.skipped_text_table >= 8 and st.skipped_repeated >= 0
    assert st.vector_candidates > 0


@pytest.mark.skipif(_corpus("openalex_effect_of_the_fiber_orientation*") is None, reason="corpus not present")
def test_harvest_margin_captions_pair(tmp_path):
    pdf = _corpus("openalex_effect_of_the_fiber_orientation*")
    b = pdf.read_bytes(); sha1 = hashlib.sha1(b).hexdigest()
    figs = F.harvest_figures(b, pdf.name, sha1, tmp_path / "figs", 20)
    p9 = sorted(f.caption[:6] for f in figs if f.page == 9)
    assert p9 == ["Fig. 5", "Fig. 6"]


# ---------------------------------------------------------------------------
# Tasks 2/3 — classify + mine (offline, scripted transport)
# ---------------------------------------------------------------------------

def _figs(tmp_path, n=2) -> list[F.Figure]:
    pdf = _synthetic_pdf(tmp_path)
    b = pdf.read_bytes()
    return F.harvest_figures(b, pdf.name, hashlib.sha1(b).hexdigest(), tmp_path / "figs", n)


def test_classify_one_call_per_pdf_and_zero_when_no_figures(tmp_path, fake_gemini):
    fg = fake_gemini(kinds=["property_plot", "photo"])
    figs = _figs(tmp_path)
    vs = F.VisionStats()
    F.classify_figures(figs, ["PPS composite"], "k", stats=vs)
    assert vs.classify_calls == 1 and fg.calls == ["classify"]
    assert figs[0].figure_kind == "property_plot" and figs[0].mining_status == "not_mined"
    assert figs[1].figure_kind == "photo" and figs[1].mining_status == "skipped_kind"
    assert figs[0].material_guess == "PPS composite"
    F.classify_figures([], ["x"], "k", stats=vs)
    assert vs.classify_calls == 1                       # 0 figures -> 0 calls


def test_classify_failure_is_counted_not_raised(tmp_path, fake_gemini):
    fake_gemini(fail_vision=True)
    figs = _figs(tmp_path)
    vs = F.VisionStats()
    F.classify_figures(figs, [], "k", stats=vs)
    assert vs.failed_calls == 1
    assert all(f.mining_status == "classify_failed" and f.figure_kind == "other" for f in figs)


def test_mine_figure_readout_and_status_precedence(tmp_path, fake_gemini):
    fake_gemini()
    fig = _figs(tmp_path, 1)[0]
    fig.figure_kind = "property_plot"
    vs = F.VisionStats()
    mf = F.mine_figure(fig, ["PPS composite", "PEKK composite"], "k", stats=vs)
    assert mf is not None and vs.mining_calls == 1 and fig.mining_status == "mined"
    assert mf.y_unit == "MPa" and fig.n_values == 6
    mats = [E.Material(material_name="PPS composite", material_abbreviation="PPS", material_class="Composite"),
            E.Material(material_name="PEKK composite", material_abbreviation="PEKK", material_class="Composite")]
    rows = F.figure_properties_to_rows([mf], mats, "x.pdf", "sha")
    assert len(rows) == 6
    by = {(r.material_name, r.property_name, r.value_raw): r for r in rows}
    ok = by[("PPS composite", "Tensile strength", "~610")]
    assert ok.origin == "figure" and ok.figure_id == fig.figure_id
    assert ok.status == "figure_estimate" and ok.flag_reason.startswith("read from Figure 1 (property_plot")
    assert ok.unit_canonical == "MPa" and ok.value_si == pytest.approx(610e6)
    assert ok.source_quote == fig.caption and ok.page == fig.page
    assert ok.prompt_version == F.FIGURE_PROMPT_VERSION
    assert ok.material_class == "Composite" and ok.material_key == "pps composite"
    # precedence: empty_value -> unit_review -> out_of_range -> figure_estimate
    assert by[("PPS composite", "Tensile strength", "")].status == "empty_value"
    assert by[("PEKK composite", "Tensile modulus", "~24")].status == "unit_review"
    assert by[("PEKK composite", "Tensile modulus", "~24000")].status == "out_of_range"
    strain = by[("PPS composite", "Strain at break", "~0.027")]
    assert strain.status == "figure_estimate" and strain.value_si == pytest.approx(0.027)  # bare fraction
    assert not any(r.status == "ok" for r in rows)       # figure rows are never ok


def test_unmatched_material_guess_becomes_its_own_material(tmp_path, fake_gemini):
    fake_gemini()
    fig = _figs(tmp_path, 1)[0]; fig.figure_kind = "property_plot"
    mf = F.mine_figure(fig, [], "k")
    rows = F.figure_properties_to_rows([mf], [], "x.pdf", "sha")   # no text materials
    names = {r.material_name for r in rows}
    assert names == {"PPS composite", "PEKK composite"}


# ---------------------------------------------------------------------------
# Tasks 4/5/6 — end to end through process_pdf into SQLite
# ---------------------------------------------------------------------------

def _run(tmp_path, fg_kwargs=None, pdf=None, mine=True, db_name="m.sqlite", max_figures=12):
    pdf = pdf or _synthetic_pdf(tmp_path)
    conn = bi.init_db(tmp_path / db_name)
    opts = bi.FigureOptions(out_dir=tmp_path / "figs", max_figures=max_figures, mine=mine)
    res = bi.process_pdf(pdf, conn, "k", figure_opts=opts)
    return conn, res


def test_end_to_end_figure_rows_persisted(tmp_path, fake_gemini):
    fg = fake_gemini()
    conn, res = _run(tmp_path)
    assert res.error is None
    assert res.figures_found >= 2 and res.figures_mined >= 1 and res.figure_rows >= 1
    assert res.vision_calls == 1 + res.figures_mined       # 1 classify + 1 per plot
    assert fg.calls.count("text") == 1 and fg.calls.count("classify") == 1
    # figures table
    figs = conn.execute("SELECT figure_id, page, bbox, caption, figure_kind, mining_status, "
                        "image_path, figure_prompt_version FROM figures").fetchall()
    assert len(figs) == res.figures_found
    assert all(json.loads(r[2]) and r[3] and r[7] == F.FIGURE_PROMPT_VERSION for r in figs)
    assert all(Path(r[6]).exists() for r in figs)
    # figure rows
    frows = conn.execute("SELECT status, origin, figure_id, source_quote, page, value_si, unit_canonical, "
                         "prompt_version FROM Composites_materials WHERE origin='figure'").fetchall()
    assert len(frows) == res.figure_rows
    assert all(r[1] == "figure" and r[2] and r[0] != "ok" and r[7] == F.FIGURE_PROMPT_VERSION for r in frows)
    fe = [r for r in frows if r[0] == "figure_estimate"]
    assert fe and any(r[5] and r[6] == "MPa" for r in fe)   # canonicalized strength survives
    assert all(r[3].startswith(("Figure", "Fig.")) for r in frows)   # source_quote = caption
    # text rows untouched and ok
    trows = conn.execute("SELECT status FROM Composites_materials WHERE IFNULL(origin,'text')='text'").fetchall()
    assert len(trows) == 2 and all(r[0] == "ok" for r in trows)


def test_rerun_zero_new_figures_zero_new_rows(tmp_path, fake_gemini):
    fake_gemini()
    pdf = _synthetic_pdf(tmp_path)
    conn, r1 = _run(tmp_path, pdf=pdf)
    n_fig = conn.execute("SELECT count(*) FROM figures").fetchone()[0]
    n_rows = conn.execute("SELECT count(*) FROM Composites_materials").fetchone()[0]
    files = sorted(p.name for p in (tmp_path / "figs").rglob("*.png"))
    opts = bi.FigureOptions(out_dir=tmp_path / "figs", max_figures=12, mine=True)
    r2 = bi.process_pdf(pdf, conn, "k", figure_opts=opts)
    assert r2.error == "skipped_seen_sha1" and r2.figures_found == 0 and r2.vision_calls == 0
    assert conn.execute("SELECT count(*) FROM figures").fetchone()[0] == n_fig
    assert conn.execute("SELECT count(*) FROM Composites_materials").fetchone()[0] == n_rows
    assert sorted(p.name for p in (tmp_path / "figs").rglob("*.png")) == files


def test_dedup_grain_includes_origin(tmp_path, fake_gemini):
    """A text row and a figure row reporting the same number both survive;
    re-inserting the same figure row is a duplicate."""
    fake_gemini()
    conn, res = _run(tmp_path)
    row = E.PropertyRow(material_name="PPS composite", material_abbreviation="PPS",
                        material_key="pps composite", material_class="Composite",
                        section="Mechanical", property_name="Tensile strength", value="608",
                        unit="MPa", english="", test_condition="", comments="", value_raw="608",
                        source_sha1=res_sha(tmp_path), status="figure_estimate", origin="figure",
                        figure_id="abc")
    assert not bi.already_inserted(conn, "Composites_materials", row)   # text 608 exists, figure 608 does not
    bi.insert_row(conn, "Composites_materials", row)
    assert bi.already_inserted(conn, "Composites_materials", row)


def res_sha(tmp_path) -> str:
    return hashlib.sha1((tmp_path / "synthetic.pdf").read_bytes()).hexdigest()


def test_no_figures_pdf_zero_vision_calls(tmp_path, fake_gemini):
    fg = fake_gemini()
    doc = fitz.open(); p = doc.new_page()
    p.insert_text((72, 72), "PPS composite tensile strength 608 MPa. " * 30, fontsize=9)
    pdf = tmp_path / "textonly.pdf"; doc.save(str(pdf)); doc.close()
    conn, res = _run(tmp_path, pdf=pdf)
    assert res.figures_found == 0 and res.vision_calls == 0 and res.figure_rows == 0
    assert fg.calls == ["text"]
    assert conn.execute("SELECT count(*) FROM figures").fetchone()[0] == 0
    assert res.inserted == 2 and res.error is None


@pytest.mark.parametrize("kind", ["http", "conn"])
def test_vision_failure_keeps_text_rows(tmp_path, fake_gemini, kind):
    fake_gemini(fail_vision=True, fail_vision_kind=kind)
    conn, res = _run(tmp_path)
    assert res.error is None                     # the PDF itself succeeded
    assert res.inserted >= 2 and res.figure_rows == 0
    assert res.figure_error and res.figure_error.startswith("vision_calls_failed")
    assert conn.execute("SELECT count(*) FROM Composites_materials WHERE IFNULL(origin,'text')='text'").fetchone()[0] == 2
    # figures were still harvested + recorded, marked classify_failed
    st = conn.execute("SELECT DISTINCT mining_status FROM figures").fetchall()
    assert st == [("classify_failed",)]


def test_no_figure_mining_mode(tmp_path, fake_gemini):
    fg = fake_gemini()
    conn, res = _run(tmp_path, mine=False)
    assert res.figures_found >= 2 and res.figures_mined == 0 and res.figure_rows == 0
    assert res.vision_calls == 1 and fg.calls.count("mine") == 0
    st = {r[0] for r in conn.execute("SELECT mining_status FROM figures")}
    assert st <= {"not_mined", "skipped_kind"}


def test_review_queue_and_promote_figure_row(tmp_path, fake_gemini):
    fake_gemini()
    conn, res = _run(tmp_path)
    review = tmp_path / "review.csv"
    n = bi.export_review_queue(conn, review)
    with review.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert n == len(rows) and "origin" in rows[0] and "figure_id" in rows[0]
    fig_rows = [r for r in rows if r["origin"] == "figure"]
    assert fig_rows and all(r["figure_id"] for r in fig_rows)
    assert all(r["status"] != "ok" for r in fig_rows)
    # promote exactly one figure_estimate row
    target = next(r for r in fig_rows if r["status"] == "figure_estimate")
    one = tmp_path / "one.csv"
    with one.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(target.keys())); w.writeheader(); w.writerow(target)
    assert bi.promote_review_queue(conn, one) == 1
    got = conn.execute("SELECT status, flag_reason, origin FROM Composites_materials WHERE figure_id=? "
                       "AND value_raw=? AND property_name=?",
                       (target["figure_id"], target["value_raw"], target["property_name"])).fetchone()
    assert got == ("ok", "promoted", "figure")
    # the text row with the same property/material was NOT touched (origin in the match)
    # and promoting is idempotent
    assert bi.promote_review_queue(conn, one) == 0


def test_promote_text_row_does_not_bless_figure_twin(tmp_path, fake_gemini):
    fake_gemini()
    conn, res = _run(tmp_path)
    # make the text row for PPS 608 flagged, and add a figure twin with the SAME grain
    conn.execute("UPDATE Composites_materials SET status='unverified' WHERE value_raw='608' AND origin='text'")
    twin = E.PropertyRow(material_name="PPS composite", material_abbreviation="PPS",
                         material_key="pps composite", material_class="Composite",
                         section="Mechanical", property_name="Tensile strength", value="608",
                         unit="MPa", english="", test_condition="", comments="", value_raw="608",
                         source_pdf="synthetic.pdf", source_sha1=res_sha(tmp_path),
                         status="figure_estimate", origin="figure", figure_id="twin")
    bi.insert_row(conn, "Composites_materials", twin); conn.commit()
    one = tmp_path / "one.csv"
    with one.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(bi._REVIEW_COLUMNS)
        w.writerow(["Composites_materials", "synthetic.pdf", 7, "unverified", "", "PPS composite",
                    "pps composite", "Composite", "Mechanical", "Tensile strength", "608", 608, "MPa",
                    "MPa", 608e6, "", "q", "", "text", ""])
    assert bi.promote_review_queue(conn, one) == 1
    st = dict(conn.execute("SELECT origin, status FROM Composites_materials WHERE value_raw='608'").fetchall())
    assert st == {"text": "ok", "figure": "figure_estimate"}


def test_backfill_figures_for_already_ingested_pdf(tmp_path, fake_gemini):
    """A PDF ingested BEFORE --figures existed: rerunning with --figures runs
    only the figure stage, using materials rebuilt from its rows."""
    fg = fake_gemini()
    pdf = _synthetic_pdf(tmp_path)
    conn = bi.init_db(tmp_path / "m.sqlite")
    r1 = bi.process_pdf(pdf, conn, "k")                    # text only
    assert r1.figures_found == 0 and fg.calls == ["text"]
    opts = bi.FigureOptions(out_dir=tmp_path / "figs", max_figures=12, mine=True)
    r2 = bi.process_pdf(pdf, conn, "k", figure_opts=opts)
    assert r2.error == "skipped_seen_sha1"
    assert r2.figures_found >= 2 and r2.figure_rows >= 1 and "text" not in fg.calls[1:]
    keys = {r[0] for r in conn.execute("SELECT material_key FROM Composites_materials WHERE origin='figure'")}
    assert keys <= {"pps composite", "pekk composite"}     # attached to the text-pass materials
    r3 = bi.process_pdf(pdf, conn, "k", figure_opts=opts)  # third run: nothing to do
    assert r3.figures_found == 0 and r3.vision_calls == 0


def test_migrate_adds_origin_figure_id_and_figures_table(tmp_path):
    from migrate import migrate
    p = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(str(p))
    for t in ("Polymers", "Fibers", "Composites_materials"):
        conn.execute(f"CREATE TABLE {t} (id INTEGER PRIMARY KEY AUTOINCREMENT, material_name TEXT,"
                     " material_abbreviation TEXT, section TEXT, property_name TEXT, value TEXT,"
                     " unit TEXT, english TEXT, test_condition TEXT, comments TEXT)")
    conn.execute("INSERT INTO Polymers (material_name, value) VALUES ('PEEK','3.5')")
    conn.commit(); conn.close()
    res = migrate(p, backup=True)
    assert (tmp_path / "legacy.sqlite.bak").exists()
    assert "origin" in res["Polymers"] and "figure_id" in res["Polymers"]
    assert res["figures"] == ["created"]
    conn = sqlite3.connect(str(p))
    assert conn.execute("SELECT origin FROM Polymers").fetchone() == ("text",)   # legacy rows read 'text'
    assert conn.execute("SELECT count(*) FROM figures").fetchone()[0] == 0
    conn.close()
    res2 = migrate(p, backup=False)
    assert res2["Polymers"] == [] and res2["figures"] == []               # idempotent


def test_summarize_reports_figures_and_vision_calls():
    r = bi.PdfResult(pdf="a", elapsed_s=1, materials=1, extracted=1, inserted=3, flagged=2,
                     duplicates=0, material_classes=["Composite"], figures_found=4, figures_mined=2,
                     figure_rows=2, vision_calls=3, figure_filters={"skipped_small": 5},
                     figure_error="vision_calls_failed:1")
    s = bi.summarize([r])
    assert s["figures"] == {"figures_found": 4, "figures_mined": 2, "figure_rows": 2,
                            "figure_rows_duplicate_skipped": 0, "vision_calls": 3,
                            "figure_errors_by_kind": {"vision_calls_failed": 1},
                            "harvest_filters": {"skipped_small": 5}}


# ---------------------------------------------------------------------------
# Task 6 — export guard invariant: a figure row can never be inserted as 'ok'
# ---------------------------------------------------------------------------

def test_figure_row_can_never_be_inserted_as_ok(tmp_path):
    conn = bi.init_db(tmp_path / "g.sqlite")
    row = E.PropertyRow(material_name="X", material_abbreviation="X", material_key="x",
                        material_class="Polymer", section="Mechanical", property_name="Tensile strength",
                        value="1", unit="MPa", english="", test_condition="", comments="", value_raw="1",
                        source_sha1="s", status="ok", origin="figure", figure_id="f")
    bi.insert_row(conn, "Polymers", row)
    st, fr = conn.execute("SELECT status, flag_reason FROM Polymers").fetchone()
    assert st == "figure_estimate" and "downgraded" in fr
    # a text row inserted as ok stays ok
    row2 = dataclasses_replace(row, origin="text", figure_id="", value_raw="2")
    bi.insert_row(conn, "Polymers", row2)
    assert conn.execute("SELECT status FROM Polymers WHERE value_raw='2'").fetchone() == ("ok",)


def dataclasses_replace(row, **kw):
    import dataclasses
    return dataclasses.replace(row, **kw)


def test_space_data_loader_gate_hides_figure_rows_until_promoted(tmp_path, fake_gemini):
    """The Space's data_loader publish gate is WHERE COALESCE(status,'ok')='ok';
    run that exact predicate against a mirror with figure rows."""
    fake_gemini()
    conn, res = _run(tmp_path)
    gate = "WHERE COALESCE(status, 'ok') = 'ok'"
    src = (ROOT / "hf_space_additions" / "data_loader.py").read_text(encoding="utf-8")
    assert gate in src
    vis = conn.execute(f"SELECT count(*) FROM Composites_materials {gate} AND origin='figure'").fetchone()[0]
    assert vis == 0 and res.figure_rows > 0
    # promote exactly one figure row (the way --promote would)
    conn.execute("UPDATE Composites_materials SET status='ok', flag_reason='promoted' WHERE id = "
                 "(SELECT id FROM Composites_materials WHERE origin='figure' AND status='figure_estimate' LIMIT 1)")
    vis = conn.execute(f"SELECT count(*) FROM Composites_materials {gate} AND origin='figure'").fetchone()[0]
    assert vis == 1
