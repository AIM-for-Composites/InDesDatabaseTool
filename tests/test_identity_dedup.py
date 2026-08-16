"""Regression tests for material identity, dedup, promotion and the sources
logbook — run against real (temporary) SQLite DBs through batch_ingest.
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest

import batch_ingest as bi
import extraction as E
from migrate import ensure_sources_sha1_unique, migrate


# --------------------------------------------------------------------------
# material_key: trade grade is part of the identity
# --------------------------------------------------------------------------

def test_material_key_includes_trade_grade():
    a = E.Material(material_name="PEEK", trade_grade="150G")
    b = E.Material(material_name="PEEK", trade_grade="450G")
    c = E.Material(material_name="PEEK")
    assert E.material_key(a) == "peek|150g"
    assert E.material_key(b) == "peek|450g"
    assert E.material_key(c) == "peek"
    assert E.material_key(a) != E.material_key(b)


def test_material_key_does_not_repeat_grade_already_in_name():
    m = E.Material(material_name="Toray Cetex TC1200 PEEK", trade_grade="TC1200")
    assert E.material_key(m) == "toray cetex tc1200 peek"


def test_material_key_fallback_to_abbreviation():
    m = E.Material(material_name="", material_abbreviation="CF/PEEK", trade_grade="X1")
    assert E.material_key(m) == "cf/peek|x1"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _row(**kw) -> E.PropertyRow:
    base = dict(
        material_name="PEEK", material_abbreviation="PEEK", material_key="peek",
        material_class="Polymer", section="Physical", property_name="Density",
        value="1.30", unit="g/cm3", english="", test_condition="", comments="",
        value_raw="1.30", value_num=1.30, source_pdf="ds.pdf", source_sha1="sha_A",
        page=1, source_quote="Density 1.30 g/cm3", status="ok",
    )
    base.update(kw)
    return E.PropertyRow(**base)


@pytest.fixture
def db(tmp_path: Path) -> sqlite3.Connection:
    conn = bi.init_db(tmp_path / "t.sqlite")
    yield conn
    conn.close()


# --------------------------------------------------------------------------
# dedup: two grades from one PDF sharing a value both survive
# --------------------------------------------------------------------------

def test_two_grades_same_value_both_insert(db):
    r1 = _row(material_key="peek|150g", trade_grade="150G")
    r2 = _row(material_key="peek|450g", trade_grade="450G")
    assert not bi.already_inserted(db, "Polymers", r1)
    bi.insert_row(db, "Polymers", r1)
    assert not bi.already_inserted(db, "Polymers", r2)   # was: True -> dropped
    bi.insert_row(db, "Polymers", r2)
    assert db.execute("SELECT count(*) FROM Polymers").fetchone()[0] == 2


def test_true_reingest_is_deduped(db):
    r = _row()
    bi.insert_row(db, "Polymers", r)
    assert bi.already_inserted(db, "Polymers", _row())


# --------------------------------------------------------------------------
# promote: full grain + status guard
# --------------------------------------------------------------------------

def _write_review(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(bi._REVIEW_COLUMNS)
        for r in rows:
            w.writerow([r.get(c, "") for c in bi._REVIEW_COLUMNS])


def test_promote_does_not_touch_ok_sibling_in_other_section(db, tmp_path):
    # flagged row (Mechanical) + already-ok sibling with the same
    # property_name/value in another section (Physical)
    bi.insert_row(db, "Polymers", _row(section="Mechanical", property_name="Modulus",
                                       value_raw="3", status="unverified",
                                       flag_reason="value_not_in_pdf_text"))
    bi.insert_row(db, "Polymers", _row(section="Physical", property_name="Modulus",
                                       value_raw="3", status="ok", flag_reason=""))
    csv_path = tmp_path / "review.csv"
    _write_review(csv_path, [dict(table_name="Polymers", source_pdf="ds.pdf",
                                  material_key="peek", section="Mechanical",
                                  property_name="Modulus", test_condition="",
                                  value_raw="3")])
    n = bi.promote_review_queue(db, csv_path)
    assert n == 1                                   # was: 2
    rows = db.execute("SELECT section, status, flag_reason FROM Polymers ORDER BY section").fetchall()
    assert rows == [("Mechanical", "ok", "promoted"), ("Physical", "ok", "")]


def test_promote_is_idempotent(db, tmp_path):
    bi.insert_row(db, "Polymers", _row(status="unverified", flag_reason="x"))
    csv_path = tmp_path / "review.csv"
    _write_review(csv_path, [dict(table_name="Polymers", source_pdf="ds.pdf",
                                  material_key="peek", section="Physical",
                                  property_name="Density", test_condition="",
                                  value_raw="1.30")])
    assert bi.promote_review_queue(db, csv_path) == 1
    assert bi.promote_review_queue(db, csv_path) == 0   # status guard


# --------------------------------------------------------------------------
# sources: identity is the sha1, not the basename
# --------------------------------------------------------------------------

def test_sources_two_pdfs_same_basename_both_recorded(db):
    bi.record_source(db, Path("vendorA/datasheet.pdf"), "sha_AAA", "Polymer", "PEEK")
    bi.record_source(db, Path("vendorB/datasheet.pdf"), "sha_BBB", "Polymer", "PPS")
    assert bi.seen_sha1(db, "sha_AAA")
    assert bi.seen_sha1(db, "sha_BBB")               # was: False -> re-extracted forever
    assert db.execute("SELECT count(*) FROM sources").fetchone()[0] == 2


def test_sources_true_reingest_ignored(db):
    bi.record_source(db, Path("a.pdf"), "sha_AAA", "Polymer", "PEEK")
    bi.record_source(db, Path("a.pdf"), "sha_AAA", "Polymer", "PEEK")
    assert db.execute("SELECT count(*) FROM sources").fetchone()[0] == 1


def test_migrate_rebuilds_legacy_sources_table(tmp_path):
    p = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(str(p))
    conn.executescript(
        """
        CREATE TABLE Polymers (id INTEGER PRIMARY KEY AUTOINCREMENT, material_name TEXT,
            material_abbreviation TEXT, section TEXT, property_name TEXT, value TEXT,
            unit TEXT, english TEXT, test_condition TEXT, comments TEXT);
        CREATE TABLE Fibers (id INTEGER PRIMARY KEY AUTOINCREMENT, material_name TEXT,
            material_abbreviation TEXT, section TEXT, property_name TEXT, value TEXT,
            unit TEXT, english TEXT, test_condition TEXT, comments TEXT);
        CREATE TABLE Composites_materials (id INTEGER PRIMARY KEY AUTOINCREMENT,
            material_name TEXT, material_abbreviation TEXT, section TEXT,
            property_name TEXT, value TEXT, unit TEXT, english TEXT,
            test_condition TEXT, comments TEXT);
        CREATE TABLE sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_filename TEXT UNIQUE, pdf_sha1 TEXT, ingested_at TEXT,
            material_class TEXT, material_abbreviation TEXT);
        INSERT INTO sources (pdf_filename, pdf_sha1, ingested_at) VALUES ('a.pdf','sha_A','t1');
        INSERT INTO sources (pdf_filename, pdf_sha1, ingested_at) VALUES ('b.pdf','sha_B','t2');
        INSERT INTO sources (pdf_filename, pdf_sha1, ingested_at) VALUES ('c.pdf','sha_A','t3');
        """
    )
    conn.commit()
    conn.close()

    result = migrate(p, backup=True)
    assert (tmp_path / "legacy.sqlite.bak").exists()
    assert result["sources"]                         # rebuilt
    conn = sqlite3.connect(str(p))
    ddl = conn.execute("SELECT sql FROM sqlite_master WHERE name='sources'").fetchone()[0]
    assert "pdf_sha1 TEXT UNIQUE" in ddl and "pdf_filename TEXT UNIQUE" not in ddl
    # duplicate sha kept the earliest row; distinct rows preserved
    rows = conn.execute("SELECT pdf_filename, pdf_sha1 FROM sources ORDER BY id").fetchall()
    assert rows == [("a.pdf", "sha_A"), ("b.pdf", "sha_B")]
    # now two different PDFs with one basename both fit
    conn.execute("INSERT INTO sources (pdf_filename, pdf_sha1) VALUES ('a.pdf','sha_Z')")
    conn.commit()
    # idempotent
    assert ensure_sources_sha1_unique(conn) is False
    conn.close()
    assert migrate(p, backup=False)["sources"] == []


def test_fresh_db_sources_keyed_on_sha1(db):
    ddl = db.execute("SELECT sql FROM sqlite_master WHERE name='sources'").fetchone()[0]
    assert "pdf_sha1 TEXT UNIQUE" in ddl
