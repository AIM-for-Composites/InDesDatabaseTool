"""Offline tests for pdf_crawler's download bookkeeping and validation.

Proves: transient failures are retried on later runs (not blacklisted),
deterministic rejections and saves are final, state.json round-trips the
`failed` map, and the magic-byte/PyMuPDF validation still rejects junk.
No network: SESSION.head / http_get are monkeypatched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

import pdf_crawler as C


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

class _Head:
    def __init__(self, status=200, ctype="application/pdf", length=None):
        self.status_code = status
        self.headers = {"Content-Type": ctype}
        if length is not None:
            self.headers["Content-Length"] = str(length)


class _Get:
    def __init__(self, data: bytes):
        self._data = data

    def iter_content(self, chunk_size=65536):
        for i in range(0, len(self._data), chunk_size):
            yield self._data[i:i + chunk_size]


def _real_pdf_bytes() -> bytes:
    """A tiny genuine PDF (>= MIN_PDF_BYTES so the size floor passes)."""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "AIM test PDF " * 5)
    data = doc.tobytes()
    doc.close()
    # pad with a trailing comment stream so it clears MIN_PDF_BYTES but stays valid
    if len(data) < C.MIN_PDF_BYTES:
        data += b"\n%" + b"x" * (C.MIN_PDF_BYTES - len(data) + 16) + b"\n"
    return data


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    """Patch throttling + HTTP; return a dict the tests mutate to script responses."""
    monkeypatch.setattr(C.THROTTLE, "wait", lambda url: None)
    script = {"head": _Head(), "get": None, "head_exc": None}

    def fake_head(url, timeout=10, allow_redirects=True):
        if script["head_exc"]:
            raise script["head_exc"]
        return script["head"]

    def fake_get(url, **kw):
        return script["get"]

    monkeypatch.setattr(C.SESSION, "head", fake_head)
    monkeypatch.setattr(C, "http_get", fake_get)
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    state = C.CrawlerState(tmp_path / "state.json")
    cand = C.Candidate(title="Test paper", pdf_url="http://x/paper.pdf", source="openalex")
    return dict(script=script, pdf_dir=pdf_dir, state=state, cand=cand, tmp=tmp_path)


# --------------------------------------------------------------------------
# transient vs final
# --------------------------------------------------------------------------

def test_transient_head_block_is_not_blacklisted(env):
    # was: URL added to seen_urls before the attempt -> a momentary 403/503
    # permanently blacklisted the PDF
    env["script"]["head"] = _Head(status=503)
    assert C.download_pdf(env["cand"], env["pdf_dir"], env["state"]) is None
    st = env["state"]
    assert env["cand"].pdf_url not in st.seen_urls
    assert st.failed[env["cand"].pdf_url] == 1


def _next_run(state: C.CrawlerState) -> C.CrawlerState:
    """Simulate a fresh crawler process: persisted state survives, the
    per-run `attempted` set does not."""
    state.save()
    return C.CrawlerState(state.path)


def test_transient_get_failure_counts_then_succeeds_next_run(env):
    url = env["cand"].pdf_url
    env["script"]["get"] = None                        # http_get exhausted retries
    C.download_pdf(env["cand"], env["pdf_dir"], env["state"])
    assert env["state"].failed[url] == 1
    # "next run": server is back
    st = _next_run(env["state"])
    env["script"]["get"] = _Get(_real_pdf_bytes())
    row = C.download_pdf(env["cand"], env["pdf_dir"], st)
    assert row is not None and row["url"] == url
    assert url in st.seen_urls
    assert url not in st.failed


def test_gives_up_after_max_attempts(env):
    url = env["cand"].pdf_url
    env["script"]["get"] = None
    st = env["state"]
    for _ in range(C.MAX_URL_ATTEMPTS):
        C.download_pdf(env["cand"], env["pdf_dir"], st)
        st = _next_run(st)                              # one attempt per RUN
    assert url in st.seen_urls
    assert url not in st.failed


def test_same_url_twice_in_one_run_is_one_attempt(env):
    url = env["cand"].pdf_url
    env["script"]["get"] = None
    for _ in range(4):
        C.download_pdf(env["cand"], env["pdf_dir"], env["state"])
    assert env["state"].failed[url] == 1
    assert url not in env["state"].seen_urls


def test_deterministic_rejections_are_final(env):
    url = env["cand"].pdf_url
    env["script"]["get"] = _Get(b"<html>paywall</html>" + b" " * C.MIN_PDF_BYTES)
    assert C.download_pdf(env["cand"], env["pdf_dir"], env["state"]) is None
    assert url in env["state"].seen_urls
    assert url not in env["state"].failed


def test_head_404_is_final_but_403_is_transient(env):
    env["script"]["head"] = _Head(status=404)
    C.download_pdf(env["cand"], env["pdf_dir"], env["state"])
    assert env["cand"].pdf_url in env["state"].seen_urls

    cand2 = C.Candidate(title="t2", pdf_url="http://x/2.pdf", source="openalex")
    env["script"]["head"] = _Head(status=403)
    C.download_pdf(cand2, env["pdf_dir"], env["state"])
    assert cand2.pdf_url not in env["state"].seen_urls
    assert env["state"].failed[cand2.pdf_url] == 1


def test_head_html_content_type_is_final(env):
    env["script"]["head"] = _Head(status=200, ctype="text/html; charset=utf-8")
    C.download_pdf(env["cand"], env["pdf_dir"], env["state"])
    assert env["cand"].pdf_url in env["state"].seen_urls


def test_seen_url_is_skipped_without_fetch(env):
    env["state"].mark_final(env["cand"].pdf_url)
    env["script"]["get"] = _Get(_real_pdf_bytes())
    assert C.download_pdf(env["cand"], env["pdf_dir"], env["state"]) is None
    assert not list(env["pdf_dir"].iterdir())


# --------------------------------------------------------------------------
# state persistence
# --------------------------------------------------------------------------

def test_state_roundtrips_failed_map(env):
    st = env["state"]
    st.mark_transient_failure("http://a")
    st.mark_transient_failure("http://a")
    st.mark_final("http://b")
    st.seen_hashes.add("deadbeef")
    st.save()
    data = json.loads(st.path.read_text())
    assert data["failed"] == {"http://a": 2}
    assert data["seen_urls"] == ["http://b"]
    st2 = C.CrawlerState(st.path)
    assert st2.failed == {"http://a": 2}
    assert st2.seen_urls == {"http://b"}
    assert st2.seen_hashes == {"deadbeef"}


def test_legacy_state_without_failed_key_loads(tmp_path):
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"seen_urls": ["u"], "seen_hashes": ["h"]}))
    st = C.CrawlerState(p)
    assert st.seen_urls == {"u"} and st.failed == {}


# --------------------------------------------------------------------------
# validation (RUNBOOK L1 fixtures) still holds
# --------------------------------------------------------------------------

@pytest.mark.parametrize("data", [
    b"<html><body>Please log in</body></html>" + b" " * 20000,   # HTML as pdf
    b"%PDF-1.4 truncated" + b"\x00" * 20000,                     # magic but broken
    b"",                                                          # empty
], ids=["html-as-pdf", "truncated-magic", "empty"])
def test_junk_bytes_rejected_and_final(env, data):
    env["script"]["get"] = _Get(data)
    assert C.download_pdf(env["cand"], env["pdf_dir"], env["state"]) is None
    assert env["cand"].pdf_url in env["state"].seen_urls
    assert not list(env["pdf_dir"].iterdir())


def test_real_pdf_saved_with_provenance(env):
    env["script"]["get"] = _Get(_real_pdf_bytes())
    row = C.download_pdf(env["cand"], env["pdf_dir"], env["state"])
    assert row is not None
    assert row["source"] == "openalex" and row["sha256"] in env["state"].seen_hashes
    assert (env["pdf_dir"] / row["filename"]).exists()


def test_duplicate_content_skipped_but_url_final(env):
    env["script"]["get"] = _Get(_real_pdf_bytes())
    C.download_pdf(env["cand"], env["pdf_dir"], env["state"])
    cand2 = C.Candidate(title="Same bytes", pdf_url="http://y/other.pdf", source="arxiv")
    assert C.download_pdf(cand2, env["pdf_dir"], env["state"]) is None
    assert cand2.pdf_url in env["state"].seen_urls
    assert len(list(env["pdf_dir"].iterdir())) == 1


# --------------------------------------------------------------------------
# generate_queries imports (was ImportError at module load)
# --------------------------------------------------------------------------

def test_generate_queries_imports():
    import generate_queries  # noqa: F401
    assert generate_queries.GEMINI_MODEL
