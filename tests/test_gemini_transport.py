"""Offline tests for the Gemini transport layer: retry/backoff on
generateContent AND on the File API upload/poll path (extraction.gemini_request,
extraction._request_with_retry, extraction._upload_pdf_file).

All HTTP is mocked; no network, no key.
"""

from __future__ import annotations

from typing import Any

import pytest
import requests

import extraction as E


class _Resp:
    def __init__(self, status: int, json_body: Any = None, headers: dict | None = None,
                 text: str = ""):
        self.status_code = status
        self._json = json_body if json_body is not None else {}
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)


class _Transport:
    """Scripted responses per (method, url-substring); records calls."""

    def __init__(self, script: list[Any]):
        self.script = list(script)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, method, url, timeout=None, **kw):
        self.calls.append((method, url))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def no_sleep():
    slept: list[float] = []
    yield slept


def _patch(monkeypatch, transport):
    monkeypatch.setattr(E.requests, "request", transport)


# --------------------------------------------------------------------------
# gemini_request (generateContent)
# --------------------------------------------------------------------------

def test_gemini_request_retries_then_succeeds(monkeypatch, no_sleep):
    t = _Transport([_Resp(503), _Resp(429, headers={"Retry-After": "0"}), _Resp(200, {"ok": 1})])
    _patch(monkeypatch, t)
    r = E.gemini_request("http://x", {"a": 1}, _sleep=no_sleep.append)
    assert r is not None and r.json() == {"ok": 1}
    assert len(t.calls) == 3
    assert len(no_sleep) == 2


def test_gemini_request_gives_up_after_max_retries(monkeypatch, no_sleep):
    t = _Transport([_Resp(500)] * (E.MAX_RETRIES + 1))
    _patch(monkeypatch, t)
    with pytest.raises(requests.HTTPError):
        E.gemini_request("http://x", {}, _sleep=no_sleep.append)
    assert len(t.calls) == E.MAX_RETRIES + 1


def test_gemini_request_non_retryable_raises_immediately(monkeypatch, no_sleep):
    t = _Transport([_Resp(400, text="bad")])
    _patch(monkeypatch, t)
    with pytest.raises(requests.HTTPError):
        E.gemini_request("http://x", {}, _sleep=no_sleep.append)
    assert len(t.calls) == 1


def test_gemini_request_connection_error_retries(monkeypatch, no_sleep):
    t = _Transport([requests.ConnectionError("boom"), _Resp(200, {"ok": 1})])
    _patch(monkeypatch, t)
    assert E.gemini_request("http://x", {}, _sleep=no_sleep.append) is not None


# --------------------------------------------------------------------------
# _upload_pdf_file (File API): retries on every hop + exponential ACTIVE poll
# --------------------------------------------------------------------------

def _start_ok():
    return _Resp(200, headers={"X-Goog-Upload-URL": "http://up"})


def _upload_ok(state="PROCESSING"):
    return _Resp(200, {"file": {"name": "files/abc", "uri": "gs://abc", "state": state}})


def _poll(state):
    return _Resp(200, {"name": "files/abc", "uri": "gs://abc", "state": state})


def test_upload_retries_a_429_on_start(monkeypatch, no_sleep):
    # was: bare requests.post + raise_for_status -> one 429 failed the whole PDF
    t = _Transport([_Resp(429), _start_ok(), _upload_ok("ACTIVE")])
    _patch(monkeypatch, t)
    uri = E._upload_pdf_file(b"%PDF-", "big.pdf", "k", _sleep=no_sleep.append)
    assert uri == "gs://abc"
    assert [m for m, _ in t.calls] == ["POST", "POST", "POST"]


def test_upload_polls_with_backoff_until_active(monkeypatch, no_sleep):
    t = _Transport([_start_ok(), _upload_ok("PROCESSING"),
                    _poll("PROCESSING"), _poll("PROCESSING"), _poll("ACTIVE")])
    _patch(monkeypatch, t)
    uri = E._upload_pdf_file(b"%PDF-", "big.pdf", "k", _sleep=no_sleep.append)
    assert uri == "gs://abc"
    # exponential: 1, 2, 4
    assert no_sleep == [1.0, 2.0, 4.0]


def test_upload_waits_longer_than_30s(monkeypatch, no_sleep):
    # was: fixed 30 x 1 s cap -> slow processing = None = empty_extraction forever
    n_polls = 8   # sleeps 1+2+4+8+15+15+15+15 = 75 s > 30 s
    t = _Transport([_start_ok(), _upload_ok("PROCESSING")]
                   + [_poll("PROCESSING")] * (n_polls - 1) + [_poll("ACTIVE")])
    _patch(monkeypatch, t)
    uri = E._upload_pdf_file(b"%PDF-", "big.pdf", "k", _sleep=no_sleep.append)
    assert uri == "gs://abc"
    assert sum(no_sleep) > 30


def test_upload_gives_up_after_wait_budget(monkeypatch, no_sleep):
    t = _Transport([_start_ok(), _upload_ok("PROCESSING")] + [_poll("PROCESSING")] * 100)
    _patch(monkeypatch, t)
    uri = E._upload_pdf_file(b"%PDF-", "big.pdf", "k", _sleep=no_sleep.append)
    assert uri is None
    assert sum(no_sleep) >= E.FILE_ACTIVE_WAIT_S


def test_upload_failed_state_returns_none(monkeypatch, no_sleep):
    t = _Transport([_start_ok(), _upload_ok("PROCESSING"), _poll("FAILED")])
    _patch(monkeypatch, t)
    assert E._upload_pdf_file(b"%PDF-", "big.pdf", "k", _sleep=no_sleep.append) is None


def test_upload_missing_upload_url_returns_none(monkeypatch, no_sleep):
    t = _Transport([_Resp(200, headers={})])
    _patch(monkeypatch, t)
    assert E._upload_pdf_file(b"%PDF-", "big.pdf", "k", _sleep=no_sleep.append) is None
