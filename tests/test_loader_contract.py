"""Loader contract tests (docs/implementation.md Phase 2).

The fetch path is exercised against a stubbed transport so these tests never hit
the network. The property under test is FR-1.5: a single bad URL must not stop
the run, and it must be recorded rather than silently dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from ingest import load as loader
from rag.manifest import enabled_rows, load_manifest
from rag.schemas import Document, ManifestRow

ROWS = [
    ManifestRow("a__ok", "hdfc_large_cap", "large_cap", "scheme_page", "A", "groww",
                "https://example.com/a", 1),
    ManifestRow("b__404", "hdfc_elss", "elss", "scheme_page", "B", "groww",
                "https://example.com/missing", 1),
    ManifestRow("c__boom", "hdfc_small_cap", "small_cap", "scheme_page", "C", "groww",
                "https://example.com/boom", 1),
]


class _Response:
    def __init__(self, status_code: int, content: bytes = b"<html><body><main><p>hi</p></main></body></html>"):
        self.status_code = status_code
        self.content = content


class _FakeSession:
    """Minimal stand-in for ``requests.Session``."""

    def __init__(self, plan: dict[str, object]) -> None:
        self.plan = plan
        self.calls: list[str] = []
        self.headers: dict[str, str] = {}

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def get(self, url, timeout=None, allow_redirects=True, **kwargs):
        self.calls.append(url)
        outcome = self.plan.get(url)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, int):
            return _Response(outcome)
        return _Response(200, outcome or b"<html></html>")


@pytest.fixture
def allow_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass robots.txt so tests never touch the network for policy."""
    monkeypatch.setattr(loader._Robots, "_fetch_parser", lambda self, origin: None)
    monkeypatch.setattr(loader._Robots, "wait", lambda self, url: None)
    monkeypatch.setattr(loader, "USER_AGENT", "test-agent")


@pytest.fixture
def fake_session(monkeypatch: pytest.MonkeyPatch) -> _FakeSession:
    plan = {
        "https://example.com/a": b"<html><body><main><p>Expense ratio 1.24%</p></main></body></html>",
        "https://example.com/missing": 404,
        "https://example.com/boom": requests.ConnectionError("dns failure"),
    }
    session = _FakeSession(plan)
    monkeypatch.setattr(loader.requests, "Session", lambda: session)
    return session


def test_fetch_returns_one_document_per_row(fake_session, allow_all, tmp_path: Path) -> None:
    docs = loader.fetch(ROWS, raw_dir=tmp_path, timeout=5, max_attempts=1)
    assert len(docs) == len(ROWS)


def test_a_failed_url_does_not_stop_the_run(fake_session, allow_all, tmp_path: Path) -> None:
    docs = loader.fetch(ROWS, raw_dir=tmp_path, timeout=5, max_attempts=1)
    assert all(isinstance(doc, Document) for doc in docs)
    assert "https://example.com/boom" in fake_session.calls


def test_http_status_is_always_recorded(fake_session, allow_all, tmp_path: Path) -> None:
    docs = {doc.source_id: doc for doc in loader.fetch(ROWS, raw_dir=tmp_path, timeout=5, max_attempts=1)}
    assert docs["a__ok"].http_status == 200
    assert docs["b__404"].http_status == 404
    assert docs["c__boom"].http_status == 0


def test_a_404_is_not_retried(fake_session, allow_all, tmp_path: Path) -> None:
    loader.fetch([ROWS[1]], raw_dir=tmp_path, timeout=5, max_attempts=3)
    assert fake_session.calls.count("https://example.com/missing") == 1


def test_a_transient_5xx_is_retried(monkeypatch, allow_all, tmp_path: Path) -> None:
    session = _FakeSession({"https://example.com/a": 503})
    monkeypatch.setattr(loader.requests, "Session", lambda: session)
    monkeypatch.setattr(loader.time, "sleep", lambda *_a, **_k: None)
    loader.fetch([ROWS[0]], raw_dir=tmp_path, timeout=5, max_attempts=3)
    assert session.calls.count("https://example.com/a") == 3


def test_successful_fetch_writes_a_raw_snapshot_and_hash(fake_session, allow_all, tmp_path: Path) -> None:
    docs = loader.fetch(ROWS, raw_dir=tmp_path, timeout=5, max_attempts=1)
    ok = next(doc for doc in docs if doc.source_id == "a__ok")
    assert Path(ok.raw_path).is_file()
    assert Path(ok.raw_path).read_bytes().startswith(b"<html>")
    assert len(ok.sha256) == 64


def test_every_document_gets_a_retrieved_at(fake_session, allow_all, tmp_path: Path) -> None:
    for doc in loader.fetch(ROWS, raw_dir=tmp_path, timeout=5, max_attempts=1):
        assert doc.retrieved_at


def test_load_all_populates_content_chars_and_clean_path(fake_session, allow_all, tmp_path: Path) -> None:
    docs = loader.load_all(ROWS, raw_dir=tmp_path / "raw", clean_dir=tmp_path / "clean", max_attempts=1)
    ok = next(doc for doc in docs if doc.source_id == "a__ok")
    assert ok.content_chars > 0
    assert Path(ok.clean_path).is_file()
    assert "1.24%" in Path(ok.clean_path).read_text(encoding="utf-8")


def test_a_thin_page_is_flagged(monkeypatch, allow_all, tmp_path: Path) -> None:
    session = _FakeSession({"https://example.com/a": b"<html><body><main><p>x</p></main></body></html>"})
    monkeypatch.setattr(loader.requests, "Session", lambda: session)
    doc = loader.load_all([ROWS[0]], raw_dir=tmp_path / "raw", clean_dir=tmp_path / "clean", max_attempts=1)[0]
    assert doc.content_chars < 200
    assert doc.thin is True


def test_write_manifest_records_the_embedding_model(fake_session, allow_all, tmp_path: Path) -> None:
    docs = loader.fetch(ROWS, raw_dir=tmp_path, timeout=5, max_attempts=1)
    out = loader.write_manifest(docs, path=tmp_path / "manifest.json", embedding_model="m", dimension=384)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["embedding_model"] == "m"
    assert payload["dimension"] == 384
    assert payload["n_documents"] == len(docs)
    assert len(payload["documents"]) == len(docs)


def test_url_allowlist_excludes_failed_fetches(fake_session, allow_all, tmp_path: Path) -> None:
    docs = loader.fetch(ROWS, raw_dir=tmp_path, timeout=5, max_attempts=1)
    path = loader.write_manifest(docs, path=tmp_path / "manifest.json")
    allow = loader.url_allowlist(path)
    assert allow == {"https://example.com/a"}


def test_url_allowlist_is_empty_without_a_manifest(tmp_path: Path) -> None:
    assert loader.url_allowlist(tmp_path / "nope.json") == set()


def test_read_manifest_on_missing_file_returns_empty(tmp_path: Path) -> None:
    assert loader.read_manifest(tmp_path / "nope.json") == {}


def test_load_documents_round_trips_through_the_manifest(fake_session, allow_all, tmp_path: Path) -> None:
    docs = loader.load_all(ROWS, raw_dir=tmp_path / "raw", clean_dir=tmp_path / "clean", max_attempts=1)
    path = loader.write_manifest(docs, path=tmp_path / "manifest.json")
    restored = loader.load_documents(manifest=path)
    assert [doc.source_id for doc in restored] == [doc.source_id for doc in docs]
    assert restored[0].content_chars == docs[0].content_chars


def test_robots_disallow_is_respected(monkeypatch, tmp_path: Path) -> None:
    import urllib.robotparser

    session = _FakeSession({"https://example.com/a": b"<html></html>"})
    monkeypatch.setattr(loader.requests, "Session", lambda: session)
    parser = urllib.robotparser.RobotFileParser()
    parser.parse(["User-agent: *", "Disallow: /a"])
    monkeypatch.setattr(loader._Robots, "_fetch_parser", lambda self, origin: parser)

    doc = loader.fetch([ROWS[0]], raw_dir=tmp_path, timeout=5, max_attempts=1)[0]

    assert doc.http_status == 0
    assert session.calls == [], "a disallowed URL must not be requested at all"


def test_log_event_drops_unknown_keys(capsys: pytest.CaptureFixture[str]) -> None:
    loader.log_event("load", source_id="a", outcome="ok", response_body="SECRET PAGE TEXT")
    line = capsys.readouterr().out.strip()
    record = json.loads(line)
    assert record == {"stage": "load", "source_id": "a", "outcome": "ok"}


# ---------------------------------------------------------------------------
# The committed manifest
# ---------------------------------------------------------------------------


def test_manifest_is_valid_and_has_five_enabled_rows() -> None:
    rows = load_manifest()
    assert loader.log_event is not None  # module import sanity
    assert len(enabled_rows(rows)) == 5
