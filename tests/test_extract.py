"""Tests for the cleaner (docs/implementation.md Phase 2).

``extract`` is pure and offline, so these run against fixture HTML with no
network access.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ingest.load import extract
from rag.schemas import Document


def write_doc(tmp_path: Path, html: str, *, source_id: str = "fixture") -> Document:
    """Write fixture HTML and return a Document pointing at it."""
    raw = tmp_path / f"{source_id}.html"
    raw.write_text(html, encoding="utf-8")
    return Document(
        source_id=source_id,
        scheme="hdfc_large_cap",
        category="large_cap",
        doc_type="scheme_page",
        title="Fixture",
        publisher="groww",
        source_url="https://example.com/scheme",
        retrieved_at="2026-09-27T00:00:00+00:00",
        sha256="0" * 64,
        http_status=200,
        content_chars=0,
        raw_path=str(raw),
    )


PAGE = """
<html><head><title>t</title><style>.x{color:red}</style></head>
<body>
  <nav>Home Products Login</nav>
  <header>Site header</header>
  <div class="cookie-banner">We use cookies. Accept all</div>
  <div class="ad-slot">Advertisement</div>
  <main>
    <h1>HDFC Large Cap Fund</h1>
    <p>Expense ratio: 1.24%</p>
    <p>Exit load: 1% if redeemed within 12 months</p>
  </main>
  <footer>Copyright 2026</footer>
  <script>var tracking = 1;</script>
  <noscript>Enable JS</noscript>
</body></html>
"""


def test_extract_prefers_main_content(tmp_path: Path) -> None:
    text, chars = extract(write_doc(tmp_path, PAGE))
    assert "Expense ratio: 1.24%" in text
    assert chars == len(text)


@pytest.mark.parametrize(
    "noise",
    ["Home Products Login", "Site header", "We use cookies", "Advertisement",
     "Copyright 2026", "var tracking", "Enable JS", "color:red"],
)
def test_extract_removes_chrome_and_noise(tmp_path: Path, noise: str) -> None:
    text, _ = extract(write_doc(tmp_path, PAGE))
    assert noise not in text


def test_extract_falls_back_to_body(tmp_path: Path) -> None:
    html = "<html><body><div><p>Minimum SIP is 500</p></div></body></html>"
    text, _ = extract(write_doc(tmp_path, html))
    assert "Minimum SIP is 500" in text


def test_extract_handles_article_and_role_main(tmp_path: Path) -> None:
    article = "<html><body><article><p>Lock-in period: 3 years</p></article></body></html>"
    role = "<html><body><div role='main'><p>Benchmark: Nifty 100</p></div></body></html>"
    assert "Lock-in period: 3 years" in extract(write_doc(tmp_path, article))[0]
    assert "Benchmark: Nifty 100" in extract(write_doc(tmp_path, role, source_id="b"))[0]


def test_extract_collapses_blank_runs(tmp_path: Path) -> None:
    html = "<html><body><main><p>one</p><p></p><p></p><p></p><p>two</p></main></body></html>"
    text, _ = extract(write_doc(tmp_path, html))
    assert "\n\n\n" not in text


def test_extract_returns_empty_for_missing_snapshot(tmp_path: Path) -> None:
    doc = Document(
        source_id="x", scheme="hdfc_large_cap", category="large_cap", doc_type="scheme_page",
        title="t", publisher="groww", source_url="https://example.com",
        retrieved_at="", sha256="", http_status=0, content_chars=0, raw_path="",
    )
    assert extract(doc) == ("", 0)
    assert extract(write_doc(tmp_path, PAGE).__class__(
        **{**write_doc(tmp_path, PAGE).__dict__, "raw_path": str(tmp_path / "gone.html")}
    )) == ("", 0)


def test_extract_survives_malformed_html(tmp_path: Path) -> None:
    raw = tmp_path / "broken.html"
    raw.write_text("<<<not really html", encoding="utf-8")
    doc = write_doc(tmp_path, "", source_id="broken")
    doc = Document(**{**doc.__dict__, "raw_path": str(raw)})
    text, chars = extract(doc)
    assert isinstance(text, str) and chars == len(text)
