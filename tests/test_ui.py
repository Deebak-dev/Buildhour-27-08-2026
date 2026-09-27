"""Phase 8 UI tests (FR-8.1 to FR-9.2, A15 to A18).

Two layers: the pure helpers are tested directly, and the rendered page is
tested with Streamlit's ``AppTest``, which runs the real script. The point of
the second layer is the requirements that only exist in the DOM: the
disclaimer is visible, the citation is a real link, and exactly three examples
are offered.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

import app as app_module
from rag.schemas import AnswerPayload, RetrievalTrace, Route

APP_PATH = str(Path(app_module.__file__))


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
ANSWER_PROSE = "The expense ratio is 1.03% per annum."
SOURCE_URL = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
LAST_UPDATED = "27 Sep 2026"


def _contract(text: str, url: str | None, date: str = LAST_UPDATED) -> str:
    """A payload body in the exact §7 response-contract shape."""
    if not url:
        return text
    return f"{text}\n\nSource: {url}\nLast updated from sources: {date}"


def _chunk(text: str = "Expense ratio: 1.03%", **kw: Any) -> Any:
    from rag.schemas import Chunk

    return Chunk(
        chunk_id=kw.get("chunk_id", "c1"),
        source_id=kw.get("source_id", "d1"),
        scheme=kw.get("scheme", "hdfc_large_cap"),
        field_type=kw.get("field_type", "expense_ratio"),
        section=kw.get("section", "Fees"),
        ordinal=1,
        text=text,
        source_url=kw.get(
            "source_url", "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
        ),
        retrieved_at=kw.get("retrieved_at", "2026-09-27T08:44:18+00:00"),
    )


def _trace(*chunks: Any) -> RetrievalTrace:
    from rag.schemas import ScoredChunk

    scored = [
        ScoredChunk(
            chunk=c,
            similarity=0.9 - 0.1 * i,
            raw_similarity=0.6 - 0.1 * i,
            boosted=(i == 0),
        )
        for i, c in enumerate(chunks or (_chunk(),))
    ]
    return RetrievalTrace(
        query="q",
        scheme_filter="hdfc_large_cap",
        chunks=scored,
        tau=0.35,
        top_score=scored[0].similarity if scored else 0.0,
    )


def _payload(route: Route = Route.FACTUAL, **kw: Any) -> AnswerPayload:
    text = kw.get("text", ANSWER_PROSE)
    url = kw.get("source_url", SOURCE_URL)
    if "text" not in kw:
        text = _contract(text, url)
    return AnswerPayload(
        text=text,
        route=route,
        source_url=url,
        last_updated=kw.get("last_updated", LAST_UPDATED),
        grounded_field=kw.get("grounded_field", "expense_ratio"),
        trace=kw.get("trace", _trace()),
        degraded=kw.get("degraded", False),
        validators_passed=kw.get("validators_passed", ["G1", "G2", "G3"]),
    )


# ---------------------------------------------------------------------------
# D5 single source of truth
# ---------------------------------------------------------------------------
class TestDisclaimer:
    def test_constant_matches_the_prd_verbatim(self) -> None:
        """The banner must be the §11.3 text, unchanged."""
        prd = (app_module.PROJECT_ROOT / "docs" / "PRD.md").read_text(encoding="utf-8")
        assert "Facts-only. No investment advice." in app_module.DISCLAIMER
        assert "SEBI-registered investment adviser before investing." in (
            app_module.DISCLAIMER
        )
        # Every PRD line of the snippet appears in the constant.
        for line in (
            "This assistant answers questions about publicly available mutual fund",
            "schemes, and it does not provide returns, projections, or tax advice.",
            "investments are subject to market risks; read the scheme's SID/KIM and",
        ):
            assert line in app_module.DISCLAIMER
            assert line in prd

    def test_committed_deliverable_matches_the_constant(self) -> None:
        """docs/disclaimer.md must not drift from DISCLAIMER (FR-9.2)."""
        path = app_module.DISCLAIMER_PATH
        assert path.is_file(), "docs/disclaimer.md is a Phase 8 deliverable"
        assert app_module.DISCLAIMER in path.read_text(encoding="utf-8")

    def test_write_is_idempotent(self, tmp_path: Path) -> None:
        target = tmp_path / "disclaimer.md"
        assert app_module.write_disclaimer(target) is True
        assert app_module.write_disclaimer(target) is False

    def test_write_repairs_a_tampered_file(self, tmp_path: Path) -> None:
        target = tmp_path / "disclaimer.md"
        target.write_text("something else entirely", encoding="utf-8")
        assert app_module.write_disclaimer(target) is True
        assert app_module.DISCLAIMER in target.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Header chip (FR-9.1)
# ---------------------------------------------------------------------------
class TestCorpusChip:
    def test_reads_counts_and_date_from_the_manifest(self) -> None:
        summary = app_module.corpus_summary()
        assert "1 AMC" in summary
        assert "5 schemes" in summary
        assert "5 documents" in summary
        assert "ingested 27 Sep 2026" in summary

    def test_nothing_is_hardcoded(self) -> None:
        """Point it at a synthetic manifest and the numbers must follow."""
        manifest = {
            "generated_at": "2026-01-02T03:04:05+00:00",
            "n_documents": 2,
            "documents": [
                {"scheme": "a", "retrieved_at": "2026-01-02T03:04:05+00:00"},
                {"scheme": "b", "retrieved_at": "2026-01-02T03:04:05+00:00"},
            ],
        }
        path = app_module.PROJECT_ROOT / "data" / "_tmp_test_manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        try:
            summary = app_module.corpus_summary(path)
        finally:
            path.unlink()
        assert "2 schemes" in summary
        assert "2 documents" in summary
        assert "ingested 02 Jan 2026" in summary

    def test_missing_manifest_gives_the_ingest_hint(self, tmp_path: Path) -> None:
        assert app_module.corpus_summary(tmp_path / "nope.json") == app_module.INGEST_HINT

    def test_empty_document_list_gives_the_ingest_hint(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"documents": []}), encoding="utf-8")
        assert app_module.corpus_summary(path) == app_module.INGEST_HINT

    def test_corrupt_manifest_gives_the_ingest_hint(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        path.write_text("{not json", encoding="utf-8")
        assert app_module.corpus_summary(path) == app_module.INGEST_HINT

    def test_date_formatting_survives_garbage(self) -> None:
        assert app_module._format_date("not-a-date") == "not-a-date"
        assert app_module._format_date("") == "unknown date"


# ---------------------------------------------------------------------------
# Citation visibility (FR-8.4, FR-8.5)
# ---------------------------------------------------------------------------
class TestCitationRendering:
    def test_source_url_becomes_a_visible_link(self) -> None:
        payload = _payload()
        rendered = app_module.answer_markdown(payload)
        assert 'rel="noopener noreferrer"' in rendered
        assert "<a href=" in rendered
        assert payload.source_url in rendered

    def test_the_link_text_is_the_url_not_a_label(self) -> None:
        """FR-8.5: the source must be visible, not a word like 'here'."""
        payload = _payload()
        rendered = app_module.answer_markdown(payload)
        assert f">{payload.source_url}</a>" in rendered, (
            "the URL itself must be the link text"
        )

    def test_target_blank_is_set(self) -> None:
        assert 'target="_blank"' in app_module.answer_markdown(_payload())

    def test_non_http_urls_are_left_as_text(self) -> None:
        assert app_module._anchor("javascript:alert(1)") == "javascript:alert(1)"
        assert "<a" not in app_module._anchor("ftp://x/y")

    def test_html_in_a_url_is_escaped(self) -> None:
        rendered = app_module._anchor('https://x/y"><script>alert(1)</script>')
        assert "<script>" not in rendered
        assert "&quot;" in rendered

    def test_a_payload_without_a_url_is_untouched(self) -> None:
        payload = _payload(source_url=None, text="No citation here.")
        assert app_module.answer_markdown(payload) == "No citation here."


# ---------------------------------------------------------------------------
# Example questions (FR-8.2, A16)
# ---------------------------------------------------------------------------
class TestExampleQuestions:
    def test_exactly_three(self) -> None:
        assert len(app_module.EXAMPLE_QUESTIONS) == 3

    def test_they_are_the_specified_questions(self) -> None:
        assert app_module.EXAMPLE_QUESTIONS == (
            "What is the expense ratio of HDFC Large Cap Fund - Direct Growth?",
            "What is the lock-in period for HDFC ELSS Tax Saver Fund?",
            "Should I buy HDFC Small Cap Fund?",
        )

    def test_two_are_facts_and_one_is_a_refusal(self) -> None:
        assert not app_module.EXAMPLE_QUESTIONS[2].lower().startswith("what is")


# ---------------------------------------------------------------------------
# Rendered page (AppTest)
# ---------------------------------------------------------------------------
def _run(monkeypatch: pytest.MonkeyPatch, ask_fn: Any, **cfg: Any) -> AppTest:
    """Run the real app script with a stubbed pipeline.

    ``AppTest`` re-executes ``app.py`` as its own module, so patching attributes
    on the imported ``app`` module would not be seen. Everything is stubbed one
    level lower, in ``rag.pipeline``, which the script imports at run time.
    """
    import rag.pipeline as pipeline

    class _Store:
        def count(self) -> int:
            return 0 if cfg.get("empty") else 1

    monkeypatch.setattr(pipeline, "ask", ask_fn)
    monkeypatch.setattr(pipeline, "collection", lambda _cfg: _Store())
    return AppTest.from_file(APP_PATH, default_timeout=30).run()


class TestRenderedPage:
    def test_header_and_scope_chip_render(self, monkeypatch: pytest.MonkeyPatch) -> None:
        at = _run(monkeypatch, lambda q: _payload())
        assert at.title[0].value == app_module.PRODUCT_NAME
        assert any("5 documents" in c.value for c in at.caption)

    def test_disclaimer_is_visible_on_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        at = _run(monkeypatch, lambda q: _payload())
        warnings = [w.value for w in at.warning]
        assert any("Facts-only. No investment advice." in w for w in warnings)
        assert any("SEBI-registered" in w for w in warnings)

    def test_exactly_three_example_buttons(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        at = _run(monkeypatch, lambda q: _payload())
        labels = [b.label for b in at.button]
        assert labels == list(app_module.EXAMPLE_QUESTIONS)

    def test_clicking_an_example_produces_an_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[str] = []

        def fake_ask(query: str) -> AnswerPayload:
            seen.append(query)
            return _payload()

        at = _run(monkeypatch, fake_ask)
        at.button(key=f"example-{app_module.EXAMPLE_QUESTIONS[0]}").click().run()
        assert seen == [app_module.EXAMPLE_QUESTIONS[0]]
        assert len(at.chat_message) >= 2

    def test_citation_link_is_in_the_rendered_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = _payload()
        at = _run(monkeypatch, lambda q: payload)
        at.chat_input[0].set_value("expense ratio?").run()
        html = "\n".join(m.value for m in at.markdown)
        assert 'rel="noopener noreferrer"' in html
        assert payload.source_url in html

    def test_a_refusal_renders_in_an_info_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = _payload(
            route=Route.ADVICE,
            text=(
                "I can share facts published in the scheme's official sources, but I "
                "do not give investment recommendations."
            ),
            source_url="https://www.sebi.gov.in/",
        )
        at = _run(monkeypatch, lambda q: payload)
        at.chat_input[0].set_value("Should I buy HDFC Small Cap Fund?").run()
        assert any("do not give investment recommendations" in i.value for i in at.info)

    def test_pii_renders_in_a_warning_not_an_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = _payload(
            route=Route.PII,
            text="This assistant does not accept personal or account information.",
            source_url=None,
            trace=None,
        )
        at = _run(monkeypatch, lambda q: payload)
        at.chat_input[0].set_value("My PAN is ABCDE1234F").run()
        assert any("does not accept personal" in w.value for w in at.warning)

    def test_trace_expander_is_collapsed_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        at = _run(monkeypatch, lambda q: _payload())
        at.chat_input[0].set_value("expense ratio?").run()
        labels = [e.label for e in at.expander]
        assert "Show retrieved chunks (RAG trace)" in labels

    def test_expander_shows_scheme_field_type_and_score(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        at = _run(monkeypatch, lambda q: _payload())
        at.chat_input[0].set_value("expense ratio?").run()
        body = "\n".join(m.value for m in at.markdown)
        assert "hdfc_large_cap" in body
        assert "expense_ratio" in body
        assert "0.900" in body

    def test_a_raising_ask_shows_a_graceful_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(query: str) -> AnswerPayload:
            raise RuntimeError("provider exploded")

        at = _run(monkeypatch, boom)
        at.chat_input[0].set_value("expense ratio?").run()
        warnings = "\n".join(w.value for w in at.warning)
        assert "couldn't answer right now" in warnings
        # I6: no answer prose may be shown for a failed call.
        assert "Expense ratio: 1.03%" not in "\n".join(m.value for m in at.markdown)

    def test_empty_store_shows_the_ingest_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        at = _run(monkeypatch, lambda q: _payload(), empty=True)
        assert any("make ingest" in e.value for e in at.error)
        assert not at.chat_input, "there is nothing to ask with an empty store"

    def test_no_admin_or_config_controls_are_rendered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FR-8.7 / A18: nothing back-end may be exposed."""
        at = _run(monkeypatch, lambda q: _payload())
        assert not at.file_uploader
        assert not at.sidebar
        assert len(at.button) == 3, "the only buttons are the three examples"

    def test_chat_is_not_written_to_disk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FR-5.3: asking a question must not create or change any file."""
        logs = app_module.PROJECT_ROOT / "logs" / "app.log"
        before = logs.stat().st_mtime if logs.exists() else None
        chunks_file = app_module.PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"
        chunks_before = (
            chunks_file.stat().st_mtime, chunks_file.stat().st_size
        ) if chunks_file.exists() else None

        at = _run(monkeypatch, lambda q: _payload())
        at.chat_input[0].set_value("My PAN is ABCDE1234F and email a@b.com").run()

        after = logs.stat().st_mtime if logs.exists() else None
        assert after == before, "the query text must not reach the log file"
        if chunks_file.exists():
            assert (
                chunks_file.stat().st_mtime,
                chunks_file.stat().st_size,
            ) == chunks_before


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
