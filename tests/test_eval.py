"""Tests for the Phase 9 eval harness and the generated deliverables.

The harness is the only thing standing between a fabricated pass rate and an
honest one, so the checks below are mostly about whether it can be fooled: a
harness that passes a broken answer, or that silently drops a failing row, is
worse than no harness.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import pytest

from eval.run_eval import (
    ADVERSARIAL_PATH,
    DEFAULT_KS,
    GOLDEN_PATH,
    SAMPLE_QA_IDS,
    Check,
    RowResult,
    build_report,
    check_adversarial_row,
    check_golden_row,
    is_refusal_prose,
    load_rows,
    prose_of,
    recall_hits,
    run_suite,
    sentence_count,
    source_urls,
    summarise,
    write_sample_qa,
    write_source_list,
)
from rag.schemas import Route

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
def _payload(route: Route = Route.FACTUAL, **kw: Any) -> Any:
    from rag.schemas import AnswerPayload, Chunk, RetrievalTrace, ScoredChunk

    url = kw.get("url", "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth")
    field = kw.get("field_type", "expense_ratio")
    scheme = kw.get("scheme", "hdfc_large_cap")
    chunk = Chunk(
        chunk_id="c1",
        source_id="d1",
        scheme=scheme,
        field_type=field,
        section="Fees",
        ordinal=1,
        text=f"{field}: 1.03%",
        source_url=url,
        retrieved_at="2026-09-27T08:44:18+00:00",
    )
    trace = RetrievalTrace(
        query="q",
        scheme_filter=scheme,
        chunks=[
            ScoredChunk(
                chunk=chunk, similarity=0.9, raw_similarity=0.6, boosted=True
            )
        ],
        tau=0.35,
        top_score=0.9,
    )
    text = kw.get(
        "text",
        f"The expense ratio is 1.03% per annum.\n\nSource: {url}\n"
        "Last updated from sources: 27 Sep 2026",
    )
    return AnswerPayload(
        text=text,
        route=route,
        source_url=url,
        last_updated="27 Sep 2026",
        grounded_field=field,
        trace=kw.get("trace", trace),
        degraded=kw.get("degraded", False),
        validators_passed=["G1"],
    )


# ---------------------------------------------------------------------------
# Suite files
# ---------------------------------------------------------------------------
class TestSuiteFiles:
    def test_golden_has_ten_rows_with_the_required_fields(self) -> None:
        rows = load_rows(GOLDEN_PATH)
        assert len(rows) == 10
        for row in rows:
            assert {"id", "query", "scheme", "expect_field_type", "expect_source_url"} <= set(row)

    def test_adversarial_has_fifteen_rows(self) -> None:
        rows = load_rows(ADVERSARIAL_PATH)
        assert len(rows) == 15
        for row in rows:
            assert {"id", "query", "expect_route"} <= set(row)

    def test_comments_and_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "s.jsonl"
        path.write_text('# header\n\n{"id": "a"}\n', encoding="utf-8")
        assert load_rows(path) == [{"id": "a"}]

    def test_ids_are_unique(self) -> None:
        ids = [r["id"] for r in load_rows(GOLDEN_PATH)] + [
            r["id"] for r in load_rows(ADVERSARIAL_PATH)
        ]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
class TestTextHelpers:
    def test_prose_strips_the_contract_lines(self) -> None:
        text = "The answer.\n\nSource: https://x/y\nLast updated from sources: 27 Sep 2026"
        assert prose_of(text) == "The answer."

    def test_sentence_count_ignores_contract_lines(self) -> None:
        text = "One. Two.\n\nSource: https://x/y\nLast updated from sources: 27 Sep 2026"
        assert sentence_count(text) == 2

    def test_decimals_do_not_split_sentences(self) -> None:
        assert sentence_count("The ratio is 1.03% per annum.") == 1

    def test_empty_prose_counts_zero(self) -> None:
        assert sentence_count("") == 0
        assert sentence_count("Source: https://x/y") == 0

    def test_source_urls_reads_every_source_line(self) -> None:
        text = "A.\n\nSource: https://x/1\nLast updated from sources: 27 Sep 2026"
        assert source_urls(text) == ["https://x/1"]

    def test_source_urls_is_empty_without_a_source_line(self) -> None:
        assert source_urls("Just prose.") == []


# ---------------------------------------------------------------------------
# Golden checks
# ---------------------------------------------------------------------------
class TestGoldenChecks:
    def test_a_correct_answer_passes_every_applicable_check(self) -> None:
        row = load_rows(GOLDEN_PATH)[0]
        payload = _payload(
            field_type=row["expect_field_type"],
            scheme=row["scheme"],
            url=row["expect_source_url"],
        )
        checks = check_golden_row(row, payload)
        assert all(c.passed for c in checks), [c.detail for c in checks if not c.passed]

    def test_a_missing_citation_fails(self) -> None:
        row = load_rows(GOLDEN_PATH)[0]
        payload = _payload(text="The expense ratio is 1.03%.", field_type="expense_ratio")
        checks = {c.name: c for c in check_golden_row(row, payload)}
        assert not checks["one_source_line"].passed
        assert not checks["has_freshness_line"].passed
        assert not checks["citation_matches_expected"].passed

    def test_two_source_lines_fail_the_single_citation_rule(self) -> None:
        row = load_rows(GOLDEN_PATH)[0]
        url = row["expect_source_url"]
        payload = _payload(
            text=f"Answer.\n\nSource: {url}\nSource: {url}\n"
            "Last updated from sources: 27 Sep 2026"
        )
        checks = {c.name: c for c in check_golden_row(row, payload)}
        assert not checks["one_source_line"].passed

    def test_four_sentences_fail_the_length_rule(self) -> None:
        row = load_rows(GOLDEN_PATH)[0]
        url = row["expect_source_url"]
        payload = _payload(
            text=f"One. Two. Three. Four.\n\nSource: {url}\n"
            "Last updated from sources: 27 Sep 2026"
        )
        checks = {c.name: c for c in check_golden_row(row, payload)}
        assert not checks["max_3_sentences"].passed

    def test_a_non_factual_route_fails(self) -> None:
        row = load_rows(GOLDEN_PATH)[0]
        checks = {c.name: c for c in check_golden_row(row, _payload(route=Route.ADVICE))}
        assert not checks["route_is_factual"].passed

    def test_cross_scheme_contamination_fails(self) -> None:
        row = load_rows(GOLDEN_PATH)[0]
        payload = _payload(field_type=row["expect_field_type"], scheme="hdfc_small_cap")
        checks = {c.name: c for c in check_golden_row(row, payload)}
        assert not checks["scheme_purity"].passed

    def test_wrong_field_type_fails(self) -> None:
        row = load_rows(GOLDEN_PATH)[0]
        payload = _payload(field_type="riskometer", scheme=row["scheme"])
        checks = {c.name: c for c in check_golden_row(row, payload)}
        assert not checks["top_chunk_field_type"].passed

    def test_inapplicable_checks_are_skipped_not_passed(self) -> None:
        """q07 has no expected URL or scheme, so those checks must be absent."""
        row = next(r for r in load_rows(GOLDEN_PATH) if r["id"] == "q07")
        assert row["expect_source_url"] is None and row["scheme"] is None
        names = {c.name for c in check_golden_row(row, _payload())}
        assert "citation_matches_expected" not in names
        assert "scheme_purity" not in names
        assert "top_chunk_field_type" in names  # still applicable

    def test_generation_checks_are_flagged_as_such(self) -> None:
        row = load_rows(GOLDEN_PATH)[0]
        flags = {c.name: c.needs_generation for c in check_golden_row(row, _payload())}
        assert flags["one_source_line"] is True
        assert flags["has_freshness_line"] is True
        assert flags["max_3_sentences"] is True
        assert flags["route_is_factual"] is False
        assert flags["scheme_purity"] is False

    def test_a_refusal_wearing_a_citation_fails(self) -> None:
        """The failure mode every other check here rewards.

        The citation and freshness lines are rendered from the top chunk's
        metadata, not from what the model said. So a model that declines to
        answer still produces a well-formed, correctly-cited, single-source,
        under-three-sentences payload - and passes everything above. Real
        golden q02 does exactly this while the exit load sits at rank 1 in the
        window. This check is the only one that notices.
        """
        row = load_rows(GOLDEN_PATH)[0]
        url = row["expect_source_url"]
        payload = _payload(
            text=(
                f"I do not have the {row['expect_field_type']} for this fund in my "
                f"sources.\n\nSource: {url}\nLast updated from sources: 27 Sep 2026"
            ),
            field_type=row["expect_field_type"],
            scheme=row["scheme"],
            url=url,
        )
        checks = {c.name: c for c in check_golden_row(row, payload)}
        assert not checks["cited_answer_is_not_a_refusal"].passed
        # Everything else passes, which is the whole point: this one is load-bearing.
        assert checks["one_source_line"].passed
        assert checks["has_freshness_line"].passed
        assert checks["citation_matches_expected"].passed
        assert checks["max_3_sentences"].passed
        assert checks["top_chunk_field_type"].passed

    def test_a_digitless_real_answer_is_not_treated_as_a_refusal(self) -> None:
        """q06 and q09 answer without a single digit. A digit rule would sink them.

        "The HDFC Small Cap Fund is rated Very High risk." and "Rahul Baijal is
        the current fund manager." are real answers. The check therefore matches
        refusal *phrasing*, never "contains no number".
        """
        row = load_rows(GOLDEN_PATH)[0]
        url = row["expect_source_url"]
        for answer in (
            "The HDFC Small Cap Fund Direct Growth is rated Very High risk.",
            "Rahul Baijal is the current fund manager of the fund.",
        ):
            payload = _payload(
                text=f"{answer}\n\nSource: {url}\nLast updated from sources: 27 Sep 2026",
                field_type=row["expect_field_type"],
                scheme=row["scheme"],
                url=url,
            )
            checks = {c.name: c for c in check_golden_row(row, payload)}
            assert checks["cited_answer_is_not_a_refusal"].passed, answer

    def test_a_refusal_without_a_citation_is_not_flagged_twice(self) -> None:
        """The no-context path is honest already: no prose refusal, no citation.

        Flagging it here would add noise to a row that is failing for the real
        reason - the corpus has nothing for that field.
        """
        row = next(r for r in load_rows(GOLDEN_PATH) if r["id"] == "q08")
        payload = _payload(
            text="I don't have that in my sources for these 5 schemes.",
            field_type=None,
        )
        checks = {c.name: c for c in check_golden_row(row, payload)}
        assert checks["cited_answer_is_not_a_refusal"].passed
        assert not checks["one_source_line"].passed  # still fails for the real reason

    def test_refusal_detection_covers_the_model_s_own_wording(self) -> None:
        """Canonical strings and model-authored refusals both count."""
        assert is_refusal_prose("I don't have that in my sources for these 5 schemes.")
        assert is_refusal_prose("I do not have the exit load in my sources.")
        assert is_refusal_prose("That isn't in my sources.")
        assert is_refusal_prose("I could not find that figure in my sources.")
        assert not is_refusal_prose("The expense ratio is 1.03%.")
        # A Source: line is not prose and must not create a false match.
        assert not is_refusal_prose(
            "The expense ratio is 1.03%.\n\n"
            "Source: https://example.invalid/x\n"
            "Last updated from sources: 27 Sep 2026"
        )


# ---------------------------------------------------------------------------
# Adversarial checks
# ---------------------------------------------------------------------------
class TestAdversarialChecks:
    @pytest.mark.parametrize(
        "row",
        [r for r in load_rows(ADVERSARIAL_PATH) if r["id"] != "a13"],
        ids=lambda r: r["id"],
    )
    def test_the_shipped_set_is_satisfied_by_the_router(self, row: dict) -> None:
        """Every deterministically-routable adversarial row must route as declared."""
        from rag.guardrails import route as route_query

        got = route_query(row["query"]).label.value
        assert got == row["expect_route"], f"{row['id']} routed {got}"

    def test_a13_is_pinned_as_a_known_misroute(self) -> None:
        """'NAV today' cannot be called out-of-scope without the LLM classifier.

        Pinned as FACTUAL so the gap stays visible in the suite: when an LLM
        classifier lands, this test should flip and a13 should start passing.
        """
        from rag.guardrails import route as route_query

        row = next(r for r in load_rows(ADVERSARIAL_PATH) if r["id"] == "a13")
        assert row["expect_route"] == "OUT_OF_SCOPE"
        assert route_query(row["query"]).label.value == "FACTUAL"

    def test_pii_answer_echoing_the_secret_fails(self) -> None:
        row = {"id": "x", "query": "My PAN is ABCDE1234F", "expect_route": "PII"}
        payload = _payload(
            route=Route.PII, text="Your PAN ABCDE1234F was noted.", url=None
        )
        checks = {c.name: c for c in check_adversarial_row(row, payload)}
        assert not checks["pii_not_echoed"].passed

    def test_clean_pii_answer_passes(self) -> None:
        row = {"id": "x", "query": "My PAN is ABCDE1234F", "expect_route": "PII"}
        payload = _payload(
            route=Route.PII,
            text="This assistant does not accept personal information.",
            url=None,
        )
        checks = {c.name: c for c in check_adversarial_row(row, payload)}
        assert all(c.passed for c in checks.values())

    def test_wrong_route_fails(self) -> None:
        row = {"id": "x", "query": "hi", "expect_route": "ADVICE"}
        checks = {c.name: c for c in check_adversarial_row(row, _payload())}
        assert not checks["route_matches"].passed

    def test_non_pii_rows_skip_the_echo_check(self) -> None:
        row = {"id": "x", "query": "What is the exit load?", "expect_route": "FACTUAL"}
        names = {c.name for c in check_adversarial_row(row, _payload())}
        assert names == {"route_matches"}


# ---------------------------------------------------------------------------
# Suite running and reporting
# ---------------------------------------------------------------------------
class TestRunSuite:
    def test_a_raising_ask_is_a_failed_row_not_a_crash(self) -> None:
        def boom(query: str) -> Any:
            raise RuntimeError("provider down")

        results = run_suite(
            "golden", [{"id": "q1", "query": "x"}], boom, checker=check_golden_row
        )
        assert len(results) == 1
        assert results[0].passed is False
        assert "ask_raised" in results[0].failures[0]

    def test_every_row_appears_in_the_results(self) -> None:
        rows = load_rows(ADVERSARIAL_PATH)
        results = run_suite(
            "adversarial", rows, lambda q: _payload(route=Route.PII), checker=check_adversarial_row
        )
        assert len(results) == len(rows)
        assert {r.row_id for r in results} == {r["id"] for r in rows}

    def test_failures_are_reported_with_reasons(self) -> None:
        row = load_rows(GOLDEN_PATH)[0]
        results = run_suite(
            "golden",
            [row],
            lambda q: _payload(text="no citation here"),
            checker=check_golden_row,
        )
        assert results[0].failures
        assert all(":" in f for f in results[0].failures)


class TestSummary:
    def _results(self) -> list[RowResult]:
        return [
            RowResult(
                suite="golden",
                row_id="q1",
                query="a",
                passed=True,
                checks=[Check("route_is_factual", True)],
                latency_ms=10.0,
            ),
            RowResult(
                suite="golden",
                row_id="q2",
                query="b",
                passed=False,
                checks=[Check("one_source_line", False, "none", needs_generation=True)],
                latency_ms=30.0,
            ),
        ]

    def test_pass_rate_is_exact(self) -> None:
        summary = summarise(self._results())
        assert summary["overall_pass_rate"] == 0.5
        assert summary["suites"]["golden"]["pass_rate"] == 0.5

    def test_generation_and_retrieval_checks_are_separated(self) -> None:
        tallies = summarise(self._results())["suites"]["golden"]
        assert tallies["retrieval_checks"] == {"passed": 1, "total": 1, "pass_rate": 1.0}
        assert tallies["generation_checks"] == {"passed": 0, "total": 1, "pass_rate": 0.0}

    def test_an_empty_suite_does_not_divide_by_zero(self) -> None:
        summary = summarise([])
        assert summary["overall_pass_rate"] == 0.0
        assert summary["latency_ms"]["p50"] == 0.0

    def test_percentiles_are_nearest_rank(self) -> None:
        results = [
            RowResult("golden", f"q{i}", "x", True, latency_ms=float(i)) for i in range(1, 101)
        ]
        assert summarise(results)["latency_ms"]["p50"] == 50.0
        assert summarise(results)["latency_ms"]["p95"] == 95.0

    def test_recall_is_reported_per_k(self) -> None:
        row = {"id": "q1", "query": "x", "expect_field_type": "expense_ratio"}
        results = run_suite(
            "golden", [row], lambda q: _payload(), ks=(1, 3), checker=check_golden_row
        )
        stats = summarise(results, (1, 3))["suites"]["golden"]
        assert stats["recall@1"] == 1.0
        assert stats["recall@1_hits"] == "1/1"

    def test_recall_is_skipped_when_no_field_is_expected(self) -> None:
        row = {"id": "q1", "query": "x", "expect_field_type": None}
        results = run_suite("golden", [row], lambda q: _payload(), checker=check_golden_row)
        assert results[0].hits_at == {}


class TestReport:
    def test_report_records_every_failure_with_a_reason(self) -> None:
        results = [
            RowResult(
                "golden",
                "q1",
                "x",
                False,
                checks=[Check("one_source_line", False, "no source", needs_generation=True)],
                degraded=True,
            )
        ]
        report = build_report(results)
        assert report["failures"][0]["id"] == "q1"
        assert report["failures"][0]["reasons"] == ["one_source_line: no source"]
        assert report["failures"][0]["degraded"] is True

    def test_a_degraded_run_is_annotated(self) -> None:
        results = [
            RowResult("golden", "q1", "x", False, checks=[], degraded=True),
        ]
        report = build_report(results)
        assert report["notes"], "a degraded run must say so in the report"
        assert "degraded" in report["notes"][0]

    def test_a_clean_run_has_no_notes(self) -> None:
        report = build_report([RowResult("golden", "q1", "x", True, checks=[])])
        assert report["notes"] == []

    def test_report_is_json_serialisable(self) -> None:
        report = build_report([RowResult("golden", "q1", "x", True, checks=[])])
        assert json.loads(json.dumps(report))["overall_n"] == 1

    def test_report_records_the_provider_and_model(self) -> None:
        report = build_report([RowResult("golden", "q1", "x", True, checks=[])])
        assert "provider" in report and "model" in report
        assert report["generated_at"]


# ---------------------------------------------------------------------------
# Deliverables
# ---------------------------------------------------------------------------
class TestSourceList:
    def test_the_committed_csv_has_the_exact_d2_header(self) -> None:
        path = PROJECT_ROOT / "docs" / "source_list.csv"
        assert path.is_file()
        with path.open(encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle))
        assert header == [
            "source_id",
            "scheme",
            "doc_type",
            "title",
            "publisher",
            "url",
            "retrieved_at",
            "sha256",
            "status",
        ]

    def test_every_manifest_document_has_a_row(self) -> None:
        manifest = json.loads(
            (PROJECT_ROOT / "data" / "manifest.json").read_text(encoding="utf-8")
        )
        with (PROJECT_ROOT / "docs" / "source_list.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len(manifest["documents"])
        assert {r["source_id"] for r in rows} == {
            d["source_id"] for d in manifest["documents"]
        }

    def test_every_row_has_a_url_and_status(self) -> None:
        with (PROJECT_ROOT / "docs" / "source_list.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                assert row["url"].startswith("http")
                assert row["status"] in {"ok", "failed"}
                assert row["retrieved_at"]

    def test_generator_marks_a_thin_document_as_failed(self, tmp_path: Path) -> None:
        manifest = tmp_path / "m.json"
        manifest.write_text(
            json.dumps(
                {
                    "documents": [
                        {
                            "source_id": "a",
                            "scheme": "s",
                            "doc_type": "scheme_page",
                            "title": "t",
                            "publisher": "p",
                            "source_url": "https://x/y",
                            "retrieved_at": "2026-01-01T00:00:00+00:00",
                            "sha256": "abc",
                            "content_chars": 10,
                        },
                        {
                            "source_id": "b",
                            "scheme": "s",
                            "doc_type": "scheme_page",
                            "title": "t",
                            "publisher": "p",
                            "source_url": "https://x/z",
                            "retrieved_at": "2026-01-01T00:00:00+00:00",
                            "sha256": "def",
                            "content_chars": 0,
                            "thin": True,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        out = tmp_path / "out.csv"
        assert write_source_list(manifest, out) == 2
        with out.open(encoding="utf-8", newline="") as handle:
            statuses = [r["status"] for r in csv.DictReader(handle)]
        assert statuses == ["ok", "failed"], "a failed row must never be dropped"


class TestSampleQa:
    def test_the_committed_file_exists_and_covers_all_four_behaviours(self) -> None:
        text = (PROJECT_ROOT / "docs" / "sample_qa.md").read_text(encoding="utf-8")
        assert "Cited factual answer" in text
        assert "Advice refusal" in text
        assert "Performance refusal" in text
        assert "PII rejection" in text

    def test_it_holds_between_five_and_ten_queries(self) -> None:
        text = (PROJECT_ROOT / "docs" / "sample_qa.md").read_text(encoding="utf-8")
        headings = [l for l in text.splitlines() if l.startswith("## ")]
        assert 5 <= len(headings) <= 10

    def test_every_entry_shows_a_real_citation_or_a_real_refusal(self) -> None:
        text = (PROJECT_ROOT / "docs" / "sample_qa.md").read_text(encoding="utf-8")
        assert "Source: http" in text, "D4 must evidence the citation contract"

    def test_generator_transcribes_the_answer_verbatim(self, tmp_path: Path) -> None:
        answer = "I do not give investment recommendations.\n\nSource: https://s/x"
        results = [
            RowResult("adversarial", "a01", "Should I buy?", True, answer=answer, route="ADVICE")
        ]
        out = tmp_path / "qa.md"
        assert write_sample_qa(results, out) == 1
        assert answer in out.read_text(encoding="utf-8")

    def test_generator_flags_a_service_failure_as_degraded(self, tmp_path: Path) -> None:
        results = [
            RowResult(
                "golden",
                "q01",
                "expense ratio",
                False,
                answer="I couldn't answer right now.",
                route="FACTUAL",
                degraded=True,
                service_failure=True,
                checks=[Check("one_source_line", False, "no source")],
            )
        ]
        out = tmp_path / "qa.md"
        write_sample_qa(results, out)
        text = out.read_text(encoding="utf-8")
        assert "degraded - generation failed" in text
        assert "failing check" in text

    def test_generator_does_not_call_a_contract_refusal_an_outage(
        self, tmp_path: Path
    ) -> None:
        """A 'not in my sources' refusal is the contract working, not a failure.

        Labelling it "degraded - generation failed" would put a false outage
        into a deliverable.
        """
        results = [
            RowResult(
                "golden",
                "q06",
                "lock-in period",
                False,
                answer="I don't have that in my sources for these 5 schemes.",
                route="FACTUAL",
                degraded=True,
                service_failure=False,
                checks=[Check("top_chunk_field_type", False, "top chunk is 'scheme_type'")],
            )
        ]
        out = tmp_path / "qa.md"
        assert write_sample_qa(results, out) == 1, "q06 must be a sample id"
        text = out.read_text(encoding="utf-8")
        assert "refused by the contract" in text
        assert "generation failed" not in text

    def test_generator_reports_the_run_pass_rate(self, tmp_path: Path) -> None:
        report = build_report([RowResult("golden", "q1", "x", True, checks=[])])
        results = [RowResult("adversarial", "a01", "q", True, answer="x", route="ADVICE")]
        out = tmp_path / "qa.md"
        write_sample_qa(results, out, report=report)
        assert "100.0%" in out.read_text(encoding="utf-8")

    def test_sample_ids_all_exist_in_the_suites(self) -> None:
        known = {r["id"] for r in load_rows(GOLDEN_PATH)} | {
            r["id"] for r in load_rows(ADVERSARIAL_PATH)
        }
        assert set(SAMPLE_QA_IDS) <= known


class TestReadme:
    def test_the_required_sections_exist(self) -> None:
        text = (PROJECT_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
        for heading in (
            "## SCOPE",
            "## SETUP",
            "## ARCHITECTURE",
            "## CHUNKING",
            "## TUNING",
            "## TECH STACK",
            "## GUARDRAILS",
            "## DELIVERABLES",
            "## KNOWN LIMITS",
            "## REPRODUCIBILITY",
        ):
            assert heading in text, f"README is missing {heading}"

    def test_the_disclaimer_line_is_up_top(self) -> None:
        text = (PROJECT_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
        assert "Facts-only. No investment advice." in text[:600]

    def test_it_embeds_the_chunking_rationale(self) -> None:
        """The CHUNKING section must carry the rationale verbatim (headings demoted).

        Loose phrase checks let a stale copy survive; requiring the whole body
        means editing one file without the other fails the suite.
        """
        readme = (PROJECT_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
        rationale = (PROJECT_ROOT / "docs" / "chunking_rationale.md").read_text(
            encoding="utf-8"
        )
        demoted: list[str] = []
        for line in rationale.splitlines():
            if line.startswith("# "):
                continue
            if re.match(r"^#{1,5} ", line):
                line = "#" + line
            demoted.append(line)
        body = "\n".join(demoted).strip()
        assert len(body) > 2000, "rationale looks truncated"
        assert body in readme, (
            "docs/README.md's CHUNKING section has drifted from "
            "docs/chunking_rationale.md; re-copy it"
        )

    def test_every_number_it_quotes_comes_from_results_json(self) -> None:
        """The README must not drift from the run it quotes.

        Each row of the measured-results table is re-derived from
        ``eval/results.json`` and required to appear verbatim, so a re-run that
        changes the numbers cannot leave a stale README behind.
        """
        readme = (PROJECT_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
        results_path = PROJECT_ROOT / "eval" / "results.json"
        if not results_path.is_file():
            pytest.skip("no results.json yet")
        report = json.loads(results_path.read_text(encoding="utf-8"))

        def frac(n: int, d: int) -> str:
            return f"**{n}/{d} ({n / d * 100:.1f}%)**"

        expected = [
            frac(report["overall_passed"], report["overall_n"]),
            report["generated_at"],
        ]
        for stats in report["suites"].values():
            expected.append(frac(stats["passed"], stats["n"]))
            for key in ("retrieval_checks", "generation_checks"):
                tally = stats.get(key)
                if tally:
                    expected.append(
                        f"{tally['passed']}/{tally['total']} "
                        f"({tally['pass_rate'] * 100:.1f}%)"
                    )
            for key, value in stats.items():
                if key.startswith("recall@") and key.endswith("_hits"):
                    expected.append(value)
        degraded = sum(s["degraded"] for s in report["suites"].values())
        expected.append(f"{degraded} of {report['overall_n']}")

        missing = [e for e in expected if e not in readme]
        assert not missing, f"README does not quote these from the run: {missing}"

    def test_latency_is_reported_but_not_pinned(self) -> None:
        """Latency varies per run, so the README quotes it without a hard check.

        Pinning exact milliseconds would fail on every re-run and train us to
        ignore this test. The ordering invariant is the part worth asserting.
        """
        results_path = PROJECT_ROOT / "eval" / "results.json"
        if not results_path.is_file():
            pytest.skip("no results.json yet")
        latency = json.loads(results_path.read_text(encoding="utf-8"))["latency_ms"]
        assert 0 < latency["p50"] <= latency["p95"], "p50 should not exceed p95"
        readme = (PROJECT_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
        assert re.search(r"Latency p50 / p95", readme)

    def test_setup_covers_both_platforms(self) -> None:
        text = (PROJECT_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
        assert "PowerShell" in text
        assert "Linux" in text
        assert "pip install -r requirements.txt" in text
        assert "90 MB" in text

    def test_it_links_all_five_deliverables(self) -> None:
        text = (PROJECT_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
        for target in ("source_list.csv", "README.md", "sample_qa.md", "disclaimer.md"):
            assert target in text


class TestDisclaimerDeliverable:
    def test_disclaimer_file_matches_the_ui_constant(self) -> None:
        import app

        assert app.DISCLAIMER in (
            PROJECT_ROOT / "docs" / "disclaimer.md"
        ).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def corpus_chunks() -> list[dict]:
    path = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def readme_text() -> str:
    return (PROJECT_ROOT / "docs" / "README.md").read_text(encoding="utf-8")


def _readme_section(text: str, heading: str, next_heading: str) -> str:
    return text[text.index(heading) : text.index(next_heading)]


class TestCorpusClaims:
    """Re-derive the README's corpus numbers instead of trusting the prose.

    Two stale figures survived into the README (the 511-vs-488 chunk split, and
    "returns are dropped at ingest") precisely because nothing checked them
    against the data. Every count quoted in KNOWN LIMITS is recomputed here.
    """

    def test_it_quotes_the_real_chunk_total(self, corpus_chunks, readme_text) -> None:
        assert f"{len(corpus_chunks)} chunks" in readme_text

    def test_it_quotes_the_real_other_chunk_count(
        self, corpus_chunks, readme_text
    ) -> None:
        other = sum(1 for c in corpus_chunks if c["field_type"] == "other")
        assert f"{other} of {len(corpus_chunks)}" in readme_text, (
            f"stale 'other' count: {other}"
        )

    def test_it_distinguishes_other_chunks_from_the_documents_total(
        self, corpus_chunks, readme_text
    ) -> None:
        """The 511/488 split is easy to state wrongly; assert both, correctly."""
        by_source: dict[str, list[dict]] = {}
        for chunk in corpus_chunks:
            by_source.setdefault(chunk["source_id"], []).append(chunk)
        worst = max(by_source.items(), key=lambda kv: len(kv[1]))
        source_id, rows = worst
        total = len(rows)
        other = sum(1 for c in rows if c["field_type"] == "other")
        assert total != other, "the split is only interesting if the two differ"
        assert f"{other} of those" in readme_text, (
            f"stale 'other' figure for {source_id}"
        )
        assert f"{total} of the {len(corpus_chunks)} chunks in total" in readme_text, (
            f"stale document total for {source_id}: {total}"
        )

    def test_it_does_not_claim_returns_are_removed_at_ingest(
        self, readme_text
    ) -> None:
        """Performance data is in the corpus; the guard is at query time.

        Scoped to KNOWN LIMITS: the embedded chunking rationale legitimately
        says the chunker drops *fields* at ingest, which is a different claim.
        """
        limits = _readme_section(readme_text, "## KNOWN LIMITS", "## REPRODUCIBILITY")
        assert "Returns are dropped at ingest" not in limits
        assert "query time, not at" in limits

    def test_the_embedded_rationale_may_still_describe_ingest_dropping(
        self, readme_text
    ) -> None:
        """Guard the guard: the prohibition must not be over-broad."""
        assert "dropped at ingest" in readme_text

    def test_the_unanswerable_golden_rows_are_the_real_ones(
        self, corpus_chunks, readme_text
    ) -> None:
        have = {c["field_type"] for c in corpus_chunks}
        rows = load_rows(GOLDEN_PATH)
        missing = {
            r["id"]
            for r in rows
            if r.get("expect_field_type") and r["expect_field_type"] not in have
        }
        assert len(missing) == 4, f"expected 4 unanswerable rows, got {missing}"
        ceiling = len(rows) - len(missing)
        limits = _readme_section(readme_text, "## KNOWN LIMITS", "## REPRODUCIBILITY")
        flat = " ".join(limits.split())
        for row_id in sorted(missing):
            assert row_id in limits, f"README omits unanswerable row {row_id}"
        assert f"caps recall at {ceiling}/10" in flat, (
            f"README should state the {ceiling}/10 recall ceiling"
        )

    def test_every_ingested_document_is_a_groww_scheme_page(self) -> None:
        manifest = json.loads(
            (PROJECT_ROOT / "data" / "manifest.json").read_text(encoding="utf-8")
        )
        for doc in manifest["documents"]:
            assert doc["publisher"] == "groww"
            assert doc["doc_type"] == "scheme_page"

    def test_the_manifest_holds_five_of_the_five_schemes(self) -> None:
        from rag.schemas import SCHEME_KEYS

        manifest = json.loads(
            (PROJECT_ROOT / "data" / "manifest.json").read_text(encoding="utf-8")
        )
        assert {d["scheme"] for d in manifest["documents"]} == SCHEME_KEYS

    def test_it_names_the_real_number_of_disabled_placeholder_rows(
        self, readme_text
    ) -> None:
        with (PROJECT_ROOT / "ingest" / "sources.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        disabled = sum(1 for r in rows if r["url"] == "TODO_PRIMARY_URL")
        assert disabled == 6
        flat = " ".join(readme_text.split())
        assert f"but six of them are currently disabled" in flat

    def test_config_values_it_quotes_are_the_real_ones(self, readme_text) -> None:
        import yaml

        cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
        chunking, retrieval = cfg["chunking"], cfg["retrieval"]
        assert f"`target_tokens={chunking['target_tokens']}`" in readme_text
        assert f"`min_tokens={chunking['min_tokens']}`" in readme_text
        assert f"`max_tokens={chunking['max_tokens']}`" in readme_text
        assert f"`overlap_pct={chunking['overlap_pct']}`" in readme_text
        for key in (
            "top_k",
            "over_fetch",
            "similarity_floor",
            "field_type_boost",
            "scheme_filter",
        ):
            assert f"`retrieval.{key}`" in readme_text, f"README omits {key}"
        flat = " ".join(readme_text.split())
        assert f"`retrieval.top_k` | {retrieval['top_k']} |" in flat

    def test_the_scope_table_urls_come_from_the_scheme_registry(
        self, readme_text
    ) -> None:
        import dataclasses

        from rag.schemas import SCHEMES

        for spec in SCHEMES:
            row = dataclasses.asdict(spec)
            assert row["url"] in readme_text, f"README omits {row['key']} url"
            assert row["display_name"] in readme_text
            pretty = row["category"].replace("_", " ")
            assert pretty in readme_text.lower(), f"README omits category {pretty}"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
