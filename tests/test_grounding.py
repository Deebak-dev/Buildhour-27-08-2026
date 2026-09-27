"""Figure fidelity, gate G6 — the defence against numeric drift (risk R7).

A sentence embedding happily supports "the expense ratio is 1.85%" when the
context says 1.03%: the two sentences are about the same thing, so the model is
being faithful to the *topic* and wrong about the *fact*. Nothing upstream stops
that, so G6 compares every numeric token in the answer against the context and
refuses on any mismatch.
"""

from __future__ import annotations

import pytest

from rag.answer import build_context
from rag.config import AnswerConfig, GenerationConfig
from rag.schemas import Chunk, RetrievalTrace, ScoredChunk
from rag.validate import _numbers_in, render_contract, validate
from tests.test_contract import (
    ALLOWLIST,
    GOOD_URL,
    make_chunk,
    make_trace,
    payload_text,
)

ANSWER_CFG = AnswerConfig()
GEN_CFG = GenerationConfig()


class TestNumberExtraction:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("1.03%", {"1.03"}),
            ("1,447.38", {"1447.38"}),
            ("Rs 500", {"500"}),
            ("₹100 and 1.03%", {"100", "1.03"}),
            ("no digits here", set()),
            ("", set()),
            ("3 years", {"3"}),
        ],
    )
    def test_extracts_and_normalises(self, text: str, expected: set[str]) -> None:
        assert _numbers_in(text) == expected

    def test_strips_commas_and_percent(self) -> None:
        assert _numbers_in("1,447.38") == _numbers_in("1447.38%")

    def test_distinguishes_different_values(self) -> None:
        assert _numbers_in("1.03%") != _numbers_in("1.85%")


class TestInventedNumbers:
    def test_invented_number_is_refused(self) -> None:
        """The R7 case: context says 1.03%, the draft says 1.85%."""
        trace = make_trace(make_chunk(text="Expense ratio: 1.03%"))
        draft = payload_text()
        draft.text = "The expense ratio is 1.85%."

        payload = validate(
            draft, "expense ratio", trace, allowlist=ALLOWLIST, gen_cfg=GEN_CFG
        )
        assert payload.degraded is True
        assert "G6" not in payload.validators_passed
        assert "1.85" not in payload.text

    def test_faithful_numbers_pass(self) -> None:
        trace = make_trace(make_chunk(text="Expense ratio: 1.03%"))
        draft = payload_text()
        draft.text = "The expense ratio is 1.03%."

        payload = validate(
            draft, "expense ratio", trace, allowlist=ALLOWLIST, gen_cfg=GEN_CFG
        )
        assert payload.degraded is False
        assert "1.03%" in payload.text

    def test_every_number_must_be_present_not_just_one(self) -> None:
        """One faithful number does not excuse a second invented one."""
        trace = make_trace(make_chunk(text="Expense ratio: 1.03%"))
        draft = payload_text()
        draft.text = "The expense ratio is 1.03% and the minimum SIP is ₹500."

        payload = validate(
            draft, "expense ratio", trace, allowlist=ALLOWLIST, gen_cfg=GEN_CFG
        )
        assert payload.degraded is True
        assert "G6" not in payload.validators_passed

    def test_a_number_from_any_retrieved_chunk_is_allowed(self) -> None:
        """Fidelity is judged against the whole context, not just the winner."""
        trace = make_trace(
            make_chunk(text="Expense ratio: 1.03%", chunk_id="a" * 16),
            make_chunk(
                text="Min. for SIP: ₹100",
                field_type="min_sip",
                chunk_id="b" * 16,
            ),
        )
        draft = payload_text()
        draft.text = "The expense ratio is 1.03% and the minimum SIP is ₹100."

        payload = validate(
            draft, "expense ratio", trace, allowlist=ALLOWLIST, gen_cfg=GEN_CFG
        )
        assert payload.degraded is False

    def test_rounding_is_still_drift(self) -> None:
        """1.0 is not 1.03. G6 compares tokens, not tolerances."""
        trace = make_trace(make_chunk(text="Expense ratio: 1.03%"))
        draft = payload_text()
        draft.text = "The expense ratio is about 1.0%."

        payload = validate(
            draft, "expense ratio", trace, allowlist=ALLOWLIST, gen_cfg=GEN_CFG
        )
        assert payload.degraded is True
        assert "G6" not in payload.validators_passed

    def test_thousands_separators_do_not_trip_the_gate(self) -> None:
        """A formatting difference is not figure drift."""
        trace = make_trace(make_chunk(text="NAV as of 27 Sep 2026 is ₹1,447.38."))
        draft = payload_text(grounded_field="nav")
        draft.text = "The NAV is ₹1,447.38."

        payload = validate(
            draft, "nav", trace, allowlist=ALLOWLIST, gen_cfg=GEN_CFG
        )
        assert payload.degraded is False

    def test_date_in_the_answer_must_come_from_the_context(self) -> None:
        trace = make_trace(make_chunk(text="NAV as of 27 Sep 2026 is ₹1,447.38."))
        draft = payload_text()
        draft.text = "The NAV is ₹1,447.38 on 01 Jan 2020."

        payload = validate(
            draft, "nav", trace, allowlist=ALLOWLIST, gen_cfg=GEN_CFG
        )
        assert payload.degraded is True
        assert "G6" not in payload.validators_passed

    def test_an_answer_with_no_numbers_always_passes_g6(self) -> None:
        trace = make_trace(make_chunk(text="Fund benchmark: NIFTY 100 Total Return Index"))
        draft = payload_text(grounded_field="benchmark")
        draft.text = "The fund benchmark is the NIFTY 100 Total Return Index."

        payload = validate(
            draft, "benchmark", trace, allowlist=ALLOWLIST, gen_cfg=GEN_CFG
        )
        assert payload.degraded is False


class TestGroundingGate:
    def test_mismatched_grounded_field_downgrades(self) -> None:
        """G8: the model claims a different field than the question asked for."""
        trace = make_trace(make_chunk(text="Expense ratio: 1.03%"))
        draft = payload_text(grounded_field="exit_load")

        payload = validate(
            draft, "expense ratio", trace, allowlist=ALLOWLIST, gen_cfg=GEN_CFG
        )
        assert payload.degraded is True
        assert "G8" not in payload.validators_passed

    def test_null_grounded_field_is_tolerated(self) -> None:
        trace = make_trace(make_chunk(text="Expense ratio: 1.03%"))
        draft = payload_text(grounded_field=None)

        payload = validate(
            draft, "expense ratio", trace, allowlist=ALLOWLIST, gen_cfg=GEN_CFG
        )
        assert payload.degraded is False

    def test_matching_grounded_field_passes(self) -> None:
        trace = make_trace(make_chunk(text="Expense ratio: 1.03%"))
        draft = payload_text(grounded_field="expense_ratio")

        payload = validate(
            draft, "expense ratio", trace, allowlist=ALLOWLIST, gen_cfg=GEN_CFG
        )
        assert payload.degraded is False
        assert "G8" in payload.validators_passed


class TestContextIsWhatGetsCompared:
    def test_context_contains_the_chunk_numbers(self) -> None:
        trace = make_trace(make_chunk(text="Expense ratio: 1.03%"))
        context = build_context(trace)
        assert "1.03" in _numbers_in(context)

    def test_explicit_context_text_overrides_the_rebuild(self) -> None:
        """A caller that generated with different context can pass it in."""
        trace = make_trace(make_chunk(text="Expense ratio: 1.03%"))
        payload = validate(
            payload_text(),
            "expense ratio",
            trace,
            allowlist=ALLOWLIST,
            gen_cfg=GEN_CFG,
            context_text="[1] (scheme=hdfc_large_cap field_type=expense_ratio)\nExpense ratio: 9.99%",
        )
        assert payload.degraded is True
        assert "G6" not in payload.validators_passed


class TestEndToEndShape:
    def test_final_text_has_the_documented_shape(self) -> None:
        trace = make_trace(make_chunk(text="Expense ratio: 1.03%"))
        payload = validate(
            payload_text(), "expense ratio", trace, allowlist=ALLOWLIST, gen_cfg=GEN_CFG
        )
        # Prose / blank / one citation / freshness, exactly as the guide shows.
        lines = payload.text.splitlines()
        assert lines[0].endswith(".")
        assert lines[1] == ""
        assert lines[2] == f"Source: {GOOD_URL}"
        assert lines[3].startswith("Last updated from sources: ")
        assert len(lines) == 4

    def test_freshness_comes_from_metadata_not_the_model(self) -> None:
        trace = make_trace(make_chunk(retrieved_at="2026-03-04T05:06:07+00:00"))
        payload = validate(
            payload_text(), "expense ratio", trace, allowlist=ALLOWLIST, gen_cfg=GEN_CFG
        )
        assert "04 Mar 2026" in payload.text
        assert payload.last_updated == "04 Mar 2026"

    def test_render_contract_is_pure(self) -> None:
        """Rendering must not depend on a trace, a draft or any global state."""
        rendered = render_contract(
            "A fact.",
            source_url=GOOD_URL,
            retrieved_at="2026-01-01T00:00:00+00:00",
            answer_cfg=ANSWER_CFG,
        )
        assert rendered.startswith("A fact.")
        assert rendered.count("http") == 1


class TestMultiChunkRealism:
    def test_noise_chunks_do_not_license_invented_numbers(self) -> None:
        """Holdings rows are in the context, but they do not cover a fee figure."""
        trace = make_trace(
            make_chunk(
                text="Expense ratio: 1.03%\n\nEquity: 0.63%\n\nRepo: 2.55%",
                chunk_id="a" * 16,
            ),
        )
        draft = payload_text()
        draft.text = "The expense ratio is 1.03% and the repo holding is 9.99%."

        payload = validate(
            draft, "expense ratio", trace, allowlist=ALLOWLIST, gen_cfg=GEN_CFG
        )
        assert payload.degraded is True
        assert "G6" not in payload.validators_passed

    def test_number_from_a_holdings_row_is_still_faithful(self) -> None:
        trace = make_trace(
            make_chunk(
                text="Expense ratio: 1.03%\n\nEquity: 0.63%\n\nRepo: 2.55%",
                chunk_id="a" * 16,
            ),
        )
        draft = payload_text()
        draft.text = "The equity holding is 0.63%."

        payload = validate(
            draft, "expense ratio", trace, allowlist=ALLOWLIST, gen_cfg=GEN_CFG
        )
        assert payload.degraded is False


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
