"""Router regression tests for over-broad keyword triggers (Phase 8 pre-work).

Every case here is a mis-route found by actually asking the corpus questions.
The corpus is five *Direct Growth* plans, so a bare ``growth`` trigger sent
almost every real question to PERFORMANCE and refused it before retrieval ran.
"""

from __future__ import annotations

import pytest

from rag.guardrails import Route, detect_pii, route


def _label(query: str) -> str:
    return route(query, allowlist=set()).label.value


class TestPlanNameIsNotAPerformanceClaim:
    """"Direct Growth" names a plan. It must not route as performance."""

    @pytest.mark.parametrize(
        "query",
        [
            "What is the expense ratio of HDFC Large Cap Fund - Direct Growth?",
            "What is the exit load on Direct Growth?",
            "What is the NAV of Direct Growth?",
            "What is the AUM of Direct Growth?",
            "What is the expense ratio of Direct Plan Growth?",
            "What is the exit load of the Growth Option?",
            "expense ratio of Direct Growth",
            "Direct Growth",
        ],
    )
    def test_plan_name_stays_factual(self, query: str) -> None:
        assert _label(query) == Route.FACTUAL.value

    def test_a_real_growth_question_still_routes_to_performance(self) -> None:
        assert _label("What is the growth of HDFC Large Cap?") == Route.PERFORMANCE.value

    @pytest.mark.parametrize(
        "query",
        [
            "How has the fund grown?",
            "How did HDFC Large Cap perform?",
            "How does this perform against its benchmark?",
            "What is the 5 year return of HDFC Large Cap?",
            "What is the CAGR of HDFC Large Cap?",
        ],
    )
    def test_genuine_performance_intent_is_preserved(self, query: str) -> None:
        assert _label(query) == Route.PERFORMANCE.value


class TestWordBoundaryMatching:
    """A trigger must not match inside an unrelated word."""

    @pytest.mark.parametrize(
        ("text", "term"),
        [
            ("is this a bargain?", "gain"),
            ("how does it perform against peers?", "gain"),
            ("frankly speaking", "rank"),
            ("the target is 5 years", "get"),
        ],
    )
    def test_substring_collisions_do_not_trigger(self, text: str, term: str) -> None:
        from rag.guardrails import _has_term

        assert not _has_term(text, (term,)), f"{term!r} matched inside {text!r}"

    def test_a_real_gain_question_still_matches(self) -> None:
        assert _label("What is the gain so far?") == Route.PERFORMANCE.value


class TestGenericWordsAreNotAdviceTriggers:
    """'target' is far too generic to imply advice on its own."""

    def test_target_expense_ratio_is_factual(self) -> None:
        assert _label("What is the target expense ratio?") == Route.FACTUAL.value

    def test_target_allocation_is_still_advice(self) -> None:
        assert _label("What is my target allocation?") == Route.ADVICE.value

    def test_allocation_alone_is_still_advice(self) -> None:
        assert _label("How should I allocate 5 lakhs?") == Route.ADVICE.value


class TestDemoScriptRoutes:
    """The three Phase 8 example questions must route as the demo expects."""

    def test_example_a_is_a_cited_fact(self) -> None:
        assert (
            _label("What is the expense ratio of HDFC Large Cap Fund - Direct Growth?")
            == Route.FACTUAL.value
        )

    def test_example_b_is_a_cited_fact(self) -> None:
        assert (
            _label("What is the lock-in period for HDFC ELSS Tax Saver Fund?")
            == Route.FACTUAL.value
        )

    def test_example_c_is_a_refusal(self) -> None:
        assert _label("Should I buy HDFC Small Cap Fund?") == Route.ADVICE.value


class TestPiiStillWinsOverEverything:
    def test_pii_outranks_a_plan_name(self) -> None:
        assert _label("Direct Growth PAN ABCDE1234F") == Route.PII.value

    def test_pii_outranks_performance(self) -> None:
        assert _label("5 year return, mail me at a@b.com") == Route.PII.value


class TestLongCardNumbersAreCaught:
    """A 13-19 digit card number matched no PII class before this was fixed.

    It is too long for OTP, too long for Aadhaar, and a 16-digit run has no
    6-12 digit sub-run bounded by non-digits, so it slipped through to the LLM
    and the logs.
    """

    @pytest.mark.parametrize(
        "query",
        [
            "my card is 4111111111111111",
            "card number 4539578763621486",
            "credit card 5234567890123456",
        ],
    )
    def test_card_number_is_detected(self, query: str) -> None:
        assert "ACCOUNT" in detect_pii(query)

    def test_card_number_routes_to_pii_not_factual(self) -> None:
        assert _label("What is the NAV? card 4111111111111111") == Route.PII.value

    def test_a_real_pan_still_matches_pan(self) -> None:
        assert "PAN" in detect_pii("My PAN is ABCDE1234F")


class TestEveryCorpusSlugIsPlanNamed:
    """Guards the class of bug: all five corpus pages are Direct Growth plans."""

    @pytest.mark.parametrize(
        "query",
        [
            "expense ratio of hdfc-large-cap-fund-direct-growth",
            "exit load of hdfc-small-cap-fund-direct-growth",
            "lock-in of hdfc-elss-tax-saver-fund-direct-plan-growth",
            "NAV of hdfc-balanced-advantage-fund-direct-growth",
        ],
    )
    def test_url_style_plan_names_are_factual(self, query: str) -> None:
        assert _label(query) == Route.FACTUAL.value


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
