"""The PII screen and the hybrid router (Phase 7 exit gate, FR-7.1 to FR-7.5).

These are deterministic tests: no network, no API key, and the classifier is a
stub wherever the deterministic rules are supposed to decide. That split is the
point of the design, so the tests assert *which stage* produced each label rather
than only what the label was.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from rag.guardrails import PII_CLASSES, detect_pii, pii_response, route
from rag.schemas import Route

#: The adversarial fixtures carry fabricated identifiers; see the file header.
ADVERSARIAL = pathlib.Path(__file__).resolve().parents[1] / "eval" / "adversarial_set.jsonl"


class StubClassifier:
    """A classifier that returns a fixed label and counts its calls."""

    def __init__(self, reply: str = "FACTUAL") -> None:
        self.reply = reply
        self.calls = 0

    def complete(self, system: str, user: str, *, json_mode: bool = True) -> str:
        self.calls += 1
        return self.reply


class ExplodingClassifier:
    """A classifier that fails, standing in for an unreachable provider."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, user: str, *, json_mode: bool = True) -> str:
        self.calls += 1
        raise RuntimeError("provider unreachable")


# ---------------------------------------------------------------------------
# The six classes
# ---------------------------------------------------------------------------


class TestPiiClasses:
    def test_all_six_classes_are_defined(self) -> None:
        assert PII_CLASSES == ("PAN", "AADHAAR", "ACCOUNT", "OTP", "EMAIL", "PHONE")

    def test_pan(self) -> None:
        assert "PAN" in detect_pii("My PAN is ABCDE1234F")

    def test_aadhaar_with_a_valid_verhoef_checksum(self) -> None:
        assert "AADHAAR" in detect_pii("my aadhaar is 2345 6789 0121")

    def test_account_and_folio(self) -> None:
        assert "ACCOUNT" in detect_pii("my account number is 12345678")
        assert "ACCOUNT" in detect_pii("folio number 12345678")
        assert "ACCOUNT" in detect_pii("demat number 12345678")

    def test_otp(self) -> None:
        assert "OTP" in detect_pii("OTP is 448291")

    def test_email(self) -> None:
        assert "EMAIL" in detect_pii("email me at ramesh@example.com")

    def test_phone(self) -> None:
        assert "PHONE" in detect_pii("call me on 9876543210")
        assert "PHONE" in detect_pii("call me on +91 9876543210")

    def test_multiple_classes_are_all_reported(self) -> None:
        found = detect_pii("OTP is 448291, and my account number is 76543210")
        assert found == ["ACCOUNT", "OTP"]

    def test_result_order_is_stable(self) -> None:
        query = "PAN ABCDE1234F and email ramesh@example.com and phone 9876543210"
        assert detect_pii(query) == ["PAN", "EMAIL", "PHONE"]


# ---------------------------------------------------------------------------
# Safeguard 1: keyword gating
# ---------------------------------------------------------------------------


class TestKeywordGating:
    def test_a_bare_number_is_not_an_account(self) -> None:
        assert detect_pii("the figure is 12345678") == []

    def test_a_bare_number_is_not_an_otp(self) -> None:
        assert detect_pii("the value is 448291") == []

    @pytest.mark.parametrize(
        "query",
        [
            "minimum SIP is \u20b9500",
            "expense ratio is 1.5%",
            "lock-in is 3 years",
            "80C limit is \u20b9150000",
            "10 year horizon",
            "SIP of 5000 per month",
        ],
    )
    def test_the_six_safe_context_strings_are_clean(self, query: str) -> None:
        """The exit gate names these six explicitly."""
        assert detect_pii(query) == []

    def test_account_keyword_is_required(self) -> None:
        assert "ACCOUNT" in detect_pii("account 12345678")
        assert detect_pii("amount 12345678") == []

    def test_otp_keyword_is_required(self) -> None:
        assert "OTP" in detect_pii("one time password 448291")
        assert "OTP" in detect_pii("verification code 448291")
        assert detect_pii("pin 448291") == []


# ---------------------------------------------------------------------------
# Safeguard 2: safe-context whitelisting
# ---------------------------------------------------------------------------


class TestSafeContext:
    def test_currency_prefix_is_whitelisted(self) -> None:
        assert detect_pii("\u20b9150000 is the 80C limit") == []

    def test_percent_suffix_is_whitelisted(self) -> None:
        assert detect_pii("the ratio is 1.5%") == []

    def test_six_digit_run_beside_a_keyword_is_still_flagged(self) -> None:
        """Whitelisting must not become a blanket escape for ACCOUNT."""
        assert "ACCOUNT" in detect_pii("my account number is 76543210")


# ---------------------------------------------------------------------------
# Safeguard 3: the Aadhaar checksum, both ways
# ---------------------------------------------------------------------------


class TestVerhoef:
    def test_valid_checksum_is_accepted(self) -> None:
        assert "AADHAAR" in detect_pii("aadhaar 2345 6789 0121")

    def test_invalid_checksum_is_rejected(self) -> None:
        """The last digit is wrong, so this is not treated as an Aadhaar."""
        assert "AADHAAR" not in detect_pii("aadhaar 2345 6789 0123")

    def test_leading_zero_is_rejected(self) -> None:
        assert "AADHAAR" not in detect_pii("aadhaar 0345 6789 0121")

    def test_a_separator_style_does_not_matter(self) -> None:
        assert "AADHAAR" in detect_pii("aadhaar 2345-6789-0121")

    def test_check_digit_helper_agrees_with_the_validator(self) -> None:
        from rag.guardrails import _verhoef_check_digit, _verhoef_valid

        first_eleven = "23456789012"
        digit = _verhoef_check_digit(first_eleven)
        assert _verhoef_valid(first_eleven + str(digit)) is True


# ---------------------------------------------------------------------------
# Router precedence
# ---------------------------------------------------------------------------


class TestRouterPrecedence:
    def test_pii_beats_everything(self) -> None:
        result = route("My PAN is ABCDE1234F, should I buy and what were the returns?")
        assert result.label is Route.PII
        assert result.matched_classes == ("PAN",)
        assert result.scheme_hint is None

    def test_performance_beats_advice(self) -> None:
        """The spec's worked example: factsheet redirect, not a generic refusal."""
        result = route("Which fund has the best returns and should I buy it?")
        assert result.label is Route.PERFORMANCE

    def test_should_i_buy_is_advice(self) -> None:
        assert route("Should I buy HDFC Small Cap Fund?").label is Route.ADVICE

    def test_right_for_me_is_advice(self) -> None:
        assert route("Is HDFC Large Cap Fund right for me?").label is Route.ADVICE

    def test_ten_year_horizon_buy_is_advice(self) -> None:
        assert route("I have a 10 year horizon, what should I buy?").label is Route.ADVICE

    def test_good_time_is_advice(self) -> None:
        assert route("Is now a good time to invest in the ELSS fund?").label is Route.ADVICE

    def test_return_question_is_performance(self) -> None:
        result = route("How much did HDFC Large Cap Fund return last year?")
        assert result.label is Route.PERFORMANCE

    def test_best_cagr_is_performance(self) -> None:
        assert route("Which fund has the best CAGR?").label is Route.PERFORMANCE

    def test_rank_is_performance(self) -> None:
        assert route("Rank these funds by performance").label is Route.PERFORMANCE

    def test_exit_load_falls_through_to_the_classifier(self) -> None:
        result = route("What is the exit load on HDFC Large Cap?", llm=StubClassifier("FACTUAL"))
        assert result.label is Route.FACTUAL
        assert result.scheme_hint == "hdfc_large_cap"

    def test_todays_nav_is_out_of_scope(self) -> None:
        result = route("What is the NAV today?", llm=StubClassifier("OUT_OF_SCOPE"))
        assert result.label is Route.OUT_OF_SCOPE

    def test_scheme_hint_is_resolved_for_factual(self) -> None:
        result = route(
            "What is the benchmark of HDFC Balanced Advantage?",
            llm=StubClassifier("FACTUAL"),
        )
        assert result.scheme_hint == "hdfc_balanced_advantage"

    def test_scheme_hint_is_none_for_an_unnamed_scheme(self) -> None:
        result = route("What is the expense ratio?", llm=StubClassifier("FACTUAL"))
        assert result.scheme_hint is None


# ---------------------------------------------------------------------------
# The classifier is a fallback, never a gate
# ---------------------------------------------------------------------------


class TestClassifier:
    def test_keyword_routes_never_reach_the_model(self) -> None:
        llm = ExplodingClassifier()
        assert route("Should I buy HDFC Small Cap Fund?", llm=llm).label is Route.ADVICE
        assert route("Which fund has the best CAGR?", llm=llm).label is Route.PERFORMANCE
        assert llm.calls == 0, "deterministic routes must not cost a model call"

    def test_pii_never_reaches_the_model(self) -> None:
        llm = ExplodingClassifier()
        route("My PAN is ABCDE1234F", llm=llm)
        assert llm.calls == 0

    def test_a_parse_failure_defaults_to_factual(self) -> None:
        result = route("What is the exit load?", llm=StubClassifier("banana"))
        assert result.label is Route.FACTUAL

    def test_an_empty_response_defaults_to_factual(self) -> None:
        result = route("What is the exit load?", llm=StubClassifier(""))
        assert result.label is Route.FACTUAL

    def test_an_unknown_label_defaults_to_factual(self) -> None:
        result = route("What is the exit load?", llm=StubClassifier("SOMETHING_ELSE"))
        assert result.label is Route.FACTUAL

    def test_a_provider_failure_defaults_to_factual(self) -> None:
        """A broken classifier must not take down the query path."""
        result = route("What is the exit load?", llm=ExplodingClassifier())
        assert result.label is Route.FACTUAL

    def test_no_classifier_still_routes_deterministically(self) -> None:
        assert route("My PAN is ABCDE1234F").label is Route.PII
        assert route("Should I buy HDFC Small Cap Fund?").label is Route.ADVICE
        assert route("What is the exit load?").label is Route.FACTUAL

    def test_classifier_is_asked_without_the_json_schema(self) -> None:
        seen: dict[str, object] = {}

        class Recorder:
            def complete(self, system: str, user: str, *, json_mode: bool = True) -> str:
                seen["json_mode"] = json_mode
                seen["system"] = system
                return "FACTUAL"

        route("What is the exit load?", llm=Recorder())
        assert seen["json_mode"] is False
        assert "FACTUAL" in str(seen["system"])


# ---------------------------------------------------------------------------
# The rejection text
# ---------------------------------------------------------------------------


class TestPiiResponse:
    def test_two_lines_and_the_closing_disclaimer(self) -> None:
        text = pii_response(["PAN"])
        assert text.splitlines()[-1] == "Facts-only. No investment advice."

    def test_does_not_echo_the_identifier(self) -> None:
        assert "ABCDE1234F" not in pii_response(["PAN"])

    def test_does_not_name_the_class_that_matched(self) -> None:
        """A constant message cannot leak which pattern fired.

        The text names every category the assistant declines, so the categories
        are not a signal - only a per-class message would be.
        """
        for classes in (["PAN"], ["OTP"], ["EMAIL"], ["PHONE"], ["AADHAAR"]):
            assert pii_response(classes) == pii_response(["PAN"])

    def test_does_not_echo_the_matched_value(self) -> None:
        text = pii_response(["ACCOUNT"])
        assert "12345678" not in text
        assert "448291" not in text
        assert "ramesh@example.com" not in text

    def test_is_the_same_for_every_class(self) -> None:
        assert pii_response(["PAN"]) == pii_response(["EMAIL"])

    def test_accepts_no_classes(self) -> None:
        assert pii_response().endswith("Facts-only. No investment advice.")


# ---------------------------------------------------------------------------
# The adversarial set itself
# ---------------------------------------------------------------------------


class TestRefusalPayloads:
    """A6/A7: the advice and performance routes carry their specified links."""

    def test_advice_carries_the_education_link(self) -> None:
        from rag.config import AnswerConfig
        from rag.guardrails import refusal_payload

        payload = refusal_payload(
            "ADVICE", answer_cfg=AnswerConfig(), route=Route.ADVICE
        )
        assert payload.route is Route.ADVICE
        assert payload.source_url == "https://www.sebi.gov.in/investor-education"
        assert "Source: https://www.sebi.gov.in/investor-education" in payload.text
        assert "Last updated from sources:" in payload.text

    def test_performance_states_the_exact_required_sentence(self) -> None:
        from rag.config import AnswerConfig
        from rag.guardrails import refusal_payload

        payload = refusal_payload(
            "PERFORMANCE", answer_cfg=AnswerConfig(), route=Route.PERFORMANCE
        )
        assert payload.route is Route.PERFORMANCE
        assert "I don't provide returns, CAGR or performance comparisons." in payload.text

    def test_every_refusal_is_at_most_three_sentences_of_prose(self) -> None:
        from rag.config import AnswerConfig
        from rag.guardrails import refusal_payload
        from rag.validate import count_sentences

        for label in ("ADVICE", "PERFORMANCE", "OUT_OF_SCOPE"):
            payload = refusal_payload(label, answer_cfg=AnswerConfig())
            prose = payload.text.split("\n\n")[0]
            assert count_sentences(prose) <= 3, label


class TestAdversarialSet:
    @staticmethod
    def rows() -> list[dict[str, str]]:
        parsed: list[dict[str, str]] = []
        for line in ADVERSARIAL.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                parsed.append(json.loads(line))
        return parsed

    def test_file_exists_and_has_at_least_twelve_rows(self) -> None:
        assert len(self.rows()) >= 12

    def test_every_row_is_well_formed(self) -> None:
        for row in self.rows():
            assert set(row) == {"id", "query", "expect_route"}
            assert row["expect_route"] in {
                "FACTUAL",
                "ADVICE",
                "PERFORMANCE",
                "OUT_OF_SCOPE",
                "PII",
            }

    def test_ids_are_unique(self) -> None:
        ids = [row["id"] for row in self.rows()]
        assert len(ids) == len(set(ids))

    def test_the_file_declares_its_identifiers_fabricated(self) -> None:
        header = ADVERSARIAL.read_text(encoding="utf-8")[:800]
        assert "FABRICATED" in header

    def test_every_pii_row_is_actually_caught(self) -> None:
        for row in self.rows():
            if row["expect_route"] == "PII":
                assert detect_pii(row["query"]), row["id"]

    def test_every_deterministic_row_routes_as_expected(self) -> None:
        """PII, ADVICE and PERFORMANCE are decided without the model."""
        for row in self.rows():
            expected = row["expect_route"]
            if expected in {"PII", "ADVICE", "PERFORMANCE"}:
                assert route(row["query"]).label.value == expected, row["id"]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
