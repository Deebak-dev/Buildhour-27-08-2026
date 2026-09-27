"""Nothing sensitive reaches disk, a log, or the model (FR-5.3, A8, NFR-4).

This suite treats the log directory as hostile: after driving real PII queries
through :func:`rag.pipeline.ask`, it reads *every* file under ``tmp_path`` and
asserts no PAN pattern, no email and no long digit run appears anywhere. A leak
anywhere - a stray field, a traceback, a repr - fails the test.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from rag import logging_utils, pipeline
from rag.config import load_config
from rag.schemas import Route

#: The identifiers used below. All fabricated; see eval/adversarial_set.jsonl.
PAN_QUERY = "My PAN is ABCDE1234F"
AADHAAR_QUERY = "my aadhaar is 2345 6789 0121 and folio 12345678"
OTP_QUERY = "OTP is 448291, and my account number is 76543210"
EMAIL_QUERY = "email me at ramesh@example.com"
PHONE_QUERY = "call me on 9876543210 about the exit load"

#: Every one of these must be absent from anything written to disk.
PAN_PATTERN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
EMAIL_PATTERN = re.compile(r"[\w.%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
TWELVE_DIGIT_PATTERN = re.compile(r"(?<!\d)\d{12}(?!\d)")


def read_everything(root: Path) -> str:
    """Concatenate every file under ``root`` as text."""
    blobs: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            blobs.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(blobs)


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the logger at ``tmp_path`` and clear any cached client."""
    logging_utils.set_log_path(tmp_path / "logs" / "app.log")
    monkeypatch.setattr(pipeline, "_CLIENT", None)
    yield tmp_path
    logging_utils.set_log_path(Path("logs") / "app.log")


class TestNothingIsPersisted:
    def test_pan_query_writes_nothing_sensitive(self, sandbox: Path) -> None:
        payload = pipeline.ask(PAN_QUERY)
        assert payload.route is Route.PII
        assert not PAN_PATTERN.search(read_everything(sandbox))

    def test_aadhaar_query_writes_nothing_sensitive(self, sandbox: Path) -> None:
        payload = pipeline.ask(AADHAAR_QUERY)
        assert payload.route is Route.PII
        assert not TWELVE_DIGIT_PATTERN.search(read_everything(sandbox))

    def test_otp_query_writes_nothing_sensitive(self, sandbox: Path) -> None:
        payload = pipeline.ask(OTP_QUERY)
        assert payload.route is Route.PII
        assert not TWELVE_DIGIT_PATTERN.search(read_everything(sandbox))

    def test_email_query_writes_nothing_sensitive(self, sandbox: Path) -> None:
        pipeline.ask(EMAIL_QUERY)
        assert not EMAIL_PATTERN.search(read_everything(sandbox))

    def test_phone_query_writes_nothing_sensitive(self, sandbox: Path) -> None:
        pipeline.ask(PHONE_QUERY)
        assert "9876543210" not in read_everything(sandbox)

    def test_nothing_written_at_all_for_a_pii_query(self, sandbox: Path) -> None:
        """The strongest form: a blocked query produces no text on disk.

        A ``matched_classes`` list of class names is permitted and auditable; no
        fragment of the query itself is.
        """
        pipeline.ask(PAN_QUERY)
        written = read_everything(sandbox)
        assert "ABCDE1234F" not in written
        for line in written.splitlines():
            record = json.loads(line)
            assert set(record) <= logging_utils.ALLOWED
            assert "query" not in record
            assert "text" not in record


class TestTheLogShape:
    def test_a_block_is_auditable_by_class_only(self, sandbox: Path) -> None:
        pipeline.ask(PAN_QUERY)
        records = [
            json.loads(line)
            for line in read_everything(sandbox).splitlines()
            if line.strip()
        ]
        assert records, "the block must still be logged, or the rate is invisible"
        assert records[-1]["route"] == "PII"
        assert records[-1]["matched_classes"] == ["PAN"]
        assert records[-1]["outcome"] == "blocked"

    def test_the_log_is_valid_jsonl(self, sandbox: Path) -> None:
        pipeline.ask(PAN_QUERY)
        for line in read_everything(sandbox).splitlines():
            if line.strip():
                json.loads(line)


class TestTheResponse:
    @pytest.mark.parametrize(
        "query", [PAN_QUERY, AADHAAR_QUERY, OTP_QUERY, EMAIL_QUERY, PHONE_QUERY]
    )
    def test_the_answer_does_not_echo_the_input(self, query: str, sandbox: Path) -> None:
        payload = pipeline.ask(query)
        assert query not in payload.text
        assert query.lower() not in payload.text.lower()

    def test_the_answer_carries_no_citation(self, sandbox: Path) -> None:
        """A PII rejection cites nothing, because it retrieved nothing."""
        payload = pipeline.ask(PAN_QUERY)
        assert payload.source_url is None
        assert "Source:" not in payload.text

    def test_the_answer_ends_with_the_disclaimer(self, sandbox: Path) -> None:
        payload = pipeline.ask(PAN_QUERY)
        assert payload.text.splitlines()[-1] == "Facts-only. No investment advice."

    def test_the_payload_carries_no_trace(self, sandbox: Path) -> None:
        """A trace would be a place the query could be persisted downstream."""
        assert pipeline.ask(PAN_QUERY).trace is None

    def test_the_payload_is_not_degraded(self, sandbox: Path) -> None:
        """A blocked query is a correct outcome, not a failure to report."""
        assert pipeline.ask(PAN_QUERY).degraded is False


class TestThePiiPathCallsNoModel:
    def test_the_model_is_never_invoked(self, sandbox: Path) -> None:
        """The screen is deterministic and runs before the model (D3, I4)."""

        class Exploding:
            def complete(self, system: str, user: str, *, json_mode: bool = True) -> str:
                raise AssertionError("the PII path must not call the LLM")

        pipeline._CLIENT = Exploding()
        payload = pipeline.ask(PAN_QUERY)
        assert payload.route is Route.PII

    def test_retrieval_is_never_reached(
        self, sandbox: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing to retrieve means no vector-store round trip either."""

        def explode(*args: object, **kwargs: object) -> None:
            raise AssertionError("the PII path must not retrieve")

        monkeypatch.setattr(pipeline, "collection", explode)
        assert pipeline.ask(PAN_QUERY).route is Route.PII


class TestRedactionIsIndependent:
    """The redaction pass is defence in depth behind the key allowlist."""

    def test_redacts_a_pan(self) -> None:
        assert "ABCDE1234F" not in logging_utils.redact("My PAN is ABCDE1234F")

    def test_redacts_an_email(self) -> None:
        assert "ramesh@example.com" not in logging_utils.redact(
            "email me at ramesh@example.com"
        )

    def test_redacts_a_long_digit_run(self) -> None:
        assert "234567890121" not in logging_utils.redact("aadhaar 2345 6789 0121")

    def test_recurses_through_lists(self) -> None:
        cleaned = logging_utils.redact(["ABCDE1234F", "safe"])
        assert cleaned == [logging_utils.REDACTED, "safe"]

    def test_leaves_numbers_alone(self) -> None:
        assert logging_utils.redact(0.83) == 0.83
        assert logging_utils.redact(5) == 5
        assert logging_utils.redact(None) is None

    def test_does_not_mangle_an_allowlisted_url(self) -> None:
        """A source_id survives; a long digit run in free text does not."""
        assert logging_utils.redact("hdfc-large-cap__groww_scheme") == (
            "hdfc-large-cap__groww_scheme"
        )

    def test_a_query_field_is_dropped_not_redacted(self) -> None:
        """The allowlist is the primary control; redaction never sees the query."""
        logging_utils.set_log_path(Path("logs") / "app.log")
        assert "query" not in logging_utils.ALLOWED


class TestNonPiiQueriesStillLog:
    """A blocked query is not the only case: factual ones log their shape too."""

    def test_a_factual_query_logs_shape_but_not_the_query(
        self, sandbox: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tests.test_contract import make_chunk, make_trace

        monkeypatch.setattr(pipeline, "collection", lambda _cfg: object())
        monkeypatch.setattr(
            pipeline, "retrieve", lambda query, **kw: make_trace(make_chunk())
        )

        class StubLLM:
            def complete(self, system: str, user: str, *, json_mode: bool = True) -> str:
                return json.dumps(
                    {"answer": "The expense ratio is 1.03%.", "grounded_field": "expense_ratio"}
                )

        monkeypatch.setattr(pipeline, "_CLIENT", StubLLM())

        payload = pipeline.ask("What is the expense ratio of HDFC Large Cap?")
        assert payload.degraded is False
        assert "1.03%" in payload.text

        records = [
            json.loads(line)
            for line in read_everything(sandbox).splitlines()
            if line.strip()
        ]
        assert records, "a factual query must be observable"
        record = records[-1]
        assert record["route"] == "FACTUAL"
        assert record["n_chunks"] == 1
        assert set(record) <= logging_utils.ALLOWED
        # The shape is logged; the words the user typed are not.
        assert "What is the expense ratio" not in read_everything(sandbox)

    def test_raw_query_logging_stays_off(self, sandbox: Path) -> None:
        cfg = load_config()
        assert cfg.guardrails.log_raw_queries is False
        assert "query" not in logging_utils.ALLOWED


class TestDegradesWithoutAProvider:
    """A missing or unconfigured provider must refuse, never raise (I6, NFR-3).

    The CLI has to stay usable before an API key is set: keyword routes still
    answer correctly, and a factual query degrades to the not-in-corpus refusal
    instead of surfacing an ``LLMError`` traceback to the user.
    """

    def test_a_factual_query_refuses_instead_of_raising(
        self, sandbox: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rag.config import load_config as _load
        from rag.llm_client import LLMError, build_client
        from tests.test_contract import make_chunk, make_trace

        def explode(_cfg: object) -> object:
            raise LLMError("no provider configured")

        monkeypatch.setattr(build_client, "__module__", build_client.__module__)
        monkeypatch.setattr("rag.llm_client.build_client", explode)
        monkeypatch.setattr(pipeline, "collection", lambda _cfg: object())
        monkeypatch.setattr(
            pipeline, "retrieve", lambda query, **kw: make_trace(make_chunk())
        )

        payload = pipeline.ask("What is the minimum SIP for HDFC Large Cap?")

        # Retrieval succeeded here, so claiming the sources lack the fact would
        # be false. The refusal must name the real cause.
        assert payload.text.startswith("I couldn't answer right now")
        assert "don't have that" not in payload.text
        assert payload.degraded is True

    def test_keyword_routes_are_unaffected(self, sandbox: Path) -> None:
        """Advice and PII never need a model, so they still work."""
        assert pipeline.ask("Should I buy HDFC Small Cap Fund?").route is Route.ADVICE
        assert pipeline.ask("My PAN is ABCDE1234F").route is Route.PII


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
