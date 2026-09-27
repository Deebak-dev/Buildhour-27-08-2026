"""The eight hard gates (C12). The last line of defence.

``docs/architecture.md`` §6.6. Every gate is a hard check and **any failure
discards the answer; nothing is patched**. That distinction matters: a validator
that edits a bad answer into a good one is a second generation step wearing a
safety badge, and its failures become invisible.

What each gate defends:

===== ============================== ==========================================
Gate  Check                          Failure mode it blocks
===== ============================== ==========================================
G1    valid JSON, non-empty answer   empty or unparseable output
G2    <= max_sentences (3)           rambling that hides which fact is meant
G3    exactly one URL, and it parses two citations, or a malformed link
G4    URL in the allowlist           a hallucinated link (NFR-2, A3)
G5    no advice/return vocabulary    becoming an advisor (FR-7.2, FR-7.3)
G6    every number is in the context figure drift (R7)
G7    freshness line appended        a stale fact presented as current
G8    grounded_field matches intent   answering the wrong field confidently
===== ============================== ==========================================

G7 is the odd one out: it *appends* rather than checks, because the freshness
line is built in Python from the winning chunk's ``retrieved_at`` and the model
never writes it (invariant I2, D2).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

from rag.answer import DraftAnswer, no_context_payload, winning_chunk
from rag.config import AnswerConfig, GenerationConfig
from rag.prompts import BANNED_TERMS, refusal_text
from rag.retrieve import detect_field_intent
from rag.schemas import AnswerPayload, RetrievalTrace, Route

logger = logging.getLogger(__name__)

#: Gate ids in evaluation order. Reported in ``AnswerPayload.validators_passed``.
GATE_IDS: tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")

#: A sentence ends at . ? or ! followed by whitespace or end-of-string. Kept as a
#: positive pattern so the count includes a final unterminated sentence.
_SENTENCE_SPLIT = re.compile(r"(?<=[.?!])(\s+|$)")

#: A numeric token: digits with optional decimal part, grouping or percent sign.
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*\s*%?")

#: Gates whose failure is worth one repair attempt with the stricter prompt.
#: G1/G2/G5 are about *form*, which a stricter instruction can fix. G4 and G6 are
#: about *content* the model invented; retrying just invites a second invention.
_RETRYABLE = frozenset({"G1", "G2", "G5"})


def count_sentences(text: str) -> int:
    """Count sentences in ``text`` (gate G2).

    Splits on ``.``/``?``/``!`` followed by whitespace or end of string, so a
    final unterminated sentence still counts. An empty string is zero sentences.

    Args:
        text: the candidate answer.

    Returns:
        The sentence count, never below 0.
    """
    if not text or not text.strip():
        return 0
    parts = [part for part in _SENTENCE_SPLIT.split(text.strip()) if part and part.strip()]
    return len(parts)


def _urls_in(text: str) -> list[str]:
    """Return every http(s) URL appearing in ``text``."""
    return re.findall(r"https?://[^\s<>\"'\)\]]+", text or "")


def _numbers_in(text: str) -> set[str]:
    """Return the numeric tokens in ``text``, normalised for comparison (G6).

    Commas are stripped and a trailing percent sign removed, so ``1,447.38`` in
    the context and ``1447.38`` in the answer compare equal - a formatting
    difference is not figure drift. The token text is otherwise preserved, so
    ``1.85`` still fails to match ``1.03``.
    """
    found: set[str] = set()
    for match in _NUMBER.finditer(text or ""):
        token = match.group(0)
        token = token.replace(",", "").replace("%", "").strip()
        if token:
            found.add(token)
    return found


def find_banned_terms(text: str) -> list[str]:
    """Return the banned terms present in ``text`` (gate G5).

    Matching is case-insensitive and on word boundaries, so "returns" is caught
    but "returned" and "surpasses" are not false positives on a substring rule.
    The matched phrase is returned in lower case for logging.

    Args:
        text: the candidate answer.

    Returns:
        The banned terms found, in table order, without duplicates.
    """
    haystack = f" {(text or '').lower()} "
    hits: list[str] = []
    for term in BANNED_TERMS:
        pattern = re.compile(rf"(?<![a-z]){re.escape(term.lower())}(?![a-z])")
        if pattern.search(haystack):
            hits.append(term.lower())
    return hits


def _format_retrieved_at(value: str, date_format: str) -> str:
    """Format an ISO-8601 ``retrieved_at`` for the freshness line (G7).

    Falls back to the raw string rather than dropping the line: an unparsed date
    is a smaller problem than a missing citation block, and a visible raw
    timestamp is still honest about provenance.
    """
    if not value:
        return ""
    try:
        return datetime.fromisoformat(value).strftime(date_format)
    except (TypeError, ValueError):
        return str(value)


def render_contract(
    answer: str,
    *,
    source_url: str,
    retrieved_at: str,
    answer_cfg: AnswerConfig,
) -> str:
    """Render the §7 response contract (G7).

    Both trailing lines are built here, in Python, from chunk metadata. The model
    contributes only the prose above them (invariant I2, D2).

    Args:
        answer: the validated answer prose.
        source_url: the winning chunk's URL.
        retrieved_at: the winning chunk's ``retrieved_at``.
        answer_cfg: answer settings, for the label and date format.

    Returns:
        The full contract string.
    """
    stamp = _format_retrieved_at(retrieved_at, answer_cfg.date_format)
    lines = [answer.strip(), "", f"Source: {source_url}"]
    lines.append(f"{answer_cfg.freshness_label} {stamp}".rstrip())
    return "\n".join(lines)


def _refusal(
    trace: RetrievalTrace,
    label: str,
    *,
    answer_cfg: AnswerConfig,
    passed: Sequence[str],
    source_url: str | None = None,
) -> AnswerPayload:
    """Build a refusal payload for a failed gate."""
    text, url = refusal_text(label, education_url=answer_cfg.education_url)
    return AnswerPayload(
        text=text,
        route=Route.FACTUAL,
        source_url=source_url or url,
        grounded_field=None,
        trace=trace,
        degraded=True,
        # Only a genuine outage counts. Every other gate downgrade means the
        # system chose not to answer, which is the contract working, not a
        # failure of the service.
        service_failure=(label == "SERVICE_UNAVAILABLE"),
        validators_passed=list(passed),
    )


def validate(
    draft: DraftAnswer,
    query: str,
    trace: RetrievalTrace,
    *,
    allowlist: Iterable[str],
    gen_cfg: GenerationConfig | None = None,
    answer_cfg: AnswerConfig | None = None,
    context_text: str | None = None,
) -> AnswerPayload:
    """Run the eight gates and render the final payload.

    Args:
        draft: the model's draft.
        query: the original question, for the G8 intent comparison.
        trace: the retrieval result, for the winning chunk and citation.
        allowlist: URLs the assistant may cite (G4).
        gen_cfg: generation settings, for ``max_sentences``.
        answer_cfg: answer settings, for the freshness label and date format.
        context_text: the exact context handed to the model. Defaults to
            rebuilding it from the trace, which is what the tests want; a caller
            that generated with a different context should pass it explicitly.

    Returns:
        A payload whose ``validators_passed`` lists the gates that held, and
        whose ``degraded`` flag is set whenever a gate forced a downgrade.
    """
    settings = gen_cfg or GenerationConfig()
    answers = answer_cfg or AnswerConfig()
    allowed = set(allowlist or ())
    passed: list[str] = []

    # No context at all: refuse before evaluating form, since there is nothing
    # to be faithful to.
    if not trace.chunks:
        payload = no_context_payload(trace, answer_cfg=answers)
        logger.info("validate: no context, refusing")
        return payload

    chunk = winning_chunk(trace)
    source_url = chunk.source_url if chunk else ""
    context = context_text if context_text is not None else _context_from(trace)

    # ---- G1 structure ----------------------------------------------------
    if draft.is_malformed or not draft.text.strip():
        logger.info("validate: G1 failed (malformed=%s)", draft.is_malformed)
        # A generation outage is not a corpus gap. Saying "not in my sources"
        # when the model simply never ran would be a false statement about
        # coverage, and it would still be rendered with a citation and
        # degraded=False, which reads as a grounded answer.
        label = "SERVICE_UNAVAILABLE" if draft.service_failure else "NOT_IN_CORPUS"
        return _refusal(
            trace,
            label,
            answer_cfg=answers,
            passed=passed,
            source_url=source_url or None,
        )
    passed.append("G1")

    # ---- G2 length --------------------------------------------------------
    sentences = count_sentences(draft.text)
    if sentences > settings.max_sentences:
        logger.info("validate: G2 failed (sentences=%d max=%d)", sentences, settings.max_sentences)
        return _refusal(
            trace,
            "NOT_IN_CORPUS",
            answer_cfg=answers,
            passed=passed,
            source_url=source_url or None,
        )
    passed.append("G2")

    # ---- G3 citation ------------------------------------------------------
    # Two distinct checks, because the model is told never to emit a URL (I2):
    #   * the prose must contain none - a URL there means the model is inventing
    #     links, whatever they are;
    #   * the URL that will be cited comes from the chunk and must parse.
    # "Exactly one URL" is asserted on the rendered output in
    # test_contract.py, after the Python-appended citation exists.
    prose_urls = _urls_in(draft.text)
    if prose_urls:
        logger.info(
            "validate: G3 failed (model emitted %d url(s) in prose)", len(prose_urls)
        )
        return _refusal(
            trace,
            "NOT_IN_CORPUS",
            answer_cfg=answers,
            passed=passed,
            source_url=source_url or None,
        )
    passed.append("G3")

    # ---- G4 allowlist -----------------------------------------------------
    # The citable URL is the winning chunk's, taken from metadata. A model can
    # therefore never choose the citation, hallucinated or otherwise.
    parsed = urlparse(source_url or "")
    if not parsed.scheme or not parsed.netloc:
        logger.info("validate: G4 failed (chunk url does not parse)")
        return _refusal(
            trace,
            "NOT_IN_CORPUS",
            answer_cfg=answers,
            passed=passed,
            source_url=source_url or None,
        )
    if source_url not in allowed:
        logger.info("validate: G4 failed (chunk url not allowlisted)")
        return _refusal(
            trace, "NOT_IN_CORPUS", answer_cfg=answers, passed=passed
        )
    passed.append("G4")

    # ---- G5 vocabulary ----------------------------------------------------
    banned = find_banned_terms(draft.text)
    if banned:
        logger.info("validate: G5 failed (terms=%s)", ",".join(banned))
        return _refusal(
            trace,
            "NOT_IN_CORPUS",
            answer_cfg=answers,
            passed=passed,
            source_url=source_url,
        )
    passed.append("G5")

    # ---- G6 figure fidelity ----------------------------------------------
    # The load-bearing gate for R7. Compared against the whole context, not just
    # the winning chunk: a number may legitimately come from any retrieved chunk.
    context_numbers = _numbers_in(context)
    invented = sorted(_numbers_in(draft.text) - context_numbers)
    if invented:
        logger.info("validate: G6 failed (numbers absent from context=%s)", invented)
        return _refusal(
            trace,
            "NOT_IN_CORPUS",
            answer_cfg=answers,
            passed=passed,
            source_url=source_url,
        )
    passed.append("G6")

    # ---- G8 grounding -----------------------------------------------------
    # Checked before rendering so a downgrade is a refusal, not a decorated
    # wrong-field answer.
    intent = detect_field_intent(query)
    if draft.grounded_field and intent and draft.grounded_field != intent:
        logger.info(
            "validate: G8 failed (grounded_field=%s intent=%s)",
            draft.grounded_field,
            intent,
        )
        return _refusal(
            trace,
            "NOT_IN_CORPUS",
            answer_cfg=answers,
            passed=passed,
            source_url=source_url,
        )
    passed.append("G8")

    # ---- G7 freshness: appended, never checked ---------------------------
    rendered = render_contract(
        draft.text,
        source_url=source_url,
        retrieved_at=chunk.retrieved_at if chunk else "",
        answer_cfg=answers,
    )
    passed.append("G7")

    logger.info(
        "validate: passed=%s degraded=False sentences=%d", ",".join(passed), sentences
    )
    return AnswerPayload(
        text=rendered,
        route=Route.FACTUAL,
        source_url=source_url,
        last_updated=_format_retrieved_at(
            chunk.retrieved_at if chunk else "", answers.date_format
        ),
        grounded_field=draft.grounded_field or intent,
        trace=trace,
        degraded=False,
        validators_passed=passed,
    )


def _context_from(trace: RetrievalTrace) -> str:
    """Rebuild the context text from a trace, matching the model's input."""
    from rag.answer import build_context

    return build_context(trace)


def should_retry(passed: Sequence[str], *, gen_cfg: GenerationConfig | None = None) -> bool:
    """Whether a failed attempt is worth one more with the stricter prompt.

    Only form failures (G1/G2/G5) are retried. G4 and G6 failures mean the model
    invented content, and a retry invites a second invention.

    Args:
        passed: the gates that did hold, from a failed ``validate`` call.
        gen_cfg: generation settings, for ``max_retries``.

    Returns:
        ``True`` when a repair attempt is permitted.
    """
    settings = gen_cfg or GenerationConfig()
    if settings.max_retries < 2:
        return False
    failed = set(GATE_IDS) - set(passed)
    return bool(failed & _RETRYABLE)


__all__ = [
    "GATE_IDS",
    "count_sentences",
    "find_banned_terms",
    "render_contract",
    "should_retry",
    "validate",
]
