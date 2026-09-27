"""Context assembly and draft generation (C10, C11).

Flow B, stage 3. Two responsibilities, and the split between them is the point:

* :func:`build_context` renders retrieved chunks into numbered, labelled text.
* :func:`generate` asks the model for prose **only** - never a citation.

The citation and the freshness line are attached later in Python from chunk
metadata (:mod:`rag.validate`). That is invariant I2 / ADR-1, and it is why a
hallucinated URL is structurally impossible rather than merely discouraged.

When retrieval found nothing above the similarity floor, this module short-circuits
to the ``NO_CONTEXT`` refusal **without calling the model at all**: an LLM asked
to answer from an empty context will do exactly that, from its priors.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from rag.config import AnswerConfig, GenerationConfig
from rag.llm_client import LLMClient, LLMError, complete_with_retry
from rag.prompts import REPAIR_PROMPT, SYSTEM_PROMPT, refusal_text
from rag.schemas import AnswerPayload, RetrievalTrace, Route

logger = logging.getLogger(__name__)


@dataclass
class DraftAnswer:
    """What the model returned, before any validation.

    Attributes:
        text: the answer prose. Empty when the draft is malformed.
        grounded_field: the ``field_type`` the model claims to have used, or
            ``None``. Compared against the query intent by gate G8.
        is_malformed: the model did not return the expected JSON. Gate G1 turns
            this into a refusal rather than an exception.
        raw: the unparsed response, kept for the logs and the tests. Never logged
            in production: it can echo context.
        error: why generation failed, if it did.
        service_failure: generation itself failed (no client, provider error,
            rate limit) rather than the model replying badly. Gate G1 uses this
            to say the assistant is unavailable instead of claiming the corpus
            lacks the fact.
    """

    text: str = ""
    grounded_field: str | None = None
    is_malformed: bool = False
    raw: str = ""
    error: str | None = None
    service_failure: bool = False


def build_context(trace: RetrievalTrace) -> str:
    """Render retrieved chunks as numbered, labelled context for the model.

    Each line carries the chunk's scheme and ``field_type`` so the model can set
    ``grounded_field`` correctly and so a human reading the trace can see why a
    chunk was retrieved.

    Args:
        trace: the retrieval result.

    Returns:
        Numbered context text, or ``""`` when the trace holds no chunks. The
        empty string is the signal to refuse without calling the model.
    """
    if not trace.chunks:
        return ""

    blocks: list[str] = []
    for index, item in enumerate(trace.chunks, start=1):
        chunk = item.chunk
        text = " ".join((chunk.text or "").split())
        blocks.append(
            f"[{index}] (scheme={chunk.scheme} field_type={chunk.field_type})\n{text}"
        )
    return "\n\n".join(blocks)


def winning_chunk(trace: RetrievalTrace) -> Any | None:
    """The chunk an answer should be cited against.

    The highest-scoring chunk that actually names a field, because a citation
    pointing at a holdings row would be technically allowlisted and useless.
    Falls back to the top chunk.
    """
    if not trace.chunks:
        return None
    for item in trace.chunks:
        if item.chunk.field_type and item.chunk.field_type != "other":
            return item.chunk
    return trace.chunks[0].chunk


def no_context_payload(trace: RetrievalTrace, *, answer_cfg: AnswerConfig) -> AnswerPayload:
    """Build the ``NO_CONTEXT`` refusal payload.

    Args:
        trace: the empty retrieval result, kept so the UI can still show why.
        answer_cfg: answer settings, for the education URL.

    Returns:
        A payload carrying the refusal and the trace.
    """
    text, url = refusal_text("NO_CONTEXT", education_url=answer_cfg.education_url)
    return AnswerPayload(
        text=text,
        route=Route.FACTUAL,
        source_url=url,
        grounded_field=None,
        trace=trace,
        degraded=True,
        validators_passed=[],
    )


def _parse(raw: str) -> DraftAnswer:
    """Parse the model's JSON response into a :class:`DraftAnswer`.

    A malformed response is a flag, not an exception: gate G1 decides what to do
    with it, so the failure is reported in the same place as every other gate.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return DraftAnswer(is_malformed=True, raw=raw, error="json_decode_error")

    if not isinstance(data, dict):
        return DraftAnswer(is_malformed=True, raw=raw, error="not_an_object")

    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return DraftAnswer(
            is_malformed=True, raw=raw, error="missing_or_empty_answer"
        )

    grounded = data.get("grounded_field")
    if grounded is not None and not isinstance(grounded, str):
        return DraftAnswer(is_malformed=True, raw=raw, error="bad_grounded_field")

    return DraftAnswer(text=answer.strip(), grounded_field=grounded, raw=raw)


def generate(
    query: str,
    trace: RetrievalTrace,
    *,
    llm: LLMClient | None = None,
    gen_cfg: GenerationConfig | None = None,
    answer_cfg: AnswerConfig | None = None,
    repair: bool = False,
) -> DraftAnswer:
    """Produce a draft answer for a grounded query.

    Args:
        query: the user's question.
        trace: the retrieval result.
        llm: the LLM client. Required unless the trace is empty.
        gen_cfg: generation settings. Defaults to ``config.yaml``.
        answer_cfg: answer settings. Defaults to ``config.yaml``.
        repair: use the stricter :data:`~rag.prompts.REPAIR_PROMPT` for the
            second attempt after a G2/G5 failure.

    Returns:
        A :class:`DraftAnswer`. An empty trace yields the refusal text with
        ``is_malformed`` left False and the model never called. An LLM failure
        yields the safe refusal rather than any model knowledge (invariant I6).
    """
    settings = gen_cfg or GenerationConfig()
    answers = answer_cfg or AnswerConfig()

    if not trace.chunks:
        # No context, no model. Asking a model to answer from nothing produces a
        # confident answer from its priors, which is exactly invariant I6.
        text, _url = refusal_text("NO_CONTEXT", education_url=answers.education_url)
        return DraftAnswer(text=text, grounded_field=None)

    if llm is None:
        return DraftAnswer(
            is_malformed=True,
            error="no_llm_client",
            service_failure=True,
            text=refusal_text("SERVICE_UNAVAILABLE", education_url=answers.education_url)[
                0
            ],
        )

    context = build_context(trace)
    user_prompt = f"QUESTION:\n{query}\n\nCONTEXT:\n{context}"
    system = REPAIR_PROMPT if repair else SYSTEM_PROMPT

    try:
        raw = complete_with_retry(llm, system, user_prompt, cfg=settings)
    except LLMError as exc:
        # Never answer from model knowledge after a failure (I6). The draft is
        # flagged malformed so validate downgrades it instead of passing the
        # refusal prose through all eight gates, which would attach a citation
        # and report degraded=False for what is really an outage.
        logger.warning("generation failed, returning the safe refusal: %s", exc)
        return DraftAnswer(
            is_malformed=True,
            service_failure=True,
            text=refusal_text("SERVICE_UNAVAILABLE", education_url=answers.education_url)[
                0
            ],
            grounded_field=None,
            error=str(exc),
        )

    draft = _parse(raw)
    logger.info(
        "draft produced: chars=%d malformed=%s grounded_field=%s",
        len(draft.text),
        draft.is_malformed,
        draft.grounded_field,
    )
    return draft
