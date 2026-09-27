"""Flow B end to end: screen, route, retrieve, generate, validate (FR-5, FR-7).

The order of the first three steps is the security property. :func:`ask` screens
for PII and routes *before* it touches the vector store, so a PAN never reaches
retrieval, the model, or a log line (architecture.md invariant I4, FR-7.4).

The PII branch returns early and drops the query on the floor: the returned
payload has no ``trace``, and the one log event it emits carries
``matched_classes`` but not the text (invariant I5).
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from rag.answer import generate as generate_draft
from rag.answer import no_context_payload
from rag.config import AppConfig, load_config
from rag.guardrails import pii_response, refusal_payload, route
from rag.logging_utils import log_event
from rag.retrieve import retrieve
from rag.schemas import AnswerPayload, RetrievalTrace, Route
from rag.validate import validate

#: The LLM classifier and the answer generator are the same client; built once.
_CLIENT: Any = None
_COLLECTION: Any = None


def _client(cfg: AppConfig) -> Any:
    """Return the process-wide LLM client, building it on first use."""
    global _CLIENT
    if _CLIENT is None:
        from rag.llm_client import build_client

        _CLIENT = build_client(cfg.generation)
    return _CLIENT


def collection(cfg: AppConfig) -> Any:
    """Open (once) and return the persistent Chroma collection."""
    global _COLLECTION
    if _COLLECTION is None:
        from ingest.store import open_collection

        _COLLECTION = open_collection(
            persist_dir=cfg.vectorstore.persist_dir,
            collection_name=cfg.vectorstore.collection,
            dimension=cfg.embedding.dimension,
            embedding_model=cfg.embedding.model,
        )
    return _COLLECTION


def _allowlist(cfg: AppConfig) -> set[str]:
    """The ingested URL set, which is also the citation allowlist (G4)."""
    if not cfg.guardrails.allowlist_from_manifest:
        return set()
    from ingest.load import url_allowlist

    return url_allowlist()


def ask(query: str, *, cfg: AppConfig | None = None) -> AnswerPayload:
    """Answer one user question, applying every guardrail in order.

    Args:
        query: the raw user question. If it screens as PII it is discarded
            immediately and never logged, stored, or sent to the model.
        cfg: application settings; loaded from ``config.yaml`` when omitted.

    Returns:
        A contract-compliant :class:`~rag.schemas.AnswerPayload`.
    """
    settings = cfg or load_config()
    started = time.perf_counter()
    allowlist = _allowlist(settings)

    # Resolve the client once. It may be None when no provider is configured, in
    # which case the router falls back to its keyword rules and the generator
    # produces a refusal rather than an exception (invariant I6).
    llm = _safe_client(settings)

    # Stage 0 + 1. The PII screen runs inside route(), ahead of any model call.
    decision = route(query, llm=llm, allowlist=allowlist)

    if decision.label is Route.PII:
        # Log the class names so the block rate is auditable (A8) and the text so
        # it is not.
        log_event(
            "route",
            route=Route.PII.value,
            matched_classes=list(decision.matched_classes),
            outcome="blocked",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        return AnswerPayload(
            text=pii_response(decision.matched_classes),
            route=Route.PII,
            source_url=None,
            degraded=False,
        )

    if decision.label is Route.ADVICE:
        log_event(
            "route",
            route=decision.label.value,
            scheme=decision.scheme_hint,
            outcome="refused",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        return refusal_payload(
            "ADVICE", answer_cfg=settings.answer, route=decision.label
        )

    if decision.label is Route.PERFORMANCE:
        log_event(
            "route",
            route=decision.label.value,
            scheme=decision.scheme_hint,
            outcome="refused",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        return refusal_payload(
            "PERFORMANCE", answer_cfg=settings.answer, route=decision.label
        )

    if decision.label is Route.OUT_OF_SCOPE:
        # Live NAV, login, calculators and non-corpus funds are refusals, not
        # retrieval: there is no source to ground them in.
        log_event(
            "route",
            route=decision.label.value,
            scheme=decision.scheme_hint,
            outcome="refused",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        return refusal_payload(
            "OUT_OF_SCOPE", answer_cfg=settings.answer, route=decision.label
        )

    # Stage 2.
    trace = retrieve(query, collection=collection(settings), cfg=settings.retrieval)

    if not trace.chunks:
        log_event(
            "query",
            route=decision.label.value,
            scheme=trace.scheme_filter,
            n_chunks=0,
            top_score=trace.top_score,
            outcome="no_context",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        return no_context_payload(trace, answer_cfg=settings.answer)

    # Stage 3 + 4.
    draft = generate_draft(
        query,
        trace,
        llm=llm,
        gen_cfg=settings.generation,
        answer_cfg=settings.answer,
    )
    payload = validate(
        draft,
        query,
        trace,
        allowlist=allowlist,
        gen_cfg=settings.generation,
        answer_cfg=settings.answer,
    )

    log_event(
        "query",
        route=payload.route.value,
        scheme=trace.scheme_filter,
        n_chunks=len(trace.chunks),
        top_score=trace.top_score,
        validators_passed=payload.validators_passed,
        degraded=payload.degraded,
        outcome="degraded" if payload.degraded else "answered",
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
    return payload


def _safe_client(cfg: AppConfig) -> Any:
    """Return an LLM client, or ``None`` when none can be built.

    The router degrades to keyword rules without a client, so a missing API key
    still answers advice and performance questions correctly. Only a genuinely
    factual question needs the model.
    """
    try:
        return _client(cfg)
    except Exception:  # noqa: BLE001 - a missing key must not break the app
        return None


def cli(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m rag.pipeline --q "..."``.

    Args:
        argv: argument list; defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="python -m rag.pipeline",
        description="Ask the HDFC facts assistant one question.",
    )
    parser.add_argument("--q", "--query", dest="query", default=None)
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="keep asking questions in this terminal until you type 'q'",
    )
    args = parser.parse_args(argv)

    if not args.interactive and not args.query:
        parser.error("pass a question with --q, or use -i to ask several in a row")

    if args.interactive:
        return _interactive()

    payload = ask(args.query)
    print(payload.text)
    if payload.route is Route.FACTUAL and payload.trace is not None:
        print(
            f"\n[route={payload.route.value} chunks={len(payload.trace.chunks)} "
            f"top_score={payload.trace.top_score:.3f} "
            f"degraded={payload.degraded}]"
        )
    else:
        print(f"\n[route={payload.route.value}]")
    return 0


def _interactive() -> int:
    """Ask questions repeatedly in one terminal, so one command covers a session.

    The model client and the collection are built once and reused, because
    ``ask`` caches both; only the embedding of each new query should be paid for.
    """
    print("HDFC facts assistant. Type a question, or 'q' to quit.")
    print("Facts only - no investment advice. Answers cite their source.\n")
    while True:
        try:
            query = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if query.lower() in {"q", "quit", "exit", ":q"}:
            return 0
        if not query:
            continue
        try:
            payload = ask(query)
        except Exception as exc:  # never drop the session on one bad question
            print(f"\nsomething went wrong: {type(exc).__name__}: {exc}\n")
            continue
        print(f"\n{_render(payload)}\n")
        print("-" * 72 + "\n")


def _render(payload: Any) -> str:
    """The answer text plus a one-line trace, shared by both CLI modes."""
    lines = [payload.text]
    if payload.route is Route.FACTUAL and payload.trace is not None:
        lines.append(
            f"\n[route={payload.route.value} chunks={len(payload.trace.chunks)} "
            f"top_score={payload.trace.top_score:.3f} "
            f"degraded={payload.degraded}]"
        )
    else:
        lines.append(f"\n[route={payload.route.value}]")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(cli(sys.argv[1:]))


__all__ = ["ask", "cli", "collection"]
