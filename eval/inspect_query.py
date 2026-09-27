"""Inspect what retrieval actually returns for any query, with no API key.

Recall@k tells you *how often* retrieval works; this tells you *why* a particular
query failed. It runs Stage 2 only, so it needs neither an LLM provider nor a
network call - which makes it the fastest way to sanity-check a query, compare
two threshold settings, or see whether a missing field type is a retrieval
problem or a corpus problem.

    python -m eval.inspect_query -q "exit load on HDFC Large Cap"
    python -m eval.inspect_query -q "tax benefit of ELSS" --floor 0.0 --k 10
    python -m eval.inspect_query -q "nav of flexi cap" --json

Read the ``raw`` column when a score looks surprising. ``score`` is the final
value after the field_type boost and the similarity floor; ``raw`` is the cosine
similarity before boosting, so ``raw < tau`` means the floor dropped the chunk.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from typing import Any

from eval.recall import open_collection
from rag.config import load_config
from rag.retrieve import retrieve


def _clip(text: str, width: int) -> str:
    """Shorten ``text`` to ``width`` characters on a word boundary."""
    flat = " ".join(text.split())
    if len(flat) <= width:
        return flat
    return flat[: width - 1].rstrip() + "\u2026"


def main(argv: list[str] | None = None) -> int:
    """Run one retrieval and print the ranked chunks."""
    parser = argparse.ArgumentParser(
        prog="eval.inspect_query",
        description="Show the chunks retrieval returns for one query.",
    )
    parser.add_argument(
        "-q", "--query", default=None, help="the query to test (omit with -i)"
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="prompt for a query repeatedly, so a few can be tried in one session",
    )
    parser.add_argument(
        "-k", type=int, default=None, help="override retrieval.top_k (default 10)"
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=None,
        help="override retrieval.similarity_floor, to see what the floor hides",
    )
    parser.add_argument(
        "--boost",
        type=float,
        default=None,
        help="override retrieval.field_type_boost (default 0.35)",
    )
    parser.add_argument(
        "--mode",
        choices=("vector", "hybrid"),
        default=None,
        help="override retrieval.mode; hybrid needs the optional rank-bm25 package",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead")
    parser.add_argument(
        "--width", type=int, default=60, help="text column width (default 60)"
    )
    args = parser.parse_args(argv)

    if not args.interactive and not args.query:
        parser.error("give a query with -q/--query, or use -i/--interactive")

    cfg = load_config().retrieval
    overrides: dict[str, Any] = {}
    if args.k is not None:
        overrides["top_k"] = args.k
    if args.floor is not None:
        overrides["similarity_floor"] = args.floor
    if args.boost is not None:
        overrides["field_type_boost"] = args.boost
    if args.mode is not None:
        overrides["mode"] = args.mode
    if overrides:
        cfg = replace(cfg, **overrides)

    collection = open_collection()

    if args.interactive:
        return _interactive(collection, cfg, args.width)

    trace = retrieve(args.query, collection=collection, cfg=cfg)

    if args.json:
        print(
            json.dumps(
                {
                    "query": args.query,
                    "scheme_filter": trace.scheme_filter,
                    "intent": trace.intent,
                    "tau": trace.tau,
                    "top_score": trace.top_score,
                    "n_chunks": len(trace.chunks),
                    "chunks": [
                        {
                            "score": round(item.similarity, 4),
                            "raw": round(item.raw_similarity, 4),
                            "boosted": item.boosted,
                            "chunk_id": item.chunk.chunk_id,
                            "scheme": item.chunk.scheme,
                            "field_type": item.chunk.field_type,
                            "source_url": item.chunk.source_url,
                            "text": item.chunk.text,
                        }
                        for item in trace.chunks
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    print(f"query      : {args.query}")
    print(f"scheme     : {trace.scheme_filter or '(unscoped)'}")
    print(f"intent     : {trace.intent or '(none)'}")
    print(
        f"settings   : top_k={cfg.top_k} floor={cfg.similarity_floor} "
        f"boost={cfg.field_type_boost} mode={cfg.mode}"
    )
    print(f"chunks     : {len(trace.chunks)}   top={trace.top_score:.3f}  tau={trace.tau}")

    if not trace.chunks:
        print(
            "\nNo chunk cleared the floor. Either nothing resembles the query, or\n"
            "similarity_floor is too high - retry with --floor 0.0 to see the\n"
            "raw ranking before the floor is applied."
        )
        return 0

    print()
    header = f"{'#':<3} {'score':>6} {'raw':>6}  {'scheme':<24} {'field_type':<14} text"
    print(header)
    print("-" * len(header))
    for i, item in enumerate(trace.chunks, 1):
        mark = "*" if item.boosted else " "
        print(
            f"{i:<3} {item.similarity:>6.3f} {item.raw_similarity:>6.3f}{mark} "
            f"{item.chunk.scheme:<24} {item.chunk.field_type:<14} "
            f"{_clip(item.chunk.text, args.width)}"
        )
    print("\n* = field_type boost applied; raw is the pre-boost cosine similarity.")


def _inspect_one(query: str, collection: Any, cfg: Any, width: int) -> None:
    """Retrieve and print one query. Shared by single-shot and interactive mode."""
    trace = retrieve(query, collection=collection, cfg=cfg)

    print(f"query      : {query}")
    print(f"scheme     : {trace.scheme_filter or '(unscoped)'}")
    print(f"intent     : {trace.intent or '(none)'}")
    print(f"chunks     : {len(trace.chunks)}   top={trace.top_score:.3f}  tau={trace.tau}")

    if not trace.chunks:
        print(
            "\nNo chunk cleared the floor. Either nothing resembles the query, or\n"
            "similarity_floor is too high - retry with --floor 0.0 to see the\n"
            "raw ranking before the floor is applied."
        )
        return

    print()
    header = f"{'#':<3} {'score':>6} {'raw':>6}  {'scheme':<24} {'field_type':<14} text"
    print(header)
    print("-" * len(header))
    for i, item in enumerate(trace.chunks, 1):
        mark = "*" if item.boosted else " "
        print(
            f"{i:<3} {item.similarity:>6.3f} {item.raw_similarity:>6.3f}{mark} "
            f"{item.chunk.scheme:<24} {item.chunk.field_type:<14} "
            f"{_clip(item.chunk.text, width)}"
        )
    print("\n* = field_type boost applied; raw is the pre-boost cosine similarity.")


def _interactive(collection: Any, cfg: Any, width: int) -> int:
    """Prompt for queries until the user quits, so a few can be tried in a row.

    The collection is opened once, which keeps each turn to a single embedding
    call instead of reloading the store per question.
    """
    print("Interactive retrieval check. Type a question, or 'q' to quit.")
    print("Retrieval only: no LLM call, so no API key is needed.\n")
    while True:
        try:
            query = input("ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if query.lower() in {"q", "quit", "exit", ":q"}:
            return 0
        if not query:
            continue
        try:
            _inspect_one(query, collection, cfg, width)
        except Exception as exc:  # keep the loop alive on a bad query
            print(f"\nretrieval failed: {type(exc).__name__}: {exc}")
        print("\n" + "-" * 72 + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
