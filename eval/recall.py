"""Recall@k sweep over eval/golden_set.jsonl.

Phase 5's exit gate says to measure recall before freezing the retrieval
thresholds, and Phase 9 needs the same measurement again, so it lives here
rather than in a throwaway script.

    python -m eval.recall                 # default k values
    python -m eval.recall --k 1 3 5 10    # specific cutoffs
    python -m eval.recall --verbose       # per-query detail

A query counts as a hit when any of the top-k chunks carries the expected
``field_type`` (or, when the golden row has no ``expect_source_url``, when the
expected field is present at all). Reporting the per-field breakdown matters
because the aggregate hides which kind of question retrieval is failing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from rag.config import RetrievalConfig, load_config
from rag.retrieve import resolve_scheme, retrieve

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)
GOLDEN_PATH = Path(__file__).resolve().parent / "golden_set.jsonl"


def load_golden(path: Path = GOLDEN_PATH) -> list[dict[str, Any]]:
    """Read the golden set, one JSON object per line."""
    if not path.is_file():
        raise SystemExit(f"golden set not found: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"golden set is empty: {path}")
    return rows


def is_hit(trace: Any, row: dict[str, Any], k: int) -> bool:
    """Whether the expected evidence appears in the top ``k`` chunks."""
    expected_field = row.get("expect_field_type")
    expected_scheme = row.get("scheme")
    for item in trace.chunks[:k]:
        chunk = item.chunk
        if expected_field and chunk.field_type != expected_field:
            continue
        # A golden row that names a scheme also requires that scheme, which is
        # the A14 bleed check applied to the eval set rather than the test suite.
        if expected_scheme and chunk.scheme != expected_scheme:
            continue
        return True
    return False


def open_collection() -> Any:
    """Open the persisted collection named in config.yaml."""
    import chromadb

    config = load_config()
    persist_dir = Path(config.vectorstore.persist_dir)
    if not persist_dir.is_dir():
        raise SystemExit(
            f"no vector store at {persist_dir}; run `python -m ingest.run_ingest` first"
        )
    client = chromadb.PersistentClient(path=str(persist_dir))
    names = {c.name for c in client.list_collections()}
    if config.vectorstore.collection not in names:
        raise SystemExit(
            f"collection {config.vectorstore.collection!r} absent; run "
            "`python -m ingest.run_ingest` first"
        )
    return client.get_collection(config.vectorstore.collection)


def measure(
    collection: Any,
    golden: Sequence[dict[str, Any]],
    ks: Sequence[int],
    cfg: RetrievalConfig,
) -> dict[int, float]:
    """Return ``{k: recall}`` over the golden set."""
    totals: dict[int, int] = {k: 0 for k in ks}
    for row in golden:
        trace = retrieve(row["query"], collection=collection, cfg=cfg)
        for k in ks:
            if is_hit(trace, row, k):
                totals[k] += 1
    count = len(golden)
    return {k: totals[k] / count for k in ks}


def main(argv: list[str] | None = None) -> int:
    """Run the sweep and print the table."""
    parser = argparse.ArgumentParser(
        prog="eval.recall", description="Recall@k over the golden set."
    )
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--golden", default=str(GOLDEN_PATH))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="override similarity_floor for a sweep, without editing config.yaml",
    )
    args = parser.parse_args(argv)

    golden = load_golden(Path(args.golden))
    cfg = load_config().retrieval
    if args.threshold is not None:
        from dataclasses import replace

        cfg = replace(cfg, similarity_floor=args.threshold)

    collection = open_collection()

    if args.verbose:
        print(f"{'id':<5} {'hit@5':<7} {'top':<7} {'scheme':<22} query")
        for row in golden:
            trace = retrieve(row["query"], collection=collection, cfg=cfg)
            hit = is_hit(trace, row, 5)
            print(
                f"{row['id']:<5} {'yes' if hit else 'NO':<7} "
                f"{trace.top_score:<7.3f} "
                f"{(trace.scheme_filter or '-'):<22} {row['query'][:52]}"
            )
        print()

    recalls = measure(collection, golden, args.k, cfg)

    print(
        f"corpus={collection.count()}  floor={cfg.similarity_floor}  "
        f"boost={cfg.field_type_boost}  mode={cfg.mode}"
    )
    print(f"golden queries: {len(golden)}")
    for k in args.k:
        hits = round(recalls[k] * len(golden))
        print(f"  recall@{k:<3} {hits}/{len(golden)}  = {recalls[k]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
