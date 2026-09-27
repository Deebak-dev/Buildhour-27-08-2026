"""Flow A CLI — manifest → load → chunk → embed → store.

Usage::

    python -m ingest.run_ingest
    python -m ingest.run_ingest --timeout 30 --rebuild
    python -m ingest.run_ingest --skip-fetch        # re-embed from cached HTML
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ingest.chunk import build_chunks, field_type_histogram, write_chunks
from ingest.embed import embed_chunks
from ingest.load import load_all, log_event, write_manifest
from ingest.store import collection_stats, open_collection, upsert
from rag.config import load_config
from rag.manifest import enabled_rows, load_manifest, placeholder_rows, validate_manifest


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the ingest CLI."""
    parser = argparse.ArgumentParser(
        prog="ingest.run_ingest",
        description="Run Flow A: load the corpus, clean it, chunk it, embed it, and store it.",
    )
    parser.add_argument("--manifest", default=None, help="path to sources.csv")
    parser.add_argument("--timeout", type=int, default=None, help="per-request timeout (s)")
    parser.add_argument("--max-attempts", type=int, default=None, help="attempts per URL")
    parser.add_argument("--crawl-delay", type=float, default=None, help="seconds between requests per host")
    parser.add_argument("--limit", type=int, default=None, help="only load the first N rows")
    parser.add_argument("--skip-fetch", action="store_true", help="reuse the existing raw snapshots")
    parser.add_argument("--rebuild", action="store_true", help="rebuild derived artifacts")
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="stop after chunking; useful when iterating on the loader",
    )
    return parser


def _suffixed(path: Path, suffix: str) -> Path:
    """Return ``path`` with ``suffix`` inserted before the final extension."""
    if not suffix:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def main(argv: list[str] | None = None) -> int:
    """Run the Flow A pipeline.

    Returns:
        ``0`` on success, ``1`` only when nothing could be loaded.
    """
    args = build_parser().parse_args(argv)
    config = load_config()
    started = time.monotonic()

    manifest_path = Path(args.manifest) if args.manifest else config.manifest_path
    rows = load_manifest(manifest_path)

    problems = validate_manifest(rows)
    if problems:
        for problem in problems:
            print(f"manifest problem: {problem}", file=sys.stderr)
        return 1

    selected = enabled_rows(rows)
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        print("no enabled rows in the manifest", file=sys.stderr)
        return 1

    pending = placeholder_rows(rows)
    if pending:
        log_event("load", outcome="placeholders_pending", n_documents=len(pending))

    if args.skip_fetch:
        from ingest.load import load_from_raw

        # Rebuilt from the raw HTML snapshots, not from data/manifest.json:
        # the manifest is a derived artifact that this run is about to overwrite,
        # so reading it back first is circular and destroys the full-corpus
        # record whenever --limit is used.
        documents = load_from_raw(
            selected,
            raw_dir=config.raw_dir,
            clean_dir=config.clean_dir,
            thin_chars=config.corpus.thin_chars,
        )
        log_event(
            "load",
            n_documents=len(documents),
            outcome="reused_raw",
            requested=len(selected),
        )
    else:
        documents = load_all(
            selected,
            timeout=args.timeout if args.timeout is not None else config.corpus.timeout,
            max_attempts=args.max_attempts or config.corpus.max_attempts,
            crawl_delay=args.crawl_delay if args.crawl_delay is not None else config.corpus.crawl_delay,
            raw_dir=config.raw_dir,
            clean_dir=config.clean_dir,
            thin_chars=config.corpus.thin_chars,
        )

    succeeded = [doc for doc in documents if doc.http_status == 200]
    if not succeeded:
        print("no documents were fetched successfully", file=sys.stderr)
        return 1

    # A --limit run processes a subset, so writing it to the canonical paths
    # would destroy the full-corpus manifest, chunks and index. Partial runs get
    # their own suffixed artifacts instead, leaving the real ones untouched.
    suffix = f".limit{args.limit}" if args.limit else ""
    manifest_json = _suffixed(config.manifest_json, suffix)
    chunks_jsonl = _suffixed(config.chunks_jsonl, suffix)
    collection_name = (
        f"{config.vectorstore.collection}{suffix}" if args.limit else config.vectorstore.collection
    )
    if suffix:
        print(
            f"--limit {args.limit}: writing suffixed artifacts "
            f"(manifest={manifest_json.name}, chunks={chunks_jsonl.name}, "
            f"collection={collection_name})",
            file=sys.stderr,
        )

    manifest_out = write_manifest(
        documents,
        path=manifest_json,
        embedding_model=config.embedding.model,
        dimension=config.embedding.dimension,
    )

    chunks = build_chunks(documents, cfg=config.chunking)
    chunks_out = write_chunks(chunks, chunks_jsonl)
    if not chunks:
        print("no chunks were produced", file=sys.stderr)
        return 1

    vectors_written = 0
    collection = None
    if not args.skip_embed:
        # Opened before embedding, not after. The dimension/model guard lives in
        # open_collection, so embedding first would spend a full CPU pass
        # (seconds to minutes) only to raise. Nothing is written either way.
        collection = open_collection(
            persist_dir=config.vectorstore.persist_dir,
            collection_name=collection_name,
            dimension=config.embedding.dimension,
            embedding_model=config.embedding.model,
            rebuild=args.rebuild,
        )
        vectors = embed_chunks(chunks, cfg=config.embedding)
        vectors_written = upsert(collection, chunks, vectors)

    _summary(
        documents,
        chunks,
        manifest_out,
        chunks_out,
        vectors_written=vectors_written,
        collection=collection,
        elapsed=time.monotonic() - started,
    )
    log_event(
        "ingest",
        n_documents=len(documents),
        n_chunks=len(chunks),
        n_vectors=vectors_written,
        collection_count=collection_stats(collection)["count"] if collection else 0,
        outcome="ok",
        elapsed_s=round(time.monotonic() - started, 2),
    )
    return 0


def _summary(
    documents,
    chunks,
    manifest_out: Path,
    chunks_out: Path,
    *,
    vectors_written: int = 0,
    collection=None,
    elapsed: float = 0.0,
) -> None:
    """Print the ingest summary table."""
    print("\n=== Ingest summary ===")
    print(f"{'source_id':<42} {'status':>6} {'chars':>7} {'thin':>5}")
    print("-" * 64)
    for doc in documents:
        print(
            f"{doc.source_id:<42} {doc.http_status:>6} "
            f"{doc.content_chars:>7} {str(doc.thin):>5}"
        )
    thin = sum(1 for doc in documents if doc.thin)
    print("-" * 64)
    print(f"documents loaded : {len(documents)}  (thin: {thin})")
    print(f"chunks created   : {len(chunks)}")
    if chunks:
        print("field types      :")
        for name, count in field_type_histogram(chunks).items():
            print(f"    {name:<22} {count}")
    print(f"vectors written  : {vectors_written}")
    if collection is not None:
        stats = collection_stats(collection)
        print(f"collection       : {stats['name']}  count={stats['count']}")
        print(f"collection meta  : {stats['metadata']}")
    print(f"manifest         : {manifest_out}")
    print(f"chunks           : {chunks_out}")
    print(f"wall clock       : {elapsed:.2f}s")


if __name__ == "__main__":
    raise SystemExit(main())
