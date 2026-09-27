"""Human-readable dumps of the corpus, chunks, and embeddings.

Everything the pipeline produces is machine-readable (JSONL, a binary vector
index), which makes it hard to eyeball whether the pipeline actually did the
right thing. This module writes the same data as plain text.

Usage::

    python -m ingest.inspect_chunks                  # both dumps
    python -m ingest.inspect_chunks --chunks-only
    python -m ingest.inspect_chunks --embeddings-only
    python -m ingest.inspect_chunks --limit 20       # first 20 of each
    python -m ingest.inspect_chunks --dims 12        # show 12 dims per vector
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from rag.config import load_config
from rag.schemas import Chunk

#: Defaults chosen so the file stays readable. A full 384-d vector on one line
#: is unreadable, and 1055 of them is unreadable for a different reason.
DEFAULT_LIMIT = 40
DEFAULT_DIMS = 16


def _rule(char: str = "-", width: int = 78) -> str:
    return char * width


def _interesting_first(entries: Sequence[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """Order entries so recognisable facts come before bulk holdings rows.

    Each entry is ``(chunk_id, field_type, ...)``; only the first two are read,
    so callers may attach a payload.

    Chunks are keyed by content hash, so a plain id sort shows whichever holding
    row happens to hash first - technically representative, useless for
    checking that expense ratio or exit load made it into the store.
    """
    def sort_key(entry: tuple[Any, ...]) -> tuple[int, Any]:
        field_type = entry[1] if len(entry) > 1 else None
        return (0 if field_type and field_type != "other" else 1, entry[0])

    return sorted(entries, key=sort_key)


def load_chunks(path: str | Path) -> list[Chunk]:
    """Read chunks back from the JSONL file written by :func:`ingest.chunk.write_chunks`."""
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(
            f"{file} does not exist. Run `python -m ingest.run_ingest` first."
        )
    chunks: list[Chunk] = []
    for line in file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            chunks.append(Chunk(**json.loads(line)))
    return chunks

def write_chunks_txt(
    chunks: Sequence[Chunk],
    path: str | Path,
    *,
    limit: int | None = None,
) -> Path:
    """Write a readable dump of the chunks.

    Args:
        chunks: the chunks to describe.
        path: destination ``.txt`` file.
        limit: only describe the first ``limit`` chunks.

    Returns:
        The path written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = _interesting_first(
        [(chunk.chunk_id, chunk.field_type, chunk) for chunk in chunks]
    )
    shown = ordered if limit is None else ordered[:limit]

    lines: list[str] = [
        "CHUNKS",
        "=" * 78,
        f"total chunks : {len(chunks)}",
        f"showing      : {len(shown)}",
        "note         : recognised facts are listed first, then bulk rows",
        "",
    ]
    for _, _, chunk in shown:
        lines.append(_rule("="))
        lines.append(f"chunk_id   : {chunk.chunk_id}")
        lines.append(f"source_id  : {chunk.source_id}")
        lines.append(f"scheme     : {chunk.scheme}")
        lines.append(f"field_type : {chunk.field_type}")
        lines.append(f"ordinal    : {chunk.ordinal}")
        lines.append(f"section    : {chunk.section or '-'}")
        lines.append(f"chars      : {len(chunk.text)}")
        lines.append(f"citation   : {chunk.source_url}")
        lines.append(_rule())
        lines.append(chunk.text)
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_embeddings_txt(
    chunks: Sequence[Chunk],
    path: str | Path,
    *,
    limit: int | None = None,
    dims: int = DEFAULT_DIMS,
) -> Path:
    """Write a readable dump of the stored vectors.

    Each entry shows the chunk, its first ``dims`` components, and the L2 norm
    and cosine similarity of the *full* vector against the first chunk's - the
    norm matters because the vectors are normalised, so a norm drifting from 1.0
    means the embedding pipeline changed under you.

    Args:
        chunks: chunks whose vectors are stored, aligned with the collection.
        path: destination ``.txt`` file.
        limit: only describe the first ``limit`` vectors.
        dims: how many leading components to print per vector.

    Returns:
        The path written.
    """
    import chromadb

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    config = load_config()

    client = chromadb.PersistentClient(path=str(config.vectorstore.persist_dir))
    collection = client.get_collection(config.vectorstore.collection)

    stored = collection.get(include=["embeddings", "documents", "metadatas"])
    ids: list[str] = stored.get("ids", [])
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    total = len(ids)

    lines: list[str] = [
        "EMBEDDINGS",
        "=" * 78,
        f"collection     : {config.vectorstore.collection}",
        f"persist_dir    : {config.vectorstore.persist_dir}",
        f"stored vectors : {total}",
        f"model          : {config.embedding.model}",
        f"dimension      : {config.embedding.dimension}",
        f"normalised     : {config.embedding.normalize}",
        f"showing        : {min(limit, total) if limit else total}",
        f"dims per vector: first {dims} of {config.embedding.dimension}",
        "",
    ]
    if not total:
        lines.append("no vectors stored. Run `python -m ingest.run_ingest`.")
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    # Sorted so the dump is stable across runs, facts before bulk rows. The
    # collection's own position is carried along as the payload.
    metadatas = stored.get("metadatas") or [{}] * total
    order = [
        entry[2]
        for entry in _interesting_first(
            [
                (ids[i], (metadatas[i] or {}).get("field_type"), i)
                for i in range(total)
            ]
        )
    ]
    chosen = order if limit is None else order[:limit]

    embeddings = stored.get("embeddings")
    reference = list(embeddings[order[0]]) if embeddings is not None else None

    lines.append("first vector in full, so the truncated rows below can be compared:")
    if reference is not None:
        lines.append(f"  {ids[order[0]]}")
        lines.append(f"  {_format_vector(reference, 16)}")
        lines.append(f"  ... ({len(reference) - 16} more)")
        lines.append(f"  norm={_norm(reference):.6f}")
        lines.append("")

    for position in chosen:
        chunk_id = ids[position]
        chunk = by_id.get(chunk_id)
        metadata = (stored.get("metadatas") or [{}] * total)[position] or {}
        vector = list(embeddings[position]) if embeddings is not None else []

        lines.append(_rule("="))
        lines.append(f"chunk_id   : {chunk_id}")
        lines.append(f"field_type : {metadata.get('field_type', '?')}")
        lines.append(f"scheme     : {metadata.get('scheme', '?')}")
        lines.append(f"ordinal    : {metadata.get('ordinal', '?')}")
        if chunk is not None:
            preview = chunk.text.replace("\n", " ")
            if len(preview) > 150:
                preview = preview[:150] + "..."
            lines.append(f"text       : {preview}")
        lines.append(f"dims       : {len(vector)}")
        lines.append(f"vector     : {_format_vector(vector, dims)}")
        if len(vector) > dims:
            lines.append(f"             ... ({len(vector) - dims} more)")
        lines.append(f"norm       : {_norm(vector):.6f}")
        if reference is not None and vector:
            lines.append(f"cosine(v0) : {_cosine(vector, reference):.6f}")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _format_vector(vector: Sequence[float], dims: int) -> str:
    shown = ", ".join(f"{value:+.5f}" for value in vector[:dims])
    if len(vector) <= dims:
        return f"[{shown}]"
    return f"[{shown}, ..."


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    denom = _norm(left) * _norm(right)
    return dot / denom if denom else 0.0


def main(argv: list[str] | None = None) -> int:
    """Write the txt dumps requested on the command line."""
    parser = argparse.ArgumentParser(
        prog="ingest.inspect_chunks",
        description="Write human-readable dumps of the chunks and embeddings.",
    )
    parser.add_argument("--chunks-only", action="store_true")
    parser.add_argument("--embeddings-only", action="store_true")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--dims", type=int, default=DEFAULT_DIMS)
    parser.add_argument("--out-dir", default=None, help="defaults to data/inspect")
    args = parser.parse_args(argv)

    config = load_config()
    out_dir = Path(args.out_dir) if args.out_dir else config.root / "data" / "inspect"
    chunks = load_chunks(config.chunks_jsonl)
    print(f"loaded {len(chunks)} chunks from {config.chunks_jsonl}")

    do_chunks = not args.embeddings_only
    do_embeddings = not args.chunks_only

    if do_chunks:
        target = out_dir / "chunks.txt"
        write_chunks_txt(chunks, target, limit=args.limit)
        print(f"wrote {target}")

    if do_embeddings:
        target = out_dir / "embeddings.txt"
        write_embeddings_txt(chunks, target, limit=args.limit, dims=args.dims)
        print(f"wrote {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
