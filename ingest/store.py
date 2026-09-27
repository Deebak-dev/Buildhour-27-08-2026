"""Vector store persistence.

Flow A, stage 4 — "Vector store" (docs/architecture.md §5.5, PRD FR-4.1–FR-4.3).

ChromaDB in persistent mode, cosine space, written idempotently: the same chunks
embedded twice must leave the collection at the same size (PRD FR-4.2, invariant
I7).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

from rag.schemas import Chunk

# Phase 7 moves log_event into rag/logging_utils.py; importing it from here
# until then keeps a single definition rather than a second copy.
from ingest.load import log_event

logger = logging.getLogger(__name__)
#: The exact metadata keys written for every chunk. Chroma rejects None and
#: nested objects, so every value is coerced to str or int by _metadata.
METADATA_KEYS: tuple[str, ...] = (
    "scheme",
    "category",
    "doc_type",
    "field_type",
    "section",
    "source_url",
    "publisher",
    "retrieved_at",
    "ordinal",
    "source_id",
)

#: Stand-in for a missing value. Chroma raises on None rather than storing it,
#: and the literal string sorts and displays better than a sentinel object.
NONE_TEXT = "none"


class VectorDimensionError(RuntimeError):
    """Raised when the existing collection was built with a different width.

    Writing mismatched vectors into a persisted HNSW index produces results that
    look plausible and are wrong, so this refuses instead (PRD FR-4.2).
    """


def open_collection(
    *,
    persist_dir: str | Path,
    collection_name: str,
    dimension: int,
    embedding_model: str,
    rebuild: bool = False,
):
    """Open, or create, the persistent collection.

    Args:
        persist_dir: directory ChromaDB persists into.
        collection_name: collection name from ``config.yaml``.
        dimension: expected vector width.
        embedding_model: recorded in collection metadata for provenance.
        rebuild: delete and recreate the collection first.

    Returns:
        A ``chromadb.Collection``.

    Raises:
        VectorDimensionError: if the collection exists with a different stored
            dimension or model, and ``rebuild`` was not requested.
    """
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - environment problem
        raise RuntimeError(
            "chromadb is not installed. Run `make setup` "
            "(python -m pip install -r requirements.txt)."
        ) from exc

    path = Path(persist_dir)
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))

    if rebuild:
        # Checked rather than caught: chromadb has raised ValueError and
        # NotFoundError from delete_collection across versions, and a missing
        # collection on a first run is normal, not an error.
        existing = {c.name for c in client.list_collections()}
        if collection_name in existing:
            client.delete_collection(collection_name)
            logger.info("deleted collection %s (--rebuild)", collection_name)

    metadata = {
        "hnsw:space": "cosine",
        "embedding_model": embedding_model,
        "dimension": int(dimension),
    }
    collection = client.get_or_create_collection(
        name=collection_name, metadata=metadata
    )

    existing = dict(collection.metadata or {})
    stored_dimension = existing.get("dimension")
    stored_model = existing.get("embedding_model")
    if stored_dimension is not None and int(stored_dimension) != int(dimension):
        raise VectorDimensionError(
            f"collection {collection_name!r} already holds {stored_dimension}-d "
            f"vectors but config.yaml declares {dimension}-d. The two indexes "
            "are not comparable, so nothing was written. Re-run with --rebuild "
            "to discard the old collection and start clean."
        )
    if stored_model is not None and stored_model != embedding_model:
        raise VectorDimensionError(
            f"collection {collection_name!r} was built with embedding model "
            f"{stored_model!r} but config.yaml now specifies "
            f"{embedding_model!r}. Vectors from different models are not "
            "comparable, so nothing was written. Re-run with --rebuild."
        )

    return collection


def _metadata(chunk: Chunk) -> dict[str, str | int]:
    """Build the Chroma metadata dict for one chunk.

    Every value is a ``str`` or ``int``. A ``None`` anywhere in here crashes the
    insert, which is why ``section`` and friends are coerced rather than trusted.
    """
    raw: dict[str, Any] = {
        "scheme": chunk.scheme,
        "category": chunk.category,
        "doc_type": chunk.doc_type,
        "field_type": chunk.field_type,
        "section": chunk.section,
        "source_url": chunk.source_url,
        "publisher": chunk.publisher,
        "retrieved_at": chunk.retrieved_at,
        "ordinal": chunk.ordinal,
        "source_id": chunk.source_id,
    }
    return {
        key: NONE_TEXT if value is None else value
        for key, value in raw.items()
    }


def upsert(
    collection: Any,
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
) -> int:
    """Write chunks and vectors into the collection, idempotently.

    Args:
        collection: an open Chroma collection.
        chunks: the chunks to store.
        vectors: positionally aligned embeddings for ``chunks``.

    Returns:
        The number of chunks written (0 when the collection was already current).

    Raises:
        ValueError: if the lengths of ``chunks`` and ``vectors`` disagree.
    """
    if len(chunks) != len(vectors):
        raise ValueError(
            f"chunks and vectors must be the same length: got {len(chunks)} "
            f"chunks and {len(vectors)} vectors"
        )
    if not chunks:
        return 0

    if _is_current(collection, chunks):
        logger.info("collection already current, skipping write")
        # Structured twin of the line above: run_ingest emits stage/outcome
        # events, and the store is the one stage whose skip is a normal,
        # expected outcome rather than something to debug.
        log_event("store", outcome="already_current", n_chunks=len(chunks))
        return 0

    collection.upsert(
        ids=[chunk.chunk_id for chunk in chunks],
        embeddings=[list(vector) for vector in vectors],
        documents=[chunk.text for chunk in chunks],
        metadatas=[_metadata(chunk) for chunk in chunks],
    )
    logger.info("upserted %d chunk(s) into %s", len(chunks), collection.name)
    return len(chunks)


def _is_current(collection: Any, chunks: Sequence[Chunk]) -> bool:
    """Whether the collection already holds exactly these chunk ids.

    Count alone is not enough to prove the corpus is unchanged — the same number
    of chunks with different content would pass a count check — so the ids are
    compared as well (PRD FR-4.2).
    """
    try:
        stored = collection.count()
    except Exception:  # noqa: BLE001 - a missing/empty collection is not current
        logger.debug("collection count unavailable, treating as not current")
        return False
    if stored != len(chunks):
        return False

    wanted = {chunk.chunk_id for chunk in chunks}
    try:
        have = set(collection.get(include=[]).get("ids", []))
    except Exception:  # noqa: BLE001 - fall back to "not current"
        return False
    return have == wanted


def collection_stats(collection: Any) -> dict[str, Any]:
    """Return a small summary of a collection, for the CLI and the tests."""
    return {
        "name": collection.name,
        "count": collection.count(),
        "metadata": dict(collection.metadata or {}),
    }
