"""Vector store tests: idempotency, dimension guard, and metadata safety.

Phase 4 (docs/implementation.md). Everything here is offline and uses a tmp_path
Chroma directory, so no model download and no network is involved.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from ingest.store import (
    METADATA_KEYS,
    NONE_TEXT,
    VectorDimensionError,
    open_collection,
    upsert,
)
from rag.schemas import Chunk

DIMENSION = 8


def make_chunk(index: int, **overrides) -> Chunk:
    """Build a chunk with a deterministic id, for store-level tests."""
    base = dict(
        chunk_id=f"chunk-{index:04d}",
        source_id=f"src-{index % 2}",
        scheme="hdfc_large_cap",
        text=f"Expense ratio: {index}.0%",
        ordinal=index,
        category="large_cap",
        doc_type="scheme_page",
        field_type="expense_ratio",
        section="",
        source_url="https://example.com/scheme",
        publisher="groww",
        retrieved_at="2026-09-27T00:00:00+00:00",
    )
    base.update(overrides)
    return Chunk(**base)


def dummy_vectors(count: int, dimension: int = DIMENSION) -> list[list[float]]:
    """Unit-length random vectors, standing in for MiniLM output."""
    rng = random.Random(1234)
    vectors = []
    for _ in range(count):
        raw = [rng.uniform(-1, 1) for _ in range(dimension)]
        length = sum(value * value for value in raw) ** 0.5
        vectors.append([value / length for value in raw])
    return vectors


def test_upsert_then_reupsert_keeps_the_count(tmp_path: Path) -> None:
    """The same chunks written twice must not double the collection (FR-4.2, I7)."""
    collection = open_collection(
        persist_dir=tmp_path,
        collection_name="test_collection",
        dimension=DIMENSION,
        embedding_model="fake-model",
    )
    chunks = [make_chunk(i) for i in range(5)]
    vectors = dummy_vectors(5)

    assert upsert(collection, chunks, vectors) == 5
    assert collection.count() == 5

    # Second identical write.
    assert upsert(collection, chunks, vectors) == 0
    assert collection.count() == 5, "re-upserting the same chunks must be a no-op"


def test_changed_chunk_count_is_written(tmp_path: Path) -> None:
    """A different corpus must actually be written, not skipped as 'current'."""
    collection = open_collection(
        persist_dir=tmp_path,
        collection_name="test_collection",
        dimension=DIMENSION,
        embedding_model="fake-model",
    )
    upsert(collection, [make_chunk(i) for i in range(5)], dummy_vectors(5))
    assert collection.count() == 5

    # Same count, different ids: a count-only check would wrongly skip this.
    changed = [make_chunk(i + 100) for i in range(5)]
    assert upsert(collection, changed, dummy_vectors(5)) == 5
    assert collection.count() == 10, "different ids must be added, not skipped"


def test_open_rejects_a_mismatched_dimension(tmp_path: Path) -> None:
    """Writing 8-d vectors into a 16-d collection must raise, not corrupt (FR-4.2)."""
    open_collection(
        persist_dir=tmp_path,
        collection_name="test_collection",
        dimension=DIMENSION,
        embedding_model="fake-model",
    )

    with pytest.raises(VectorDimensionError) as excinfo:
        open_collection(
            persist_dir=tmp_path,
            collection_name="test_collection",
            dimension=DIMENSION * 2,
            embedding_model="fake-model",
        )
    assert "--rebuild" in str(excinfo.value)


def test_open_rejects_a_mismatched_model(tmp_path: Path) -> None:
    """Vectors from two different models are not comparable, so refuse them."""
    open_collection(
        persist_dir=tmp_path,
        collection_name="test_collection",
        dimension=DIMENSION,
        embedding_model="model-a",
    )
    with pytest.raises(VectorDimensionError) as excinfo:
        open_collection(
            persist_dir=tmp_path,
            collection_name="test_collection",
            dimension=DIMENSION,
            embedding_model="model-b",
        )
    assert "--rebuild" in str(excinfo.value)


def test_rebuild_replaces_a_mismatched_collection(tmp_path: Path) -> None:
    """--rebuild is the documented escape hatch from the dimension guard."""
    open_collection(
        persist_dir=tmp_path,
        collection_name="test_collection",
        dimension=DIMENSION,
        embedding_model="fake-model",
    )

    # Same dimension, different model, with rebuild: allowed and replaces the old.
    collection = open_collection(
        persist_dir=tmp_path,
        collection_name="test_collection",
        dimension=DIMENSION,
        embedding_model="other-model",
        rebuild=True,
    )
    upsert(collection, [make_chunk(0)], dummy_vectors(1))
    assert collection.count() == 1


def test_metadata_has_no_none_and_exact_keys(tmp_path: Path) -> None:
    """Chroma crashes on None, so every value must be str or int (Phase 4 gate)."""
    collection = open_collection(
        persist_dir=tmp_path,
        collection_name="test_collection",
        dimension=DIMENSION,
        embedding_model="fake-model",
    )
    # section="" and ordinal=0 are the realistic awkward cases.
    chunks = [make_chunk(i, section="") for i in range(3)]
    upsert(collection, chunks, dummy_vectors(3))

    stored = collection.get(include=["metadatas"])
    for metadata in stored["metadatas"]:
        assert set(metadata) == set(METADATA_KEYS)
        for key, value in metadata.items():
            assert value is not None, f"{key} is None"
            assert isinstance(value, (str, int)), f"{key} is {type(value)}"
            if not value:
                assert value in ("", 0, NONE_TEXT)


def test_missing_optional_fields_become_none_text(tmp_path: Path) -> None:
    """A chunk with no section still stores, rather than crashing the insert."""
    collection = open_collection(
        persist_dir=tmp_path,
        collection_name="test_collection",
        dimension=DIMENSION,
        embedding_model="fake-model",
    )
    upsert(collection, [make_chunk(0, section=None)], dummy_vectors(1))
    stored = collection.get(include=["metadatas"])
    assert stored["metadatas"][0]["section"] == NONE_TEXT


def test_upsert_rejects_misaligned_vectors(tmp_path: Path) -> None:
    """A length mismatch is a programming error and must be loud."""
    collection = open_collection(
        persist_dir=tmp_path,
        collection_name="test_collection",
        dimension=DIMENSION,
        embedding_model="fake-model",
    )
    with pytest.raises(ValueError, match="same length"):
        upsert(collection, [make_chunk(0), make_chunk(1)], dummy_vectors(1))


def test_upsert_of_nothing_is_a_noop(tmp_path: Path) -> None:
    collection = open_collection(
        persist_dir=tmp_path,
        collection_name="test_collection",
        dimension=DIMENSION,
        embedding_model="fake-model",
    )
    assert upsert(collection, [], []) == 0
    assert collection.count() == 0


def test_documents_round_trip_unchanged(tmp_path: Path) -> None:
    """The stored document must be the chunk text, byte for byte."""
    collection = open_collection(
        persist_dir=tmp_path,
        collection_name="test_collection",
        dimension=DIMENSION,
        embedding_model="fake-model",
    )
    chunks = [make_chunk(i, text=f"Fund benchmark: NIFTY {i}") for i in range(3)]
    upsert(collection, chunks, dummy_vectors(3))
    stored = collection.get(include=["documents"])
    assert sorted(stored["documents"]) == sorted(c.text for c in chunks)
