"""Shared pytest fixtures (docs/architecture.md, Phase 0).

Two groups live here:

* ``project_root`` / ``manifest_path`` / ``config`` — the Phase 0 path and config
  fixtures every layer of the test suite can rely on.
* the Phase 5 retriever fixtures — the real collection, the golden queries, and
  a synthetic-collection helper for testing retrieval logic without a corpus.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterator

import pytest

# Allow `import rag...` when pytest is run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the repository root."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def manifest_path() -> Path:
    """Absolute path to the corpus manifest CSV."""
    return PROJECT_ROOT / "ingest" / "sources.csv"


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    """Parsed config.yaml, skipped when it is absent."""
    pytest.importorskip("yaml", reason="PyYAML not installed")
    path = PROJECT_ROOT / "config.yaml"
    if not path.is_file():
        pytest.skip("config.yaml not found")
    return yaml_safe_load(path.read_text(encoding="utf-8"))


def yaml_safe_load(text: str) -> dict[str, Any]:
    """Parse YAML, importing the library lazily so the module imports without it."""
    import yaml

    return yaml.safe_load(text)


# ---------------------------------------------------------------------------
# Phase 5 retriever fixtures
# ---------------------------------------------------------------------------


class FakeEmbeddingFunction:
    """Deterministic stand-in for a sentence-transformer.

    Chroma requires an embedding function on the collection to answer
    ``query_texts``. This one is a hashing bag-of-words: each token is hashed
    into a bucket and counted, then the vector is L2-normalised.

    That choice matters for what the tests can prove. A random-per-text vector
    would put every similarity near zero, so the similarity floor would drop
    everything and the floor tests would pass for the wrong reason. With
    bag-of-words, text that shares words scores high and unrelated text scores
    near zero - the same qualitative behaviour as the real embedding, without a
    90 MB model download.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in (text or "").lower().split():
            # Strip punctuation so "ratio:" and "ratio" land in one bucket.
            token = token.strip(".,:;!?()[]{}\"'%₹/-")
            if not token:
                continue
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dimension] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            # An all-zero vector is not a point on the unit sphere; Chroma
            # rejects it. Park it on an arbitrary axis instead.
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return [self._vector(text) for text in input]

    def embed_query(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        """Chroma calls this for query_texts; the real models use the same path."""
        return self(input)

    @staticmethod
    def is_legacy() -> bool:
        """Chroma protocol requirement; this is a modern-config EF."""
        return False

    @staticmethod
    def name() -> str:
        """Chroma protocol requirement."""
        return "test_fake"

    def get_config(self) -> dict[str, Any]:
        """Chroma protocol requirement."""
        return {"dimension": self.dimension}

    @classmethod
    def build_from_config(
        cls, config: dict[str, Any]
    ) -> "FakeEmbeddingFunction":  # noqa: D102
        return cls(int(config["dimension"]))

    def default_space(self) -> str:
        """Chroma protocol requirement."""
        return "cosine"

    def supported_spaces(self) -> list[str]:
        """Chroma protocol requirement."""
        return ["cosine", "l2", "ip"]


@pytest.fixture(scope="session")
def cfg():
    """Retrieval thresholds from config.yaml."""
    from rag.config import RetrievalConfig, load_config

    return load_config().retrieval


@pytest.fixture(scope="session")
def golden_set() -> list[dict[str, Any]]:
    """The 10 golden queries, parsed."""
    path = PROJECT_ROOT / "eval" / "golden_set.jsonl"
    if not path.is_file():
        pytest.skip("no eval/golden_set.jsonl")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows, "eval/golden_set.jsonl is empty"
    return rows


@pytest.fixture(scope="session")
def corpus_chunks() -> list[dict[str, Any]]:
    """The stored chunks, as dicts, read from data/chunks/chunks.jsonl."""
    path = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"
    if not path.is_file():
        pytest.skip("no chunk corpus; run `make ingest` first")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="session")
def live_collection() -> Iterator[Any]:
    """The real persisted collection, opened read-only for the session."""
    chromadb = pytest.importorskip("chromadb")
    from rag.config import load_config

    config_obj = load_config()
    persist_dir = config_obj.vectorstore.persist_dir
    if not Path(persist_dir).is_dir():
        pytest.skip("no vector store; run `make ingest` first")

    client = chromadb.PersistentClient(path=str(persist_dir))
    names = {c.name for c in client.list_collections()}
    if config_obj.vectorstore.collection not in names:
        pytest.skip("collection absent; run `make ingest` first")

    yield client.get_collection(config_obj.vectorstore.collection)


def synthetic_collection(
    tmp_path: Path,
    chunks: list[dict[str, Any]],
    *,
    dimension: int = 32,
) -> Any:
    """A tiny in-tmp_path collection with deterministic fake vectors.

    Lets the retrieval logic be tested without the real corpus or the real
    model. A :class:`FakeEmbeddingFunction` is attached so ``query_texts``
    works: :func:`rag.retrieve.retrieve` passes text, not vectors, and Chroma
    would otherwise reach for its default 384-d model against an 8-d index.

    The fake vectors are pseudo-random unit vectors derived from a hash of the
    text, so identical text always yields an identical vector and different
    text yields an unrelated one - close enough to real cosine behaviour for the
    floor, the boost and the top-k cut to be exercised honestly.
    """
    chromadb = pytest.importorskip("chromadb")
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection(
        name="synthetic",
        metadata={"hnsw:space": "cosine", "dimension": dimension},
        embedding_function=FakeEmbeddingFunction(dimension),
    )
    embedder = FakeEmbeddingFunction(dimension)
    collection.upsert(
        ids=[chunk["chunk_id"] for chunk in chunks],
        embeddings=embedder([chunk["text"] for chunk in chunks]),
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[
            {
                "scheme": chunk["scheme"],
                "category": chunk.get("category", ""),
                "doc_type": chunk.get("doc_type", ""),
                "field_type": chunk.get("field_type", "other"),
                "section": chunk.get("section", ""),
                "source_url": chunk.get("source_url", ""),
                "publisher": chunk.get("publisher", ""),
                "retrieved_at": chunk.get("retrieved_at", ""),
                "ordinal": int(chunk.get("ordinal", 0)),
                "source_id": chunk.get("source_id", ""),
            }
            for chunk in chunks
        ],
    )
    return collection
