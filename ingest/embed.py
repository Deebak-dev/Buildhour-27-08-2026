"""Chunk → vector conversion.

Flow A, stage 3 — "Embedding" (docs/architecture.md §5.5, PRD FR-3.1–FR-3.3).

The model is ``sentence-transformers/all-MiniLM-L6-v2`` (384-d), CPU only. It is
downloaded once on first use and fully offline afterwards (architecture.md
NFR-9), so the corpus can be rebuilt with no network.

Nothing here touches the vector store; :mod:`ingest.store` owns that.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Sequence

from rag.config import EmbedConfig
from rag.schemas import Chunk

if TYPE_CHECKING:  # pragma: no cover - import cost avoided at runtime
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingDimensionError(RuntimeError):
    """Raised when the model returns vectors of an unexpected width.

    This guard exists to catch a silent model swap. Without it, writing 768-d
    vectors into a collection declared as 384-d either fails deep inside ChromaDB
    or, worse, succeeds against the wrong index and returns nonsense neighbours
    with no error at all.
    """


@functools.lru_cache(maxsize=2)
def get_model(model_name: str) -> "SentenceTransformer":
    """Load and cache the embedding model.

    Args:
        model_name: a sentence-transformers model id, read from ``config.yaml``.

    Returns:
        The cached model. Repeated calls with the same name are free.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - environment problem
        raise RuntimeError(
            "sentence-transformers is not installed. Run `make setup` "
            "(python -m pip install -r requirements.txt)."
        ) from exc

    logger.info("loading embedding model %s (cpu)", model_name)
    try:
        # device="cpu" is explicit: NFR-2 forbids silently using a GPU, and an
        # absent argument means the answer changes with the host hardware.
        return SentenceTransformer(model_name, device="cpu")
    except Exception as exc:  # noqa: BLE001 - re-raised with actionable advice
        raise RuntimeError(
            f"could not load embedding model {model_name!r}. The weights are "
            "downloaded on first use and need network access once; after that "
            f"the model is cached locally. Original error: {exc}"
        ) from exc


def embed_chunks(
    chunks: Sequence[Chunk], *, cfg: EmbedConfig | None = None
) -> list[list[float]]:
    """Embed chunk texts, returning one plain float list per chunk.

    Args:
        chunks: the chunks to embed, in the order they should be stored.
        cfg: embedding settings. Defaults to ``config.yaml``.

    Returns:
        A list of ``cfg.dimension``-length lists of floats, positionally aligned
        with ``chunks``.

    Raises:
        EmbeddingDimensionError: if the model returns a different width, or if
            it returns a different number of vectors than chunks.
    """
    settings = cfg or EmbedConfig()
    if not chunks:
        return []

    model = get_model(settings.model)
    texts = [chunk.text for chunk in chunks]
    vectors = model.encode(
        texts,
        batch_size=settings.batch_size,
        normalize_embeddings=settings.normalize,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    if len(vectors) != len(chunks):
        raise EmbeddingDimensionError(
            f"embedding count mismatch: got {len(vectors)} vectors for "
            f"{len(chunks)} chunks. The model skipped empty or failed texts."
        )

    width = len(vectors[0])
    if width != settings.dimension:
        raise EmbeddingDimensionError(
            f"embedding dimension mismatch: model {settings.model!r} produced "
            f"{width}-d vectors but config.yaml declares "
            f"dimension={settings.dimension}. Either the model was swapped or "
            "the config is stale; do not write these vectors to the store."
        )

    logger.info(
        "embedded %d chunk(s) into %d-d vectors with %s",
        len(chunks),
        width,
        settings.model,
    )
    return [vector.tolist() for vector in vectors]
