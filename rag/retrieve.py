"""Query → scoped, top-k, thresholded chunks, with a trace.

Flow B, stage 2 (docs/architecture.md §6.4, docs/PRD.md FR-5.1–FR-5.4).

Three things happen here, in this order, and each one is load-bearing:

1. **Scheme scoping.** If the query names a scheme, the vector search is filtered
   to that scheme. This is the defence against cross-scheme bleed (PRD A14, risk
   R4) — answering "what is the exit load on the flexi cap fund" with the ELSS
   fund's "Exit load: Nil" is the specific failure this project exists to avoid.
2. **Field-type boosting.** A question about exit load gets a small similarity
   boost for chunks tagged ``exit_load``, so a fact chunk outranks a page of
   holdings rows that happens to mention the words.
3. **A similarity floor.** Chunks below the floor are dropped. A query with
   nothing relevant returns *zero chunks*, which is the correct signal for the
   "not in my sources" answer — not an error.

No LLM call and no network access happens in this module. Field intent is read
with the same precedence table the chunker used (:mod:`rag.fieldmap`).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from rag.config import RetrievalConfig
from rag.fieldmap import match_field_type
from rag.schemas import SCHEMES, Chunk, RetrievalTrace, ScoredChunk

logger = logging.getLogger(__name__)

#: ``field_type`` values whose chunks describe *other schemes* rather than a
#: named field of the scheme being asked about.
#:
#: A ``scheme_type`` chunk is a list of sibling funds ("HDFC Equity Savings Direct
#: Plan Growth: HDFC Multi Asset Fund..."), so it embeds close to any query that
#: merely names a fund. Measured on golden ``q02`` ("exit load on HDFC Equity Fund
#: - Direct Growth"), such a chunk scored 0.773 raw while the correct
#: ``exit_load`` chunk scored 0.401 raw and 0.751 after the boost - losing by
#: 0.022 and answering a field question with a fund list.
#:
#: Boosting alone cannot fix that: closing a 0.37 raw gap would need a boost far
#: larger than the similarity floor, which would distort genuine matches. The
#: honest fix is to recognise that for a *field* question these chunks are
#: off-topic by construction, and demote them by
#: :attr:`RetrievalConfig.field_misfit_penalty`.
#:
#: The value is not knife-edge: golden top-1 is 5/10 at penalty 0.0 and a stable
#: 6/10 for every value from 0.05 to 0.40, with no other query changing state.
#: 6/10 is the corpus ceiling (four golden queries have no evidence at all).
OFF_TOPIC_FIELD_TYPES: frozenset[str] = frozenset({"scheme_type"})

#: Keys copied out of a Chroma metadata dict onto a :class:`Chunk`. Chroma
#: returns metadata as plain strings and ints, which is exactly what Chunk wants.
_METADATA_FIELDS = (
    "source_id",
    "scheme",
    "category",
    "doc_type",
    "field_type",
    "section",
    "source_url",
    "publisher",
    "retrieved_at",
)

#: Reciprocal Rank Fusion constant (FR-5.4). 60 is the value from the original
#: RRF paper and is deliberately not configurable: it is a smoothing term, not a
#: threshold, and tuning it would mean tuning against a single golden set.
_RRF_K = 60


# ---------------------------------------------------------------------------
# Query understanding (pure, no model)
# ---------------------------------------------------------------------------


def resolve_scheme(query: str) -> str | None:
    """Return the scheme key a query names, or ``None``.

    Tolerant by design: "large cap", "largecap", "HDFC Large Cap Fund" and
    "large cap fund" all resolve to ``hdfc_large_cap``. Longer aliases win over
    shorter ones so that "balanced advantage" is not swallowed by "balanced".

    Args:
        query: the raw user question.

    Returns:
        A key from :data:`~rag.schemas.SCHEME_KEYS`, or ``None`` when the query
        names no scheme — a general question must stay unfiltered.
    """
    text = (query or "").lower()
    if not text.strip():
        return None

    candidates: list[tuple[int, int, str]] = []
    for spec in SCHEMES:
        # Every spelling we know about: the aliases, the display name, and the
        # name the source page itself uses.
        spellings = {
            *spec.aliases,
            spec.display_name,
            spec.page_title,
            spec.category.replace("_", " "),
            spec.key.replace("_", " "),
        }
        for spelling in spellings:
            needle = spelling.lower().strip()
            if not needle:
                continue
            at = text.find(needle)
            if at < 0:
                continue
            # Longest match at the earliest position wins, so a specific alias
            # beats a generic one that merely appears later in the sentence.
            candidates.append((at, -len(needle), spec.key))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def detect_field_intent(query: str) -> str | None:
    """Return the field type a question is asking about, or ``None``.

    Uses the shared precedence table so this cannot disagree with the tagger
    that labelled the chunks being boosted.

    Args:
        query: the raw user question.

    Returns:
        A field-type name, or ``None`` when the question names no field.
    """
    return match_field_type(query or "")


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def _chunk_from_metadata(
    chunk_id: str, document: str, metadata: dict[str, Any] | None
) -> Chunk:
    """Rebuild a :class:`Chunk` from a Chroma record.

    Chroma stores metadata as strings and ints only, so the ordinal has to be
    coerced back to an int and anything missing falls back to the empty string
    rather than ``None`` (``Chunk`` is typed as ``str``).
    """
    data = dict(metadata or {})
    fields: dict[str, Any] = {
        name: data.get(name, "") or "" for name in _METADATA_FIELDS
    }
    try:
        ordinal = int(data.get("ordinal", 0) or 0)
    except (TypeError, ValueError):
        ordinal = 0
    return Chunk(
        chunk_id=chunk_id,
        text=document or "",
        ordinal=ordinal,
        **fields,
    )


def _query_collection(
    collection: Any, query: str, where: dict[str, str] | None, n_results: int
) -> list[tuple[str, str, dict[str, Any], float]]:
    """Run the vector search and return ``(id, document, metadata, similarity)``.

    Chroma returns cosine *distance* in a ``0..2`` range; similarity is
    ``1 - distance`` so that "higher is better" and "closer to 1 is better" are
    the same statement, which keeps the floor and the boost in one unit.
    """
    result = collection.query(
        query_texts=[query],
        n_results=max(1, n_results),
        where=where or None,
        include=["documents", "metadatas", "distances"],
    )

    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    rows: list[tuple[str, str, dict[str, Any], float]] = []
    for index, chunk_id in enumerate(ids):
        document = documents[index] if index < len(documents) else ""
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = float(distances[index]) if index < len(distances) else 1.0
        rows.append((chunk_id, document or "", dict(metadata or {}), 1.0 - distance))
    return rows


def retrieve(
    query: str, *, collection: Any, cfg: RetrievalConfig | None = None
) -> RetrievalTrace:
    """Resolve a question to a scoped, boosted, floored set of chunks.

    Args:
        query: the raw user question.
        collection: an open Chroma collection.
        cfg: retrieval thresholds. Defaults to ``config.yaml``.

    Returns:
        A :class:`~rag.schemas.RetrievalTrace`. ``trace.chunks`` is empty when
        nothing cleared the floor, which is a valid answer rather than a failure.
    """
    settings = cfg or RetrievalConfig()

    scheme = resolve_scheme(query)
    # No scheme named → no filter. A general question like "how do I download my
    # capital gains statement" must still be able to reach a statement guide.
    where = {"scheme": scheme} if scheme else None

    intent = detect_field_intent(query)

    rows = _query_collection(
        collection,
        query,
        where,
        settings.top_k + settings.over_fetch,
    )

    scored: list[ScoredChunk] = []
    for chunk_id, document, metadata, similarity in rows:
        chunk = _chunk_from_metadata(chunk_id, document, metadata)
        boosted = False
        final = similarity
        field = metadata.get("field_type")
        if intent and field == intent:
            final += settings.field_type_boost
            boosted = True
        elif intent and field in OFF_TOPIC_FIELD_TYPES:
            final -= settings.field_misfit_penalty
        scored.append(
            ScoredChunk(
                chunk=chunk,
                similarity=final,
                raw_similarity=similarity,
                boosted=boosted,
            )
        )

    # Highest boosted similarity first.
    scored.sort(key=lambda item: item.similarity, reverse=True)
    scored = scored[: settings.top_k]

    # The floor is applied after the boost and after truncation, so a boosted
    # chunk can clear it but nothing below it survives regardless of rank.
    kept = [item for item in scored if item.similarity >= settings.similarity_floor]

    trace = RetrievalTrace(
        query=query,
        scheme_filter=scheme,
        chunks=kept,
        tau=settings.similarity_floor,
        top_score=kept[0].similarity if kept else 0.0,
        intent=intent,
    )
    logger.info(
        "retrieve: scheme=%s intent=%s candidates=%d kept=%d top=%.3f tau=%.3f",
        scheme or "-",
        intent or "-",
        len(rows),
        len(kept),
        trace.top_score,
        trace.tau,
    )
    return trace


# ---------------------------------------------------------------------------
# Optional hybrid mode (FR-5.4, open question Q5)
# ---------------------------------------------------------------------------


def _bm25_scores(
    query: str, collection: Any, cfg: RetrievalConfig
) -> dict[str, float]:
    """Score every stored chunk with BM25 over its text.

    Optional per FR-5.4. BM25 is exact-token matching, so it catches the literal
    field names ("Min. for SIP") that a 384-d sentence embedding tends to blur
    into its neighbours.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "retrieval.mode is 'hybrid', which needs rank_bm25. Install it with "
            "`python -m pip install rank_bm25`, or set retrieval.mode back to "
            "'vector' in config.yaml."
        ) from exc

    stored = collection.get(include=["documents"])
    ids = stored.get("ids") or []
    documents = stored.get("documents") or []
    if not documents:
        return {}

    corpus = [str(document or "").lower().split() for document in documents]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores((query or "").lower().split())
    return {
        chunk_id: float(score)
        for chunk_id, score in zip(ids, scores)
        if float(score) > 0.0
    }


def hybrid_search(
    query: str, *, collection: Any, cfg: RetrievalConfig | None = None
) -> RetrievalTrace:
    """Fuse the dense and lexical rankings, then apply the same floor.

    Optional per FR-5.4 and open question Q5; the shipped default is ``vector``.
    Merging is Reciprocal Rank Fusion (``score = sum(1 / (k + rank))``, k=60),
    chosen because it needs no score calibration between two incomparable
    scoring systems — cosine similarity and BM25 are not on a shared scale.

    Args:
        query: the raw user question.
        collection: an open Chroma collection.
        cfg: retrieval thresholds. Defaults to ``config.yaml``.

    Returns:
        A trace whose ``chunks`` are ranked by fused score and floored exactly as
        in :func:`retrieve`.
    """
    settings = cfg or RetrievalConfig()
    scheme = resolve_scheme(query)
    where = {"scheme": scheme} if scheme else None
    intent = detect_field_intent(query)

    dense = _query_collection(
        collection, query, where, settings.top_k + settings.over_fetch
    )
    lexical = _bm25_scores(query, collection, settings)
    if where:
        allowed = {chunk_id for chunk_id, _, _, _ in dense}
        lexical = {k: v for k, v in lexical.items() if k in allowed}

    stored_meta = collection.get(ids=list(lexical), include=["documents", "metadatas"])
    meta_by_id = dict(zip(stored_meta.get("ids") or [], stored_meta.get("metadatas") or []))
    docs_by_id = dict(
        zip(stored_meta.get("ids") or [], stored_meta.get("documents") or [])
    )
    dense_by_id = {chunk_id: (doc, meta, sim) for chunk_id, doc, meta, sim in dense}

    fused: dict[str, float] = {}
    for rank, (chunk_id, _doc, _meta, _sim) in enumerate(dense, start=1):
        fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (_RRF_K + rank)
    for rank, chunk_id in enumerate(lexical, start=1):
        fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (_RRF_K + rank)

    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)[
        : settings.top_k
    ]

    # The fused score is a rank statistic in roughly 0..0.033, so it cannot be
    # compared with the vector floor directly. Map it back onto the dense
    # similarity scale so the configured threshold keeps its meaning.
    scored: list[ScoredChunk] = []
    for chunk_id, fused_score in ordered:
        document, metadata, similarity = dense_by_id.get(
            chunk_id, (docs_by_id.get(chunk_id, ""), meta_by_id.get(chunk_id, {}), 0.0)
        )
        boosted = bool(intent and (meta_by_id.get(chunk_id) or {}).get("field_type") == intent)
        final = similarity + (settings.field_type_boost if boosted else 0.0)
        # Fused evidence acts as a gate, not a replacement: a chunk that only BM25
        # found still has to clear the floor on its own similarity.
        if chunk_id not in dense_by_id and fused_score <= 0.0:
            continue
        scored.append(
            ScoredChunk(
                chunk=_chunk_from_metadata(chunk_id, document, metadata),
                similarity=final,
                raw_similarity=similarity,
                boosted=boosted,
            )
        )

    kept = [item for item in scored if item.similarity >= settings.similarity_floor]
    return RetrievalTrace(
        query=query,
        scheme_filter=scheme,
        chunks=kept,
        tau=settings.similarity_floor,
        top_score=kept[0].similarity if kept else 0.0,
        intent=intent,
    )


def search(
    query: str, *, collection: Any, cfg: RetrievalConfig | None = None
) -> RetrievalTrace:
    """Dispatch to the configured retrieval mode.

    Args:
        query: the raw user question.
        collection: an open Chroma collection.
        cfg: retrieval thresholds. Defaults to ``config.yaml``.

    Returns:
        A trace from :func:`retrieve` or :func:`hybrid_search`.
    """
    settings = cfg or RetrievalConfig()
    if settings.mode == "hybrid":
        return hybrid_search(query, collection=collection, cfg=settings)
    return retrieve(query, collection=collection, cfg=settings)


def recall_at_k(
    trace: RetrievalTrace, k: int, *, field_type: str | None = None
) -> bool:
    """Whether a trace's top ``k`` contains the expected evidence.

    Args:
        trace: the trace to judge.
        k: how many results to consider.
        field_type: when given, a hit requires this ``field_type``.

    Returns:
        ``True`` when at least one of the top ``k`` chunks matches.
    """
    for item in trace.chunks[:k]:
        if field_type is None or item.chunk.field_type == field_type:
            return True
    return False
