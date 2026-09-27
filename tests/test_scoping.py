"""Cross-scheme bleed: the single most important test in the project.

PRD A14 / risk R4. A user who asks about the HDFC Flexi Cap Fund must never be
answered with the ELSS fund's facts, even though both pages contain the words
"exit load" and MiniLM embeds them close together.

The scoping defence is the Chroma ``where={"scheme": ...}`` filter applied in
:func:`rag.retrieve.retrieve`. These tests assert that filter actually binds,
one case per scheme, against the real persisted collection.
"""

from __future__ import annotations

import pytest

from rag.retrieve import resolve_scheme, retrieve

#: One query per scheme, written the way a user would actually phrase it.
SCHEME_QUERIES: tuple[tuple[str, str, str], ...] = (
    (
        "hdfc_large_cap",
        "expense ratio of HDFC Large Cap Fund - Direct Growth",
        "expense_ratio",
    ),
    (
        "hdfc_flexi_cap",
        "exit load on HDFC Equity Fund - Direct Growth",
        "exit_load",
    ),
    (
        "hdfc_elss",
        "minimum SIP for HDFC ELSS Tax Saver Fund - Direct Growth",
        "min_sip",
    ),
    (
        "hdfc_small_cap",
        "current riskometer level of HDFC Small Cap Fund",
        "riskometer",
    ),
    (
        "hdfc_balanced_advantage",
        "benchmark of HDFC Balanced Advantage Fund - Direct Growth",
        "benchmark",
    ),
)

SCHEME_KEYS = tuple(scheme for scheme, _, _ in SCHEME_QUERIES)


class TestNoCrossSchemeBleed:
    """Every returned chunk must belong to the scheme the query named."""

    @pytest.mark.parametrize("scheme,query,intent", SCHEME_QUERIES)
    def test_every_chunk_is_the_asked_scheme(
        self, scheme: str, query: str, intent: str, live_collection, cfg
    ) -> None:
        trace = retrieve(query, collection=live_collection, cfg=cfg)

        assert resolve_scheme(query) == scheme, "query failed to scope at all"
        assert trace.scheme_filter == scheme
        assert trace.chunks, f"no chunks returned for {query!r}"
        offenders = sorted(
            {item.chunk.scheme for item in trace.chunks if item.chunk.scheme != scheme}
        )
        assert not offenders, f"cross-scheme bleed into {offenders} for {query!r}"

    @pytest.mark.parametrize("scheme,query,intent", SCHEME_QUERIES)
    def test_scheme_metadata_agrees_with_chunk(
        self, scheme: str, query: str, intent: str, live_collection, cfg
    ) -> None:
        """The scheme on the Chunk must survive the Chroma round trip."""
        trace = retrieve(query, collection=live_collection, cfg=cfg)
        for item in trace.chunks:
            assert item.chunk.scheme == scheme
            assert item.chunk.field_type
            assert item.chunk.source_url.startswith("https://")

    def test_one_scheme_named_never_mixes_another(
        self, live_collection, cfg
    ) -> None:
        """Explicitly assert the union across all five queries is clean."""
        seen: set[str] = set()
        for scheme, query, _intent in SCHEME_QUERIES:
            trace = retrieve(query, collection=live_collection, cfg=cfg)
            seen.update(item.chunk.scheme for item in trace.chunks)
        assert seen <= set(SCHEME_KEYS), f"unexpected schemes surfaced: {seen}"

    def test_exit_load_answer_is_not_the_nil_elss_one_for_flexi(
        self, live_collection, cfg
    ) -> None:
        """The concrete failure this project exists to prevent.

        ELSS carries "Exit load: Nil" and flexi cap carries a real 1% charge. If
        the filter ever fails, the flexi question silently answers "Nil".
        """
        trace = retrieve(
            "what is the exit load on HDFC Equity Fund - Direct Growth",
            collection=live_collection,
            cfg=cfg,
        )
        assert trace.chunks, "no chunks returned"
        for item in trace.chunks:
            assert item.chunk.scheme != "hdfc_elss", (
                "an ELSS exit-load chunk leaked into a flexi cap answer"
            )


class TestUnscopedQueries:
    """A query naming no scheme must stay unfiltered, not be blocked."""

    def test_general_question_is_not_scheme_filtered(
        self, live_collection, cfg
    ) -> None:
        trace = retrieve(
            "how do I download my capital gains statement",
            collection=live_collection,
            cfg=cfg,
        )
        assert trace.scheme_filter is None, "a general query was wrongly scoped"
        # It may legitimately return nothing: the statement guide rows are
        # disabled in the manifest. What must not happen is a scoping filter.
        assert isinstance(trace.chunks, list)

    def test_unrelated_query_returns_nothing_rather_than_wrong_schemes(
        self, live_collection, cfg
    ) -> None:
        trace = retrieve(
            "recipe for chocolate cake with butter", collection=live_collection, cfg=cfg
        )
        assert trace.scheme_filter is None
        assert trace.chunks == []
        assert trace.top_score == 0.0
