"""Retriever unit tests: scheme resolution, field intent, floor, and boosting.

These cover the pure query-understanding functions exhaustively (no corpus
needed) and the retrieval mechanics against a synthetic collection, so a
regression is reported precisely rather than as a fuzzy recall number.
"""

from __future__ import annotations

import pytest

from rag.config import RetrievalConfig
from rag.fieldmap import match_field_type
from rag.retrieve import (
    OFF_TOPIC_FIELD_TYPES,
    detect_field_intent,
    hybrid_search,
    recall_at_k,
    resolve_scheme,
    retrieve,
    search,
)
from rag.schemas import SCHEME_KEYS


class TestResolveScheme:
    """Tolerant scheme resolution (Phase 5 §1)."""

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("expense ratio of HDFC Large Cap Fund - Direct Growth", "hdfc_large_cap"),
            ("what is the exit load on large cap", "hdfc_large_cap"),
            ("largecap", "hdfc_large_cap"),
            ("HDFC Large Cap Fund", "hdfc_large_cap"),
            ("exit load on HDFC Equity Fund - Direct Growth", "hdfc_flexi_cap"),
            ("flexi cap fund", "hdfc_flexi_cap"),
            ("flexicap", "hdfc_flexi_cap"),
            ("equity fund", "hdfc_flexi_cap"),
            ("minimum SIP for HDFC ELSS Tax Saver Fund - Direct Growth", "hdfc_elss"),
            ("elss", "hdfc_elss"),
            ("tax saver", "hdfc_elss"),
            ("80c fund", "hdfc_elss"),
            ("current riskometer level of HDFC Small Cap Fund", "hdfc_small_cap"),
            ("small cap", "hdfc_small_cap"),
            ("smallcap", "hdfc_small_cap"),
            (
                "benchmark of HDFC Balanced Advantage Fund - Direct Growth",
                "hdfc_balanced_advantage",
            ),
            ("balanced advantage", "hdfc_balanced_advantage"),
        ],
    )
    def test_resolves_known_spellings(self, query: str, expected: str) -> None:
        assert resolve_scheme(query) == expected

    @pytest.mark.parametrize(
        "query",
        [
            "",
            "   ",
            "how do I download my capital gains statement",
            "what is a mutual fund",
            "recipe for chocolate cake",
        ],
    )
    def test_returns_none_when_no_scheme_named(self, query: str) -> None:
        assert resolve_scheme(query) is None

    def test_result_is_always_a_known_key_or_none(self) -> None:
        for query in ("large cap", "elss", "balanced", "nothing here"):
            result = resolve_scheme(query)
            assert result is None or result in SCHEME_KEYS

    def test_specific_alias_beats_generic_one(self) -> None:
        # "balanced" alone is a hdfc_balanced_advantage alias, but a query
        # naming "small cap" must not be captured by it.
        assert resolve_scheme("small cap fund") == "hdfc_small_cap"
        assert resolve_scheme("balanced advantage fund") == "hdfc_balanced_advantage"


class TestDetectFieldIntent:
    """Field intent must come from the shared table (Phase 5 §2)."""

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("what is the exit load", "exit_load"),
            ("exit load on the flexi cap fund", "exit_load"),
            ("lock-in period for HDFC ELSS Tax Saver Fund", "lock_in"),
            ("what is the lock in period", "lock_in"),
            ("minimum SIP", "min_sip"),
            ("what is the minimum sip amount", "min_sip"),
            ("expense ratio of the fund", "expense_ratio"),
            ("total expense ratio", "expense_ratio"),
            ("what is the benchmark", "benchmark"),
            ("current riskometer level", "riskometer"),
            ("who manages the fund", "fund_manager"),
            ("section 80C limit", "tax_benefit"),
            ("direct plan vs regular plan", "direct_vs_regular"),
        ],
    )
    def test_detects_intent(self, query: str, expected: str) -> None:
        assert detect_field_intent(query) == expected

    @pytest.mark.parametrize(
        "query", ["", "tell me about the fund", "recipe for chocolate cake"]
    )
    def test_returns_none_when_no_field_named(self, query: str) -> None:
        assert detect_field_intent(query) is None

    def test_uses_the_same_table_as_the_chunker(self) -> None:
        """The single-source-of-truth guardrail (Phase 5 exit gate).

        If these two ever disagree, a query for exit load boosts chunks the
        chunker tagged as something else and the boost silently does nothing.
        """
        from ingest.chunk import tag_field_type

        probes = [
            "what is the exit load",
            "lock-in period",
            "minimum SIP",
            "expense ratio",
            "benchmark",
            "riskometer",
        ]
        for probe in probes:
            assert match_field_type(probe) == tag_field_type(probe)

    def test_chunker_falls_back_to_other_where_retriever_returns_none(self) -> None:
        from ingest.chunk import tag_field_type

        assert match_field_type("tell me about the fund") is None
        assert tag_field_type("tell me about the fund") == "other"


class TestFloorAndBoost:
    """The floor and the boost, against a synthetic collection."""

    @staticmethod
    def _chunks() -> list[dict]:
        return [
            {
                "chunk_id": "a" * 16,
                "source_id": "s1",
                "scheme": "hdfc_large_cap",
                "text": "Expense ratio: 1.03%",
                "ordinal": 1,
                "field_type": "expense_ratio",
                "category": "large_cap",
                "doc_type": "scheme_page",
                "section": "Fees",
                "source_url": "https://example.invalid/a",
                "publisher": "groww",
                "retrieved_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "chunk_id": "b" * 16,
                "source_id": "s2",
                "scheme": "hdfc_elss",
                "text": "Exit load: Nil",
                "ordinal": 2,
                "field_type": "exit_load",
                "category": "elss",
                "doc_type": "scheme_page",
                "section": "Exit load, stamp duty and tax",
                "source_url": "https://example.invalid/b",
                "publisher": "groww",
                "retrieved_at": "2026-01-01T00:00:00+00:00",
            },
        ]

    def test_floor_drops_everything_when_nothing_is_similar(
        self, tmp_path
    ) -> None:
        from tests.conftest import synthetic_collection

        collection = synthetic_collection(tmp_path, self._chunks())
        # A floor of 1.5 is unreachable: cosine similarity tops out at 1.0.
        cfg = RetrievalConfig(top_k=5, over_fetch=2, similarity_floor=1.5)
        trace = retrieve("expense ratio", collection=collection, cfg=cfg)
        assert trace.chunks == []
        assert trace.top_score == 0.0

    def test_zero_chunks_is_not_an_error(self, tmp_path) -> None:
        """An empty trace is the 'not in my sources' signal, never a raise."""
        from tests.conftest import synthetic_collection

        collection = synthetic_collection(tmp_path, self._chunks())
        cfg = RetrievalConfig(similarity_floor=1.5)
        trace = retrieve("recipe for chocolate cake", collection=collection, cfg=cfg)
        assert trace.chunks == []
        assert trace.tau == cfg.similarity_floor

    def test_trace_records_tau_and_top_score(self, tmp_path) -> None:
        from tests.conftest import synthetic_collection

        collection = synthetic_collection(tmp_path, self._chunks())
        cfg = RetrievalConfig(similarity_floor=0.0)
        trace = retrieve("expense ratio of large cap", collection=collection, cfg=cfg)
        assert trace.tau == cfg.similarity_floor
        assert trace.top_score == max(item.similarity for item in trace.chunks)

    def test_boost_is_recorded_on_the_scored_chunk(self, tmp_path) -> None:
        from tests.conftest import synthetic_collection

        collection = synthetic_collection(tmp_path, self._chunks())
        cfg = RetrievalConfig(similarity_floor=0.0, field_type_boost=0.05)
        trace = retrieve(
            "expense ratio of HDFC Large Cap Fund", collection=collection, cfg=cfg
        )
        assert trace.intent == "expense_ratio"
        boosted = [item for item in trace.chunks if item.boosted]
        assert boosted, "the expense_ratio chunk should have been boosted"
        for item in boosted:
            assert item.chunk.field_type == "expense_ratio"
            assert item.similarity == pytest.approx(
                item.raw_similarity + cfg.field_type_boost
            )

    def test_raw_and_boosted_scores_are_both_kept(self, tmp_path) -> None:
        from tests.conftest import synthetic_collection

        collection = synthetic_collection(tmp_path, self._chunks())
        cfg = RetrievalConfig(similarity_floor=0.0, field_type_boost=0.05)
        trace = retrieve(
            "expense ratio of HDFC Large Cap Fund", collection=collection, cfg=cfg
        )
        for item in trace.chunks:
            assert 0.0 <= item.raw_similarity <= 1.0
            assert item.similarity >= item.raw_similarity

    def test_similarities_stay_within_the_cosine_range(self, tmp_path) -> None:
        from tests.conftest import synthetic_collection

        collection = synthetic_collection(tmp_path, self._chunks())
        cfg = RetrievalConfig(similarity_floor=-1.0)
        trace = retrieve("expense ratio", collection=collection, cfg=cfg)
        for item in trace.chunks:
            assert -1.0 <= item.raw_similarity <= 1.0

    def test_top_k_is_respected(self, tmp_path) -> None:
        from tests.conftest import synthetic_collection

        many = []
        for index in range(12):
            many.append(
                {
                    **self._chunks()[0],
                    "chunk_id": f"{index:016x}",
                    "ordinal": index,
                    "text": f"Expense ratio: {index}.0%",
                }
            )
        collection = synthetic_collection(tmp_path, many, dimension=12)
        cfg = RetrievalConfig(top_k=3, over_fetch=2, similarity_floor=-1.0)
        trace = retrieve("expense ratio of large cap", collection=collection, cfg=cfg)
        assert len(trace.chunks) <= 3

    def test_chunk_reconstructed_from_metadata(self, tmp_path) -> None:
        from tests.conftest import synthetic_collection

        collection = synthetic_collection(tmp_path, self._chunks())
        cfg = RetrievalConfig(similarity_floor=-1.0)
        trace = retrieve("expense ratio of large cap", collection=collection, cfg=cfg)
        assert trace.chunks
        chunk = trace.chunks[0].chunk
        assert chunk.text
        assert chunk.ordinal >= 0
        assert chunk.scheme in SCHEME_KEYS
        assert chunk.source_url.startswith("https://")


class TestFieldMisfitPenalty:
    """Off-topic "other schemes" chunks are demoted when a field intent is found.

    Reproduces golden q02, where a ``scheme_type`` chunk listing sibling funds
    outranked the correct ``exit_load`` chunk by 0.022 and would have answered a
    field question with a fund list.
    """

    @staticmethod
    def _chunks() -> list[dict]:
        base = {
            "source_id": "s1",
            "scheme": "hdfc_flexi_cap",
            "category": "flexi_cap",
            "doc_type": "scheme_page",
            "section": "Fund details",
            "source_url": "https://example.invalid/f",
            "publisher": "groww",
            "retrieved_at": "2026-01-01T00:00:00+00:00",
        }
        return [
            {
                **base,
                "chunk_id": "c" * 16,
                "ordinal": 1,
                "field_type": "exit_load",
                # closer to "exit load" in plain text than the sibling-fund list
                "text": "Exit load: Exit load of 1% if redeemed within 1 year",
            },
            {
                **base,
                "chunk_id": "d" * 16,
                "ordinal": 2,
                "field_type": "scheme_type",
                "text": (
                    "HDFC Equity Fund Direct Growth is a Flexi Cap scheme. "
                    "HDFC Multi Asset Fund Direct Growth is a hybrid scheme."
                ),
            },
        ]

    def _trace(self, tmp_path, **overrides):
        from tests.conftest import synthetic_collection

        collection = synthetic_collection(tmp_path, self._chunks())
        cfg = RetrievalConfig(
            top_k=5, over_fetch=4, similarity_floor=-1.0, **overrides
        )
        return retrieve(
            "exit load on HDFC Equity Fund - Direct Growth",
            collection=collection,
            cfg=cfg,
        )

    def test_intent_is_detected(self, tmp_path) -> None:
        assert self._trace(tmp_path).intent == "exit_load"

    def test_a_zero_penalty_leaves_the_ranking_untouched(self, tmp_path) -> None:
        """The behaviour is opt-out-able, so it can be A/B'd on a real corpus."""
        trace = self._trace(tmp_path, field_misfit_penalty=0.0)
        off_topic = [c for c in trace.chunks if c.chunk.field_type == "scheme_type"]
        assert off_topic, "the scheme_type chunk should still be retrieved"
        for item in off_topic:
            assert item.similarity == pytest.approx(item.raw_similarity)
            assert item.boosted is False

    def test_penalty_demotes_the_off_topic_chunk(self, tmp_path) -> None:
        trace = self._trace(tmp_path, field_misfit_penalty=0.15)
        off_topic = [c for c in trace.chunks if c.chunk.field_type == "scheme_type"]
        assert off_topic, "the scheme_type chunk should still be retrieved"
        for item in off_topic:
            assert item.similarity == pytest.approx(item.raw_similarity - 0.15)
            assert item.similarity < item.raw_similarity

    def test_the_matching_chunk_is_never_penalised(self, tmp_path) -> None:
        trace = self._trace(tmp_path, field_misfit_penalty=0.15)
        matching = [c for c in trace.chunks if c.chunk.field_type == "exit_load"]
        assert matching, "the exit_load chunk should be boosted, not penalised"
        for item in matching:
            assert item.boosted is True
            assert item.similarity == pytest.approx(
                item.raw_similarity + 0.35
            )

    def test_no_intent_means_no_penalty(self, tmp_path) -> None:
        """Without a detected field, nothing is demoted - a general question
        must still be able to reach a scheme_type chunk."""
        from tests.conftest import synthetic_collection

        collection = synthetic_collection(tmp_path, self._chunks())
        cfg = RetrievalConfig(
            top_k=5, over_fetch=4, similarity_floor=-1.0, field_misfit_penalty=0.15
        )
        trace = retrieve(
            "tell me about HDFC Equity Fund",
            collection=collection,
            cfg=cfg,
        )
        assert trace.intent is None
        for item in trace.chunks:
            assert item.similarity == pytest.approx(item.raw_similarity)
            assert item.boosted is False

    def test_the_off_topic_set_is_explicit(self) -> None:
        assert OFF_TOPIC_FIELD_TYPES == frozenset({"scheme_type"})

    def test_the_penalty_is_configurable_and_documented(self) -> None:
        assert RetrievalConfig().field_misfit_penalty == 0.15
        from rag.config import load_config

        assert load_config().retrieval.field_misfit_penalty == 0.15


class TestThresholdsComeFromConfig:
    """No magic numbers in the logic (Phase 5 exit gate)."""

    def test_defaults_match_config_yaml(self) -> None:
        import yaml

        from rag.config import load_config

        raw = yaml.safe_load(
            (load_config().root / "config.yaml").read_text(encoding="utf-8")
        )["retrieval"]
        default = RetrievalConfig()
        assert default.top_k == raw["top_k"]
        assert default.over_fetch == raw["over_fetch"]
        assert default.similarity_floor == raw["similarity_floor"]
        assert default.field_type_boost == raw["field_type_boost"]
        assert default.mode == raw["mode"]

    #: Metadata every chunk in the context-window tests shares, so only the
    #: text differs between them.
    _BASE = {
        "source_id": "s1",
        "scheme": "hdfc_flexi_cap",
        "category": "flexi_cap",
        "doc_type": "scheme_page",
        "section": "Fund details",
        "source_url": "https://example.invalid/f",
        "publisher": "groww",
        "retrieved_at": "2026-01-01T00:00:00+00:00",
        "field_type": "exit_load",
    }

    @classmethod
    def _many_chunks(cls, count: int = 14) -> list[dict]:
        """More chunks than the widest window we ship, so the cut is visible."""
        return [
            {
                **cls._BASE,
                "chunk_id": f"{index:016x}",
                "ordinal": index,
                "text": f"Exit load record number {index} for this scheme.",
            }
            for index in range(count)
        ]

    def test_context_window_is_the_configured_top_k(self, tmp_path) -> None:
        """The prompt must show exactly ``top_k`` chunks - there is no second knob.

        ``build_context`` renders every chunk in the trace, so retrieval depth
        *is* the model's context window. Were a separate ``context_k`` ever added
        that silently disagrees with ``top_k``, the model would see more or less
        than retrieval was measured at, and the eval's recall@k numbers would
        stop describing what the model actually read.
        """
        from rag.answer import build_context
        from tests.conftest import synthetic_collection

        collection = synthetic_collection(tmp_path, self._many_chunks())

        for k in (5, 10):
            trace = retrieve(
                "exit load on HDFC Equity Fund - Direct Growth",
                collection=collection,
                cfg=RetrievalConfig(top_k=k, over_fetch=4, similarity_floor=-1.0),
            )
            blocks = build_context(trace)

            assert len(trace.chunks) == k, "trace must hold exactly top_k chunks"
            for index in range(1, k + 1):
                assert f"[{index}] (scheme=" in blocks
            assert blocks.count("field_type=") == k, "no chunk outside the window"

            # Every trace chunk is present, whitespace-normalised, verbatim.
            for item in trace.chunks:
                assert " ".join((item.chunk.text or "").split()) in blocks

    def test_a_wider_context_window_does_not_weaken_the_floor(self, tmp_path) -> None:
        """Widening the window must not smuggle sub-floor chunks back in.

        The floor is the only thing between a low-similarity chunk and the
        prompt. If a wider window leaked chunks below it, the "not in my
        sources" refusal would stop triggering exactly when the corpus is
        thinnest - which is what the 4 zero-evidence golden queries rely on.
        """
        from tests.conftest import synthetic_collection

        collection = synthetic_collection(tmp_path, self._many_chunks())

        for k in (5, 10):
            trace = retrieve(
                "exit load on HDFC Equity Fund - Direct Growth",
                collection=collection,
                # Unreachable: nothing in this corpus is this similar.
                cfg=RetrievalConfig(top_k=k, over_fetch=4, similarity_floor=0.99),
            )
            assert trace.chunks == [], (
                f"an unreachable floor must refuse at top_k={k}, not widen"
            )


class TestHybridMode:
    """Hybrid is optional (FR-5.4, Q5); the default stays vector."""

    def test_vector_is_the_default(self) -> None:
        assert RetrievalConfig().mode == "vector"

    def test_search_dispatches_on_mode(self, tmp_path) -> None:
        from tests.conftest import synthetic_collection

        collection = synthetic_collection(
            tmp_path,
            [
                {
                    "chunk_id": "a" * 16,
                    "source_id": "s1",
                    "scheme": "hdfc_large_cap",
                    "text": "Expense ratio: 1.03%",
                    "ordinal": 1,
                    "field_type": "expense_ratio",
                }
            ],
        )
        # Dispatch must not raise in either mode, and vector mode is the default.
        vector_trace = search(
            "expense ratio", collection=collection, cfg=RetrievalConfig(mode="vector")
        )
        assert vector_trace.query == "expense ratio"
        assert vector_trace.tau == RetrievalConfig().similarity_floor

    def test_missing_bm25_raises_a_clear_import_error(self, tmp_path, monkeypatch) -> None:
        """A clear message beats an obscure ImportError from deep inside."""
        import builtins

        from tests.conftest import synthetic_collection

        collection = synthetic_collection(
            tmp_path,
            [
                {
                    "chunk_id": "a" * 16,
                    "source_id": "s1",
                    "scheme": "hdfc_large_cap",
                    "text": "Expense ratio: 1.03%",
                    "ordinal": 1,
                    "field_type": "expense_ratio",
                }
            ],
        )

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "rank_bm25":
                raise ImportError("no module named rank_bm25")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)

        with pytest.raises(ImportError, match="rank_bm25"):
            hybrid_search("expense ratio", collection=collection, cfg=RetrievalConfig())


class TestRecallAtK:
    def test_reports_a_hit_within_k(self, tmp_path) -> None:
        from tests.conftest import synthetic_collection

        collection = synthetic_collection(
            tmp_path,
            [
                {
                    "chunk_id": "a" * 16,
                    "source_id": "s1",
                    "scheme": "hdfc_large_cap",
                    "text": "Expense ratio: 1.03%",
                    "ordinal": 1,
                    "field_type": "expense_ratio",
                }
            ],
        )
        trace = retrieve(
            "expense ratio of large cap",
            collection=collection,
            cfg=RetrievalConfig(similarity_floor=-1.0),
        )
        assert recall_at_k(trace, 5, field_type="expense_ratio") is True
        assert recall_at_k(trace, 5, field_type="lock_in") is False

    def test_empty_trace_reports_no_hit(self) -> None:
        from rag.schemas import RetrievalTrace

        assert recall_at_k(RetrievalTrace(query="x"), 5) is False


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
