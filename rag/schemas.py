"""Shared data types for the Mutual Fund FAQ Assistant.

Every module in the project imports its data contracts from this module so that
there is exactly one definition of each shape. See docs/architecture.md §9.

This module holds:

* controlled vocabularies (routes, document types, publishers, field types)
* the canonical scheme registry (the 5 HDFC schemes in scope)
* the frozen dataclasses passed between pipeline stages

It performs no I/O and imports nothing outside the standard library, so it is
safe to import from tests with no corpus, no model, and no API key present.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from enum import Enum
from typing import Sequence

# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------


class Route(str, Enum):
    """Pre-retrieval routing labels (docs/architecture.md §6.3).

    ``PII`` is screened deterministically before any model call, so it can never
    be talked past (architecture.md invariant I4).
    """

    FACTUAL = "FACTUAL"
    ADVICE = "ADVICE"
    PERFORMANCE = "PERFORMANCE"
    PII = "PII"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class DocShape(str, Enum):
    """Detected document shapes that drive the chunking strategy (§5.4).

    ``LABEL_VALUE`` was added after inspecting the real corpus in Phase 3: the
    published scheme pages extract as a label/value stream one token per line
    (``Expense ratio`` / ``1.03%``), which none of the other three splitters
    keep paired. Emitting ``Expense ratio: 1.03%`` is what makes the fact
    retrievable at all.
    """

    FAQ = "FAQ"
    TABLE = "TABLE"
    NARRATIVE = "NARRATIVE"
    LABEL_VALUE = "LABEL_VALUE"


#: Document types permitted in the corpus manifest (docs/PRD.md §4.2).
DOC_TYPES: frozenset[str] = frozenset(
    {
        "scheme_page",
        "factsheet",
        "kim",
        "sid",
        "faq",
        "fees",
        "riskometer",
        "benchmark",
        "statement_guide",
    }
)

#: Publishers permitted in the corpus manifest. ``groww`` is a registered
#: distributor platform and is retained only because the brief supplies its URLs;
#: primary AMC/SEBI/AMFI documents are preferred wherever they carry the same
#: fact (architecture.md ADR-4, PRD risk R1).
PUBLISHERS: frozenset[str] = frozenset({"hdfc_amc", "sebi", "amfi", "groww"})

#: Scheme key used by publisher-level documents (SEBI/AMFI pages) that are not
#: tied to a single scheme.
ALL_SCHEMES_KEY = "all"

#: Sentinel URL for a manifest row whose real primary-source URL has not been
#: filled in yet. Rows carrying it must stay ``enabled=0``.
PLACEHOLDER_URL = "TODO_PRIMARY_URL"


# ---------------------------------------------------------------------------
# Canonical scheme registry — the 5 schemes in scope (docs/PRD.md §4.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchemeSpec:
    """One scheme in scope.

    Attributes:
        key: canonical scheme key used in metadata and manifest rows.
        display_name: how we refer to the scheme in our own answers.
        page_title: the name the source pages use, which is frequently *not* the
            display name - Groww titles the flexi cap fund "HDFC Flexi Cap"
            while it is listed as "HDFC Equity Fund". Chunking needs the page
            title to tell one scheme's facts from another's (PRD A14).
        category: fund category slug.
        url: seed source URL from docs/PRD.md §4.1.
        aliases: tolerated query spellings for Phase 5 scheme resolution.
    """

    key: str
    display_name: str
    page_title: str
    category: str
    url: str
    aliases: tuple[str, ...]


SCHEMES: tuple[SchemeSpec, ...] = (
    SchemeSpec(
        key="hdfc_large_cap",
        display_name="HDFC Large Cap Fund - Direct Growth",
        page_title="HDFC Large Cap Fund Direct Growth",
        category="large_cap",
        url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        aliases=("large cap", "largecap", "hdfc large cap", "large cap fund"),
    ),
    SchemeSpec(
        key="hdfc_flexi_cap",
        display_name="HDFC Equity Fund - Direct Growth",
        page_title="HDFC Flexi Cap Direct Plan Growth",
        category="flexi_cap",
        url="https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
        aliases=("flexi cap", "flexicap", "equity fund", "hdfc equity", "large and mid"),
    ),
    SchemeSpec(
        key="hdfc_elss",
        display_name="HDFC ELSS Tax Saver Fund - Direct Growth",
        page_title="HDFC ELSS Tax Saver Fund Direct Plan Growth",
        category="elss",
        url="https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
        aliases=("elss", "tax saver", "tax saver fund", "80c", "80c fund"),
    ),
    SchemeSpec(
        key="hdfc_small_cap",
        display_name="HDFC Small Cap Fund - Direct Growth",
        page_title="HDFC Small Cap Fund Direct Growth",
        category="small_cap",
        url="https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
        aliases=("small cap", "smallcap", "hdfc small cap", "small cap fund"),
    ),
    SchemeSpec(
        key="hdfc_balanced_advantage",
        display_name="HDFC Balanced Advantage Fund - Direct Growth",
        page_title="HDFC Balanced Advantage Fund Direct Growth",
        category="hybrid",
        url="https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
        aliases=(
            "balanced advantage",
            "balanced advantage fund",
            "balanced",
            "hybrid",
        ),
    ),
)

#: The 5 canonical scheme keys.
SCHEME_KEYS: frozenset[str] = frozenset(spec.key for spec in SCHEMES)

#: Scheme keys accepted in a manifest row: the 5 schemes plus the
#: publisher-level ``all`` key.
VALID_SCHEME_VALUES: frozenset[str] = SCHEME_KEYS | {ALL_SCHEMES_KEY}


# ---------------------------------------------------------------------------
# Field-type vocabulary (docs/architecture.md §5.3)
# ---------------------------------------------------------------------------

#: ``field_type`` tags drive retrieval boosting and are the main defence against
#: MiniLM conflating *direct vs regular* plans and *exit-load* tiers (PRD risk R4).
FIELD_TYPES: frozenset[str] = frozenset(
    {
        "expense_ratio",
        "exit_load",
        "min_sip",
        "min_lumpsum",
        "lock_in",
        "tax_benefit",
        "riskometer",
        "benchmark",
        "nav",
        "fund_manager",
        "scheme_type",
        "direct_vs_regular",
        "document_download",
        "statement_guide",
        "tax_report",
    }
)

#: Fallback tag when no rule matches. Kept outside :data:`FIELD_TYPES` so the
#: vocabulary stays exactly the 15 specific buckets named in
#: docs/architecture.md §5.3, while still being a legal value on a Chunk.
FIELD_TYPE_OTHER: str = "other"

#: Every value a ``Chunk.field_type`` may legally hold.
FIELD_TYPES_ALL: frozenset[str] = FIELD_TYPES | {FIELD_TYPE_OTHER}


# ---------------------------------------------------------------------------
# Manifest row (docs/architecture.md §5.2, Phase 1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestRow:
    """One row of ``ingest/sources.csv``.

    Attributes:
        source_id: stable unique slug, e.g. ``hdfc-large-cap__groww_scheme``.
        scheme: a key from :data:`VALID_SCHEME_VALUES`.
        category: fund category slug, or ``general`` for publisher-level pages.
        doc_type: one of :data:`DOC_TYPES`.
        title: human-readable document title.
        publisher: one of :data:`PUBLISHERS`.
        url: source URL, or :data:`PLACEHOLDER_URL` when not yet filled in.
        enabled: ``1`` to ingest, ``0`` to skip.
    """

    source_id: str
    scheme: str
    category: str
    doc_type: str
    title: str
    publisher: str
    url: str
    enabled: int

    @property
    def is_placeholder(self) -> bool:
        """Whether the row still carries the placeholder URL."""
        return self.url.strip() == PLACEHOLDER_URL

    @property
    def is_enabled(self) -> bool:
        """Whether the row is enabled for ingestion."""
        return int(self.enabled) == 1


# ---------------------------------------------------------------------------
# Pipeline data carriers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Document:
    """A fetched, cleaned source document (Flow A output, stage 1)."""

    source_id: str
    scheme: str
    category: str
    doc_type: str
    title: str
    publisher: str
    source_url: str
    retrieved_at: str
    sha256: str
    http_status: int
    content_chars: int
    raw_path: str = ""
    clean_path: str = ""
    thin: bool = False


@dataclass(frozen=True)
class Chunk:
    """An embedded text unit (Flow A output, stage 2).

    Every document field needed to build a citation travels with the chunk, so
    the answer contract can be assembled in Python without re-reading the corpus
    (architecture.md ADR-1).
    """

    chunk_id: str
    source_id: str
    scheme: str
    text: str
    ordinal: int
    category: str = ""
    doc_type: str = ""
    field_type: str = "other"
    section: str = ""
    source_url: str = ""
    publisher: str = ""
    retrieved_at: str = ""


@dataclass(frozen=True)
class ScoredChunk:
    """A retrieved chunk with its similarity before and after boosting."""

    chunk: Chunk
    similarity: float
    raw_similarity: float
    boosted: bool = False


@dataclass
class RetrievalTrace:
    """Everything retrieval decided, surfaced in the UI expander (FR-8.6)."""

    query: str
    scheme_filter: str | None = None
    chunks: list[ScoredChunk] = _field(default_factory=list)
    tau: float = 0.0
    top_score: float = 0.0
    intent: str | None = None


@dataclass(frozen=True)
class RouteResult:
    """Outcome of the pre-retrieval router (docs/architecture.md §6.3)."""

    label: Route
    reason: str
    matched_classes: tuple[str, ...] = ()
    scheme_hint: str | None = None


@dataclass
class AnswerPayload:
    """The final, contract-compliant response returned by Flow B.

    For a ``PII`` route the originating query is deliberately absent from this
    object so it cannot be persisted downstream (architecture.md invariant I5).
    """

    text: str
    route: Route
    source_url: str | None = None
    last_updated: str = ""
    grounded_field: str | None = None
    trace: RetrievalTrace | None = None
    degraded: bool = False
    #: True only when generation could not run at all - no client, provider
    #: error, or rate limit. This is deliberately narrower than ``degraded``:
    #: a "not in my sources" refusal is also a non-answer, but it is the
    #: system working correctly, and reporting it as a failure would be a lie
    #: in eval/results.json.
    service_failure: bool = False
    validators_passed: list[str] = _field(default_factory=list)


def scheme_display_name(scheme_key: str) -> str:
    """Return the display name for a scheme key.

    Args:
        scheme_key: a key from :data:`SCHEME_KEYS`.

    Returns:
        The scheme's display name, or the key itself when unknown.
    """
    for spec in SCHEMES:
        if spec.key == scheme_key:
            return spec.display_name
    return scheme_key


def scheme_keys() -> Sequence[str]:
    """Return the 5 canonical scheme keys in registry order."""
    return [spec.key for spec in SCHEMES]
