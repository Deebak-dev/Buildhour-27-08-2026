"""The one field-type precedence table, shared by chunking and retrieval.

Phase 5 §2 of docs/implementation.md requires this table to live in exactly one
module and be imported by both :mod:`ingest.chunk` (which tags corpus chunks)
and :mod:`rag.retrieve` (which reads a user's intent out of a question). A
divergence between the two is a correctness bug, not a style problem: a query
for "exit load" boosts ``exit_load`` chunks, and those chunks are only tagged
``exit_load`` if the chunker used the same rules.

Matching is ordered most-specific-first and the first bucket that matches wins,
so ``exit_load`` beats the broad ``document_download``.

This module is pure: no I/O, no model, no network.
"""

from __future__ import annotations

import re

#: Ordered most-specific-first. The first bucket that matches wins, so
#: ``exit_load`` beats the broad ``document_download``.
FIELD_TYPE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "exit_load",
        (
            r"\bexit\s+load\b", r"\bexit\s+charge", r"\bredemption\s+charge",
            r"\bpremium\s+on\s+exit", r"\bif\s+redeemed\s+within\b",
        ),
    ),
    (
        "direct_vs_regular",
        (
            # A *comparison* is required. A bare mention of "direct plan" appears
            # in almost every chunk of this corpus (every scheme is a Direct
            # Growth plan), so matching it here would retag the whole document
            # and make the retrieval boost in Phase 5 useless.
            r"\bdirect\s+(?:vs\.?|versus)\s+regular\b",
            r"\bregular\s+(?:vs\.?|versus)\s+direct\b",
            r"\bdirect\s+plan\s+(?:vs\.?|versus)\s+regular\b",
            r"\bregular\s+plan\s+(?:vs\.?|versus)\s+direct\b",
            r"\bdirect\s+and\s+regular\b",
            r"\bregular\s+plan\b",
            r"\bplan\s+option", r"\bgrowth\s+option", r"\bdividend\s+option",
            r"\bdifference\s+between\s+direct\b",
        ),
    ),
    (
        "min_sip",
        (
            r"\bminimum\s+sip\b", r"\bmin\.?\s*(?:for\s+)?sip\b",
            r"\bminimum\s+monthly\s+investment", r"\bsip\s+starts?\b",
            r"\bstart\s+(?:your\s+)?sip\b", r"\bsip\s+of\b", r"\bmonthly\s+sip\b",
            r"\bsip\s+amount\b", r"\bmin\.?\s*monthly\b",
        ),
    ),
    (
        "lock_in",
        (
            r"\block[\s-]?in\s+period\b", r"\block[\s-]in\b",
            r"\bminimum\s+holding\b", r"\bholding\s+period\b",
        ),
    ),
    (
        "expense_ratio",
        (
            r"\bexpense\s+ratio\b", r"\btotal\s+expense\s+ratio\b", r"\bannual\s+recurring\s+cost",
            r"\bter\b", r"\bongoing\s+charge", r"\bcharges?\s+and\s+expenses?\b",
        ),
    ),
    (
        "riskometer",
        (
            r"\briskometer\b", r"\brisk\s+level\b", r"\briskometer\s+category\b",
            # "The HDFC ELSS ... is rated Very High risk."
            r"\bis\s+rated\b", r"\brated\s+(?:\w+\s+)?risk\b",
            r"\b(?:very\s+low|very\s+high|moderate|low|high)\s+risk\b",
        ),
    ),
    (
        "benchmark",
        (
            r"\bbenchmark\b", r"\bbenchmark\s+index\b", r"\bnifty\s*\d*\b",
            r"\bbse\s*\d*\b", r"\bsensex\b", r"\btrix\b", r"\bindex\s+that\s+the\s+fund\b",
        ),
    ),
    (
        "tax_benefit",
        (
            r"\bsection\s*80c\b", r"\b80c\b", r"\btax\s+benefit", r"\btax\s+saving\b",
            r"\beligib", r"\bdeduction\b",
        ),
    ),
    (
        # Distinct from tax_report so both buckets stay reachable.
        "statement_guide",
        (
            r"\bstatement\s+download\b", r"\bdownload\s+(?:your\s+|the\s+)?statements?\b",
            r"\be[\s-]?statements?\b", r"\bhow\s+to\s+download\b",
            r"\bstatement\s+request\b", r"\bncas\b",
            r"\bconsolidated\s+account\s+statement\b", r"\baccount\s+statement\b",
        ),
    ),
    (
        "tax_report",
        (
            r"\btax\s+report", r"\bcapital\s+gains\s+statement", r"\btax\s+statement",
            r"\btax\s+proof", r"\b26as\b", r"\bincome\s+tax\b", r"\btax\s+slip\b",
        ),
    ),
    (
        "document_download",
        (r"\bdownload\b", r"\bfetch\b", r"\bview\s+statement\b"),
    ),
    (
        "min_lumpsum",
        (
            r"\bminimum\s+lump\s?sum\b", r"\bmin\s+lump\s?sum\b",
            r"\bminimum\s+(?:one[\s-]?time\s+)?investment\s+amount\b", r"\blump\s?sum\b",
        ),
    ),
    ("nav", (r"\bnet\s+asset\s+value\b", r"\bnavs?\b")),
    (
        "fund_manager",
        (
            r"\bfund\s+manager", r"\bmanaged\s+by\b", r"\bportfolio\s+manager\b",
            # PRD golden query q09 is "who manages HDFC Large Cap Fund"; without
            # these the intent detector returned None and the fund_manager boost
            # never fired on the project's own eval set.
            r"\bwho\s+manages\b", r"\bmanages?\s+the\s+fund\b",
            r"\bmanager(?:'s)?\s+name\b", r"\bfund\s+is\s+managed\b",
        ),
    ),
    ("scheme_type", (r"\bdirect\s+plan\b", r"\bgrowth\s+plan\b", r"\bdividend\b", r"\bregular\b")),
)

#: The same table, pre-compiled. Building these once matters: chunking calls the
#: matcher once per candidate chunk across ~1000 chunks and a dozen regexes each.
COMPILED_RULES: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = tuple(
    (name, tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns))
    for name, patterns in FIELD_TYPE_RULES
)

#: A lock-in is a duration, but a bare "3 years" on these pages is far more often
#: an axis label of the growth chart. The duration only counts as a lock-in when
#: something in the same text says which kind of period it is.
LOCK_VOCABULARY = re.compile(
    r"\block[\s-]?in\b|80\s*c\b|\belss\b|tax\s+saver|tax\s+saving", re.IGNORECASE
)
LOCK_DURATION = re.compile(r"\b\d+\s*(?:year|yr)s?\b", re.IGNORECASE)


def match_field_type(text: str, section: str = "") -> str | None:
    """Return the field type this text is about, or ``None``.

    The section heading is searched first because it is the strongest signal,
    then the body.

    Args:
        text: the body to classify.
        section: the heading path, searched before ``text``.

    Returns:
        A field-type name, or ``None`` when no rule and no lock-in vocabulary
        match. Callers that need a total function (the chunker) substitute
        ``"other"`` for the ``None``.
    """
    for haystack in (section or "", text or ""):
        if not haystack:
            continue
        best: tuple[int, int, int, str] | None = None
        for priority, (name, patterns) in enumerate(COMPILED_RULES):
            for pattern in patterns:
                match = pattern.search(haystack)
                if match is None:
                    continue
                # Earliest mention wins, so "is rated Very High risk. Minimum SIP
                # Investment is set to Rs500" is a riskometer fact rather than a
                # min_sip one - the first thing the sentence talks about is the
                # thing it is a fact about. Ties go to the longer match, so
                # "direct plan versus regular" beats a bare "direct plan".
                span = match.end() - match.start()
                candidate = (match.start(), -span, priority, name)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
        if best is not None:
            return best[3]
        if LOCK_VOCABULARY.search(haystack) and LOCK_DURATION.search(haystack):
            return "lock_in"
    return None
