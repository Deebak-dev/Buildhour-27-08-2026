"""Structure-aware chunking with field-type tagging.

Flow A, stage 2 — "Chunking" (docs/architecture.md §5.3–5.4, PRD FR-2.1–FR-2.5).

The strategy is **decided from the observed data**: :func:`profile` detects the
shape of each document, and the splitter for that shape is applied. A fixed-size
character split is deliberately rejected because it destroys the two units every
fact lookup depends on — a FAQ question/answer pair and a table row.

Everything in this module is pure and offline: no network, no LLM, no embeddings.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Iterable, Sequence

from rag.config import ChunkConfig, load_config
from rag.schemas import (
    FIELD_TYPE_OTHER,
    FIELD_TYPES_ALL,
    SCHEMES,
    Chunk,
    Document,
    DocShape,
)

logger = logging.getLogger(__name__)

#: Words per token, used instead of a tokenizer dependency. Deliberately an
#: over-estimate so chunks stay under ``max_tokens`` on real text.
WORDS_PER_TOKEN = 1.3

#: A line that looks like a FAQ question.
QUESTION_LINE = re.compile(r"^\s*(?:Q\d+[.)]|Ques(?:tion)?\s*\d*[.:)]|Question\s*:?)\s*\S", re.IGNORECASE)
QUESTION_MARK = re.compile(r"\?\s*$")

HEADING_LINE = re.compile(
    r"^\s*(?:#{1,6}\s+\S|[A-Z][A-Za-z0-9 &/()'\-,]{2,70}:?\s*$)"
)

#: A table row: 2+ pipes, or 2+ column gaps of 2+ spaces.
PIPE_ROW = re.compile(r"\|.*\|.*\|")
SPACE_ROW = re.compile(r"\S {2,}\S.*\S {2,}\S")

#: A bare value: a number, percentage, currency amount, or an explicit null.
VALUE_TOKEN = re.compile(
    r"""^(?:
          [-–—]{1,2}                       # -- / em dash used as a null
        | n/?a | nil | none
        | (?:[₹$]|\brs\.?)?\s*\d[\d,]*(?:\.\d+)?\s*   # 1,447.38 / ₹39,933.37 / Rs 500
          (?:%|x|cr|cr\.|lakh|crore|bps)?
        | \d+(?:\.\d+)?\s*%
    )\s*(?:\(.*\))?$""",
    re.IGNORECASE | re.VERBOSE,
)

#: An index name, which is a value even though it contains letters.
INDEX_TOKEN = re.compile(
    r"\b(nifty\s*\d*|bse\s*\d*|sensex|trix|total\s+return\s+index|cn\s?x\s?nifty)\b",
    re.IGNORECASE,
)

#: Longest a line may be and still be considered a label rather than prose.
LABEL_MAX_CHARS = 80

#: A document is LABEL_VALUE when at least this share of its lines are
#: label/value pairs.
LABEL_VALUE_RATIO = 0.30

#: The exact field names the published pages use. A line matching one of these
#: is a heading, not the value of the line above it.
CANONICAL_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "aum",
        "benchmark",
        "category",
        "direct vs regular",
        "exit load",
        "expense ratio",
        "fund benchmark",
        "fund name",
        "fund size (aum)",
        "fund type",
        "lock in period",
        "lock-in period",
        "min. for investment",
        "min. for sip",
        "minimum sip",
        "nav",
        "riskometer",
        "riskometer level",
        "scheme type",
        "tax treatment",
    }
)

# Words that only appear around a returns table. Paired with a percentage count
# by _is_performance_unit so that ordinary facts are never discarded.
PERFORMANCE_HINT = re.compile(
    r"annualised|annualized|return\s*calculator|\bcagr\b|"
    r"\b1D\s|\b1M\s|\b6M\s|\b1Y\b|\b3Y\b|\b5Y\b|\b10Y\b|"
    r"\bSIP\b.*\b(?:return|growth)|\binvested\b|\bvalue\s+of\s+investment\b",
    re.IGNORECASE,
)


def _is_value_like(line: str) -> bool:
    """Whether a line is a value rather than a label."""
    stripped = line.strip()
    if not stripped:
        return False
    return bool(VALUE_TOKEN.match(stripped)) or bool(INDEX_TOKEN.search(stripped))


def _is_label_like(line: str) -> bool:
    """Whether a line is short enough to be a label rather than prose."""
    stripped = line.strip()
    return 0 < len(stripped) <= LABEL_MAX_CHARS


def estimate_tokens(text: str) -> int:
    """Estimate token count from whitespace-separated words.

    Args:
        text: the text to measure.

    Returns:
        An estimated token count, never below 1 for non-empty text.
    """
    words = len(text.split())
    if words == 0:
        return 0
    return max(1, round(words * WORDS_PER_TOKEN))


# ---------------------------------------------------------------------------
# Shape detection
# ---------------------------------------------------------------------------


def _question_like_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if QUESTION_LINE.match(line) or QUESTION_MARK.search(line)]


def _table_like_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if PIPE_ROW.search(line) or SPACE_ROW.search(line)]


def _label_value_ratio(text: str) -> float:
    """Fraction of non-empty lines that form a label/value pair."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    pairs = sum(
        1
        for index in range(len(lines) - 1)
        if _is_label_like(lines[index]) and _is_value_like(lines[index + 1])
    )
    return pairs / len(lines)


def profile(text: str) -> DocShape:
    """Detect the shape of a document so the right splitter is used.

    Args:
        text: cleaned document text.

    Returns:
        The detected :class:`~rag.schemas.DocShape`. The real corpus is a
        label/value stream, so ``LABEL_VALUE`` is tested before the fallbacks.
    """
    if not text.strip():
        return DocShape.NARRATIVE

    lines = text.splitlines()

    # FAQ: an explicit FAQ marker near the top, or several question-like lines
    # each followed by non-empty content.
    if re.search(r"frequently\s+asked|\bFAQ\b", text[:500], re.IGNORECASE):
        return DocShape.FAQ

    questions = _question_like_lines(text)
    if len(questions) >= 3:
        for index, line in enumerate(lines):
            if not (QUESTION_LINE.match(line) or QUESTION_MARK.search(line)):
                continue
            following = [nxt for nxt in lines[index + 1: index + 3] if nxt.strip()]
            if following:
                return DocShape.FAQ
        if len(questions) >= 3:
            return DocShape.FAQ

    # Label/value stream: the shape of the published scheme pages. Tested
    # before TABLE so a page mixing a fee table with a key-value header block is
    # still split into retrievable facts.
    if _label_value_ratio(text) >= LABEL_VALUE_RATIO:
        return DocShape.LABEL_VALUE

    rows = _table_like_lines(text)
    if len(rows) >= 3 and len(rows) / max(len(lines), 1) >= 0.4:
        return DocShape.TABLE

    return DocShape.NARRATIVE


# ---------------------------------------------------------------------------
# field_type tagging (deterministic rules only)
# ---------------------------------------------------------------------------

# The precedence table itself lives in rag/fieldmap.py so the chunker and the
# retriever cannot disagree about what an "exit load" question means (Phase 5
# §2). Only the chunking-specific total-function wrapper is defined here.
from rag.fieldmap import (  # noqa: E402  - kept beside the tagging helpers
    COMPILED_RULES as _COMPILED_RULES,
    LOCK_DURATION,
    LOCK_VOCABULARY,
    match_field_type,
)


def tag_field_type(text: str, section: str = "") -> str:
    """Tag a chunk with a ``field_type`` using deterministic rules.

    The section heading is searched first because it is the strongest signal,
    then the body. Returns ``"other"`` when nothing matches: the shared matcher
    in :mod:`rag.fieldmap` returns ``None`` for that case, but every chunk must
    carry a legal tag, so the fallback belongs here rather than there.

    Args:
        text: the chunk body.
        section: the chunk's heading path.

    Returns:
        One of :data:`~rag.schemas.FIELD_TYPES_ALL`.
    """
    return match_field_type(text, section) or FIELD_TYPE_OTHER


# ---------------------------------------------------------------------------
# Splitters
# ---------------------------------------------------------------------------


#: "HDFC Retirement Fund Hybrid Debt Plan Direct Growth: HDFC NIFTY 50 ETF IDCW"
#: - a link-list or benchmark-table row whose subject is a *named fund*. The
#: colon form is what distinguishes it from a real fact ("Expense ratio: 1.03%").
FUND_NAME_PREFIX = re.compile(
    r"^\s*HDFC\b[^:]{0,80}\b(?:Fund|ETF|FoF)\b[^:]{0,60}:",
    re.IGNORECASE,
)


def _is_cross_scheme(text: str, own_scheme: str) -> bool:
    """Whether a unit is about a fund other than the one it was crawled from.

    Every scheme page carries other funds' data: an "Also manages these schemes"
    link list, and benchmark/returns tables covering dozens of other HDFC funds.
    Left in the corpus they are actively dangerous - a question about the ELSS
    lock-in could be grounded in a neighbouring page's exit load and answered
    with a correct-looking citation to the wrong fund (PRD A14).

    Units whose subject is the document's own scheme, or which name no fund at
    all (holdings rows, prose, manager bios), are kept.
    """
    own = next((spec for spec in SCHEMES if spec.key == own_scheme), None)
    own_forms = (
        {_normalise_name(own.page_title), _normalise_name(own.display_name)}
        if own
        else set()
    )
    normalised_text = _normalise_name(text)

    # Another one of the five supported schemes is named.
    for spec in SCHEMES:
        if spec.key == own_scheme:
            continue
        if _normalise_name(spec.page_title) in normalised_text:
            return True

    # Some other fund entirely is the subject of a "name: value" row.
    if FUND_NAME_PREFIX.match(text):
        return not any(form in normalised_text for form in own_forms)

    return False


def _normalise_name(text: str) -> str:
    """Reduce a fund name to comparable letters and digits.

    ``"HDFC ELSS Tax Saver Fund Direct Plan Growth"`` and
    ``"HDFC ELSS Tax Saver Fund Direct Growth"`` normalise to the same string, so
    a page's own name is not mistaken for another scheme's.
    """
    return "".join(char.lower() for char in text if char.isalnum())


def _chunk_id(source_id: str, ordinal: int) -> str:
    """Return the stable content-derived id for a chunk (FR-4.2 idempotency)."""
    return hashlib.sha256(f"{source_id}::{ordinal}".encode("utf-8")).hexdigest()[:16]


def _split_faq(text: str) -> list[tuple[str, str]]:
    """Split FAQ text into ``(section, body)`` units, one question+answer each.

    A pair is never broken: if a single question's answer runs past ``max_tokens``
    the pair is kept whole rather than cut.
    """
    lines = text.splitlines()
    units: list[tuple[str, str]] = []
    buffer: list[str] = []
    current_question: str | None = None

    for line in lines:
        stripped = line.strip()
        is_question = bool(QUESTION_LINE.match(stripped) or QUESTION_MARK.search(stripped))
        if is_question and buffer:
            units.append((current_question or "", "\n".join(buffer).strip()))
            buffer = [stripped]
            current_question = stripped
        elif is_question and not buffer:
            buffer = [stripped]
            current_question = stripped
        else:
            buffer.append(stripped)

    tail = "\n".join(buffer).strip()
    if tail:
        units.append((current_question or "", tail))

    return [(question, body) for question, body in units if body]


def _looks_like_fact_label(line: str) -> bool:
    """Whether a line names a fact this assistant knows how to answer.

    Used to decide that the *next* line is that fact's value even when the value
    is a word rather than a number, e.g. ``Riskometer`` / ``Very High``.
    """
    return any(
        pattern.search(line)
        for _name, patterns in _COMPILED_RULES
        for pattern in patterns
    )


def _is_canonical_field_name(line: str) -> bool:
    """Whether a line is the *name* of a field rather than a field's value.

    Needed to tell ``Lock-in period`` / ``3 years`` (a real pair) from
    ``Exit load, stamp duty and tax`` / ``Exit load`` (a heading followed by the
    heading it announces). Matching the field rules is not enough here, because
    durations such as ``3 years`` legitimately belong to the lock-in rule.
    """
    normalised = line.strip().rstrip(":").lower()
    return normalised in CANONICAL_FIELD_NAMES


def _is_table_row(line: str) -> bool:
    """Whether a line is part of a pipe/space-aligned table.

    Table rows are left untouched by the label/value pass: a row such as
    ``Direct Dividend | 1.25% | 1% within 12 months`` would otherwise pair
    ``Direct Dividend`` with ``1.25%`` and lose the row structure that makes
    the cell self-describing.
    """
    return bool(PIPE_ROW.search(line) or SPACE_ROW.search(line))


def _split_label_value(lines: Sequence[str]) -> tuple[list[tuple[str, str]], set[int]]:
    """Extract label/value units from a line stream.

    Returns:
        ``(units, consumed)`` where ``units`` is a list of ``(section, text)``
        and ``consumed`` is the set of line indices turned into a unit, so the
        caller can split whatever is left over with the shape-appropriate
        splitter.

    A run of two or more label-like lines is treated as a section header rather
    than a fact, which also conveniently discards the "Also manages these
    schemes" cross-scheme link lists that would otherwise leak one fund's facts
    into another's answers (PRD A14).
    """
    stripped = [line.strip() for line in lines]
    units: list[tuple[str, str]] = []
    consumed: set[int] = set()
    section = ""
    index = 0
    total = len(stripped)

    while index < total:
        line = stripped[index]
        if not line or _is_table_row(line):
            # Blank lines and table rows are never label/value facts.
            index += 1
            continue

        if _is_value_like(line):
            # An orphan value with no label - nothing to attribute it to.
            consumed.add(index)
            index += 1
            continue

        following = stripped[index + 1] if index + 1 < total else ""
        if not following or _is_table_row(following):
            index += 1
            continue

        # A bare horizon is a performance axis label, never a fact label, so it
        # must not become the subject of a pair.
        if BARE_HORIZON.match(line):
            index += 1
            continue

        # A value is either a recognisable token, or the answer to a label that
        # names a fact we can speak to ("Riskometer" / "Very High", which is a
        # word rather than a number). The second clause is skipped when the
        # following line is itself a field name, so "Exit load, stamp duty and
        # tax" / "Exit load" stays a section header.
        if _is_label_like(line) and len(following) <= MAX_VALUE_CHARS and (
            _is_pair_value(following, line)
            or (_looks_like_fact_label(line) and not _is_canonical_field_name(following))
        ):
            units.append((section, f"{line}: {following}"))
            consumed.update({index, index + 1})
            index += 2
        elif _is_label_like(following):
            # Two labels in a row: the first is a section header. It is not
            # consumed, so the remainder splitter still sees it.
            section = line
            index += 1
        else:
            # Not a fact we can attribute. Leave the line for the remainder
            # splitter, which can size and overlap prose properly.
            index += 1

    return units, consumed


def _is_performance_unit(text: str) -> bool:
    """Whether a unit is a returns/performance block.

    The PRD forbids the assistant from computing or quoting returns (FR-6.2,
    §5.2). The source pages carry large performance tables, so they are dropped
    at ingest time rather than left in the corpus where a generator could ground
    a number from them.

    The test is deliberately conservative: it needs **two or more percentages**
    *and* a performance hint, so single-figure facts such as
    ``Expense ratio: 1.03%`` and ``Lock-in period: 3 years`` are never dropped.
    """
    percentages = len(re.findall(r"[-+]?\d+(?:\.\d+)?\s*%", text))
    return percentages >= 2 and bool(PERFORMANCE_HINT.search(text))


#: A fact's value is a number or a short phrase, never a paragraph. Without this
#: ceiling a label followed by a long prose line produces one enormous "value".
MAX_VALUE_CHARS = 200

#: A bare percentage or a bare number, which is how the cleaner renders a cell
#: whose "%" the page split onto its own line.
BARE_NUMBER = re.compile(r"^[-+]?\d+(?:\.\d+)?$|^%$")

#: A bare investment horizon. On these pages ``3 years`` is an axis label of the
#: "what would this have grown into" illustration, not a fact - and on its own it
#: looks exactly like a lock-in period, so left alone it will happily answer an
#: ELSS lock-in question with a growth projection (PRD FR-6.2).
#:
#: It is only a real value when a fact label owns it, e.g. ``Lock-in period``.
BARE_HORIZON = re.compile(
    r"^(?:1|3|5|10)\s*(?:years?|yrs?)\b|^(?:1D|1M|6M|1Y|3Y|5Y|10Y)$",
    re.IGNORECASE,
)


def _is_pair_value(candidate: str, label: str) -> bool:
    """Whether ``candidate`` is the value of the ``label`` above it.

    A bare horizon counts as a value only when the label above it actually names
    a fact, which is what separates ``Lock-in period`` / ``3 years`` from the
    growth illustration's bare ``3 years`` axis label.
    """
    if BARE_HORIZON.match(candidate):
        return _looks_like_fact_label(label)
    return _is_value_like(candidate)


def _table_parts(lines: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split table-ish lines into a header block and data rows."""
    table_lines = [line for line in lines if PIPE_ROW.search(line) or SPACE_ROW.search(line)]
    if not table_lines:
        return [], []
    header = table_lines[0]
    return [header], table_lines[1:]


def _merge_label_value_lines(lines: Sequence[str]) -> list[str]:
    """Join a label line with the value line that follows it.

    Applied before narrative splitting so that a value can never be orphaned
    into a chunk of its own, whatever shape the document was detected as. A
    chunk containing only ``1.03%`` is useless to a generator.
    """
    merged: list[str] = []
    index = 0
    total = len(lines)
    while index < total:
        line = lines[index].strip()
        following = lines[index + 1].strip() if index + 1 < total else ""
        if line and following and _is_label_like(line) and _is_value_like(following):
            merged.append(f"{line}: {following}")
            index += 2
        else:
            if line:
                merged.append(line)
            index += 1
    return merged


def _split_narrative(text: str, cfg: ChunkConfig) -> list[tuple[str, str]]:
    """Heading-anchored recursive split with overlap.

    Splits on headings, then paragraphs, then sentences, merging small pieces up
    to ``target_tokens`` and never exceeding ``max_tokens``.
    """
    lines = _merge_label_value_lines(text.splitlines())

    # Pass 1: group under headings, which are the natural semantic boundaries.
    sections: list[tuple[str, list[str]]] = []
    heading = ""
    buffer: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and HEADING_LINE.match(stripped) and len(stripped) <= 72:
            if buffer:
                sections.append((heading, buffer))
                buffer = []
            heading = stripped
        else:
            buffer.append(stripped)
    if buffer:
        sections.append((heading, buffer))

    # Pass 2: one piece per paragraph. Size- and tag-aware merging is
    # _merge_and_overlap's job; accumulating here first would hand it a single
    # pre-joined blob per section, leaving it nothing to choose between.
    pieces: list[tuple[str, str]] = []
    for section_heading, section_lines in sections:
        for paragraph in _group_paragraphs(section_lines, cfg):
            if paragraph:
                pieces.append((section_heading, paragraph))

    return _merge_and_overlap(pieces, cfg)


def _group_paragraphs(lines: Sequence[str], cfg: ChunkConfig) -> list[str]:
    """Join lines into paragraphs, hard-splitting any that exceed ``max_tokens``.

    The hard split has to escalate - sentence boundary, then word boundary - and
    the final word-level pass is what makes the size guarantee real. The cleaned
    corpus has no blank lines at all, so a whole section arrives here as one
    paragraph, and its holdings rows contain no sentence punctuation to break
    on. Without the word-level pass such a run cannot be split at all and lands
    in the corpus as one enormous chunk, which the performance filter then
    deletes whole - taking the legitimate prose around it with it.
    """
    paragraphs: list[str] = []
    buffer: list[str] = []
    for line in lines:
        if not line.strip():
            if buffer:
                paragraphs.append(" ".join(buffer))
                buffer = []
            continue
        buffer.append(line.strip())
    if buffer:
        paragraphs.append(" ".join(buffer))

    limit = cfg.max_tokens
    out: list[str] = []
    for paragraph in paragraphs:
        # Paragraphs are broken into sentences even when they fit. The "About"
        # paragraph of a scheme page packs the auditor, launch date, fund
        # manager, AUM, NAV, risk rating and minimum investment into one short
        # run of prose, and a chunk can only carry one field type - so leaving it
        # whole hides the risk rating behind whichever fact came first. Pieces
        # are re-merged up to the target size by _merge_and_overlap, which will
        # not combine sentences that state different facts.
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            if sentence.strip():
                out.extend(_hard_wrap(sentence, limit))
    return out


def _hard_wrap(text: str, limit: int) -> list[str]:
    """Break ``text`` into pieces of at most ``limit`` estimated tokens."""
    if estimate_tokens(text) <= limit:
        return [text] if text.strip() else []

    sentences: list[str] = []
    buffer = ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if buffer and estimate_tokens(f"{buffer} {sentence}") > limit:
            sentences.append(buffer.strip())
            buffer = sentence
        else:
            buffer = f"{buffer} {sentence}".strip()
    if buffer.strip():
        sentences.append(buffer.strip())

    out: list[str] = []
    for sentence in sentences:
        if estimate_tokens(sentence) <= limit:
            out.append(sentence)
            continue
        # Still oversized: fall back to a word-level split.
        words = sentence.split()
        current = ""
        for word in words:
            if current and estimate_tokens(f"{current} {word}") > limit:
                out.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            out.append(current)
    return out


def _overlap_tail(text: str, cfg: ChunkConfig) -> str:
    """Return the trailing slice of ``text`` used as overlap for the next chunk."""
    if cfg.overlap_pct <= 0:
        return ""
    words = text.split()
    keep = max(1, round(len(words) * cfg.overlap_pct / 100))
    return " ".join(words[-keep:]) if keep < len(words) else ""


def _merge_and_overlap(
    pieces: Sequence[tuple[str, str]], cfg: ChunkConfig
) -> list[tuple[str, str]]:
    """Merge small pieces up to the target size, then prepend overlap tails."""
    merged: list[tuple[str, str]] = []
    buffer: list[str] = []
    buffer_heading = ""

    for heading, body in pieces:
        if not body:
            continue
        if buffer and buffer_heading == heading:
            # Two sentences that state *different* facts are kept apart, so each
            # chunk carries a single meaningful field type for the Phase 5
            # retrieval boost. Untagged prose still merges freely.
            #
            # The section is deliberately excluded: a scheme page's section
            # heading is its own name, which contains "Direct Plan" and would
            # therefore tag every sentence in the section identically.
            buffered_tag = tag_field_type("\n\n".join(buffer))
            incoming_tag = tag_field_type(body)
            conflicting = (
                buffered_tag != "other"
                and incoming_tag != "other"
                and buffered_tag != incoming_tag
            )
            candidate = buffer + [body]
            if not conflicting and estimate_tokens("\n\n".join(candidate)) <= cfg.target_tokens:
                buffer = candidate
                continue
            merged.append((buffer_heading, "\n\n".join(buffer).strip()))
            buffer, buffer_heading = [body], heading
            continue
        # A different section: flush rather than merge across the boundary.
        # Carrying the first heading forward would attribute later sections'
        # text to it, and merging a short "About" paragraph into a large
        # "Return calculator" slice would get both dropped as performance.
        if buffer:
            merged.append((buffer_heading, "\n\n".join(buffer).strip()))
        buffer, buffer_heading = [body], heading

    if buffer:
        merged.append((buffer_heading, "\n\n".join(buffer).strip()))

    with_overlap: list[tuple[str, str]] = []
    for index, (heading, body) in enumerate(merged):
        if index:
            tail = _overlap_tail(merged[index - 1][1], cfg)
            if tail:
                body = f"{tail}\n{body}" if heading == merged[index - 1][0] else f"{tail}\n\n{body}"
        with_overlap.append((heading, body))
    return with_overlap


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_chunks(
    documents: Sequence[Document],
    *,
    cfg: ChunkConfig | None = None,
) -> list[Chunk]:
    """Split documents into chunks according to each document's detected shape.

    Args:
        documents: loaded documents, each with a readable ``clean_path``.
        cfg: chunking parameters. Defaults to ``config.yaml``.

    Returns:
        The chunks, in document then ordinal order, each carrying full document
        metadata so a citation can be built without re-reading the corpus.
    """
    settings = cfg or load_config().chunking
    chunks: list[Chunk] = []
    dropped_performance = 0
    dropped_cross_scheme = 0

    for doc in documents:
        text = _read_clean(doc)
        if not text:
            continue

        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            continue

        # Label/value facts are extracted first and unconditionally. The real
        # scheme pages are hybrids - a key-value block, then a holdings table,
        # then prose - so a single document-level shape switch would bury
        # "Expense ratio: 1.03%" inside a multi-kilobyte chunk that can never be
        # quoted. Emitting each fact on its own is what makes it retrievable.
        fact_units, consumed = _split_label_value(lines)
        remainder = "\n".join(
            line for index, line in enumerate(lines) if index not in consumed
        )

        # The detected shape only decides how the *leftovers* are split.
        shape = profile(remainder)
        if shape is DocShape.FAQ and settings.strategy in ("auto", "faq_pair"):
            rest_units = _split_faq(remainder)
        elif shape is DocShape.TABLE and settings.strategy in ("auto", "table_row"):
            header, rows = _table_parts(remainder.splitlines())
            rest_units = [("", "\n".join(header + [row]).strip()) for row in rows]
        elif shape is DocShape.LABEL_VALUE and settings.strategy in (
            "auto",
            "label_value",
        ):
            rest_units, _ = _split_label_value(remainder.splitlines())
        else:
            rest_units = _split_narrative(remainder, settings)

        units = fact_units + rest_units
        ordinal = 0

        for section, body in units:
            body = body.strip()
            if not body:
                continue
            if _is_performance_unit(body):
                # Any returns table that survived the line filter.
                dropped_performance += 1
                continue
            if _is_cross_scheme(body, doc.scheme):
                # Another fund's fact on this page (PRD A14).
                dropped_cross_scheme += 1
                continue
            chunk = Chunk(
                chunk_id=_chunk_id(doc.source_id, ordinal),
                source_id=doc.source_id,
                scheme=doc.scheme,
                text=body,
                ordinal=ordinal,
                category=doc.category,
                doc_type=doc.doc_type,
                field_type=tag_field_type(body, section),
                section=section,
                source_url=doc.source_url,
                publisher=doc.publisher,
                retrieved_at=doc.retrieved_at,
            )
            chunks.append(chunk)
            ordinal += 1

    if dropped_performance or dropped_cross_scheme:
        logger.info(
            "dropped %d performance unit(s) per the no-returns policy (FR-6.2) "
            "and %d cross-scheme unit(s) (PRD A14)",
            dropped_performance,
            dropped_cross_scheme,
        )

    return chunks


def _read_clean(doc: Document) -> str:
    """Read a document's cleaned text, tolerating a missing file."""
    if not doc.clean_path:
        return ""
    path = Path(doc.clean_path)
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def write_chunks(chunks: Iterable[Chunk], path: str | Path | None = None) -> Path:
    """Write chunks as JSONL so they can be inspected by eye for the demo.

    Args:
        chunks: chunks to write.
        path: destination. Defaults to ``data/chunks/chunks.jsonl``.

    Returns:
        The path written.
    """
    destination = Path(path) if path is not None else load_config().chunks_jsonl
    destination.parent.mkdir(parents=True, exist_ok=True)
    from dataclasses import asdict

    with destination.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    return destination


def field_type_histogram(chunks: Iterable[Chunk]) -> dict[str, int]:
    """Count chunks per ``field_type``, for the ingest summary."""
    histogram: dict[str, int] = {}
    for chunk in chunks:
        histogram[chunk.field_type] = histogram.get(chunk.field_type, 0) + 1
    return dict(sorted(histogram.items(), key=lambda item: -item[1]))


def assert_vocabulary(chunks: Iterable[Chunk]) -> None:
    """Raise if any chunk carries a ``field_type`` outside the vocabulary.

    Args:
        chunks: chunks to check.
    """
    unknown = {chunk.field_type for chunk in chunks} - FIELD_TYPES_ALL
    if unknown:
        raise ValueError(f"chunks carry unknown field_type values: {sorted(unknown)}")
