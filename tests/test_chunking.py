"""Chunking tests (docs/implementation.md Phase 3).

The properties that matter for fact accuracy: a FAQ question and its answer are
never separated, a fee-table row always carries its column header, and every
chunk carries a valid ``field_type``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ingest.chunk import (
    build_chunks,
    estimate_tokens,
    field_type_histogram,
    profile,
    tag_field_type,
    write_chunks,
)
from rag.config import ChunkConfig
from rag.schemas import FIELD_TYPES_ALL, Chunk, Document, DocShape

FAQ_HTML = """
<html><body><main>
Frequently Asked Questions
What is the expense ratio of the fund?
The expense ratio is 1.24% per annum for the direct growth plan.
What is the exit load?
Exit load is 1% if units are redeemed within 12 months.
What is the minimum SIP amount?
The minimum SIP amount is INR 500 per month.
How long is the lock-in period?
The lock-in period is 3 years from the date of investment.
Is a direct plan different from a regular plan?
A direct plan has a lower expense ratio than a regular plan.
</main></body></html>
"""

TABLE_HTML = """
<html><body><main>
Scheme particulars
Plan | Expense ratio | Exit load | Minimum SIP
Direct Growth | 1.24% | 1% within 12 months | INR 500
Direct Dividend | 1.25% | 1% within 12 months | INR 500
Regular Growth | 1.55% | 1% within 12 months | INR 500
</main></body></html>
"""


def make_doc(tmp_path: Path, html: str, *, source_id: str = "doc", **overrides) -> Document:
    """Build a Document whose clean_path holds ``html``."""
    clean = tmp_path / f"{source_id}.txt"
    clean.write_text(_to_text(html), encoding="utf-8")
    base = dict(
        source_id=source_id,
        scheme="hdfc_large_cap",
        category="large_cap",
        doc_type="scheme_page",
        title="Fixture",
        publisher="groww",
        source_url=f"https://example.com/{source_id}",
        retrieved_at="2026-09-27T00:00:00+00:00",
        sha256="0" * 64,
        http_status=200,
        content_chars=0,
        clean_path=str(clean),
    )
    base.update(overrides)
    return Document(**base)


def _to_text(html: str) -> str:
    """Turn fixture HTML into the plain text the cleaner would have produced."""
    from bs4 import BeautifulSoup

    return BeautifulSoup(html, "lxml").get_text(separator="\n", strip=True)


# ---------------------------------------------------------------------------
# Shape detection
# ---------------------------------------------------------------------------


def test_profile_detects_faq() -> None:
    assert profile(_to_text(FAQ_HTML)) is DocShape.FAQ


def test_profile_detects_table() -> None:
    assert profile(_to_text(TABLE_HTML)) is DocShape.TABLE


def test_profile_defaults_to_narrative() -> None:
    text = _to_text("<html><body><main><p>Fund objective paragraph. More prose here.</p></main></body></html>")
    assert profile(text) is DocShape.NARRATIVE


def test_profile_handles_empty_text() -> None:
    assert profile("") is DocShape.NARRATIVE


def test_profile_detects_the_label_value_stream_of_the_real_pages() -> None:
    """The published scheme pages extract one token per line."""
    text = "\n".join([
        "HDFC Large Cap Fund",
        "Min. for SIP",
        "₹100",
        "Fund size (AUM)",
        "₹39,933.37 Cr",
        "Expense ratio",
        "1.03%",
        "Exit load",
        "Nil",
        "Fund benchmark",
        "NIFTY 100 Total Return Index",
    ])
    assert profile(text) is DocShape.LABEL_VALUE


def test_label_value_chunks_pair_each_label_with_its_value(tmp_path: Path) -> None:
    text = "\n".join([
        "Min. for SIP", "₹100",
        "Fund size (AUM)", "₹39,933.37 Cr",
        "Expense ratio", "1.03%",
        "Exit load", "Nil",
        "Riskometer", "Very High",
        "Fund benchmark", "NIFTY 100 Total Return Index",
    ])
    doc = make_doc(
        tmp_path,
        f"<html><body><main>{text}</main></body></html>",
        source_id="lv",
    )
    chunks = build_chunks([doc])
    by_field: dict[str, list[str]] = {}
    for chunk in chunks:
        by_field.setdefault(chunk.field_type, []).append(chunk.text)

    assert "Expense ratio: 1.03%" in by_field.get("expense_ratio", [])
    assert "₹100" in " ".join(by_field.get("min_sip", []))
    assert "NIFTY 100 Total Return Index" in " ".join(by_field.get("benchmark", []))
    assert "Very High" in " ".join(by_field.get("riskometer", []))


def test_label_value_never_orphans_a_value(tmp_path: Path) -> None:
    text = "\n".join([
        "Expense ratio", "1.03%", "Exit load", "1%",
        "Lock-in period", "3 years", "Minimum SIP", "₹500",
    ])
    doc = make_doc(tmp_path, f"<html><body><main>{text}</main></body></html>", source_id="lv2")
    for chunk in build_chunks([doc]):
        assert ":" in chunk.text, f"orphaned value: {chunk.text!r}"


def test_label_value_discards_cross_scheme_link_runs(tmp_path: Path) -> None:
    """A run of scheme names is a link list, not a fact (PRD A14)."""
    text = "\n".join([
        "Expense ratio", "1.03%",
        "Minimum SIP", "₹500",
        "Fund benchmark", "NIFTY 100 Total Return Index",
        "Also manages these schemes",
        "HDFC Credit Risk Fund Direct Growth",
        "HDFC Banking and PSU Debt Fund Direct Growth",
        "HDFC Arbitrage Fund Direct Growth",
    ])
    doc = make_doc(tmp_path, f"<html><body><main>{text}</main></body></html>", source_id="lv3")
    joined = " ".join(chunk.text for chunk in build_chunks([doc]))
    assert "Credit Risk Fund" not in joined
    assert "Arbitrage Fund" not in joined
    assert "Expense ratio: 1.03%" in joined


def test_unpunctuated_runs_are_still_size_bounded(tmp_path: Path) -> None:
    """A holdings list has no sentence punctuation, so the word split must hold.

    The cleaned corpus has no blank lines either, so an entire section arrives as
    a single paragraph. Before the word-level fallback such a run could not be
    split at all and became one enormous chunk.
    """
    names = [
        "ICICI Bank Ltd.", "HDFC Bank Ltd.", "Infosys Ltd.", "Tata Motors Ltd.",
        "Reliance Industries Ltd.", "Hindustan Unilever Ltd.", "ITC Ltd.",
        "State Bank of India Ltd.", "Axis Bank Ltd.", "Kotak Mahindra Bank Ltd.",
        "Bajaj Finance Ltd.", "HDFC Life Insurance Co. Ltd.", "Britannia Ltd.",
        "Sun Pharmaceutical Ltd.", "Maruti Suzuki India Ltd.", "Tata Steel Ltd.",
        "Power Grid Corp Ltd.", "NTPC Ltd.", "Coal India Ltd.", "Grasim Ltd.",
    ]
    holdings = "\n".join(names * 3)
    doc = make_doc(
        tmp_path,
        f"<html><body><main>Portfolio\n{holdings}</main></body></html>",
        source_id="run",
    )
    cfg = ChunkConfig(strategy="auto", target_tokens=80, min_tokens=20, max_tokens=140, overlap_pct=12)
    chunks = build_chunks([doc], cfg=cfg)
    assert len(chunks) > 1
    for chunk in chunks:
        assert estimate_tokens(chunk.text) <= cfg.max_tokens * 1.5


def test_a_long_riskometer_sentence_survives(tmp_path: Path) -> None:
    """Regression: the About blurb was swallowed by the returns calculator.

    A performance filter that removed whole oversized sections also removed the
    legitimate riskometer sentence that followed them, leaving the golden
    riskometer question unanswerable.
    """
    calculator = (
        "Return calculator\n"
        "1 year\n-3.43\n%\n3 years\n%\n5 years\n%\n10 years\n%"
    )
    about = (
        "About\n"
        "The scheme is rated Very High risk. "
        "Minimum SIP Investment is set to 500. "
        "Minimum Lumpsum Investment is 500."
    )
    doc = make_doc(
        tmp_path,
        f"<html><body><main>{calculator}{about}</main></body></html>",
        source_id="abt",
        scheme="hdfc_elss",
        category="elss",
    )
    chunks = build_chunks([doc])
    assert any("rated Very High risk" in chunk.text for chunk in chunks)
    assert any(chunk.field_type == "riskometer" for chunk in chunks)


def test_a_bare_horizon_is_not_a_lock_in(tmp_path: Path) -> None:
    """``3 years`` from a growth chart must not read as a lock-in period."""
    text = "\n".join([
        "What would my investment have grown into?",
        "60,000", "57,940", "-3.43", "%", "3 years", "1,80,000", "1,87,909",
    ])
    doc = make_doc(tmp_path, f"<html><body><main>{text}</main></body></html>", source_id="hz")
    chunks = build_chunks([doc])
    assert not [c for c in chunks if c.field_type == "lock_in"]


def test_cross_scheme_facts_are_dropped(tmp_path: Path) -> None:
    """Another fund's fact on this page must not enter the corpus (PRD A14)."""
    text = "\n".join([
        "Expense ratio", "1.03%",
        "Fund benchmark", "NIFTY 100 Total Return Index",
        "HDFC Small Cap Fund Direct Growth: HDFC NIFTY 100 Total Return Index",
        "HDFC Retirement Fund Hybrid Debt Plan Direct Growth: HDFC NIFTY 50 ETF IDCW",
    ])
    doc = make_doc(tmp_path, f"<html><body><main>{text}</main></body></html>", source_id="xs")
    joined = " ".join(chunk.text for chunk in build_chunks([doc]))
    assert "Retirement Fund" not in joined
    assert "HDFC Small Cap Fund Direct Growth:" not in joined
    # The scheme's own facts survive.
    assert "Expense ratio: 1.03%" in joined
    assert "Fund benchmark: NIFTY 100 Total Return Index" in joined


def test_a_page_may_mention_its_own_name(tmp_path: Path) -> None:
    """A scheme's own name on its own page is not contamination."""
    text = "\n".join([
        "HDFC ELSS Tax Saver Fund Direct Plan Growth",
        "is an equity oriented fund with a 3 year lock-in period.",
    ])
    doc = make_doc(
        tmp_path,
        f"<html><body><main>{text}</main></body></html>",
        source_id="own",
        scheme="hdfc_elss",
        category="elss",
    )
    chunks = build_chunks([doc])
    assert any("lock-in" in chunk.text for chunk in chunks)


def test_narrative_split_never_orphans_a_value_either(tmp_path: Path) -> None:
    """Safety net: even a prose document keeps a label with its value."""
    text = (
        "The scheme invests in equity instruments.\n"
        "Expense ratio\n"
        "1.03%\n"
        "The fund follows a benchmark noted below for reference purposes.\n"
    )
    doc = make_doc(
        tmp_path, f"<html><body><main>{text}</main></body></html>", source_id="narr"
    )
    joined = " ".join(chunk.text for chunk in build_chunks([doc]))
    assert "Expense ratio: 1.03%" in joined


# ---------------------------------------------------------------------------
# field_type tagging
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Expense ratio: 1.24%", "expense_ratio"),
        ("Total expense ratio of the scheme is 1.55%", "expense_ratio"),
        ("Exit load: 1% if redeemed within 12 months", "exit_load"),
        ("Lock-in period: 3 years", "lock_in"),
        ("Minimum SIP amount is INR 500", "min_sip"),
        ("The benchmark is Nifty 100", "benchmark"),
        ("The current riskometer level is very high", "riskometer"),
        ("Section 80C deduction up to INR 150000", "tax_benefit"),
        # A download instruction outranks the document type it refers to, which
        # matches the Phase 5 golden set expectation for query q07.
        ("How to download your capital gains statement", "statement_guide"),
        ("The capital gains statement shows realised gains", "tax_report"),
        ("Managed by the fund manager since 2015", "fund_manager"),
    ],
)
def test_tag_field_type_on_body(text: str, expected: str) -> None:
    assert tag_field_type(text) == expected


def test_tag_field_type_prefers_the_section_heading() -> None:
    assert tag_field_type("Some prose about charges.", "Exit load") == "exit_load"


def test_tag_field_type_falls_back_to_other() -> None:
    assert tag_field_type("An unrelated sentence with nothing matching.") == "other"


def test_a_bare_direct_plan_mention_does_not_hijack_the_tag() -> None:
    """Every scheme is a Direct Growth plan, so a bare mention must not win.

    Otherwise the Phase 5 field_type boost would retag the whole corpus and
    become useless (docs/architecture.md Â§5.3, PRD risk R4).
    """
    assert tag_field_type("Expense ratio: 1.24% for the Direct Growth plan.") == "expense_ratio"


def test_a_direct_vs_regular_comparison_is_tagged() -> None:
    assert tag_field_type("A direct plan versus regular plan comparison.") == "direct_vs_regular"


def test_every_tag_is_in_the_vocabulary() -> None:
    samples = [
        "Exit load: 1%", "Minimum SIP: 500", "Lock-in: 3 years", "Expense ratio: 1.24%",
        "Benchmark: Nifty 100", "Riskometer: high", "80C tax benefit",
        "Capital gains statement download", "Fund manager", "NAV", "Lump sum",
        "Nothing relevant here at all.",
    ]
    for sample in samples:
        assert tag_field_type(sample) in FIELD_TYPES_ALL


# ---------------------------------------------------------------------------
# Chunking behaviour
# ---------------------------------------------------------------------------


def test_faq_pairs_are_never_split(tmp_path: Path) -> None:
    chunks = build_chunks([make_doc(tmp_path, FAQ_HTML, source_id="faq")])
    questions = [line for line in _to_text(FAQ_HTML).splitlines() if line.endswith("?")]
    assert len(questions) >= 5
    for question in questions:
        # The chunk holding the question must also hold its answer.
        holder = next((c for c in chunks if question in c.text), None)
        assert holder is not None, f"question lost: {question}"
        assert "?" in holder.text and len(holder.text) > len(question) + 20


def test_no_chunk_ends_mid_question(tmp_path: Path) -> None:
    chunks = build_chunks([make_doc(tmp_path, FAQ_HTML, source_id="faq")])
    for chunk in chunks:
        lines = [line for line in chunk.text.splitlines() if line.strip()]
        if lines and lines[-1].endswith("?"):
            pytest.fail(f"chunk ends on a bare question: {lines[-1]!r}")


def test_table_rows_each_carry_the_header(tmp_path: Path) -> None:
    chunks = build_chunks([make_doc(tmp_path, TABLE_HTML, source_id="tbl")])
    assert len(chunks) >= 3
    for chunk in chunks:
        assert "Expense ratio" in chunk.text, "row lost its column header"


def test_table_rows_are_not_concatenated(tmp_path: Path) -> None:
    chunks = build_chunks([make_doc(tmp_path, TABLE_HTML, source_id="tbl")])
    for chunk in chunks:
        body_lines = [line for line in chunk.text.splitlines() if "|" in line]
        assert len(body_lines) <= 2, "a chunk holds more than one data row"


def test_narrative_chunks_respect_max_tokens_and_overlap(tmp_path: Path) -> None:
    cfg = ChunkConfig(target_tokens=80, min_tokens=20, max_tokens=140, overlap_pct=12)
    long_text = "\n\n".join(
        f"Section {i}\n" + ". ".join(f"Sentence {i}-{j} about fund charges and terms." for j in range(6))
        for i in range(8)
    )
    doc = make_doc(tmp_path, f"<html><body><main>{long_text}</main></body></html>", source_id="nar")
    chunks = build_chunks([doc], cfg=cfg)
    assert len(chunks) > 1
    for chunk in chunks:
        assert estimate_tokens(chunk.text) <= cfg.max_tokens * 1.5
    # At least one adjacent pair shares text, i.e. overlap was applied.
    assert any(
        set(chunks[i].text.split()) & set(chunks[i + 1].text.split())
        for i in range(len(chunks) - 1)
    )


def test_chunk_ids_are_deterministic(tmp_path: Path) -> None:
    doc = make_doc(tmp_path, FAQ_HTML, source_id="faq")
    first = [chunk.chunk_id for chunk in build_chunks([doc])]
    second = [chunk.chunk_id for chunk in build_chunks([doc])]
    assert first == second
    assert len(set(first)) == len(first)


def test_every_chunk_carries_full_citation_metadata(tmp_path: Path) -> None:
    doc = make_doc(tmp_path, FAQ_HTML, source_id="faq")
    for chunk in build_chunks([doc]):
        assert chunk.field_type in FIELD_TYPES_ALL
        assert chunk.source_url == doc.source_url
        assert chunk.retrieved_at == doc.retrieved_at
        assert chunk.publisher == doc.publisher
        assert chunk.scheme == doc.scheme
        assert chunk.text.strip()


def test_ordinals_restart_per_document(tmp_path: Path) -> None:
    a = make_doc(tmp_path, FAQ_HTML, source_id="a")
    b = make_doc(tmp_path, FAQ_HTML, source_id="b")
    chunks = build_chunks([a, b])
    assert [c.ordinal for c in chunks if c.source_id == "a"] == list(
        range(sum(1 for c in chunks if c.source_id == "a"))
    )
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_a_document_with_no_clean_file_yields_no_chunks(tmp_path: Path) -> None:
    doc = make_doc(tmp_path, FAQ_HTML, source_id="x")
    doc = Document(**{**doc.__dict__, "clean_path": str(tmp_path / "missing.txt")})
    assert build_chunks([doc]) == []


def test_empty_chunks_are_dropped(tmp_path: Path) -> None:
    assert all(c.text.strip() for c in build_chunks([make_doc(tmp_path, FAQ_HTML)]))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def test_write_chunks_produces_readable_jsonl(tmp_path: Path) -> None:
    import json

    chunks = build_chunks([make_doc(tmp_path, FAQ_HTML, source_id="faq")])
    out = write_chunks(chunks, tmp_path / "chunks.jsonl")
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(chunks)
    first = json.loads(lines[0])
    assert {"chunk_id", "text", "field_type", "source_url"} <= set(first)


def test_histogram_counts_field_types(tmp_path: Path) -> None:
    chunks = build_chunks([make_doc(tmp_path, FAQ_HTML, source_id="faq")])
    histogram = field_type_histogram(chunks)
    assert sum(histogram.values()) == len(chunks)
    assert all(name in FIELD_TYPES_ALL for name in histogram)


def test_estimate_tokens_is_positive_for_text() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("one two three") >= 1
