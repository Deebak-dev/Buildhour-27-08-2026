"""Tests for the corpus manifest (implementation.md Phase 1).

The committed ``ingest/sources.csv`` must validate cleanly, and the validators
must actually catch malformed rows.
"""

from __future__ import annotations

import pytest

from rag.manifest import (
    DEFAULT_MANIFEST,
    SOURCE_LIST_COLUMNS,
    ManifestError,
    enabled_rows,
    load_manifest,
    placeholder_rows,
    to_source_list_dicts,
    validate_manifest,
    write_source_list,
)
from rag.schemas import DOC_TYPES, PUBLISHERS, SCHEME_KEYS, ManifestRow

#: The 5 seed URLs fixed by docs/PRD.md §4.1.
EXPECTED_SEED_URLS = {
    "hdfc-large-cap__groww_scheme": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    "hdfc-flexi-cap__groww_scheme": "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
    "hdfc-elss__groww_scheme": "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
    "hdfc-small-cap__groww_scheme": "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
    "hdfc-balanced-advantage__groww_scheme": (
        "https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth"
    ),
}


def make_row(**overrides) -> ManifestRow:
    """Build a valid manifest row, applying any field overrides."""
    base = {
        "source_id": "test__row",
        "scheme": "hdfc_large_cap",
        "category": "large_cap",
        "doc_type": "scheme_page",
        "title": "Test row",
        "publisher": "hdfc_amc",
        "url": "https://example.com/scheme",
        "enabled": 1,
    }
    base.update(overrides)
    return ManifestRow(**base)


# ---------------------------------------------------------------------------
# The committed manifest
# ---------------------------------------------------------------------------


def test_committed_manifest_exists() -> None:
    assert DEFAULT_MANIFEST.is_file(), f"missing manifest: {DEFAULT_MANIFEST}"


def test_committed_manifest_has_no_validation_errors() -> None:
    rows = load_manifest()
    assert rows, "manifest is empty"
    assert validate_manifest(rows) == []


def test_manifest_has_at_least_five_enabled_rows() -> None:
    assert len(enabled_rows(load_manifest())) >= 5


def test_manifest_covers_all_five_schemes() -> None:
    schemes = {row.scheme for row in load_manifest()}
    missing = SCHEME_KEYS - schemes
    assert not missing, f"schemes missing from manifest: {sorted(missing)}"


def test_manifest_has_exactly_the_five_seed_urls() -> None:
    """The enabled rows must be the 5 PRD seed URLs, verbatim."""
    by_id = {row.source_id: row.url for row in enabled_rows(load_manifest())}
    for source_id, expected_url in EXPECTED_SEED_URLS.items():
        assert by_id.get(source_id) == expected_url, f"{source_id} URL is wrong"


def test_enabled_rows_are_all_https() -> None:
    for row in enabled_rows(load_manifest()):
        assert row.url.startswith("https://"), f"{row.source_id} is not https"


def test_disabled_rows_may_carry_the_placeholder() -> None:
    """Placeholder rows ship disabled so Phase 2 can flag them loudly."""
    placeholders = placeholder_rows(load_manifest())
    assert placeholders, "expected placeholder rows for the primary sources"
    for row in placeholders:
        assert not row.is_enabled, f"{row.source_id} is a placeholder but enabled"


def test_manifest_doc_types_and_publishers_are_in_vocabularies() -> None:
    for row in load_manifest():
        assert row.doc_type in DOC_TYPES
        assert row.publisher in PUBLISHERS


# ---------------------------------------------------------------------------
# Negative tests — the validators must actually fire
# ---------------------------------------------------------------------------


def test_duplicate_source_id_is_reported() -> None:
    rows = [make_row(), make_row(title="Duplicate of the same source_id")]
    problems = validate_manifest(rows)
    assert any("duplicate source_id" in problem for problem in problems), problems


def test_unknown_scheme_is_reported() -> None:
    problems = validate_manifest([make_row(scheme="hdfc_mega_cap")])
    assert any("is not one of" in problem and "scheme" in problem for problem in problems)


def test_unknown_doc_type_is_reported() -> None:
    problems = validate_manifest([make_row(doc_type="press_release")])
    assert any("doc_type" in problem for problem in problems)


def test_unknown_publisher_is_reported() -> None:
    problems = validate_manifest([make_row(publisher="random_blog")])
    assert any("publisher" in problem for problem in problems)


def test_non_https_url_on_an_enabled_row_is_reported() -> None:
    problems = validate_manifest([make_row(url="http://insecure.example.com")])
    assert any("https://" in problem for problem in problems)


def test_placeholder_url_on_an_enabled_row_is_reported() -> None:
    problems = validate_manifest([make_row(url="TODO_PRIMARY_URL", enabled=1)])
    assert any("placeholder" in problem for problem in problems)


def test_placeholder_url_on_a_disabled_row_is_accepted() -> None:
    assert validate_manifest([make_row(url="TODO_PRIMARY_URL", enabled=0)]) == []


def test_empty_source_id_is_reported() -> None:
    problems = validate_manifest([make_row(source_id="")])
    assert any("source_id is empty" in problem for problem in problems)


def test_invalid_enabled_flag_is_reported() -> None:
    problems = validate_manifest([make_row(enabled=7)])
    assert any("enabled must be 0 or 1" in problem for problem in problems)


# ---------------------------------------------------------------------------
# Filters and the D2 export
# ---------------------------------------------------------------------------


def test_enabled_rows_filters_correctly() -> None:
    rows = [make_row(source_id="a", enabled=1), make_row(source_id="b", enabled=0)]
    assert [row.source_id for row in enabled_rows(rows)] == ["a"]


def test_placeholder_rows_filters_correctly() -> None:
    rows = [
        make_row(source_id="a", url="TODO_PRIMARY_URL", enabled=0),
        make_row(source_id="b", enabled=1),
    ]
    assert [row.source_id for row in placeholder_rows(rows)] == ["a"]


def test_source_list_rows_use_the_d2_schema() -> None:
    records = to_source_list_dicts(load_manifest())
    assert records, "no source-list records produced"
    for record in records:
        assert tuple(record.keys()) == SOURCE_LIST_COLUMNS


def test_source_list_never_drops_a_row() -> None:
    rows = load_manifest()
    assert len(to_source_list_dicts(rows)) == len(rows)


def test_source_list_marks_placeholder_and_pending_rows() -> None:
    records = {record["source_id"]: record for record in to_source_list_dicts(load_manifest())}
    assert records["sebi__investor_education"]["status"] == "placeholder"
    assert records["hdfc-elss__groww_scheme"]["status"] == "pending"


def test_write_source_list_creates_a_readable_csv(tmp_path) -> None:
    destination = write_source_list(load_manifest(), tmp_path / "source_list.csv")
    assert destination.is_file()
    header = destination.read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(SOURCE_LIST_COLUMNS)


# ---------------------------------------------------------------------------
# Malformed files
# ---------------------------------------------------------------------------


def test_missing_file_raises_manifest_error(tmp_path) -> None:
    with pytest.raises(ManifestError):
        load_manifest(tmp_path / "nope.csv")


def test_wrong_header_raises_manifest_error(tmp_path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("source_id,url\na,https://example.com\n", encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(bad)
