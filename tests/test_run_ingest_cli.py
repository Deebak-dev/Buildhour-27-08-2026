"""CLI-level regressions for ingest.run_ingest.

These cover the pipeline wiring rather than the store itself: which artifacts a
run is allowed to write, and where a limited run is allowed to write them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ingest.load import load_from_raw
from ingest.run_ingest import _suffixed, build_parser
from rag.manifest import ManifestRow


def _row(source_id: str) -> ManifestRow:
    return ManifestRow(
        source_id=source_id,
        scheme="hdfc_test",
        category="test",
        doc_type="scheme_page",
        title="Test",
        publisher="groww",
        url=f"https://example.invalid/{source_id}",
        enabled=1,
    )


class TestSuffixed:
    def test_no_suffix_returns_same_path(self, tmp_path: Path) -> None:
        target = tmp_path / "chunks.jsonl"
        assert _suffixed(target, "") == target

    def test_suffix_goes_before_extension(self, tmp_path: Path) -> None:
        # chunks.limit2.jsonl, not chunks.jsonl.limit2 - the final suffix has to
        # stay .jsonl or tools stop recognising the file.
        assert _suffixed(tmp_path / "chunks.jsonl", ".limit2").name == (
            "chunks.limit2.jsonl"
        )

    def test_suffixed_path_does_not_collide_with_canonical(self, tmp_path: Path) -> None:
        canonical = tmp_path / "chunks.jsonl"
        assert _suffixed(canonical, ".limit2") != canonical


class TestParser:
    def test_documented_flags_exist(self) -> None:
        # Every flag the Phase 4 spec lists must be accepted.
        args = build_parser().parse_args(
            ["--manifest", "m.csv", "--timeout", "5", "--rebuild", "--limit", "2"]
        )
        assert args.manifest == "m.csv"
        assert args.timeout == 5
        assert args.rebuild is True
        assert args.limit == 2

    def test_defaults_are_unset(self) -> None:
        args = build_parser().parse_args([])
        assert args.limit is None
        assert args.rebuild is False
        assert args.skip_embed is False


class TestLoadFromRaw:
    def test_missing_snapshot_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "present.html").write_text(
            "<html><body><main><p>hello world</p></main></body></html>",
            encoding="utf-8",
        )

        docs = load_from_raw(
            [_row("present"), _row("absent")],
            raw_dir=raw,
            clean_dir=tmp_path / "clean",
        )

        assert [d.source_id for d in docs] == ["present"]

    def test_rebuilds_text_and_metadata_from_snapshot(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "a.html").write_text(
            "<html><body><main><p>Fund benchmark: NIFTY 100</p></main></body></html>",
            encoding="utf-8",
        )

        doc = load_from_raw(
            [_row("a")],
            raw_dir=raw,
            clean_dir=tmp_path / "clean",
            thin_chars=10,
        )[0]

        assert doc.http_status == 200
        assert doc.content_chars > 0
        assert doc.thin is False
        assert doc.scheme == "hdfc_test"
        assert Path(doc.clean_path).is_file()
        assert "NIFTY 100" in Path(doc.clean_path).read_text(encoding="utf-8")

    def test_short_page_is_flagged_thin(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "a.html").write_text(
            "<html><body><main><p>tiny</p></main></body></html>",
            encoding="utf-8",
        )

        doc = load_from_raw(
            [_row("a")], raw_dir=raw, clean_dir=tmp_path / "clean", thin_chars=1000
        )[0]

        assert doc.thin is True

    def test_does_not_need_the_written_manifest(self, tmp_path: Path) -> None:
        """The point of reading raw: it works with no manifest.json present.

        --skip-fetch used to read data/manifest.json and then overwrite it, so
        a limited run destroyed the full-corpus record and could not be
        recovered offline.
        """
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "a.html").write_text(
            "<html><body><main><p>content</p></main></body></html>",
            encoding="utf-8",
        )
        assert not (tmp_path / "manifest.json").exists()

        docs = load_from_raw([_row("a")], raw_dir=raw, clean_dir=tmp_path / "clean")

        assert len(docs) == 1


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
