"""Corpus manifest: load, validate, filter, and export (implementation.md Phase 1).

The manifest (``ingest/sources.csv``) is the single curated list of documents the
assistant is allowed to know about. It is the input to Flow A and the origin of
the citation allowlist used by validator gate G4.

This module is pure: it reads and writes CSV only, performs no network calls,
and requires no model or API key.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from rag.schemas import (
    DOC_TYPES,
    PLACEHOLDER_URL,
    PUBLISHERS,
    VALID_SCHEME_VALUES,
    ManifestRow,
)

#: Default location of the manifest, resolved relative to this file so it works
#: from any working directory.
DEFAULT_MANIFEST: Path = Path(__file__).resolve().parent.parent / "ingest" / "sources.csv"

#: Exact CSV header required by ``ingest/sources.csv``.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "source_id",
    "scheme",
    "category",
    "doc_type",
    "title",
    "publisher",
    "url",
    "enabled",
)

#: Column order of the D2 source-list deliverable (docs/PRD.md §11.1).
SOURCE_LIST_COLUMNS: tuple[str, ...] = (
    "source_id",
    "scheme",
    "doc_type",
    "title",
    "publisher",
    "url",
    "retrieved_at",
    "sha256",
    "status",
)


class ManifestError(ValueError):
    """Raised when the manifest file itself is unusable (missing or malformed)."""


def load_manifest(path: str | Path | None = None) -> list[ManifestRow]:
    """Load the corpus manifest from CSV.

    Args:
        path: manifest path. Defaults to ``ingest/sources.csv``.

    Returns:
        The parsed rows, in file order.

    Raises:
        ManifestError: if the file is missing or the header is not exactly
            :data:`REQUIRED_COLUMNS`.
    """
    manifest_path = Path(path) if path is not None else DEFAULT_MANIFEST
    if not manifest_path.is_file():
        raise ManifestError(f"manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if header != REQUIRED_COLUMNS:
            raise ManifestError(
                f"manifest header mismatch in {manifest_path}\n"
                f"  expected: {','.join(REQUIRED_COLUMNS)}\n"
                f"  found:    {','.join(header)}"
            )
        return [
            ManifestRow(
                source_id=(row.get("source_id") or "").strip(),
                scheme=(row.get("scheme") or "").strip(),
                category=(row.get("category") or "").strip(),
                doc_type=(row.get("doc_type") or "").strip(),
                title=(row.get("title") or "").strip(),
                publisher=(row.get("publisher") or "").strip(),
                url=(row.get("url") or "").strip(),
                enabled=int((row.get("enabled") or "0").strip() or 0),
            )
            for row in reader
        ]


def validate_manifest(rows: Sequence[ManifestRow]) -> list[str]:
    """Validate manifest rows and return human-readable problems.

    An empty list means the manifest is valid.

    Strict URL checking applies only to **enabled** rows. Disabled rows are
    allowed to carry :data:`~rag.schemas.PLACEHOLDER_URL` while the team fills
    in the real primary-source URLs during Phase 1 discovery — that is the whole
    point of shipping them disabled.

    Args:
        rows: rows to validate.

    Returns:
        A list of problem descriptions; empty when the manifest is valid.
    """
    problems: list[str] = []
    seen: dict[str, int] = {}

    for index, row in enumerate(rows, start=2):  # +2: 1-based, and row 1 is the header
        where = f"row {index} ({row.source_id or '<blank source_id>'})"

        if not row.source_id:
            problems.append(f"{where}: source_id is empty")
        else:
            seen[row.source_id] = seen.get(row.source_id, 0) + 1

        if row.scheme not in VALID_SCHEME_VALUES:
            problems.append(
                f"{where}: scheme {row.scheme!r} is not one of "
                f"{sorted(VALID_SCHEME_VALUES)}"
            )
        if not row.category:
            problems.append(f"{where}: category is empty")
        if row.doc_type not in DOC_TYPES:
            problems.append(
                f"{where}: doc_type {row.doc_type!r} is not one of {sorted(DOC_TYPES)}"
            )
        if row.publisher not in PUBLISHERS:
            problems.append(
                f"{where}: publisher {row.publisher!r} is not one of {sorted(PUBLISHERS)}"
            )
        if not row.title:
            problems.append(f"{where}: title is empty")
        if not row.url:
            problems.append(f"{where}: url is empty")
        elif row.is_enabled:
            if row.url == PLACEHOLDER_URL:
                problems.append(
                    f"{where}: url is the placeholder but the row is enabled"
                )
            elif not row.url.startswith("https://"):
                problems.append(f"{where}: url must start with https://, got {row.url!r}")
        if int(row.enabled) not in (0, 1):
            problems.append(f"{where}: enabled must be 0 or 1, got {row.enabled!r}")

    for source_id, count in seen.items():
        if count > 1:
            problems.append(f"duplicate source_id {source_id!r} appears {count} times")

    return problems


def enabled_rows(rows: Iterable[ManifestRow]) -> list[ManifestRow]:
    """Return only the rows enabled for ingestion."""
    return [row for row in rows if row.is_enabled]


def placeholder_rows(rows: Iterable[ManifestRow]) -> list[ManifestRow]:
    """Return disabled rows still carrying the placeholder URL.

    Phase 2 surfaces these loudly in the ingest summary, because a scheme whose
    seed page comes back thin will need its primary source enabled (risk R2).
    """
    return [row for row in rows if row.is_placeholder]


def to_source_list_dicts(
    rows: Sequence[ManifestRow],
    documents: Mapping[str, object] | None = None,
) -> list[dict[str, str]]:
    """Render rows as the D2 source-list deliverable (docs/PRD.md §11.1).

    A row is never dropped, including failed ones — an honest list is the point
    (FR-1.5).

    Args:
        rows: manifest rows to render.
        documents: optional mapping of ``source_id`` to a
            :class:`~rag.schemas.Document` carrying real ``retrieved_at`` and
            ``sha256`` values. Phase 1 has no crawl results, so the derived
            columns stay empty.

    Returns:
        A list of dicts keyed by :data:`SOURCE_LIST_COLUMNS`.
    """
    known = documents or {}
    output: list[dict[str, str]] = []

    for row in rows:
        document = known.get(row.source_id)
        if document is not None:
            status = "failed" if getattr(document, "http_status", 200) != 200 else (
                "thin" if getattr(document, "thin", False) else "ok"
            )
            retrieved_at = str(getattr(document, "retrieved_at", "") or "")
            sha256 = str(getattr(document, "sha256", "") or "")
        elif row.is_placeholder:
            status, retrieved_at, sha256 = "placeholder", "", ""
        else:
            status, retrieved_at, sha256 = "pending", "", ""

        output.append(
            {
                "source_id": row.source_id,
                "scheme": row.scheme,
                "doc_type": row.doc_type,
                "title": row.title,
                "publisher": row.publisher,
                "url": row.url,
                "retrieved_at": retrieved_at,
                "sha256": sha256,
                "status": status,
            }
        )

    return output


def write_source_list(
    rows: Sequence[ManifestRow],
    path: str | Path,
    documents: Mapping[str, object] | None = None,
) -> Path:
    """Write the D2 source list to ``path`` as CSV.

    Args:
        rows: manifest rows to render.
        path: destination CSV path.
        documents: optional ``source_id`` → ``Document`` mapping.

    Returns:
        The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = to_source_list_dicts(rows, documents)

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_LIST_COLUMNS)
        writer.writeheader()
        writer.writerows(records)

    return destination
