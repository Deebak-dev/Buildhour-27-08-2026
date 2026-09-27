"""Corpus loading: fetch, clean, hash, and write the manifest.

Flow A, stage 1 — "Loading" (docs/architecture.md §5.1, PRD FR-1.2 to FR-1.6).

Design points:

* A single bad URL never stops the run. Failures are logged and recorded on the
  :class:`~rag.schemas.Document` so the D2 source list stays honest (FR-1.5).
* ``extract`` is **pure and offline**: it reads the raw snapshot from disk, never
  the network, so it is trivially testable.
* Response bodies are never logged.

Phase 7 moves the ``log_event`` helper here into ``rag/logging_utils.py``; until
then this module owns the one structured logger the ingest path uses.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.robotparser
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from rag.config import THIN_CHARS, AppConfig, load_config
from rag.schemas import Document, ManifestRow

#: A realistic UA is required: many public fund pages serve a stub to unknown
#: agents, which is the most common cause of a "thin" page (risk R2).
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
    "mf-faq-assistant/1.0 (educational prototype; +local research)"
)

#: Status codes worth retrying with exponential backoff.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

#: Chrome and navigation furniture removed before main-content detection.
STRIP_TAGS = (
    "script", "style", "noscript", "template", "svg",
    "nav", "header", "footer", "aside", "form", "iframe",
)

#: Class/id fragments that mark cookie walls, ad slots, and promos.
NOISE_PATTERN = re.compile(
    r"cookie|consent|ad-|advert|banner|popup|modal|newsletter|subscribe|"
    r"promo|overlay|toast|skip-link|navbar|footer|header",
    re.IGNORECASE,
)

BLANK_LINES = re.compile(r"\n{3,}")


# ---------------------------------------------------------------------------
# Logging (temporary home; moves to rag/logging_utils.py in Phase 7)
# ---------------------------------------------------------------------------


def log_event(stage: str, **fields: Any) -> None:
    """Emit one structured JSON line to stdout.

    Only metadata is ever logged. Response bodies, page text, and query text are
    excluded by construction — the allowlist is applied before serialization.

    Args:
        stage: pipeline stage name, e.g. ``"load"``.
        **fields: metadata to include; unknown keys are dropped.
    """
    allowed = {
        "source_id", "scheme", "publisher", "http_status", "outcome",
        "content_chars", "thin", "n_documents", "n_chunks", "attempts",
        "skipped", "reason", "elapsed_s",
    }
    record = {"stage": stage, **{k: v for k, v in fields.items() if k in allowed}}
    print(json.dumps(record, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------------------
# robots.txt politeness
# ---------------------------------------------------------------------------


class _Robots:
    """Cached ``robots.txt`` checker with a crawl delay (one fetch per host)."""

    def __init__(self, delay: float = 1.0, timeout: int = 30) -> None:
        self._delay = delay
        self._timeout = timeout
        self._parsers: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._last_request: dict[str, float] = {}

    def allowed(self, url: str) -> bool:
        """Return whether ``url`` may be fetched, honouring the crawl delay."""
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._parsers:
            self._parsers[origin] = self._fetch_parser(origin)
        parser = self._parsers[origin]
        if parser is None:
            return True
        return parser.can_fetch(USER_AGENT, url)

    def wait(self, url: str) -> None:
        """Sleep so consecutive requests to one host are at least ``delay`` apart."""
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        previous = self._last_request.get(origin)
        now = time.monotonic()
        if previous is not None:
            remaining = self._delay - (now - previous)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request[origin] = time.monotonic()

    def _fetch_parser(self, origin: str) -> urllib.robotparser.RobotFileParser | None:
        parser = urllib.robotparser.RobotFileParser()
        try:
            response = requests.get(
                f"{origin}/robots.txt",
                timeout=self._timeout,
                headers={"User-Agent": USER_AGENT},
            )
        except requests.RequestException:
            return None
        if response.status_code >= 400:
            # No robots.txt means no stated restrictions.
            return None
        parser.parse(response.text.splitlines())
        return parser


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _sha256(payload: bytes) -> str:
    """Return the hex SHA-256 of ``payload``."""
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fetch_one(
    row: ManifestRow,
    session: requests.Session,
    robots: _Robots,
    *,
    timeout: int,
    max_attempts: int,
    raw_dir: Path,
) -> Document:
    """Fetch a single manifest row, retrying transient failures.

    Returns a :class:`Document` in every case — on failure the document carries a
    non-200 ``http_status`` and zero content, so callers never need to branch on
    an exception.
    """
    base = {
        "source_id": row.source_id,
        "scheme": row.scheme,
        "category": row.category,
        "doc_type": row.doc_type,
        "title": row.title,
        "publisher": row.publisher,
        "source_url": row.url,
    }

    if not robots.allowed(row.url):
        log_event("load", source_id=row.source_id, outcome="skipped", reason="robots_disallowed")
        return Document(**base, retrieved_at=_utc_now(), sha256="", http_status=0,
                        content_chars=0, raw_path="", thin=True)

    raw_path = raw_dir / f"{row.source_id}.html"
    last_status = 0
    error = ""

    for attempt in range(1, max_attempts + 1):
        robots.wait(row.url)
        try:
            response = session.get(row.url, timeout=timeout, allow_redirects=True)
            last_status = response.status_code
        except requests.RequestException as exc:
            error = type(exc).__name__
            last_status = 0
            response = None  # type: ignore[assignment]

        if response is not None and last_status == 200:
            payload = response.content
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(payload)
            log_event(
                "load",
                source_id=row.source_id,
                http_status=200,
                outcome="ok",
                attempts=attempt,
            )
            return Document(
                **base,
                retrieved_at=_utc_now(),
                sha256=_sha256(payload),
                http_status=200,
                content_chars=0,
                raw_path=str(raw_path),
                thin=False,
            )

        if last_status not in RETRY_STATUSES and last_status != 0:
            break  # 404/403/etc. will not improve on retry
        if attempt < max_attempts:
            time.sleep(2 ** (attempt - 1))  # exponential backoff

    log_event(
        "load",
        source_id=row.source_id,
        http_status=last_status,
        outcome="failed",
        attempts=max_attempts,
        reason=error or f"http_{last_status}",
    )
    return Document(
        **base,
        retrieved_at=_utc_now(),
        sha256="",
        http_status=last_status,
        content_chars=0,
        raw_path="",
        thin=True,
    )


def fetch(
    rows: Sequence[ManifestRow],
    *,
    timeout: int = 30,
    max_attempts: int = 2,
    crawl_delay: float = 1.0,
    raw_dir: str | Path | None = None,
) -> list[Document]:
    """Fetch every row, snapshotting raw bytes to ``raw_dir``.

    Args:
        rows: manifest rows to fetch.
        timeout: per-request timeout in seconds.
        max_attempts: total attempts per URL, including the first.
        crawl_delay: minimum seconds between requests to the same host.
        raw_dir: destination for raw snapshots. Defaults to ``data/raw``.

    Returns:
        One :class:`Document` per input row, in order. Failed fetches carry a
        non-200 ``http_status`` rather than raising.
    """
    destination = Path(raw_dir) if raw_dir is not None else load_config().raw_dir
    destination.mkdir(parents=True, exist_ok=True)
    robots = _Robots(delay=crawl_delay, timeout=timeout)

    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-IN,en;q=0.9"})
        return [
            _fetch_one(
                row, session, robots,
                timeout=timeout, max_attempts=max_attempts, raw_dir=destination,
            )
            for row in rows
        ]


# ---------------------------------------------------------------------------
# Cleaning (pure and offline)
# ---------------------------------------------------------------------------


def _strip_noise(soup: BeautifulSoup) -> None:
    """Remove scripts, chrome, and cookie/ad furniture in place."""
    for tag in soup(list(STRIP_TAGS)):
        tag.decompose()
    for element in soup.find_all(True):
        try:
            # bs4 returns class as a list, and on some real-world markup
            # element.attrs is None, so both are read defensively.
            attrs = getattr(element, "attrs", None) or {}
            classes = attrs.get("class") or []
            if isinstance(classes, str):
                classes = [classes]
            marker = " ".join([*(str(item) for item in classes), str(attrs.get("id") or "")])
            if marker and NOISE_PATTERN.search(marker):
                element.decompose()
        except (AttributeError, TypeError):  # noqa: PERF203 - skip only this element
            continue


def _main_content(soup: BeautifulSoup):
    """Return the main content element, or the whole document as a fallback."""
    for finder in (
        lambda: soup.find("main"),
        lambda: soup.find("article"),
        lambda: soup.find(attrs={"role": "main"}),
        lambda: soup.find("body"),
    ):
        element = finder()
        if element is not None:
            return element
    return soup


def extract(doc: Document) -> tuple[str, int]:
    """Extract main readable text from a document's raw snapshot.

    This function performs no network access: it reads ``doc.raw_path`` from
    disk, which makes it fast to test against fixture HTML.

    Args:
        doc: a document whose ``raw_path`` points at a raw HTML snapshot.

    Returns:
        ``(clean_text, character_count)``. Returns ``("", 0)`` when the snapshot
        is missing or unreadable.
    """
    if not doc.raw_path:
        return "", 0

    raw_path = Path(doc.raw_path)
    if not raw_path.is_file():
        return "", 0

    try:
        soup = BeautifulSoup(raw_path.read_text(encoding="utf-8", errors="replace"), "lxml")
    except Exception:  # noqa: BLE001 - a malformed page must not stop the run
        return "", 0

    _strip_noise(soup)
    text = _main_content(soup).get_text(separator="\n", strip=True)
    text = BLANK_LINES.sub("\n\n", "\n".join(line.strip() for line in text.splitlines()))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, len(text)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def load_all(
    rows: Sequence[ManifestRow],
    *,
    timeout: int = 30,
    max_attempts: int = 2,
    crawl_delay: float = 1.0,
    raw_dir: str | Path | None = None,
    clean_dir: str | Path | None = None,
    thin_chars: int = THIN_CHARS,
) -> list[Document]:
    """Fetch and clean every row, writing clean text to ``clean_dir``.

    Args:
        rows: manifest rows to load.
        timeout: per-request timeout in seconds.
        max_attempts: total attempts per URL.
        crawl_delay: minimum seconds between requests to the same host.
        raw_dir: raw snapshot directory.
        clean_dir: cleaned text directory.
        thin_chars: character count below which a page is flagged thin (R2).

    Returns:
        The documents, each with ``content_chars``, ``clean_path``, and ``thin``
        populated.
    """
    config = load_config()
    raw = Path(raw_dir) if raw_dir is not None else config.raw_dir
    clean = Path(clean_dir) if clean_dir is not None else config.clean_dir
    clean.mkdir(parents=True, exist_ok=True)

    fetched = fetch(
        rows,
        timeout=timeout,
        max_attempts=max_attempts,
        crawl_delay=crawl_delay,
        raw_dir=raw,
    )

    results: list[Document] = []
    for doc in fetched:
        text, chars = extract(doc)
        clean_path = clean / f"{doc.source_id}.txt"
        if text:
            clean_path.write_text(text, encoding="utf-8")
        thin = chars < thin_chars
        results.append(
            Document(
                **{
                    **{k: v for k, v in asdict(doc).items()},
                    "content_chars": chars,
                    "clean_path": str(clean_path) if text else "",
                    "thin": thin,
                }
            )
        )
        if thin:
            log_event(
                "load",
                source_id=doc.source_id,
                http_status=doc.http_status,
                content_chars=chars,
                thin=True,
                outcome="thin_page",
            )

    return results


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def write_manifest(
    documents: Iterable[Document],
    *,
    path: str | Path | None = None,
    embedding_model: str | None = None,
    dimension: int | None = None,
) -> Path:
    """Write ``data/manifest.json``, the source of the citation allowlist.

    Args:
        documents: the loaded documents.
        path: destination. Defaults to ``data/manifest.json``.
        embedding_model: recorded so a model change forces a rebuild (FR-3.3).
        dimension: recorded so a model change forces a rebuild (FR-3.3).

    Returns:
        The path written.
    """
    config = load_config()
    destination = Path(path) if path is not None else config.manifest_json
    destination.parent.mkdir(parents=True, exist_ok=True)

    docs = list(documents)
    payload = {
        "generated_at": _utc_now(),
        "n_documents": len(docs),
        "n_thin": sum(1 for doc in docs if doc.thin),
        "embedding_model": embedding_model or config.embedding.model,
        "dimension": dimension if dimension is not None else config.embedding.dimension,
        "documents": [asdict(doc) for doc in docs],
    }
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log_event("load", n_documents=len(docs), outcome="manifest_written")
    return destination


def read_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Read ``data/manifest.json``.

    Args:
        path: manifest path. Defaults to ``data/manifest.json``.

    Returns:
        The parsed manifest, or an empty dict when it does not exist.
    """
    source = Path(path) if path is not None else load_config().manifest_json
    if not source.is_file():
        return {}
    return json.loads(source.read_text(encoding="utf-8"))


def url_allowlist(manifest_path: str | Path | None = None) -> set[str]:
    """Return the set of URLs the assistant is allowed to cite.

    This is the basis of validator gate G4, which refuses any citation outside
    this set (architecture.md §6.6, invariant I3). Only successfully fetched
    pages qualify; a page that returned an error is not a citable source.

    Args:
        manifest_path: manifest path. Defaults to ``data/manifest.json``.

    Returns:
        The allowlisted source URLs; empty when the manifest is missing.
    """
    manifest = read_manifest(manifest_path)
    return {
        str(entry.get("source_url"))
        for entry in manifest.get("documents", [])
        if int(entry.get("http_status", 0) or 0) == 200 and entry.get("source_url")
    }


def load_from_raw(
    rows: Sequence[ManifestRow],
    *,
    raw_dir: str | Path,
    clean_dir: str | Path | None = None,
    thin_chars: int = THIN_CHARS,
) -> list[Document]:
    """Rebuild documents from the raw HTML snapshots, with no network access.

    The raw snapshots are the source of truth; ``data/manifest.json`` is derived
    from them. Reading the snapshots directly means ``--skip-fetch`` cannot be
    broken by a stale or truncated manifest - and, more importantly, a partial
    run cannot destroy the record of a full one.

    Args:
        rows: manifest rows, normally the enabled ones.
        raw_dir: directory holding ``<source_id>.html`` snapshots.
        clean_dir: where to write ``<source_id>.txt``. Skipped when ``None``.
        thin_chars: a document below this many characters is thin.

    Returns:
        One :class:`Document` per row whose snapshot exists on disk. Rows with no
        snapshot are omitted, since there is nothing to chunk.
    """
    raw = Path(raw_dir)
    clean = Path(clean_dir) if clean_dir else None

    results: list[Document] = []
    for row in rows:
        raw_path = raw / f"{row.source_id}.html"
        if not raw_path.is_file():
            log_event(
                "load",
                source_id=row.source_id,
                outcome="skipped",
                reason="no_raw_snapshot",
            )
            continue

        doc = Document(
            source_id=row.source_id,
            scheme=row.scheme,
            category=row.category,
            doc_type=row.doc_type,
            title=row.title,
            publisher=row.publisher,
            source_url=row.url,
            retrieved_at=_utc_now(),
            sha256=_sha256(raw_path.read_bytes()),
            http_status=200,
            content_chars=0,
            raw_path=str(raw_path),
            clean_path="",
            thin=True,
        )
        text, chars = extract(doc)
        clean_path = ""
        if text and clean is not None:
            clean.mkdir(parents=True, exist_ok=True)
            target = clean / f"{row.source_id}.txt"
            target.write_text(text, encoding="utf-8")
            clean_path = str(target)
        # Document is frozen, so the measured fields are swapped in one go.
        results.append(
            replace(
                doc,
                content_chars=chars,
                clean_path=clean_path,
                thin=chars < thin_chars,
            )
        )

    return results


def load_documents(
    rows: Sequence[ManifestRow] | None = None,
    *,
    manifest: str | Path | None = None,
) -> list[Document]:
    """Rebuild :class:`Document` objects from a written manifest.

    Phase 3 (chunking) is offline and must not re-crawl, so it reads documents
    back from ``data/manifest.json`` through this helper.

    Args:
        rows: unused, accepted for signature symmetry with :func:`load_all`.
        manifest: manifest path. Defaults to ``data/manifest.json``.

    Returns:
        The reconstructed documents.
    """
    payload = read_manifest(manifest)
    documents: list[Document] = []
    for entry in payload.get("documents", []):
        known = {f for f in Document.__dataclass_fields__}  # type: ignore[attr-defined]
        documents.append(Document(**{k: v for k, v in entry.items() if k in known}))
    return documents


def default_config() -> AppConfig:
    """Return the loaded project configuration (convenience re-export)."""
    return load_config()
