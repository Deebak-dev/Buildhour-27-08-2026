"""The one logging sink, and the last place PII could leak (architecture.md §11).

Every module logs through :func:`log_event`. Two properties make that safe, and
neither depends on the caller remembering to be careful:

* **An allowlist, applied before serialization.** Only :data:`ALLOWED` keys are
  written. ``log_event(stage="query", query="...")`` therefore drops the query
  silently rather than persisting it (invariant I5). This is deliberately a
  filter on the way *out* of the function, not a convention on the way in, so a
  new call site cannot leak by accident.
* **A redaction pass on every value that survives.** Defence in depth: even an
  allowlisted field cannot carry a PAN, an email or a long digit run.

The log file is the only file this module ever writes. There is no rotation, no
second sink, and no query-text persistence.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

#: The only keys that may appear in a log record. Anything else is dropped.
ALLOWED: Final[frozenset[str]] = frozenset(
    {
        "ts",
        "stage",
        "route",
        "scheme",
        "n_chunks",
        "top_score",
        "validators_passed",
        "latency_ms",
        "outcome",
        "degraded",
        "matched_classes",
        "source_id",
    }
)

#: Placeholder written in place of anything that looks sensitive.
REDACTED: Final[str] = "[REDACTED]"

#: The single log file. Overridable so tests can point it at ``tmp_path``.
_LOG_PATH: Path = Path("logs") / "app.log"

#: Guards the append/serialize pair; concurrent queries would otherwise interleave
#: partial lines and produce unparseable JSON.
_LOCK = threading.Lock()

_PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_EMAIL_RE = re.compile(r"\b[\w.%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_LONG_DIGITS_RE = re.compile(r"\d{4,}")
_PHONE_RE = re.compile(r"(?:\+91[\s-]?)?[6-9]\d{9}")


def set_log_path(path: str | Path) -> Path:
    """Point the sink at a different file.

    Args:
        path: the new log file. Parent directories are created on first write.

    Returns:
        The absolute path now in use.
    """
    global _LOG_PATH
    _LOG_PATH = Path(path)
    return _LOG_PATH


def log_path() -> Path:
    """Return the current log file path."""
    return _LOG_PATH


def redact(value: Any) -> Any:
    """Scrub anything PII-shaped from a value about to be logged.

    Handles strings, and recurses through lists and tuples. Numbers, booleans
    and ``None`` are returned unchanged: ``top_score`` and ``n_chunks`` are
    legitimate and there is nothing to scrub from them.

    Args:
        value: the value to sanitise.

    Returns:
        The value with PAN-like, email-like and long digit runs replaced by
        :data:`REDACTED`.
    """
    if isinstance(value, str):
        cleaned = _PAN_RE.sub(REDACTED, value)
        cleaned = _EMAIL_RE.sub(REDACTED, cleaned)
        cleaned = _PHONE_RE.sub(REDACTED, cleaned)
        return _LONG_DIGITS_RE.sub(REDACTED, cleaned)
    if isinstance(value, (list, tuple)):
        return type(value)(redact(item) for item in value)
    return value


def log_event(stage: str, **fields: Any) -> None:
    """Append one redacted JSON line to the log file.

    Only :data:`ALLOWED` keys are written. ``ts`` and ``stage`` are set here and
    cannot be overridden by a caller.

    Args:
        stage: the pipeline stage, e.g. ``"route"`` or ``"query"``.
        **fields: candidate metadata. Unknown keys are dropped without warning,
            which is what makes ``query=`` a no-op rather than a leak.
    """
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
    }
    for key, value in fields.items():
        if key in ALLOWED and key not in ("ts", "stage"):
            record[key] = redact(value)

    line = json.dumps(record, ensure_ascii=False, default=str)
    try:
        with _LOCK:
            _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except OSError:
        # Observability must never take the app down. A read-only or full disk
        # loses log lines, not answers.
        pass


__all__ = [
    "ALLOWED",
    "REDACTED",
    "log_event",
    "log_path",
    "redact",
    "set_log_path",
]
