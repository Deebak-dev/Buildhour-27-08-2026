"""Typed access to ``config.yaml`` — the single source of truth for settings.

Every stage reads its thresholds from here so no magic number is hardcoded in
logic (docs/architecture.md §8, implementation.md Appendix B).

Values of the form ``${ENV_VAR}`` are expanded from the environment at load time.
When the variable is unset the literal placeholder is returned, so the pipeline
stays importable in a bare environment. A ``.env`` file in the project root is
loaded automatically at import (see :func:`_load_dotenv`).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field as _field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG: Path = PROJECT_ROOT / "config.yaml"

#: ``${NAME}`` placeholders are expanded from the environment.
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _load_dotenv() -> None:
    """Load ``.env`` into ``os.environ`` so ``${...}`` placeholders resolve.

    ``override=False`` keeps a real shell variable ahead of the file, so
    ``$env:LLM_MODEL=...`` in a terminal still wins over ``.env``. Absence of
    either the file or the package is not an error: the app must stay importable
    with no dotenv and no secrets at all.
    """
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is in requirements.txt
        return
    load_dotenv(env_path, override=False)


# Runs at import so every entry point - config loading, the LLM client, the eval
# harness - sees the same environment without repeating the call.
_load_dotenv()

#: A page yielding fewer than this many characters of text is treated as thin.
#: Thin pages usually mean JS-rendered content (docs/PRD.md risk R2).
THIN_CHARS: int = 200


def _expand(value: Any) -> Any:
    """Recursively expand ``${ENV_VAR}`` placeholders in strings.

    An unset variable and a variable that is set but blank are treated alike,
    because a ``.env`` line like ``LLM_PROVIDER=`` loads as an empty string.
    Preserving the literal placeholder in that case is what keeps the error
    message actionable: it names the variable to fill in rather than reporting
    an empty provider.
    """
    if isinstance(value, str):
        return _ENV_PATTERN.sub(_env_or_placeholder, value)
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def _env_or_placeholder(match: re.Match[str]) -> str:
    """Return the environment value for a placeholder, or the placeholder."""
    resolved = os.environ.get(match.group(1), "").strip()
    return resolved if resolved else match.group(0)


def _section(raw: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Return one config section as a plain dict, or an empty dict if absent."""
    value = raw.get(name) or {}
    return dict(value) if isinstance(value, Mapping) else {}


# ---------------------------------------------------------------------------
# Per-stage config objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusConfig:
    """Corpus locations and crawl settings."""

    manifest: str = "ingest/sources.csv"
    raw_dir: str = "data/raw"
    clean_dir: str = "data/clean"
    chunks_dir: str = "data/chunks"
    manifest_json: str = "data/manifest.json"
    source_list: str = "docs/source_list.csv"
    timeout: int = 30
    max_attempts: int = 2
    crawl_delay: float = 1.0
    thin_chars: int = THIN_CHARS


@dataclass(frozen=True)
class ChunkConfig:
    """Chunking parameters (docs/architecture.md §5.4)."""

    strategy: str = "auto"
    target_tokens: int = 600
    min_tokens: int = 200
    max_tokens: int = 800
    overlap_pct: int = 12


@dataclass(frozen=True)
class EmbedConfig:
    """Embedding model settings."""

    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    normalize: bool = True
    batch_size: int = 32


@dataclass(frozen=True)
class StoreConfig:
    """Vector store settings."""

    backend: str = "chromadb"
    persist_dir: str = "data/chroma"
    collection: str = "mf_faq_chroma"
    rebuild_on_dimension_change: bool = True


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval thresholds (tuned in Phase 5, then frozen).

    These defaults mirror the frozen operating point in ``config.yaml`` and
    ``docs/chunking_rationale.md``: recall@5 = 6/10 on ``eval/golden_set.jsonl``,
    which is the ceiling for the current corpus because four of the ten golden
    queries ask for facts that were never ingested. ``load_config`` is the real
    source of truth; these values are the fallback for a caller that constructs
    the config directly.
    """

    mode: str = "vector"
    #: The LLM context window: this many chunks are retrieved *and* rendered
    #: into the prompt by :func:`rag.answer.build_context`. There is no second
    #: knob, so raising it widens the model's view of the corpus. Measured
    #: effect is in ``docs/README.md``; 10 beats 5 mainly by letting the model
    #: see enough of a noisy chunk neighbourhood to notice when a field is
    #: genuinely absent, rather than citing a neighbouring chunk instead.
    top_k: int = 10
    over_fetch: int = 10
    similarity_floor: float = 0.35
    field_type_boost: float = 0.35
    #: Subtracted from a chunk whose ``field_type`` is about *other schemes*
    #: (:data:`rag.retrieve.OFF_TOPIC_FIELD_TYPES`) when the query's detected
    #: intent is some other field. See that module for the derivation; the
    #: value sits in the middle of a flat 0.05-0.40 plateau, so it is not
    #: balanced on a knife edge. Set to 0.0 to disable the behaviour entirely.
    field_misfit_penalty: float = 0.15
    scheme_filter: str = "strict"


@dataclass(frozen=True)
class GenerationConfig:
    """LLM settings."""

    provider: str = ""
    model: str = ""
    temperature: float = 0.0
    max_sentences: int = 3
    max_retries: int = 2
    structured_output: bool = True


@dataclass(frozen=True)
class GuardrailConfig:
    """Guardrail switches."""

    pii_screen: bool = True
    allowlist_from_manifest: bool = True
    log_raw_queries: bool = False


@dataclass(frozen=True)
class AnswerConfig:
    """The §7 response contract."""

    freshness_label: str = "Last updated from sources:"
    date_format: str = "%d %b %Y"
    education_url: str = "https://www.sebi.gov.in/investor-education"


@dataclass(frozen=True)
class AppConfig:
    """The whole configuration tree, with resolved project paths."""

    root: Path
    corpus: CorpusConfig = _field(default_factory=CorpusConfig)
    chunking: ChunkConfig = _field(default_factory=ChunkConfig)
    embedding: EmbedConfig = _field(default_factory=EmbedConfig)
    vectorstore: StoreConfig = _field(default_factory=StoreConfig)
    retrieval: RetrievalConfig = _field(default_factory=RetrievalConfig)
    generation: GenerationConfig = _field(default_factory=GenerationConfig)
    guardrails: GuardrailConfig = _field(default_factory=GuardrailConfig)
    answer: AnswerConfig = _field(default_factory=AnswerConfig)

    def path(self, relative: str | Path) -> Path:
        """Resolve a project-relative path against the repository root."""
        candidate = Path(relative)
        return candidate if candidate.is_absolute() else self.root / candidate

    @property
    def manifest_path(self) -> Path:
        """Absolute path to ``ingest/sources.csv``."""
        return self.path(self.corpus.manifest)

    @property
    def raw_dir(self) -> Path:
        """Absolute path to the raw page snapshot directory."""
        return self.path(self.corpus.raw_dir)

    @property
    def clean_dir(self) -> Path:
        """Absolute path to the cleaned-text directory."""
        return self.path(self.corpus.clean_dir)

    @property
    def chunks_dir(self) -> Path:
        """Absolute path to the chunk JSONL directory."""
        return self.path(self.corpus.chunks_dir)

    @property
    def manifest_json(self) -> Path:
        """Absolute path to ``data/manifest.json``."""
        return self.path(self.corpus.manifest_json)

    @property
    def source_list_path(self) -> Path:
        """Absolute path to the D2 source list."""
        return self.path(self.corpus.source_list)

    @property
    def chunks_jsonl(self) -> Path:
        """Absolute path to ``data/chunks/chunks.jsonl``."""
        return self.chunks_dir / "chunks.jsonl"


def _build(cls: type, values: Mapping[str, Any]):
    """Instantiate a config dataclass, ignoring unknown keys."""
    known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in values.items() if k in known})


@lru_cache(maxsize=4)
def _load_cached(path: str, mtime: float) -> AppConfig:
    """Load and cache the config. ``mtime`` participates so edits are picked up."""
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw = _expand(raw)
    root = config_path.resolve().parent

    return AppConfig(
        root=root,
        corpus=_build(CorpusConfig, _section(raw, "corpus")),
        chunking=_build(ChunkConfig, _section(raw, "chunking")),
        embedding=_build(EmbedConfig, _section(raw, "embedding")),
        vectorstore=_build(StoreConfig, _section(raw, "vectorstore")),
        retrieval=_build(RetrievalConfig, _section(raw, "retrieval")),
        generation=_build(GenerationConfig, _section(raw, "generation")),
        guardrails=_build(GuardrailConfig, _section(raw, "guardrails")),
        answer=_build(AnswerConfig, _section(raw, "answer")),
    )


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load the project configuration.

    Args:
        path: config file path. Defaults to ``config.yaml`` at the project root.

    Returns:
        The populated :class:`AppConfig`.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    mtime = config_path.stat().st_mtime if config_path.is_file() else 0.0
    return _load_cached(str(config_path.resolve()), mtime)
