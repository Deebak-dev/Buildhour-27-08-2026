"""A thin, provider-agnostic LLM wrapper (C10).

Two constraints shape this module:

* **Never log the API key or the prompt body.** A prompt body contains retrieved
  scheme facts and, on the PII path, must never be persisted at all
  (architecture.md invariant I5). Logging here is limited to shapes and counts.
* **temperature 0.0 always** (``config.yaml``), so the same context produces the
  same draft and a validator failure is reproducible rather than a coin flip.

The provider dispatch is a clean ``if``/``elif`` on ``cfg.generation.provider``
so that a second provider is an addition rather than a rewrite (ADR-11, Q2).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Protocol

from rag.config import GenerationConfig

logger = logging.getLogger(__name__)

#: Per-request timeout in seconds (NFR-3 bounds end-to-end latency).
DEFAULT_TIMEOUT_S = 30


class LLMError(RuntimeError):
    """Raised when the model could not be reached or returned nothing usable."""


class LLMClient(Protocol):
    """The single method the rest of the app depends on.

    Declared as a Protocol so tests can pass a stub without patching the SDK, and
    so an alternative provider only has to satisfy this shape.
    """

    def complete(
        self, system: str, user: str, *, json_mode: bool = True
    ) -> str:
        """Return the model's raw text response."""
        ...


def _api_key(env_var: str) -> str:
    """Read an API key from the environment.

    Raises:
        LLMError: when the variable is unset or empty. The message names the
            variable but never prints its value.
    """
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise LLMError(
            f"{env_var} is not set. Export it before starting the app, e.g. "
            f'$env:{env_var}="..." (PowerShell) or `export {env_var}=...` (bash). '
            "The key is never logged."
        )
    return key


class OpenAICompatibleClient:
    """Client for any OpenAI-compatible ``/chat/completions`` endpoint.

    OpenAI, Groq, Together, and a local Ollama all speak this dialect, so one
    implementation covers the realistic near-term options.
    """

    #: Environment variable holding the key, per provider.
    KEY_ENV = {
        "openai": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
        "together": "TOGETHER_API_KEY",
        "ollama": "",  # local server needs no key
    }

    def __init__(
        self,
        cfg: GenerationConfig,
        *,
        base_url: str | None = None,
        timeout: int = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.cfg = cfg
        self.timeout = timeout
        self.provider = (cfg.provider or "openai").lower()
        self.base_url = base_url or os.environ.get("LLM_BASE_URL") or self._default_base()

    def _default_base(self) -> str:
        return {
            "openai": "https://api.openai.com/v1",
            "groq": "https://api.groq.com/openai/v1",
            "together": "https://api.together.xyz/v1",
            "ollama": "http://localhost:11434/v1",
        }.get(self.provider, "https://api.openai.com/v1")

    def complete(self, system: str, user: str, *, json_mode: bool = True) -> str:
        """Call the chat endpoint and return the assistant message text.

        Raises:
            LLMError: on a missing SDK, a missing key, a non-2xx status, or an
                empty response.
        """
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - environment problem
            raise ImportError(
                "the LLM client needs httpx. Install it with "
                "`python -m pip install httpx`."
            ) from exc

        headers = {"Content-Type": "application/json"}
        env_var = self.KEY_ENV.get(self.provider, "")
        if env_var:
            # The value goes into the header and nowhere else; no logging below
            # ever touches `headers`.
            headers["Authorization"] = f"Bearer {_api_key(env_var)}"

        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode and self.cfg.structured_output and self.provider != "ollama":
            payload["response_format"] = {"type": "json_object"}

        logger.info(
            "llm request: provider=%s model=%s json_mode=%s system_chars=%d user_chars=%d",
            self.provider,
            self.cfg.model,
            json_mode,
            len(system),
            len(user),
        )

        try:
            response = httpx.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as LLMError
            raise LLMError(f"LLM request failed: {type(exc).__name__}: {exc}") from exc

        if response.status_code >= 400:
            # The body may echo the prompt; log only the status.
            logger.error("llm http error: status=%d", response.status_code)
            raise LLMError(
                f"LLM returned HTTP {response.status_code}. Check the model name "
                f"in config.yaml and that the key for {self.provider} is valid."
            )

        try:
            data = response.json()
            text = data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001 - malformed provider response
            raise LLMError(
                f"could not parse the LLM response: {type(exc).__name__}"
            ) from exc

        if not text or not str(text).strip():
            raise LLMError("the LLM returned an empty response")
        return str(text)


def build_client(cfg: GenerationConfig) -> LLMClient:
    """Construct the client named by ``cfg.generation.provider``.

    Args:
        cfg: generation settings from ``config.yaml``.

    Returns:
        A client satisfying :class:`LLMClient`.

    Raises:
        LLMError: when the provider is not one this build knows how to drive.
    """
    provider = (cfg.provider or "").lower()
    if provider in {"", "openai", "groq", "together", "ollama"}:
        return OpenAICompatibleClient(cfg)
    raise LLMError(
        f"unknown generation provider {cfg.provider!r}. This build supports "
        "openai, groq, together and ollama. Set generation.provider in "
        "config.yaml to one of those."
    )


def complete_with_retry(
    client: LLMClient,
    system: str,
    user: str,
    *,
    cfg: GenerationConfig,
    json_mode: bool = True,
) -> str:
    """Call the client, retrying transient failures up to ``cfg.max_retries``.

    Args:
        client: the LLM client.
        system: the system prompt.
        user: the user prompt.
        cfg: generation settings; ``max_retries`` bounds the attempts.
        json_mode: request structured output.

    Returns:
        The model's text response.

    Raises:
        LLMError: when every attempt failed. The caller turns this into a
            refusal - it must never fall back to answering from model knowledge
            (invariant I6).
    """
    attempts = max(1, cfg.max_retries)
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return client.complete(system, user, json_mode=json_mode)
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised
            last = exc
            logger.warning(
                "llm attempt %d/%d failed: %s: %s",
                attempt,
                attempts,
                type(exc).__name__,
                exc,
            )
            if attempt < attempts:
                time.sleep(0)
    raise LLMError(f"LLM call failed after {attempts} attempt(s): {last}")
