"""Environment expansion and the .env loader (rag/config.py).

The rule that matters: a blank value must behave like an unset one. A ``.env``
line of ``LLM_PROVIDER=`` loads as an empty string, and if that were treated as
"set" the app would report an empty provider instead of naming the variable the
user has to fill in.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rag.config import PROJECT_ROOT, _env_or_placeholder, _expand, load_config


class TestExpansion:
    def test_unset_variable_keeps_the_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SOME_ABSENT_VAR", raising=False)
        assert _expand("${SOME_ABSENT_VAR}") == "${SOME_ABSENT_VAR}"

    def test_set_variable_is_substituted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SOME_SET_VAR", "groq")
        assert _expand("${SOME_SET_VAR}") == "groq"

    def test_blank_variable_keeps_the_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The regression: '.env with LLM_PROVIDER=' must not expand to ''."""
        monkeypatch.setenv("SOME_BLANK_VAR", "")
        assert _expand("${SOME_BLANK_VAR}") == "${SOME_BLANK_VAR}"

    def test_whitespace_only_variable_keeps_the_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SOME_PAD_VAR", "   ")
        assert _expand("${SOME_PAD_VAR}") == "${SOME_PAD_VAR}"

    def test_value_is_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SOME_PADDED_VAR", "  openai  ")
        assert _expand("${SOME_PADDED_VAR}") == "openai"

    def test_expands_inside_a_larger_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SOME_HOST", "api.example.com")
        assert _expand("https://${SOME_HOST}/v1") == "https://api.example.com/v1"

    def test_recurses_through_containers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SOME_KEY", "k")
        assert _expand({"a": ["${SOME_KEY}", 1], "b": {"c": "${SOME_KEY}"}}) == {
            "a": ["k", 1],
            "b": {"c": "k"},
        }

    def test_leaves_non_strings_alone(self) -> None:
        assert _expand(7) == 7
        assert _expand(None) is None
        assert _expand(True) is True


class TestHelper:
    def test_env_or_placeholder_reads_the_group(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import re

        monkeypatch.setenv("SOME_HELPER_VAR", "value")
        match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", "${SOME_HELPER_VAR}")
        assert match is not None
        assert _env_or_placeholder(match) == "value"


class TestDotenvLoader:
    def test_the_env_file_exists_and_is_git_ignored(self) -> None:
        env_file = PROJECT_ROOT / ".env"
        assert env_file.is_file(), ".env should exist for the user to fill in"

        ignored = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert ".env" in ignored

    def test_the_example_file_is_not_git_ignored(self) -> None:
        ignored = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert ".env.example" not in ignored

    def test_the_example_lists_the_variables_the_code_reads(self) -> None:
        """`.env.example` must not drift from rag/llm_client.py again."""
        from rag.llm_client import OpenAICompatibleClient

        example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
        assert "LLM_PROVIDER" in example
        assert "LLM_MODEL" in example
        for env_var in OpenAICompatibleClient.KEY_ENV.values():
            if env_var:
                assert env_var in example, f"{env_var} is read by the client but undocumented"

    def test_the_example_documents_the_base_url_override(self) -> None:
        assert "LLM_BASE_URL" in (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")

    def test_a_real_key_is_never_committed(self) -> None:
        """No non-empty provider key may sit in the tracked example.

        The failure message deliberately omits the value: a test that prints a
        key into CI output has leaked it into the build log.
        """
        example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
        for line in example.splitlines():
            if line.startswith(("OPENAI_API_KEY", "GROQ_API_KEY", "TOGETHER_API_KEY")):
                name = line.split("=", 1)[0]
                assert line.split("=", 1)[1].strip() == "", (
                    f"{name} must be blank in the committed .env.example "
                    "(value withheld from this message on purpose)"
                )

    def test_a_local_env_may_hold_a_real_key(self) -> None:
        """.env is git-ignored, so a developer's real key belongs there.

        The invariant is not "the key is empty" - it is ".env is never tracked".
        Asserting emptiness would forbid the very thing a developer needs to do,
        and this test is where a filled-in key leaked into a failure message
        once already.
        """
        env_file = PROJECT_ROOT / ".env"
        assert env_file.is_file()
        gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        entries = {
            line.strip().lstrip("/")
            for line in gitignore.splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        assert ".env" in entries, (
            ".env must stay git-ignored; it is the one file that may hold a "
            "real provider key"
        )

    def test_no_key_value_appears_in_any_tracked_doc(self) -> None:
        """Belt and braces: a live key must not have been pasted into the docs."""
        env_file = PROJECT_ROOT / ".env"
        keys = []
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(("OPENAI_API_KEY=", "GROQ_API_KEY=", "TOGETHER_API_KEY=")):
                value = stripped.split("=", 1)[1].strip()
                if value:
                    keys.append(value)
        if not keys:
            pytest.skip("no local key set")
        tracked = [
            PROJECT_ROOT / "docs" / name
            for name in ("README.md", "sample_qa.md", "source_list.csv", "disclaimer.md")
        ]
        for path in tracked:
            body = path.read_text(encoding="utf-8")
            for key in keys:
                assert key not in body, f"a live API key leaked into {path.name}"

    def test_missing_dotenv_package_is_survivable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "dotenv":
                raise ImportError("no dotenv here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        # Must not raise: the app stays importable with no dotenv installed.
        import rag.config as config_module

        monkeypatch.setattr(config_module, "PROJECT_ROOT", Path(os.devnull))
        config_module._load_dotenv()

    def test_missing_env_file_is_a_no_op(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import rag.config as config_module

        monkeypatch.setattr(config_module, "PROJECT_ROOT", Path(os.devnull))
        config_module._load_dotenv()  # must not raise


class TestCurrentEnvironment:
    def test_config_loads_with_a_blank_env(self) -> None:
        cfg = load_config()
        assert cfg.generation.temperature == 0.0
        assert cfg.retrieval.top_k == 10

    def test_provider_is_either_a_name_or_the_placeholder(self) -> None:
        """Never an empty string, which would be un-actionable in an error."""
        provider = load_config().generation.provider
        assert provider in {
            "openai",
            "groq",
            "together",
            "ollama",
            "${LLM_PROVIDER}",
        }


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
