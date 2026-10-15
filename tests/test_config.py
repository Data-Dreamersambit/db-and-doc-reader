"""Tests for application configuration.

Trade-off: Isolated env monkeypatching verifies config overrides without modifying disk files.
Tests default fallbacks against spec constants.
"""
import pytest
from backend.app.config import Settings


def test_default_settings() -> None:
    """Verify default settings adhere to local-first M1 requirements."""
    settings = Settings()
    assert settings.CHAT_MODEL == "qwen2.5:3b"
    assert settings.EMBED_MODEL == "nomic-embed-text"
    assert settings.OLLAMA_URL == "http://localhost:11434"
    assert settings.DUCKDB_PATH == "data/agentic_insight.duckdb"
    assert settings.SQLITE_PATH == "data/agentic_insight.sqlite"


def test_custom_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify custom environment variables override defaults correctly."""
    monkeypatch.setenv("CHAT_MODEL", "custom-model:latest")
    monkeypatch.setenv("EMBED_MODEL", "custom-embed:latest")
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:11435")

    settings = Settings()
    assert settings.CHAT_MODEL == "custom-model:latest"
    assert settings.EMBED_MODEL == "custom-embed:latest"
    assert settings.OLLAMA_URL == "http://127.0.0.1:11435"
