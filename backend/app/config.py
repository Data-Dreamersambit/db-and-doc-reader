"""Application configuration using pydantic-settings.

Trade-off: Centralized pydantic-settings validates env at boot; prevents runtime config drift.
Avoids hardcoding model names across agent modules.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration sourced exclusively from environment variables."""

    CHAT_MODEL: str = "qwen2.5:3b"
    EMBED_MODEL: str = "nomic-embed-text"
    OLLAMA_URL: str = "http://localhost:11434"
    DUCKDB_PATH: str = "data/agentic_insight.duckdb"
    SQLITE_PATH: str = "data/agentic_insight.sqlite"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of Settings."""
    return Settings()
