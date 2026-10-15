"""Tests for the /api/health endpoint.

Trade-off: FastAPI dependency overrides isolate endpoint logic from actual network sockets.
Validates degraded state serialization without service interruption.
"""
import httpx
import pytest

from backend.app.config import Settings, get_settings
from backend.app.llm.client import OllamaClient
from backend.app.main import app, get_ollama_client


@pytest.mark.asyncio
async def test_health_endpoint_healthy() -> None:
    """Test /api/health when Ollama daemon responds successfully."""
    async def mock_ollama_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, json={"version": "0.3.12"})

    mock_client = OllamaClient(
        base_url="http://localhost:11434",
        chat_model="qwen2.5:3b",
        embed_model="nomic-embed-text",
        transport=httpx.MockTransport(mock_ollama_handler),
    )

    app.dependency_overrides[get_ollama_client] = lambda: mock_client
    app.dependency_overrides[get_settings] = lambda: Settings()

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/health")

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["ollama"]["reachable"] is True
        assert payload["ollama"]["url"] == "http://localhost:11434"
        assert payload["ollama"]["chat_model"] == "qwen2.5:3b"
        assert payload["ollama"]["embed_model"] == "nomic-embed-text"
    finally:
        app.dependency_overrides.clear()
        await mock_client.aclose()


@pytest.mark.asyncio
async def test_health_endpoint_degraded() -> None:
    """Test /api/health when Ollama daemon is unreachable."""
    async def failing_ollama_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    mock_client = OllamaClient(
        base_url="http://localhost:11434",
        chat_model="qwen2.5:3b",
        embed_model="nomic-embed-text",
        transport=httpx.MockTransport(failing_ollama_handler),
    )

    app.dependency_overrides[get_ollama_client] = lambda: mock_client
    app.dependency_overrides[get_settings] = lambda: Settings()

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/health")

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "degraded"
        assert payload["ollama"]["reachable"] is False
    finally:
        app.dependency_overrides.clear()
        await mock_client.aclose()
