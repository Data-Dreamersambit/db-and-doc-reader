"""Unit tests for OllamaClient using mocked httpx transport.

Trade-off: MockTransport avoids spinning up an Ollama daemon in CI/test suites.
Verifies strict JSON contract matching Ollama's HTTP specification.
"""
import json
import httpx
import pytest

from backend.app.llm.client import (
    ChatMessage,
    ChatResponse,
    EmbeddingResponse,
    OllamaClient,
)


@pytest.mark.asyncio
async def test_ollama_client_chat_success() -> None:
    """Test standard non-streaming chat completion."""
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        payload = json.loads(request.content)
        assert payload["model"] == "qwen2.5:3b"
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]
        assert payload["stream"] is False
        assert "format" not in payload

        return httpx.Response(
            status_code=200,
            json={
                "model": "qwen2.5:3b",
                "message": {"role": "assistant", "content": "Hello there!"},
                "done": True,
                "total_duration": 1500000,
            },
        )

    transport = httpx.MockTransport(handler)
    client = OllamaClient(
        base_url="http://localhost:11434",
        chat_model="qwen2.5:3b",
        transport=transport,
    )

    response = await client.chat([ChatMessage(role="user", content="Hello")])
    assert isinstance(response, ChatResponse)
    assert response.content == "Hello there!"
    assert response.model == "qwen2.5:3b"
    assert response.done is True
    assert response.total_duration == 1500000
    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_client_chat_json_mode() -> None:
    """Test chat with json_mode enabled."""
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload.get("format") == "json"
        return httpx.Response(
            status_code=200,
            json={
                "model": "qwen2.5:3b",
                "message": {"role": "assistant", "content": '{"action": "sql"}'},
                "done": True,
            },
        )

    transport = httpx.MockTransport(handler)
    client = OllamaClient(transport=transport)
    response = await client.chat(
        [{"role": "user", "content": "Return json"}],
        json_mode=True,
    )
    assert response.content == '{"action": "sql"}'
    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_client_stream() -> None:
    """Test streaming response chunks from Ollama API."""
    lines = [
        json.dumps({"message": {"content": "Hello "}, "done": False}) + "\n",
        json.dumps({"message": {"content": "World!"}, "done": True}) + "\n",
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        payload = json.loads(request.content)
        assert payload["stream"] is True
        return httpx.Response(
            status_code=200,
            content="".join(lines).encode("utf-8"),
        )

    transport = httpx.MockTransport(handler)
    client = OllamaClient(transport=transport)

    tokens: list[str] = []
    async for token in client.stream([ChatMessage(role="user", content="Stream hi")]):
        tokens.append(token)

    assert tokens == ["Hello ", "World!"]
    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_client_embed_batch() -> None:
    """Test batch embedding generation via /api/embed."""
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        payload = json.loads(request.content)
        assert payload["model"] == "nomic-embed-text"
        assert payload["input"] == ["doc1", "doc2"]
        return httpx.Response(
            status_code=200,
            json={
                "model": "nomic-embed-text",
                "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            },
        )

    transport = httpx.MockTransport(handler)
    client = OllamaClient(transport=transport)

    result = await client.embed(["doc1", "doc2"])
    assert isinstance(result, EmbeddingResponse)
    assert len(result.embeddings) == 2
    assert result.embeddings[0] == [0.1, 0.2, 0.3]
    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_client_embed_fallback_404() -> None:
    """Test fallback to legacy /api/embeddings when /api/embed returns 404."""
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/embed":
            return httpx.Response(status_code=404)
        if request.url.path == "/api/embeddings":
            payload = json.loads(request.content)
            return httpx.Response(
                status_code=200,
                json={"embedding": [0.7, 0.8, 0.9]},
            )
        return httpx.Response(status_code=400)

    transport = httpx.MockTransport(handler)
    client = OllamaClient(transport=transport)

    result = await client.embed(["single query"])
    assert len(result.embeddings) == 1
    assert result.embeddings[0] == [0.7, 0.8, 0.9]
    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_check_health() -> None:
    """Test health check under reachable and unreachable conditions."""
    async def ok_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, json={"version": "0.3.12"})

    ok_client = OllamaClient(transport=httpx.MockTransport(ok_handler))
    assert await ok_client.check_health() is True
    await ok_client.aclose()

    async def fail_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=500)

    fail_client = OllamaClient(transport=httpx.MockTransport(fail_handler))
    assert await fail_client.check_health() is False
    await fail_client.aclose()
