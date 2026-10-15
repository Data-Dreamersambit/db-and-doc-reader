"""Ollama API async HTTP client.

Trade-off: Direct httpx client avoids heavy framework overhead on 8GB M1 Mac.
Streaming yields raw token deltas directly to minimize memory footprint.
"""
from collections.abc import AsyncIterator
import json
from typing import Any
import httpx
from pydantic import BaseModel, Field

from backend.app.config import get_settings


class ChatMessage(BaseModel):
    """Payload representing a single chat conversation message."""

    role: str = Field(..., description="Message role (e.g., system, user, assistant)")
    content: str = Field(..., description="Message text content")


class ChatResponse(BaseModel):
    """Standardized response payload from an Ollama chat completion."""

    content: str = Field(..., description="Assistant text response")
    model: str = Field(..., description="Model identifier used for generation")
    done: bool = Field(default=True, description="Completion status flag")
    total_duration: int | None = Field(
        default=None, description="Generation duration in nanoseconds"
    )


class EmbeddingResponse(BaseModel):
    """Response payload containing generated text embeddings."""

    embeddings: list[list[float]] = Field(
        ..., description="List of embedding vectors corresponding to input texts"
    )
    model: str = Field(..., description="Model identifier used for embeddings")


class OllamaClient:
    """Async client for interacting with Ollama's HTTP API."""

    def __init__(
        self,
        base_url: str | None = None,
        chat_model: str | None = None,
        embed_model: str | None = None,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.OLLAMA_URL).rstrip("/")
        self.chat_model = chat_model or settings.CHAT_MODEL
        self.embed_model = embed_model or settings.EMBED_MODEL
        self.timeout = timeout
        self._owns_client = client is None

        if client is not None:
            self._client = client
        else:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=transport,
            )

    async def chat(
        self,
        messages: list[ChatMessage | dict[str, str]],
        json_mode: bool = False,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Send chat messages and return the full assistant response."""
        formatted_messages = [
            m.model_dump() if isinstance(m, ChatMessage) else m for m in messages
        ]
        payload: dict[str, Any] = {
            "model": model or self.chat_model,
            "messages": formatted_messages,
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"
        if options:
            payload["options"] = options

        response = await self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

        return ChatResponse(
            content=data.get("message", {}).get("content", ""),
            model=data.get("model", model or self.chat_model),
            done=data.get("done", True),
            total_duration=data.get("total_duration"),
        )

    async def stream(
        self,
        messages: list[ChatMessage | dict[str, str]],
        model: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream assistant response tokens asynchronously as they arrive."""
        formatted_messages = [
            m.model_dump() if isinstance(m, ChatMessage) else m for m in messages
        ]
        payload: dict[str, Any] = {
            "model": model or self.chat_model,
            "messages": formatted_messages,
            "stream": True,
        }

        async with self._client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content

    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> EmbeddingResponse:
        """Generate dense embeddings for a batch of text strings."""
        target_model = model or self.embed_model
        payload = {
            "model": target_model,
            "input": texts,
        }

        response = await self._client.post("/api/embed", json=payload)

        # Fallback for older Ollama daemon versions exposing /api/embeddings
        if response.status_code == 404:
            fallback_embeddings: list[list[float]] = []
            for text in texts:
                single_res = await self._client.post(
                    "/api/embeddings",
                    json={"model": target_model, "prompt": text},
                )
                single_res.raise_for_status()
                fallback_embeddings.append(single_res.json().get("embedding", []))
            return EmbeddingResponse(
                embeddings=fallback_embeddings,
                model=target_model,
            )

        response.raise_for_status()
        data = response.json()
        return EmbeddingResponse(
            embeddings=data.get("embeddings", []),
            model=data.get("model", target_model),
        )

    async def check_health(self) -> bool:
        """Verify reachability of Ollama HTTP daemon."""
        try:
            response = await self._client.get("/api/version")
            return response.status_code == 200
        except (httpx.RequestError, httpx.HTTPStatusError):
            return False

    async def aclose(self) -> None:
        """Close underlying httpx AsyncClient if created internally."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "OllamaClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()
