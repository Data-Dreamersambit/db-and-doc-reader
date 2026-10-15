"""Unit and integration tests for document RAG, DocAgent refusal, and REST endpoints.

Trade-off: MockTransport validates cosine similarity math and RRF fusion without network dependencies.
Temporary SQLite files verify atomic cascade deletions between test runs.
"""
from pathlib import Path
import json
import httpx
import pytest

from backend.app.agents.doc_agent import (
    REFUSAL_MESSAGE,
    extract_citations,
    run_doc_agent,
)
from backend.app.llm.client import OllamaClient
from backend.app.main import app, get_doc_store, get_ollama_client
from backend.app.rag.chunker import DocumentChunk, extract_and_chunk_pdf
from backend.app.rag.store import SQLiteDocStore


def test_chunker_provenance_metadata() -> None:
    """Verify recursive chunking preserves doc, page, and chunk_id metadata."""
    sample_text = (
        "Project Antigravity is a local-first assistant. "
        "It supports DuckDB for tabular data and SQLite for vectors. "
    ) * 30  # ~360 words

    _, chunks = extract_and_chunk_pdf(
        file_bytes=sample_text.encode("utf-8"),
        filename="overview.pdf",
        doc_id="doc_123",
        target_tokens=100,
        overlap_tokens=20,
    )

    assert len(chunks) >= 3
    for chunk in chunks:
        assert chunk.doc_name == "overview.pdf"
        assert chunk.doc_id == "doc_123"
        assert chunk.page_number == 1
        assert chunk.chunk_id.startswith("doc_123_p1_c")


def test_sqlite_doc_store_crud(tmp_path: Path) -> None:
    """Test SQLite document persistence, float32 BLOB vectors, and cascade deletion."""
    db_file = tmp_path / "test_store.sqlite"
    store = SQLiteDocStore(db_path=db_file)

    chunk1 = DocumentChunk(
        chunk_id="doc1_p1_c0",
        doc_id="doc1",
        doc_name="report.pdf",
        page_number=1,
        text="Alpha fact.",
        token_count=2,
    )
    chunk2 = DocumentChunk(
        chunk_id="doc1_p2_c0",
        doc_id="doc1",
        doc_name="report.pdf",
        page_number=2,
        text="Beta fact.",
        token_count=2,
    )

    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    doc_info = store.save_document(
        doc_id="doc1",
        filename="report.pdf",
        page_count=2,
        chunks=[chunk1, chunk2],
        embeddings=embeddings,
    )

    assert doc_info.id == "doc1"
    assert doc_info.chunk_count == 2

    # Verify retrieval
    stored_chunks = store.get_all_chunks()
    assert len(stored_chunks) == 2
    assert stored_chunks[0].embedding.shape == (3,)

    # Verify delete cascade
    assert store.delete_document("doc1") is True
    assert len(store.get_all_chunks()) == 0
    assert store.get_document("doc1") is None


def test_extract_citations() -> None:
    """Test regex extraction of [doc p.N] citation patterns."""
    text = "The system runs locally [architecture.pdf p.3] and supports DuckDB [specs.pdf p.12]."
    citations = extract_citations(text)
    assert citations == ["[architecture.pdf p.3]", "[specs.pdf p.12]"]


@pytest.mark.asyncio
async def test_doc_agent_retrieval_known_fact(tmp_path: Path) -> None:
    """Test DocAgent retrieves a known fact and outputs citation."""
    store = SQLiteDocStore(db_path=tmp_path / "fact.sqlite")
    known_chunk = DocumentChunk(
        chunk_id="d1_p1_c0",
        doc_id="d1",
        doc_name="apollo.pdf",
        page_number=1,
        text="Project Apollo 11 landed humans on the Moon on July 20, 1969.",
        token_count=13,
    )
    store.save_document(
        doc_id="d1",
        filename="apollo.pdf",
        page_count=1,
        chunks=[known_chunk],
        embeddings=[[0.9, 0.1, 0.0]],
    )

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        url_path = request.url.path
        if url_path == "/api/embed":
            # High cosine similarity with stored embedding
            return httpx.Response(
                status_code=200,
                json={"embeddings": [[0.95, 0.05, 0.0]], "model": "nomic-embed-text"},
            )
        # LLM chat answer
        return httpx.Response(
            status_code=200,
            json={
                "model": "qwen2.5:3b",
                "message": {
                    "role": "assistant",
                    "content": "Apollo 11 landed on the Moon on July 20, 1969 [apollo.pdf p.1].",
                },
                "done": True,
            },
        )

    client = OllamaClient(transport=httpx.MockTransport(mock_handler))

    result = await run_doc_agent(
        query="When did Apollo 11 land on the Moon?",
        store=store,
        client=client,
        threshold=0.5,
    )

    assert result.refused is False
    assert "July 20, 1969" in result.answer
    assert "[apollo.pdf p.1]" in result.citations
    assert len(result.chunks) == 1
    assert result.best_score > 0.9


@pytest.mark.asyncio
async def test_doc_agent_refusal_path(tmp_path: Path) -> None:
    """Test exact refusal when best score is below configurable threshold."""
    store = SQLiteDocStore(db_path=tmp_path / "refuse.sqlite")
    cooking_chunk = DocumentChunk(
        chunk_id="d2_p1_c0",
        doc_id="d2",
        doc_name="cookbook.pdf",
        page_number=1,
        text="Bake the chocolate cake at 350 degrees Fahrenheit for 25 minutes.",
        token_count=12,
    )
    store.save_document(
        doc_id="d2",
        filename="cookbook.pdf",
        page_count=1,
        chunks=[cooking_chunk],
        embeddings=[[1.0, 0.0, 0.0]],
    )

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        # Orthogonal embedding -> 0.0 cosine similarity
        return httpx.Response(
            status_code=200,
            json={"embeddings": [[0.0, 1.0, 0.0]], "model": "nomic-embed-text"},
        )

    client = OllamaClient(transport=httpx.MockTransport(mock_handler))

    # Query about quantum physics has 0.0 similarity to cake recipe
    result = await run_doc_agent(
        query="What is quantum tunneling?",
        store=store,
        client=client,
        threshold=0.40,
    )

    assert result.refused is True
    assert result.answer == REFUSAL_MESSAGE
    assert result.citations == []


@pytest.mark.asyncio
async def test_document_endpoints(tmp_path: Path) -> None:
    """Test POST /api/documents, GET /api/documents, and DELETE /api/documents/{id}."""
    test_store = SQLiteDocStore(db_path=tmp_path / "api_docs.sqlite")

    async def mock_embed_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        texts = body.get("input", [])
        return httpx.Response(
            status_code=200,
            json={
                "model": "nomic-embed-text",
                "embeddings": [[0.1, 0.2, 0.3] for _ in texts],
            },
        )

    mock_client = OllamaClient(transport=httpx.MockTransport(mock_embed_handler))

    app.dependency_overrides[get_doc_store] = lambda: test_store
    app.dependency_overrides[get_ollama_client] = lambda: mock_client

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            # 1. Upload document
            doc_content = b"Local-first document processing with vector search."
            res = await client.post(
                "/api/documents",
                files={"file": ("guide.pdf", doc_content, "application/pdf")},
            )
            assert res.status_code == 200
            data = res.json()
            assert data["filename"] == "guide.pdf"
            assert data["chunk_count"] >= 1
            doc_id = data["id"]

            # 2. List documents
            list_res = await client.get("/api/documents")
            assert list_res.status_code == 200
            docs = list_res.json()["documents"]
            assert len(docs) == 1
            assert docs[0]["id"] == doc_id

            # 3. Delete document
            del_res = await client.delete(f"/api/documents/{doc_id}")
            assert del_res.status_code == 200
            assert del_res.json()["status"] == "deleted"

            # 4. Verify deleted
            empty_list = await client.get("/api/documents")
            assert len(empty_list.json()["documents"]) == 0

            # 5. Delete 404
            missing_del = await client.delete(f"/api/documents/{doc_id}")
            assert missing_del.status_code == 404
    finally:
        app.dependency_overrides.clear()
