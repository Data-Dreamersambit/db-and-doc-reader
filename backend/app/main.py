"""FastAPI entrypoint and core API routes for AgenticInsight.

Trade-off: Dependency-injected OllamaClient enables isolated mock transport testing.
Non-blocking health endpoint returns HTTP 200 with 'degraded' status for frontend resilience.
"""
import asyncio
import uuid
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.app.agents.router import stream_chat_events
from backend.app.config import Settings, get_settings
from backend.app.llm.client import OllamaClient
from backend.app.observability.session import SessionManager
from backend.app.observability.tracer import (
    TraceDetailResponse,
    Tracer,
    get_global_tracer,
)
from backend.app.rag.chunker import extract_and_chunk_pdf
from backend.app.rag.store import DocumentInfo, SQLiteDocStore
from backend.app.tools.datastore import (
    DatasetListResponse,
    DatasetSchema,
    Datastore,
)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AgenticInsight API",
    description="Local-first assistant over structured DuckDB data and PDF documents",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class OllamaHealth(BaseModel):
    """Health details for the configured Ollama backend."""

    reachable: bool = Field(..., description="Whether Ollama daemon is reachable")
    url: str = Field(..., description="Target Ollama base URL")
    chat_model: str = Field(..., description="Configured chat model name")
    embed_model: str = Field(..., description="Configured embedding model name")


class HealthResponse(BaseModel):
    """System health response payload."""

    status: str = Field(
        ..., description="Overall service health status ('ok' or 'degraded')"
    )
    ollama: OllamaHealth


class DocumentListResponse(BaseModel):
    """Response containing all registered documents."""

    documents: list[DocumentInfo] = Field(default_factory=list)


class DeleteDocumentResponse(BaseModel):
    """Response payload for document deletion."""

    status: str
    id: str


class ChatRequest(BaseModel):
    """Chat invocation payload with session ID."""

    query: str = Field(..., description="User prompt or analytical inquiry")
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Session identifier for multi-turn conversational history",
    )


def get_ollama_client() -> OllamaClient:
    """FastAPI dependency yielding an OllamaClient instance."""
    return OllamaClient()


def get_datastore() -> Datastore:
    """FastAPI dependency yielding a Datastore instance."""
    return Datastore()


def get_doc_store() -> SQLiteDocStore:
    """FastAPI dependency yielding an SQLiteDocStore instance."""
    return SQLiteDocStore()


def get_session_manager() -> SessionManager:
    """FastAPI dependency yielding an SQLite SessionManager instance."""
    return SessionManager()


def get_tracer() -> Tracer:
    """FastAPI dependency yielding a Tracer instance."""
    return get_global_tracer()


@app.get("/api/health", response_model=HealthResponse)
async def health(
    settings: Settings = Depends(get_settings),
    client: OllamaClient = Depends(get_ollama_client),
) -> HealthResponse:
    """Check API and Ollama daemon reachability."""
    is_reachable = await client.check_health()
    return HealthResponse(
        status="ok" if is_reachable else "degraded",
        ollama=OllamaHealth(
            reachable=is_reachable,
            url=settings.OLLAMA_URL,
            chat_model=settings.CHAT_MODEL,
            embed_model=settings.EMBED_MODEL,
        ),
    )


# --- Datasets Endpoints ---


@app.post("/api/datasets", response_model=DatasetSchema)
async def upload_dataset(
    file: UploadFile = File(...),
    datastore: Datastore = Depends(get_datastore),
) -> DatasetSchema:
    """Upload CSV dataset, validate, ingest into DuckDB, and return inferred schema."""
    content = await file.read()
    filename = file.filename or "dataset.csv"
    try:
        return datastore.load_csv(content, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to ingest dataset: {e}"
        )


@app.get("/api/datasets", response_model=DatasetListResponse)
def list_datasets(
    datastore: Datastore = Depends(get_datastore),
) -> DatasetListResponse:
    """List all available datasets in DuckDB."""
    return DatasetListResponse(datasets=datastore.list_datasets())


@app.get("/api/datasets/{name}/schema", response_model=DatasetSchema)
def get_dataset_schema(
    name: str,
    datastore: Datastore = Depends(get_datastore),
) -> DatasetSchema:
    """Retrieve schema, sample rows, and low-cardinality values for a dataset."""
    try:
        return datastore.get_schema(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found.")


# --- Documents Endpoints ---


@app.post("/api/documents", response_model=DocumentInfo)
async def upload_document(
    file: UploadFile = File(...),
    doc_store: SQLiteDocStore = Depends(get_doc_store),
    client: OllamaClient = Depends(get_ollama_client),
) -> DocumentInfo:
    """Upload PDF/document, chunk, compute embeddings in batches, and persist to SQLite."""
    content = await file.read()
    filename = file.filename or "document.pdf"
    doc_id = str(uuid.uuid4())

    page_count, chunks = extract_and_chunk_pdf(content, filename, doc_id=doc_id)

    # Batch embedding (size 8) to preserve M1 memory limits
    all_embeddings: list[list[float]] = []
    batch_size = 8
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        res = await client.embed([c.text for c in batch])
        all_embeddings.extend(res.embeddings)

    return doc_store.save_document(
        doc_id=doc_id,
        filename=filename,
        page_count=page_count,
        chunks=chunks,
        embeddings=all_embeddings,
    )


@app.get("/api/documents", response_model=DocumentListResponse)
def list_documents(
    doc_store: SQLiteDocStore = Depends(get_doc_store),
) -> DocumentListResponse:
    """List all indexed documents and chunk counts."""
    return DocumentListResponse(documents=doc_store.list_documents())


@app.delete("/api/documents/{id}", response_model=DeleteDocumentResponse)
def delete_document(
    id: str,
    doc_store: SQLiteDocStore = Depends(get_doc_store),
) -> DeleteDocumentResponse:
    """Delete a document and cascade remove all associated chunk vectors."""
    deleted = doc_store.delete_document(id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{id}' not found.")
    return DeleteDocumentResponse(status="deleted", id=id)


# --- Traces Endpoints ---


@app.get("/api/traces/{trace_id}", response_model=TraceDetailResponse)
def get_trace_details(
    trace_id: str,
    tracer: Tracer = Depends(get_tracer),
) -> TraceDetailResponse:
    """Retrieve audit trace steps and latency metrics for an execution."""
    steps = tracer.get_trace(trace_id)
    if not steps:
        raise HTTPException(status_code=404, detail=f"Trace '{trace_id}' not found.")

    total_latency = sum(s.latency_ms for s in steps)
    return TraceDetailResponse(
        trace_id=trace_id,
        step_count=len(steps),
        total_latency_ms=total_latency,
        steps=steps,
    )


# --- Chat SSE Endpoint ---


@app.post("/api/chat")
async def chat_sse(
    request: Request,
    body: ChatRequest,
    client: OllamaClient = Depends(get_ollama_client),
    datastore: Datastore = Depends(get_datastore),
    doc_store: SQLiteDocStore = Depends(get_doc_store),
    session_mgr: SessionManager = Depends(get_session_manager),
) -> StreamingResponse:
    """Stream routed agent reasoning, SQL, data rows, and answers via Server-Sent Events."""

    async def event_generator():
        try:
            generator = stream_chat_events(
                query=body.query,
                session_id=body.session_id,
                client=client,
                datastore=datastore,
                doc_store=doc_store,
                session_mgr=session_mgr,
            )
            async for sse_chunk in generator:
                if await request.is_disconnected():
                    break
                yield sse_chunk
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
