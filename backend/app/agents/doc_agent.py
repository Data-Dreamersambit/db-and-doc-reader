"""Document RAG Agent for hybrid retrieval, citation synthesis, and strict refusal thresholding.

Trade-off: Hard similarity threshold cutoff prevents LLM hallucinations on out-of-domain questions.
Explicit [doc p.N] format enforces auditability for local enterprise documents.
"""
import re
from pydantic import BaseModel, Field

from backend.app.llm.client import ChatMessage, OllamaClient
from backend.app.rag.retriever import ChunkMatch, hybrid_retrieve
from backend.app.rag.sanitizer import sanitize_document_text
from backend.app.rag.store import SQLiteDocStore

DEFAULT_SIMILARITY_THRESHOLD: float = 0.45
REFUSAL_MESSAGE: str = "Not found in the documents"


class DocAgentResult(BaseModel):
    """Execution result returned by the Document RAG Agent."""

    answer: str = Field(..., description="Answer text with [doc p.N] citations or exact refusal")
    citations: list[str] = Field(default_factory=list, description="List of cited [doc p.N] references")
    chunks: list[ChunkMatch] = Field(default_factory=list, description="Top retrieved context chunks")
    best_score: float = Field(default=0.0, description="Highest dense similarity score among chunks")
    refused: bool = Field(default=False, description="True if query refused due to low relevance")


def extract_citations(text: str) -> list[str]:
    """Extract all [doc p.N] citations from generated text."""
    pattern = re.compile(r"\[[a-zA-Z0-9_\-\. ]+ p\.\d+\]")
    return list(dict.fromkeys(pattern.findall(text)))


async def run_doc_agent(
    query: str,
    store: SQLiteDocStore | None = None,
    client: OllamaClient | None = None,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> DocAgentResult:
    """Execute Document RAG workflow: hybrid retrieval -> threshold check -> cited answer / refusal."""
    doc_store = store or SQLiteDocStore()
    llm = client or OllamaClient()

    # Step 1: Hybrid Retrieval (Dense top-20 + BM25 top-20 -> RRF top-5)
    retrieved_chunks = await hybrid_retrieve(
        query=query,
        store=doc_store,
        client=llm,
        top_k_dense=20,
        top_k_bm25=20,
        top_k_fused=5,
    )

    if not retrieved_chunks:
        return DocAgentResult(
            answer=REFUSAL_MESSAGE,
            citations=[],
            chunks=[],
            best_score=0.0,
            refused=True,
        )

    best_score = max(c.cosine_score for c in retrieved_chunks)

    # Step 2: Refusal path if best cosine score is below threshold
    if best_score < threshold:
        return DocAgentResult(
            answer=REFUSAL_MESSAGE,
            citations=[],
            chunks=retrieved_chunks,
            best_score=best_score,
            refused=True,
        )

    # Step 3: Format context with mandatory [doc p.N] headers and instruction sanitization
    context_blocks = [
        sanitize_document_text(
            text=chunk.text,
            filename=chunk.doc_name,
            page_number=chunk.page_number,
        )
        for chunk in retrieved_chunks
    ]
    context_str = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an accurate, local-first enterprise document assistant.\n"
        "Answer the user's question using ONLY the provided document context.\n"
        "Requirements:\n"
        "- Every single factual claim MUST cite its source as [doc p.N], matching the source header.\n"
        "- Do not synthesize information outside the context.\n"
        f"- If the context does not contain sufficient details to answer, reply exactly: '{REFUSAL_MESSAGE}'."
    )

    user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"

    response = await llm.chat(
        [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]
    )

    raw_answer = response.content.strip()

    # Enforce exact refusal if LLM indicated absence of information
    if REFUSAL_MESSAGE.lower() in raw_answer.lower():
        return DocAgentResult(
            answer=REFUSAL_MESSAGE,
            citations=[],
            chunks=retrieved_chunks,
            best_score=best_score,
            refused=True,
        )

    citations = extract_citations(raw_answer)
    if not citations:
        return DocAgentResult(
            answer=REFUSAL_MESSAGE,
            citations=[],
            chunks=retrieved_chunks,
            best_score=best_score,
            refused=True,
        )

    return DocAgentResult(
        answer=raw_answer,
        citations=citations,
        chunks=retrieved_chunks,
        best_score=best_score,
        refused=False,
    )
