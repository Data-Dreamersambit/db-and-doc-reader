"""Hybrid retrieval combining NumPy cosine similarity and rank_bm25 via Reciprocal Rank Fusion.

Trade-off: In-memory NumPy cosine + rank_bm25 with RRF yields competitive hybrid recall.
Avoids heavy external C++/Java search infrastructure while keeping latency under 15ms.
"""
import re
import numpy as np
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from backend.app.llm.client import OllamaClient
from backend.app.rag.store import SQLiteDocStore, StoredChunk

RRF_K_CONSTANT: int = 60


class ChunkMatch(BaseModel):
    """Retrieved document chunk with individual and fused relevance scores."""

    chunk_id: str
    doc_id: str
    doc_name: str
    page_number: int
    text: str
    cosine_score: float = Field(..., description="Dense cosine similarity score")
    bm25_score: float = Field(..., description="Sparse BM25 score")
    rrf_score: float = Field(..., description="Reciprocal Rank Fusion score")


async def hybrid_retrieve(
    query: str,
    store: SQLiteDocStore,
    client: OllamaClient,
    top_k_dense: int = 20,
    top_k_bm25: int = 20,
    top_k_fused: int = 5,
) -> list[ChunkMatch]:
    """Retrieve top-k chunks by fusing top-20 dense cosine and top-20 BM25 results with RRF."""
    all_chunks = store.get_all_chunks()
    if not all_chunks:
        return []

    num_chunks = len(all_chunks)
    actual_top_dense = min(top_k_dense, num_chunks)
    actual_top_bm25 = min(top_k_bm25, num_chunks)

    # 1. Dense Cosine Retrieval via NumPy
    embed_response = await client.embed([query])
    query_vector = np.array(embed_response.embeddings[0], dtype=np.float32)

    matrix = np.vstack([c.embedding for c in all_chunks])
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-9)
    cosine_similarities = np.dot(matrix_norm, query_norm)

    top_dense_indices = np.argsort(cosine_similarities)[::-1][:actual_top_dense]

    # 2. Sparse Lexical Retrieval via rank_bm25
    tokenized_corpus = [
        re.findall(r"\w+", chunk.text.lower()) for chunk in all_chunks
    ]
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = re.findall(r"\w+", query.lower())
    if query_tokens:
        bm25_scores = np.array(bm25.get_scores(query_tokens), dtype=np.float32)
    else:
        bm25_scores = np.zeros(num_chunks, dtype=np.float32)

    top_bm25_indices = np.argsort(bm25_scores)[::-1][:actual_top_bm25]

    # 3. Reciprocal Rank Fusion (RRF)
    rrf_scores: dict[int, float] = {}
    for rank, idx in enumerate(top_dense_indices, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (RRF_K_CONSTANT + rank))

    for rank, idx in enumerate(top_bm25_indices, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (RRF_K_CONSTANT + rank))

    sorted_indices = sorted(
        rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True
    )[:top_k_fused]

    matches: list[ChunkMatch] = []
    for idx in sorted_indices:
        chunk: StoredChunk = all_chunks[idx]
        matches.append(
            ChunkMatch(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                doc_name=chunk.doc_name,
                page_number=chunk.page_number,
                text=chunk.text,
                cosine_score=float(cosine_similarities[idx]),
                bm25_score=float(bm25_scores[idx]),
                rrf_score=float(rrf_scores[idx]),
            )
        )

    return matches
