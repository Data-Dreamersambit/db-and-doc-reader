"""RAG module for PDF chunking, SQLite vector storage, and hybrid retrieval."""
from backend.app.rag.chunker import DocumentChunk, extract_and_chunk_pdf
from backend.app.rag.retriever import ChunkMatch, hybrid_retrieve
from backend.app.rag.store import DocumentInfo, SQLiteDocStore, StoredChunk

__all__ = [
    "ChunkMatch",
    "DocumentChunk",
    "DocumentInfo",
    "SQLiteDocStore",
    "StoredChunk",
    "extract_and_chunk_pdf",
    "hybrid_retrieve",
]
