"""SQLite document and vector storage with float32 BLOB embeddings.

Trade-off: Storing float32 blobs directly in SQLite avoids running a dedicated vector daemon.
Single-file database simplifies local backups and local-first distribution.
"""
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import threading
from typing import Any
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from backend.app.config import get_settings
from backend.app.rag.chunker import DocumentChunk


class DocumentInfo(BaseModel):
    """Metadata summary of an ingested document."""

    id: str = Field(..., description="Unique document UUID")
    filename: str = Field(..., description="Original filename")
    page_count: int = Field(..., description="Total pages in source file")
    chunk_count: int = Field(..., description="Total generated chunks")
    created_at: str = Field(..., description="ISO 8601 ingestion timestamp")


class StoredChunk(BaseModel):
    """Document chunk record containing text and deserialized numpy vector."""

    chunk_id: str
    doc_id: str
    doc_name: str
    page_number: int
    text: str
    embedding: Any = Field(..., description="NumPy float32 1D array")
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SQLiteDocStore:
    """Manages document metadata and dense vector storage inside a local SQLite file."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        raw_path = db_path or get_settings().SQLITE_PATH
        self.db_path = Path(raw_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with foreign keys and WAL enabled."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def _init_db(self) -> None:
        """Initialize documents and document_chunks tables."""
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    page_count INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    doc_name TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks(doc_id);"
            )
            conn.commit()

    def save_document(
        self,
        doc_id: str,
        filename: str,
        page_count: int,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> DocumentInfo:
        """Persist document metadata and chunk float32 vector BLOBs."""
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks count must match embeddings count.")

        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents (id, filename, page_count, chunk_count, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (doc_id, filename, page_count, len(chunks), now_iso),
            )

            # Insert all chunk records
            chunk_records = [
                (
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.doc_name,
                    chunk.page_number,
                    chunk.text,
                    np.array(vec, dtype=np.float32).tobytes(),
                    now_iso,
                )
                for chunk, vec in zip(chunks, embeddings)
            ]
            conn.executemany(
                """
                INSERT OR REPLACE INTO document_chunks
                (chunk_id, doc_id, doc_name, page_number, text, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                chunk_records,
            )
            conn.commit()

        return DocumentInfo(
            id=doc_id,
            filename=filename,
            page_count=page_count,
            chunk_count=len(chunks),
            created_at=now_iso,
        )

    def list_documents(self) -> list[DocumentInfo]:
        """List all indexed documents."""
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(
                "SELECT id, filename, page_count, chunk_count, created_at FROM documents ORDER BY created_at DESC"
            ).fetchall()
            return [
                DocumentInfo(
                    id=r["id"],
                    filename=r["filename"],
                    page_count=r["page_count"],
                    chunk_count=r["chunk_count"],
                    created_at=r["created_at"],
                )
                for r in rows
            ]

    def get_document(self, doc_id: str) -> DocumentInfo | None:
        """Retrieve metadata for a single document."""
        with self._lock, self._get_connection() as conn:
            row = conn.execute(
                "SELECT id, filename, page_count, chunk_count, created_at FROM documents WHERE id = ?",
                (doc_id,),
            ).fetchone()
            if not row:
                return None
            return DocumentInfo(
                id=row["id"],
                filename=row["filename"],
                page_count=row["page_count"],
                chunk_count=row["chunk_count"],
                created_at=row["created_at"],
            )

    def delete_document(self, doc_id: str) -> bool:
        """Delete document and its cascaded chunks from SQLite."""
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()
            return cur.rowcount > 0

    def get_all_chunks(self) -> list[StoredChunk]:
        """Fetch all chunks with deserialized float32 NumPy vectors."""
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(
                "SELECT chunk_id, doc_id, doc_name, page_number, text, embedding FROM document_chunks"
            ).fetchall()
            results: list[StoredChunk] = []
            for r in rows:
                vec = np.frombuffer(r["embedding"], dtype=np.float32)
                results.append(
                    StoredChunk(
                        chunk_id=r["chunk_id"],
                        doc_id=r["doc_id"],
                        doc_name=r["doc_name"],
                        page_number=r["page_number"],
                        text=r["text"],
                        embedding=vec,
                    )
                )
            return results
