"""SQLite session conversation history with sliding-window turn pruning.

Trade-off: SQLite session pruning enforces a strict 6-turn sliding window to bound LLM context size.
Storing turns in the unified SQLite file prevents multi-database synchronization overhead.
"""
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import threading
from backend.app.config import get_settings
from backend.app.llm.client import ChatMessage

MAX_TURNS: int = 6
MAX_MESSAGES: int = MAX_TURNS * 2  # 6 User + 6 Assistant turns


class SessionManager:
    """Manages chat session turns in SQLite with automatic pruning."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        raw_path = db_path or get_settings().SQLITE_PATH
        self.db_path = Path(raw_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with WAL and timeout configured."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def _init_db(self) -> None:
        """Initialize session_history table."""
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_history_id ON session_history(session_id, id);"
            )
            conn.commit()

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append message and enforce sliding window of the last 6 turns (12 messages)."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO session_history (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, now_iso),
            )
            # Prune turns older than the latest MAX_MESSAGES
            conn.execute(
                """
                DELETE FROM session_history
                WHERE session_id = ?
                  AND id NOT IN (
                      SELECT id FROM session_history
                      WHERE session_id = ?
                      ORDER BY id DESC
                      LIMIT ?
                  )
                """,
                (session_id, session_id, MAX_MESSAGES),
            )
            conn.commit()

    def get_history(self, session_id: str) -> list[ChatMessage]:
        """Fetch the latest conversation turns in chronological order."""
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT role, content FROM (
                    SELECT id, role, content FROM session_history
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY id ASC
                """,
                (session_id, MAX_MESSAGES),
            ).fetchall()
            return [ChatMessage(role=r["role"], content=r["content"]) for r in rows]

    def clear_session(self, session_id: str) -> None:
        """Clear all conversation history for a session."""
        with self._lock, self._get_connection() as conn:
            conn.execute(
                "DELETE FROM session_history WHERE session_id = ?", (session_id,)
            )
            conn.commit()
