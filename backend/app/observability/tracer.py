"""Async step execution tracer logging to unified SQLite audit log.

Trade-off: SQLite step logging provides lightweight, non-blocking distributed tracing without OTel daemons.
ContextVar propagation correlates nested multi-agent steps under a single trace_id.
"""
from collections.abc import Callable, Coroutine
from contextvars import ContextVar
from datetime import datetime, timezone
import functools
import inspect
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, ParamSpec, TypeVar
import uuid
from pydantic import BaseModel, Field

from backend.app.config import get_settings

current_trace_id: ContextVar[str | None] = ContextVar("current_trace_id", default=None)

P = ParamSpec("P")
R = TypeVar("R")


class TraceRecord(BaseModel):
    """Database record of an individual traced execution step."""

    id: int = Field(..., description="Unique step sequence ID")
    trace_id: str = Field(..., description="Parent distributed trace UUID")
    step: str = Field(..., description="Logical step or function identifier")
    input: str | None = Field(default=None, description="Serialized function arguments")
    output: str | None = Field(default=None, description="Serialized result payload")
    latency_ms: float = Field(..., description="Step duration in milliseconds")
    status: str = Field(..., description="'success' or 'error'")
    error: str | None = Field(default=None, description="Error message if failed")
    created_at: str = Field(..., description="ISO 8601 execution timestamp")


class TraceDetailResponse(BaseModel):
    """Aggregated trace details for API consumers."""

    trace_id: str
    step_count: int
    total_latency_ms: float
    steps: list[TraceRecord]


def _serialize_safely(data: Any) -> str | None:
    """Serialize inputs/outputs safely into JSON strings without raising exceptions."""
    if data is None:
        return None
    try:
        if isinstance(data, BaseModel):
            return data.model_dump_json()
        return json.dumps(data, default=str)
    except Exception:
        return str(data)


class Tracer:
    """Thread-safe SQLite audit tracer."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        raw_path = db_path or get_settings().SQLITE_PATH
        self.db_path = Path(raw_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with WAL mode enabled."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def _init_db(self) -> None:
        """Initialize the audit_traces table."""
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    step TEXT NOT NULL,
                    input TEXT,
                    output TEXT,
                    latency_ms REAL NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_traces_tid ON audit_traces(trace_id, id);"
            )
            conn.commit()

    def log_step(
        self,
        trace_id: str,
        step: str,
        input_data: Any,
        output_data: Any,
        latency_ms: float,
        status: str,
        error: str | None = None,
    ) -> None:
        """Persist a single execution step to SQLite."""
        now_iso = datetime.now(timezone.utc).isoformat()
        serialized_input = _serialize_safely(input_data)
        serialized_output = _serialize_safely(output_data)

        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_traces (trace_id, step, input, output, latency_ms, status, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    step,
                    serialized_input,
                    serialized_output,
                    latency_ms,
                    status,
                    error,
                    now_iso,
                ),
            )
            conn.commit()

    def get_trace(self, trace_id: str) -> list[TraceRecord]:
        """Fetch all execution steps associated with a trace_id."""
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, trace_id, step, input, output, latency_ms, status, error, created_at
                FROM audit_traces
                WHERE trace_id = ?
                ORDER BY id ASC
                """,
                (trace_id,),
            ).fetchall()
            return [
                TraceRecord(
                    id=r["id"],
                    trace_id=r["trace_id"],
                    step=r["step"],
                    input=r["input"],
                    output=r["output"],
                    latency_ms=r["latency_ms"],
                    status=r["status"],
                    error=r["error"],
                    created_at=r["created_at"],
                )
                for r in rows
            ]


_GLOBAL_TRACER: Tracer | None = None


def get_global_tracer() -> Tracer:
    """Return the global default Tracer instance."""
    global _GLOBAL_TRACER
    if _GLOBAL_TRACER is None:
        _GLOBAL_TRACER = Tracer()
    return _GLOBAL_TRACER


def trace_step(
    step_name: str,
    tracer: Tracer | None = None,
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
    """Async decorator that profiles and logs step execution metrics to SQLite."""

    def decorator(
        func: Callable[P, Coroutine[Any, Any, R]],
    ) -> Callable[P, Coroutine[Any, Any, R]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            active_tracer = tracer or get_global_tracer()
            # Inherit active trace_id or generate a new one
            explicit_tid = kwargs.get("trace_id")
            active_tid = str(explicit_tid) if explicit_tid else (current_trace_id.get() or str(uuid.uuid4()))
            token = current_trace_id.set(active_tid)

            # Clean input serialization
            safe_kwargs = {k: v for k, v in kwargs.items() if k != "trace_id"}
            input_summary = {"args": args, "kwargs": safe_kwargs}

            sig = inspect.signature(func)
            accepts_trace_id = "trace_id" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            call_kwargs = kwargs if accepts_trace_id else safe_kwargs

            start_time = time.perf_counter()
            try:
                result = await func(*args, **call_kwargs)
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                active_tracer.log_step(
                    trace_id=active_tid,
                    step=step_name,
                    input_data=input_summary,
                    output_data=result,
                    latency_ms=latency_ms,
                    status="success",
                    error=None,
                )
                return result
            except Exception as exc:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                active_tracer.log_step(
                    trace_id=active_tid,
                    step=step_name,
                    input_data=input_summary,
                    output_data=None,
                    latency_ms=latency_ms,
                    status="error",
                    error=str(exc),
                )
                raise
            finally:
                current_trace_id.reset(token)

        return wrapper

    return decorator
