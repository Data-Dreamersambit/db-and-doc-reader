"""DuckDB datastore tool for CSV ingestion, schema extraction, and read-only querying.

Trade-off: Isolated read-only connection guarantees analytical queries cannot mutate datastore.
Re-entrant thread lock coordinates DuckDB file handles to prevent in-process config collisions.
"""
from collections.abc import Generator
from contextlib import contextmanager
import datetime
from decimal import Decimal
from pathlib import Path
import re
import tempfile
import threading
from typing import Any
import duckdb
from pydantic import BaseModel, Field

from backend.app.config import get_settings

MAX_CSV_BYTES: int = 50 * 1024 * 1024  # 50 MB
LOW_CARDINALITY_THRESHOLD: int = 20
RESERVED_SQL_KEYWORDS: set[str] = {
    "select", "from", "where", "table", "group", "order", "by", "limit",
    "join", "insert", "update", "delete", "create", "drop", "alter", "with",
    "case", "when", "then", "else", "end", "having", "union", "all", "distinct",
}


class ColumnInfo(BaseModel):
    """Column definition and inferred SQL data type."""

    name: str = Field(..., description="Column identifier")
    type: str = Field(..., description="Inferred DuckDB SQL data type")


class DatasetSchema(BaseModel):
    """Complete dataset schema inspection payload."""

    table_name: str = Field(..., description="Sanitized DuckDB table name")
    row_count: int = Field(..., description="Total number of rows in the table")
    columns: list[ColumnInfo] = Field(..., description="Columns with SQL types")
    sample_rows: list[dict[str, Any]] = Field(
        ..., description="Up to 5 sample rows from the table"
    )
    low_cardinality_values: dict[str, list[Any]] = Field(
        default_factory=dict,
        description="Unique values for low-cardinality text columns (<= 20 unique)",
    )


class DatasetSummary(BaseModel):
    """Summary of a stored dataset table."""

    table_name: str = Field(..., description="Sanitized DuckDB table name")
    row_count: int = Field(..., description="Total rows in the table")
    column_count: int = Field(..., description="Total columns in the table")


class DatasetListResponse(BaseModel):
    """List of registered datasets in the DuckDB datastore."""

    datasets: list[DatasetSummary] = Field(
        default_factory=list, description="Registered datasets"
    )


def sanitize_table_name(name: str) -> str:
    """Sanitize arbitrary user input into a safe, valid SQL table identifier."""
    # Strip extension if supplied
    base_name = re.sub(r"\.[^.]+$", "", name.strip())
    # Replace non-alphanumeric characters with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", base_name).lower()
    # Collapse consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")

    # Guard against empty names or leading digits
    if not sanitized or sanitized[0].isdigit():
        sanitized = f"t_{sanitized}" if sanitized else "t_dataset"

    # Avoid reserved SQL keywords
    if sanitized in RESERVED_SQL_KEYWORDS:
        sanitized = f"t_{sanitized}"

    return sanitized[:63]


def validate_csv_content(content: bytes, max_bytes: int = MAX_CSV_BYTES) -> str:
    """Validate CSV size, binary check, and decode text safely."""
    if not content or not content.strip():
        raise ValueError("Uploaded file is empty.")

    if len(content) > max_bytes:
        raise ValueError(
            f"File size ({len(content)} bytes) exceeds the {max_bytes} bytes limit."
        )

    # Reject binary data with null bytes
    if b"\x00" in content:
        raise ValueError("Invalid CSV content: binary data detected.")

    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            decoded = content.decode(encoding)
            lines = [ln.strip() for ln in decoded.splitlines() if ln.strip()]
            if not lines:
                raise ValueError("CSV contains no valid rows.")
            return decoded
        except UnicodeDecodeError:
            continue

    raise ValueError("Unable to decode CSV using supported text encodings.")


def _make_serializable(value: Any) -> Any:
    """Convert DuckDB returned types to JSON-safe Python primitives."""
    if isinstance(value, (datetime.date, datetime.datetime, datetime.time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


class Datastore:
    """Manages DuckDB ingestion, schema discovery, and read-only query execution."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        raw_path = db_path or get_settings().DUCKDB_PATH
        self.db_path = Path(raw_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure database file exists on disk so read-only connections succeed
        duckdb.connect(str(self.db_path)).close()
        self._lock = threading.Lock()

    @contextmanager
    def read_connection(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Provide a read-only DuckDB connection context."""
        with self._lock:
            conn = duckdb.connect(str(self.db_path), read_only=True)
            try:
                yield conn
            finally:
                conn.close()

    @contextmanager
    def write_connection(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Provide a read-write DuckDB connection context."""
        with self._lock:
            conn = duckdb.connect(str(self.db_path), read_only=False)
            try:
                yield conn
            finally:
                conn.close()

    def load_csv(self, content: bytes, original_name: str) -> DatasetSchema:
        """Validate, ingest CSV into DuckDB, and return the inferred schema."""
        decoded_text = validate_csv_content(content)
        table_name = sanitize_table_name(original_name)

        # Write to temporary file for DuckDB's robust read_csv_auto parser
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", encoding="utf-8", delete=False
        ) as tmp:
            tmp.write(decoded_text)
            tmp_path = tmp.name

        try:
            with self.write_connection() as conn:
                conn.execute(
                    f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM read_csv_auto(?)',
                    [tmp_path],
                )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return self.get_schema(table_name)

    def list_datasets(self) -> list[DatasetSummary]:
        """List all tables with row and column counts using a read-only connection."""
        with self.read_connection() as conn:
            tables = conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name"
            ).fetchall()

            summaries: list[DatasetSummary] = []
            for (name,) in tables:
                row_count = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                cols = conn.execute(f'PRAGMA table_info("{name}")').fetchall()
                summaries.append(
                    DatasetSummary(
                        table_name=name,
                        row_count=row_count,
                        column_count=len(cols),
                    )
                )
            return summaries

    def get_schema(self, table_name: str) -> DatasetSchema:
        """Inspect table schema, extract sample rows, and identify low-cardinality values."""
        clean_name = sanitize_table_name(table_name)
        with self.read_connection() as conn:
            exists = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'main' AND table_name = ?",
                [clean_name],
            ).fetchone()[0]

            if not exists:
                raise KeyError(f"Dataset '{clean_name}' does not exist.")

            row_count = conn.execute(
                f'SELECT COUNT(*) FROM "{clean_name}"'
            ).fetchone()[0]

            desc = conn.execute(f'DESCRIBE "{clean_name}"').fetchall()
            columns = [ColumnInfo(name=r[0], type=r[1]) for r in desc]

            cur = conn.execute(f'SELECT * FROM "{clean_name}" LIMIT 5')
            col_names = [col[0] for col in cur.description]
            sample_rows = [
                {k: _make_serializable(v) for k, v in zip(col_names, row)}
                for row in cur.fetchall()
            ]

            low_cardinality_values: dict[str, list[Any]] = {}
            for col in columns:
                upper_type = col.type.upper()
                if any(t in upper_type for t in ("VARCHAR", "TEXT", "STRING", "CHAR")):
                    dist_count = conn.execute(
                        f'SELECT COUNT(DISTINCT "{col.name}") FROM "{clean_name}"'
                    ).fetchone()[0]
                    if dist_count <= LOW_CARDINALITY_THRESHOLD:
                        dist_rows = conn.execute(
                            f'SELECT DISTINCT "{col.name}" FROM "{clean_name}" '
                            f'WHERE "{col.name}" IS NOT NULL ORDER BY 1 LIMIT ?',
                            [LOW_CARDINALITY_THRESHOLD],
                        ).fetchall()
                        low_cardinality_values[col.name] = [
                            _make_serializable(r[0]) for r in dist_rows
                        ]

            return DatasetSchema(
                table_name=clean_name,
                row_count=row_count,
                columns=columns,
                sample_rows=sample_rows,
                low_cardinality_values=low_cardinality_values,
            )

    def execute_query(
        self, sql: str, params: list[Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a read-only query and return records as dictionaries."""
        with self.read_connection() as conn:
            cur = conn.execute(sql, params or [])
            if not cur.description:
                return []
            col_names = [col[0] for col in cur.description]
            return [
                {k: _make_serializable(v) for k, v in zip(col_names, row)}
                for row in cur.fetchall()
            ]
