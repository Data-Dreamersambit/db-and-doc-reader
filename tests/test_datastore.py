"""Tests for DuckDB Datastore tool and dataset REST endpoints.

Trade-off: Temporary DuckDB files isolate disk state between test runs.
ASGITransport tests FastAPI upload streams without running live network servers.
"""
from pathlib import Path
import duckdb
import httpx
import pytest

from backend.app.main import app, get_datastore
from backend.app.tools.datastore import (
    Datastore,
    sanitize_table_name,
    validate_csv_content,
)


def test_sanitize_table_name() -> None:
    """Test sanitization of various valid and malformed table names."""
    assert sanitize_table_name("sales.csv") == "sales"
    assert sanitize_table_name("2024 Sales (Q1) - Final.csv") == "t_2024_sales_q1_final"
    assert sanitize_table_name("123_data.csv") == "t_123_data"
    assert sanitize_table_name("select.csv") == "t_select"
    assert sanitize_table_name("where") == "t_where"
    assert sanitize_table_name("") == "t_dataset"
    assert sanitize_table_name("___") == "t_dataset"


def test_validate_csv_content() -> None:
    """Test CSV content validation including encodings, sizes, and binary checks."""
    valid_csv = b"id,name\n1,Alpha\n2,Beta\n"
    decoded = validate_csv_content(valid_csv)
    assert "id,name" in decoded

    # Empty content
    with pytest.raises(ValueError, match="empty"):
        validate_csv_content(b"   ")

    # Binary data with null bytes
    with pytest.raises(ValueError, match="binary data"):
        validate_csv_content(b"PK\x03\x04\x00\x00some zip data")

    # Size limit
    with pytest.raises(ValueError, match="exceeds"):
        validate_csv_content(b"col\nval\n", max_bytes=5)


def test_datastore_load_and_query(tmp_path: Path) -> None:
    """Test loading sales.csv into DuckDB and inspecting schema."""
    db_file = tmp_path / "test_data.duckdb"
    store = Datastore(db_path=db_file)

    sample_csv_path = Path("sample_data/sales.csv")
    content = sample_csv_path.read_bytes()

    schema = store.load_csv(content, "sales.csv")
    assert schema.table_name == "sales"
    assert schema.row_count == 25
    assert len(schema.sample_rows) == 5

    # Check columns
    col_names = [c.name for c in schema.columns]
    assert "order_id" in col_names
    assert "region" in col_names
    assert "total_revenue" in col_names

    # Check low-cardinality values
    assert "region" in schema.low_cardinality_values
    assert sorted(schema.low_cardinality_values["region"]) == [
        "East",
        "North",
        "South",
        "West",
    ]
    assert "payment_method" in schema.low_cardinality_values
    assert sorted(schema.low_cardinality_values["payment_method"]) == [
        "Credit Card",
        "Direct Debit",
        "PayPal",
    ]

    # Verify list datasets
    datasets = store.list_datasets()
    assert len(datasets) == 1
    assert datasets[0].table_name == "sales"
    assert datasets[0].row_count == 25

    # Verify read-only query execution
    records = store.execute_query(
        'SELECT region, SUM(total_revenue) as rev FROM "sales" GROUP BY region ORDER BY region'
    )
    assert len(records) == 4
    assert "region" in records[0]
    assert "rev" in records[0]

    # Verify read-only connection rejects mutations
    with pytest.raises(duckdb.InvalidInputException):
        with store.read_connection() as conn:
            conn.execute("INSERT INTO sales VALUES (9999, '2024-01-01', 'C', 'N', 'E', 'P', 1, 10, 0, 10, 'Cash')")


def test_datastore_get_schema_not_found(tmp_path: Path) -> None:
    """Test that querying a non-existent table schema raises KeyError."""
    store = Datastore(db_path=tmp_path / "empty.duckdb")
    with pytest.raises(KeyError, match="does not exist"):
        store.get_schema("unknown_table")


@pytest.mark.asyncio
async def test_dataset_endpoints(tmp_path: Path) -> None:
    """Test POST /api/datasets, GET /api/datasets, and GET /api/datasets/{name}/schema."""
    test_db = tmp_path / "api_test.duckdb"
    store = Datastore(db_path=test_db)
    app.dependency_overrides[get_datastore] = lambda: store

    try:
        sample_csv = Path("sample_data/sales.csv").read_bytes()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            # 1. Upload valid CSV
            upload_res = await client.post(
                "/api/datasets",
                files={"file": ("sales.csv", sample_csv, "text/csv")},
            )
            assert upload_res.status_code == 200
            data = upload_res.json()
            assert data["table_name"] == "sales"
            assert data["row_count"] == 25
            assert len(data["sample_rows"]) == 5

            # 2. Upload invalid CSV
            bad_upload = await client.post(
                "/api/datasets",
                files={"file": ("corrupt.csv", b"\x00\x01\x02", "application/octet-stream")},
            )
            assert bad_upload.status_code == 400

            # 3. List datasets
            list_res = await client.get("/api/datasets")
            assert list_res.status_code == 200
            datasets = list_res.json()["datasets"]
            assert len(datasets) == 1
            assert datasets[0]["table_name"] == "sales"

            # 4. Get dataset schema
            schema_res = await client.get("/api/datasets/sales/schema")
            assert schema_res.status_code == 200
            schema_data = schema_res.json()
            assert schema_data["table_name"] == "sales"
            assert "region" in schema_data["low_cardinality_values"]

            # 5. Get missing dataset schema
            missing_res = await client.get("/api/datasets/not_real/schema")
            assert missing_res.status_code == 404
    finally:
        app.dependency_overrides.clear()
