"""Unit and integration tests for the SQL Agent.

Trade-off: Direct AST validator testing verifies security boundaries independently of LLM.
Mocked LLM transport tests retry loop deterministically without external GPU dependencies.
"""
from pathlib import Path
import json
import httpx
import pytest

from backend.app.agents.sql_agent import (
    ChartSpec,
    SQLAgentExecutionError,
    SQLValidationError,
    extract_referenced_columns,
    infer_chart_spec,
    repair_columns,
    run_sql_agent,
    validate_and_format_sql,
)
from backend.app.llm.client import OllamaClient
from backend.app.tools.datastore import Datastore


def test_validate_and_format_sql_security_blocks() -> None:
    """Verify that DROP, multi-statement DELETE, and read_csv are strictly blocked."""
    # 1. DROP TABLE attempt
    with pytest.raises(SQLValidationError, match="Only single SELECT statements are allowed"):
        validate_and_format_sql("DROP TABLE sales")

    # 2. Multi-statement injection attempt
    with pytest.raises(SQLValidationError, match="Multiple SQL statements"):
        validate_and_format_sql("SELECT * FROM sales; DELETE FROM sales")

    # 3. read_csv file access attempt
    with pytest.raises(SQLValidationError, match="Forbidden statement or expression type"):
        validate_and_format_sql("SELECT * FROM read_csv('/etc/passwd')")

    # 4. read_csv_auto file access attempt
    with pytest.raises(SQLValidationError, match="Forbidden function 'read_csv_auto'"):
        validate_and_format_sql("SELECT * FROM read_csv_auto('/etc/passwd')")

    # 5. COPY statement attempt
    with pytest.raises(SQLValidationError, match="Only single SELECT statements are allowed"):
        validate_and_format_sql("COPY sales TO '/tmp/data.csv'")

    # 6. INSERT / DDL attempt
    with pytest.raises(SQLValidationError, match="Only single SELECT statements are allowed"):
        validate_and_format_sql("INSERT INTO sales VALUES (1, '2024-01-01')")


def test_validate_and_format_sql_limit_and_lineage() -> None:
    """Verify LIMIT 1000 injection, limit capping, and table/column lineage extraction."""
    # Automatic LIMIT 1000 injection
    sql1, lineage1 = validate_and_format_sql(
        "SELECT region, SUM(total_revenue) AS rev FROM sales WHERE discount = 0 GROUP BY region"
    )
    assert "LIMIT 1000" in sql1
    assert lineage1.tables == ["sales"]
    assert "region" in lineage1.columns
    assert "total_revenue" in lineage1.columns
    assert "discount" in lineage1.columns

    # Capping limit > 1000 to 1000
    sql2, _ = validate_and_format_sql("SELECT * FROM sales LIMIT 5000")
    assert "LIMIT 1000" in sql2

    # Preserving smaller limit
    sql3, _ = validate_and_format_sql("SELECT * FROM sales LIMIT 25")
    assert "LIMIT 25" in sql3


@pytest.mark.asyncio
async def test_infer_chart_spec_heuristics() -> None:
    """Test rule-based heuristics for line, bar, pie, and table charts."""
    client = OllamaClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))

    # Temporal -> line chart
    line_spec = await infer_chart_spec(
        user_query="Show revenue over time",
        columns=["order_date", "total_revenue"],
        rows=[
            {"order_date": "2024-01-01", "total_revenue": 100.0},
            {"order_date": "2024-01-02", "total_revenue": 150.0},
        ],
        client=client,
    )
    assert line_spec.chart_type == "line"
    assert line_spec.x_field == "order_date"
    assert line_spec.y_fields == ["total_revenue"]

    # Categorical + metric -> bar chart
    bar_spec = await infer_chart_spec(
        user_query="Revenue by region",
        columns=["region", "total_revenue"],
        rows=[
            {"region": "North", "total_revenue": 500.0},
            {"region": "South", "total_revenue": 300.0},
        ],
        client=client,
    )
    assert bar_spec.chart_type == "bar"
    assert bar_spec.x_field == "region"
    assert bar_spec.y_fields == ["total_revenue"]

    # Share / breakdown with few categories -> pie chart
    pie_spec = await infer_chart_spec(
        user_query="What is the distribution of sales by category?",
        columns=["product_category", "units_sold"],
        rows=[
            {"product_category": "Electronics", "units_sold": 40},
            {"product_category": "Furniture", "units_sold": 20},
        ],
        client=client,
    )
    assert pie_spec.chart_type == "pie"
    assert pie_spec.x_field == "product_category"
    assert pie_spec.y_fields == ["units_sold"]

    # Single row -> table
    table_spec = await infer_chart_spec(
        user_query="Total sales revenue",
        columns=["total_revenue"],
        rows=[{"total_revenue": 5400.0}],
        client=client,
    )
    assert table_spec.chart_type == "table"


@pytest.mark.asyncio
async def test_run_sql_agent_e2e(tmp_path: Path) -> None:
    """Test full SQL Agent execution on sample sales data with mocked LLM responses."""
    db_path = tmp_path / "sql_agent_test.duckdb"
    store = Datastore(db_path=db_path)
    store.load_csv(Path("sample_data/sales.csv").read_bytes(), "sales.csv")

    call_index = 0

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_index
        call_index += 1
        body = json.loads(request.content)
        prompt_str = str(body.get("messages", []))

        # Schema prune response
        if "relevant_tables" in prompt_str:
            return httpx.Response(
                status_code=200,
                json={
                    "model": "qwen2.5:3b",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "relevant_tables": [{"table": "sales", "columns": ["region", "total_revenue"]}]
                        }),
                    },
                    "done": True,
                },
            )

        # SQL generation response
        if "DuckDB SQL data engineer" in prompt_str:
            return httpx.Response(
                status_code=200,
                json={
                    "model": "qwen2.5:3b",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "sql": "SELECT region, SUM(total_revenue) AS total_revenue FROM sales GROUP BY region ORDER BY total_revenue DESC",
                            "reasoning": "Sum revenue grouped by region.",
                        }),
                    },
                    "done": True,
                },
            )

        # Explanation response
        return httpx.Response(
            status_code=200,
            json={
                "model": "qwen2.5:3b",
                "message": {
                    "role": "assistant",
                    "content": "Total revenue is highest in the North region, followed by East and South.",
                },
                "done": True,
            },
        )

    mock_client = OllamaClient(transport=httpx.MockTransport(mock_handler))

    result = await run_sql_agent(
        user_query="What is the total revenue by region?",
        datastore=store,
        client=mock_client,
    )

    assert "sales" in result.sql
    assert "LIMIT 1000" in result.sql
    assert result.columns == ["region", "total_revenue"]
    assert len(result.rows) == 4
    assert result.lineage.tables == ["sales"]
    assert "region" in result.lineage.columns
    assert "total_revenue" in result.lineage.columns
    assert result.chart_spec.chart_type == "bar"
    assert result.chart_spec.x_field == "region"
    assert "total_revenue" in result.chart_spec.y_fields
    assert result.attempts == 1
    assert "North" in result.explanation


@pytest.mark.asyncio
async def test_run_sql_agent_retry_on_error(tmp_path: Path) -> None:
    """Test that execution error is fed back to LLM and succeeds on retry."""
    db_path = tmp_path / "sql_retry_test.duckdb"
    store = Datastore(db_path=db_path)
    store.load_csv(Path("sample_data/sales.csv").read_bytes(), "sales.csv")

    attempt_count = 0

    async def mock_retry_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt_count
        body = json.loads(request.content)
        prompt_str = str(body.get("messages", []))

        # Schema prune response
        if "relevant_tables" in prompt_str:
            return httpx.Response(
                status_code=200,
                json={
                    "model": "qwen2.5:3b",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"relevant_tables": [{"table": "sales", "columns": ["region"]}]}),
                    },
                    "done": True,
                },
            )

        # SQL generation response: fail on attempt 1 with non-existent column, fix on attempt 2
        if "DuckDB SQL data engineer" in prompt_str:
            attempt_count += 1
            if attempt_count == 1:
                bad_sql = "SELECT non_existent_column FROM sales"
                return httpx.Response(
                    status_code=200,
                    json={
                        "model": "qwen2.5:3b",
                        "message": {"role": "assistant", "content": json.dumps({"sql": bad_sql, "reasoning": "Wrong col"})},
                        "done": True,
                    },
                )
            else:
                good_sql = "SELECT region, COUNT(*) AS count FROM sales GROUP BY region"
                return httpx.Response(
                    status_code=200,
                    json={
                        "model": "qwen2.5:3b",
                        "message": {"role": "assistant", "content": json.dumps({"sql": good_sql, "reasoning": "Fixed col"})},
                        "done": True,
                    },
                )

        # Explanation
        return httpx.Response(
            status_code=200,
            json={
                "model": "qwen2.5:3b",
                "message": {"role": "assistant", "content": "Orders distributed across 4 regions."},
                "done": True,
            },
        )

    mock_client = OllamaClient(transport=httpx.MockTransport(mock_retry_handler))

    result = await run_sql_agent(
        user_query="Count orders by region",
        datastore=store,
        client=mock_client,
    )

    assert result.attempts == 2
    assert "region" in result.columns
    assert len(result.rows) == 4


def test_validate_columns_against_schema_blocks_hallucinations() -> None:
    """Verify that hallucinated columns like delivery_estimate are blocked before hitting DuckDB."""
    from backend.app.tools.datastore import ColumnInfo, DatasetSchema

    schema = DatasetSchema(
        table_name="sales",
        row_count=25,
        sample_rows=[],
        columns=[
            ColumnInfo(name="order_id", type="BIGINT", sample_values=["1001"]),
            ColumnInfo(name="order_date", type="VARCHAR", sample_values=["2024-01-05"]),
            ColumnInfo(name="region", type="VARCHAR", sample_values=["North"]),
            ColumnInfo(name="total_revenue", type="DOUBLE", sample_values=["127.50"]),
        ],
        low_cardinality_values={"region": ["North", "South"]},
    )

    # 1. Hallucinated column 'delivery_estimate' must be rejected
    bad_sql = "SELECT region, AVG(delivery_estimate) FROM sales GROUP BY region"
    with pytest.raises(SQLValidationError, match="Column 'delivery_estimate' does not exist"):
        validate_and_format_sql(bad_sql, schemas=[schema])

    # 2. Hallucinated table 'orders' must be rejected
    bad_table_sql = "SELECT region, COUNT(*) FROM orders GROUP BY region"
    with pytest.raises(SQLValidationError, match="Table 'orders' does not exist"):
        validate_and_format_sql(bad_table_sql, schemas=[schema])

    # 3. Valid columns and valid aliases in ORDER BY must pass
    valid_sql = "SELECT region, SUM(total_revenue) AS rev FROM sales GROUP BY region ORDER BY rev DESC"
    sql, lineage = validate_and_format_sql(valid_sql, schemas=[schema])
    assert "sales" in lineage.tables
    assert "region" in lineage.columns
    assert "total_revenue" in lineage.columns


@pytest.mark.asyncio
async def test_run_sql_agent_unanswerable_immediate_exit(tmp_path: Path) -> None:
    """Verify that when a query needs data not in schema, agent returns immediately with status not_in_data."""
    store = Datastore(db_path=tmp_path / "test_unans.duckdb")
    store.load_csv(Path("sample_data/sales.csv").read_bytes(), "sales.csv")

    captured_prompt = ""

    async def mock_unanswerable_handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_prompt
        body = json.loads(request.content)
        prompt_str = str(body.get("messages", []))

        # Schema pruner
        if "relevant_tables" in prompt_str:
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5:3b",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"relevant_tables": [{"table": "sales", "columns": ["region"]}]}),
                    },
                    "done": True,
                },
            )

        # SQL generator
        if "DuckDB SQL data engineer" in prompt_str:
            captured_prompt = prompt_str
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5:3b",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "answerable": False,
                            "missing": "delivery_estimate",
                        }),
                    },
                    "done": True,
                },
            )

        return httpx.Response(200, json={"model": "qwen2.5:3b", "message": {"role": "assistant", "content": "ok"}})

    mock_client = OllamaClient(transport=httpx.MockTransport(mock_unanswerable_handler))

    result = await run_sql_agent(
        user_query="What is the average delivery estimate by region?",
        datastore=store,
        client=mock_client,
    )

    # 1. Returned immediately with not_in_data and missing info
    assert result.status == "not_in_data"
    assert result.missing == "delivery_estimate"
    assert result["status"] == "not_in_data"
    assert result["missing"] == "delivery_estimate"
    assert result.attempts == 1
    assert "delivery_estimate" in result.explanation

    # 2. Prompt contains the required rule
    assert "If the question needs data not in the listed columns, set answerable to false. Never guess column names." in captured_prompt


def test_repair_columns() -> None:
    """Test difflib column matching with cutoff 0.75."""
    valid_cols = ["unit_price_inr", "order_id", "total_revenue", "region"]

    # 1. Close match should auto-match
    fixes = repair_columns(["unit_price"], valid_cols)
    assert fixes == {"unit_price": "unit_price_inr"}

    # 2. No close match should return empty fixes
    fixes_empty = repair_columns(["delivery_estimate"], valid_cols)
    assert fixes_empty == {}

    # 3. Valid columns should not be in fixes
    fixes_valid = repair_columns(["order_id", "region"], valid_cols)
    assert fixes_valid == {}

    # 4. Mixed valid, repairable, and unmatchable
    fixes_mixed = repair_columns(["order_id", "unit_price", "delivery_estimate"], valid_cols)
    assert fixes_mixed == {"unit_price": "unit_price_inr"}


def test_extract_referenced_columns() -> None:
    """Test AST column extraction ignoring aliases and CTEs."""
    sql = "SELECT region, SUM(unit_price) AS total FROM sales WHERE status = 'Completed' GROUP BY region ORDER BY total DESC"
    cols = extract_referenced_columns(sql)
    assert "region" in cols
    assert "unit_price" in cols
    assert "status" in cols
    assert "total" not in cols


@pytest.mark.asyncio
async def test_run_sql_agent_auto_repair_close_match(tmp_path: Path) -> None:
    """Test that when model uses unit_price instead of unit_price_inr, agent auto-replaces and executes."""
    store = Datastore(db_path=tmp_path / "repair_test.duckdb")
    store.load_csv(
        b"order_id,region,unit_price_inr\n1,North,150.0\n2,South,200.0\n",
        "sales.csv",
    )

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt_str = str(body.get("messages", []))

        if "relevant_tables" in prompt_str:
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5:3b",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"relevant_tables": [{"table": "sales", "columns": ["region", "unit_price_inr"]}]}),
                    },
                    "done": True,
                },
            )

        if "DuckDB SQL data engineer" in prompt_str:
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5:3b",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "answerable": True,
                            "sql": "SELECT region, AVG(unit_price) AS avg_price FROM sales GROUP BY region",
                            "reasoning": "Average unit price",
                        }),
                    },
                    "done": True,
                },
            )

        return httpx.Response(200, json={"model": "qwen2.5:3b", "message": {"role": "assistant", "content": "Avg unit price is 175."}})

    mock_client = OllamaClient(transport=httpx.MockTransport(mock_handler))
    result = await run_sql_agent(
        user_query="What is the average unit price by region?",
        datastore=store,
        client=mock_client,
    )

    assert result.status == "success"
    assert "unit_price_inr" in result.sql
    assert result.attempts == 1
    assert len(result.rows) == 2


@pytest.mark.asyncio
async def test_run_sql_agent_stops_on_no_close_match_delivery_estimate(tmp_path: Path) -> None:
    """Test that when an unmatchable column like delivery_estimate is used, agent stops immediately without retrying."""
    store = Datastore(db_path=tmp_path / "delivery_test.duckdb")
    store.load_csv(Path("sample_data/sales.csv").read_bytes(), "sales.csv")

    attempts_seen = 0

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts_seen
        body = json.loads(request.content)
        prompt_str = str(body.get("messages", []))

        if "relevant_tables" in prompt_str:
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5:3b",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"relevant_tables": [{"table": "sales", "columns": ["region"]}]}),
                    },
                    "done": True,
                },
            )

        if "DuckDB SQL data engineer" in prompt_str:
            attempts_seen += 1
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5:3b",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "answerable": True,
                            "sql": "SELECT region, AVG(delivery_estimate) AS avg_deliv FROM sales GROUP BY region",
                            "reasoning": "Average delivery estimate",
                        }),
                    },
                    "done": True,
                },
            )

        return httpx.Response(200, json={"model": "qwen2.5:3b", "message": {"role": "assistant", "content": "ok"}})

    mock_client = OllamaClient(transport=httpx.MockTransport(mock_handler))
    result = await run_sql_agent(
        user_query="What is the estimated delivery time for North region orders?",
        datastore=store,
        client=mock_client,
    )

    assert result.status == "not_in_data"
    assert result.missing == "delivery_estimate"
    assert result.attempts == 1
    assert attempts_seen == 1


@pytest.mark.asyncio
async def test_run_sql_agent_tracks_invalid_columns_in_retry_prompt(tmp_path: Path) -> None:
    """Test that invalid columns are tracked across attempts and added to retry prompt."""
    store = Datastore(db_path=tmp_path / "track_retry.duckdb")
    store.load_csv(Path("sample_data/sales.csv").read_bytes(), "sales.csv")

    retry_prompt = ""
    attempt_count = 0

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal retry_prompt, attempt_count
        body = json.loads(request.content)
        prompt_str = str(body.get("messages", []))

        if "relevant_tables" in prompt_str:
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5:3b",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"relevant_tables": [{"table": "sales", "columns": ["region"]}]}),
                    },
                    "done": True,
                },
            )

        if "DuckDB SQL data engineer" in prompt_str:
            attempt_count += 1
            if attempt_count == 1:
                return httpx.Response(
                    200,
                    json={
                        "model": "qwen2.5:3b",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({
                                "answerable": True,
                                "sql": "SELECT non_existent_column FROM sales",
                                "reasoning": "Invalid col",
                            }),
                        },
                        "done": True,
                    },
                )
            else:
                messages = body.get("messages", [])
                retry_prompt = messages[1].get("content", "") if len(messages) > 1 else prompt_str
                return httpx.Response(
                    200,
                    json={
                        "model": "qwen2.5:3b",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({
                                "answerable": True,
                                "sql": "SELECT region, COUNT(*) AS count FROM sales GROUP BY region",
                                "reasoning": "Corrected",
                            }),
                        },
                        "done": True,
                    },
                )

        return httpx.Response(200, json={"model": "qwen2.5:3b", "message": {"role": "assistant", "content": "Done."}})

    mock_client = OllamaClient(transport=httpx.MockTransport(mock_handler))
    result = await run_sql_agent(
        user_query="Count orders by region",
        datastore=store,
        client=mock_client,
    )

    assert result.status == "success"
    assert result.attempts == 2
    assert "These columns do not exist and must not be used again: ['non_existent_column']." in retry_prompt


@pytest.mark.asyncio
async def test_run_sql_agent_stops_retrying_on_repeated_failure(tmp_path: Path) -> None:
    """Test that if model fails on attempt 2 with an invalid column, it stops immediately instead of wasting all 4 attempts."""
    store = Datastore(db_path=tmp_path / "stop_repeat.duckdb")
    store.load_csv(Path("sample_data/sales.csv").read_bytes(), "sales.csv")

    attempts_seen = 0

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts_seen
        body = json.loads(request.content)
        prompt_str = str(body.get("messages", []))

        if "relevant_tables" in prompt_str:
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5:3b",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"relevant_tables": [{"table": "sales", "columns": ["region"]}]}),
                    },
                    "done": True,
                },
            )

        if "DuckDB SQL data engineer" in prompt_str:
            attempts_seen += 1
            bad_col = f"bogus_column_{attempts_seen}"
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5:3b",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "answerable": True,
                            "sql": f"SELECT {bad_col} FROM sales",
                            "reasoning": "Wrong again",
                        }),
                    },
                    "done": True,
                },
            )

        return httpx.Response(200, json={"model": "qwen2.5:3b", "message": {"role": "assistant", "content": "ok"}})

    mock_client = OllamaClient(transport=httpx.MockTransport(mock_handler))
    result = await run_sql_agent(
        user_query="Count orders by region",
        datastore=store,
        client=mock_client,
    )

    assert result.status == "not_in_data"
    assert result.attempts == 2
    assert attempts_seen == 2

