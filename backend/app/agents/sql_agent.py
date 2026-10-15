"""SQL Agent for schema pruning, safe SQL generation, execution, and chart spec inference.

Trade-off: AST inspection via sqlglot stops injection and file access before hitting DuckDB.
Rule-based chart heuristics avoid extra LLM latency for standard aggregations.
"""
from collections.abc import Sequence
import difflib
import json
import re
from typing import Any
from pydantic import BaseModel, Field
import sqlglot
import sqlglot.expressions as exp

from backend.app.config import get_settings
from backend.app.llm.client import ChatMessage, OllamaClient
from backend.app.tools.datastore import DatasetSchema, Datastore

FORBIDDEN_FUNCTIONS: set[str] = {
    "read_csv", "read_csv_auto", "read_parquet", "parquet_scan",
    "read_json", "read_json_auto", "read_text", "read_blob",
    "scan_csv", "scan_parquet", "scan_json", "glob",
    "install", "load", "copy", "duckdb_settings", "duckdb_secrets",
    "system", "shell", "getenv", "current_setting",
}

FORBIDDEN_EXPRESSIONS = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create,
    exp.Alter, exp.Copy, exp.Command, exp.Transaction, exp.Into,
    exp.Pragma, exp.ReadCSV, exp.ReadParquet,
)


class Lineage(BaseModel):
    """Lineage metadata showing queried tables and referenced columns."""

    tables: list[str] = Field(default_factory=list, description="Referenced table names")
    columns: list[str] = Field(default_factory=list, description="Referenced column names")


class ChartSpec(BaseModel):
    """Visualization configuration specification for frontend rendering."""

    chart_type: str = Field(..., description="Chart visualization type: bar | line | pie | table")
    x_field: str | None = Field(default=None, description="Dimension or temporal field on X-axis")
    y_fields: list[str] = Field(default_factory=list, description="Metric fields on Y-axis")
    title: str | None = Field(default=None, description="Descriptive chart title")


class SQLAgentResult(BaseModel):
    """Complete execution payload returned by the SQL Agent."""

    status: str = Field(default="success", description="Execution status: 'success' | 'not_in_data'")
    missing: str | None = Field(default=None, description="Missing column or data name if unanswerable")
    sql: str = Field(default="", description="Executed and validated DuckDB SQL query")
    columns: list[str] = Field(default_factory=list, description="Result column names")
    rows: list[dict[str, Any]] = Field(default_factory=list, description="Result rows")
    explanation: str = Field(default="", description="Plain-language explanation of findings")
    chart_spec: ChartSpec = Field(default_factory=lambda: ChartSpec(chart_type="table"), description="Inferred chart specification")
    lineage: Lineage = Field(default_factory=Lineage, description="Referenced database objects")
    attempts: int = Field(default=1, description="Number of generation/execution attempts")

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self):
        return self.model_dump().keys()

    def values(self):
        return self.model_dump().values()

    def items(self):
        return self.model_dump().items()


class SQLGenerationResult(BaseModel):
    """Internal result payload from SQL generation step."""

    answerable: bool = True
    sql: str = ""
    reasoning: str = ""
    missing: str | None = None


class SQLValidationError(ValueError):
    """Raised when an SQL query fails syntax, single SELECT, or safety validation."""


class SQLAgentExecutionError(RuntimeError):
    """Raised when the agent fails to produce a valid, executable SQL query within retry limits."""


def validate_columns_against_schema(
    stmt: exp.Select,
    schemas: Sequence[DatasetSchema],
) -> None:
    """Validate that all referenced tables and columns exist in the provided schemas."""
    if not schemas:
        return

    table_schema_map: dict[str, list[str]] = {
        s.table_name.lower(): [c.name for c in s.columns]
        for s in schemas
    }
    table_schema_map_lower: dict[str, set[str]] = {
        tbl: {c.lower() for c in cols}
        for tbl, cols in table_schema_map.items()
    }

    # CTE names defined within query
    cte_names = {
        cte.alias_or_name.lower()
        for cte in stmt.find_all(exp.CTE)
        if cte.alias_or_name
    }

    # Validate physical tables referenced
    tables_in_query = [
        t.name.lower()
        for t in stmt.find_all(exp.Table)
        if t.name and t.name.lower() not in cte_names
    ]
    for tbl in tables_in_query:
        if tbl not in table_schema_map_lower:
            available_tables = ", ".join(repr(s.table_name) for s in schemas)
            raise SQLValidationError(
                f"Table '{tbl}' does not exist in the database. Available tables are: {available_tables}."
            )

    # Collect aliases defined in query (e.g., SELECT col AS alias, or GROUP BY alias)
    aliases = {a.alias.lower() for a in stmt.find_all(exp.Alias) if a.alias}

    # Determine valid column names from active tables (or all schemas if table omitted)
    active_tables = [tbl for tbl in tables_in_query if tbl in table_schema_map_lower]
    if active_tables:
        valid_cols_lower: set[str] = set().union(*(table_schema_map_lower[t] for t in active_tables))
        sample_table = active_tables[0]
        available_columns = ", ".join(repr(c) for c in table_schema_map[sample_table])
    else:
        valid_cols_lower = set().union(*table_schema_map_lower.values())
        first_table = list(table_schema_map.keys())[0]
        available_columns = ", ".join(repr(c) for c in table_schema_map[first_table])

    # Check referenced columns
    for c in stmt.find_all(exp.Column):
        col_name = c.name
        if not col_name or col_name == "*":
            continue
        col_lower = col_name.lower()
        if col_lower in aliases:
            continue

        if c.table and c.table.lower() not in cte_names:
            c_tbl = c.table.lower()
            if c_tbl in table_schema_map_lower and col_lower not in table_schema_map_lower[c_tbl]:
                cols_str = ", ".join(repr(x) for x in table_schema_map[c_tbl])
                raise SQLValidationError(
                    f"Column '{col_name}' does not exist in table '{c.table}'. "
                    f"Available columns are: {cols_str}. Only use columns that exist in the schema."
                )
        elif col_lower not in valid_cols_lower:
            table_desc = f"table '{active_tables[0]}'" if len(active_tables) == 1 else "available tables"
            raise SQLValidationError(
                f"Column '{col_name}' does not exist in {table_desc}. "
                f"Available columns are: {available_columns}. Only use columns that exist in the schema."
            )


def validate_and_format_sql(
    raw_sql: str,
    schemas: Sequence[DatasetSchema] | None = None,
) -> tuple[str, Lineage]:
    """Parse raw SQL with sqlglot, enforce safety rules, inject LIMIT 1000, and extract lineage."""
    cleaned = raw_sql.strip().rstrip(";")
    try:
        parsed_statements = sqlglot.parse(cleaned, read="duckdb")
    except Exception as e:
        raise SQLValidationError(f"SQL syntax parse error: {e}") from e

    if not parsed_statements:
        raise SQLValidationError("Empty SQL query provided.")

    if len(parsed_statements) > 1:
        raise SQLValidationError("Multiple SQL statements are strictly prohibited.")

    stmt = parsed_statements[0]
    if not isinstance(stmt, exp.Select):
        raise SQLValidationError(
            f"Only single SELECT statements are allowed; received {type(stmt).__name__}."
        )

    # Walk AST to detect forbidden expressions, commands, and functions
    for node in stmt.walk():
        if isinstance(node, FORBIDDEN_EXPRESSIONS):
            raise SQLValidationError(
                f"Forbidden statement or expression type detected: {type(node).__name__}."
            )
        if isinstance(node, (exp.Func, exp.Anonymous)):
            fname = (node.name or type(node).__name__).lower()
            if fname in FORBIDDEN_FUNCTIONS:
                raise SQLValidationError(
                    f"Forbidden function '{fname}' detected for security reasons."
                )

    # Validate against known database schemas if provided
    if schemas:
        validate_columns_against_schema(stmt, schemas)

    # Enforce LIMIT 1000
    existing_limit = stmt.args.get("limit")
    if existing_limit is None:
        stmt = stmt.limit(1000)
    else:
        try:
            val = int(str(existing_limit.expression.this))
            if val > 1000:
                stmt.set("limit", exp.Limit(expression=exp.Literal.number(1000)))
        except (ValueError, AttributeError):
            stmt.set("limit", exp.Limit(expression=exp.Literal.number(1000)))

    # Extract lineage
    tables = list(dict.fromkeys(t.name.lower() for t in stmt.find_all(exp.Table) if t.name))
    columns = list(dict.fromkeys(c.name.lower() for c in stmt.find_all(exp.Column) if c.name))

    return stmt.sql(dialect="duckdb"), Lineage(tables=tables, columns=columns)


def extract_referenced_columns(sql_or_stmt: str | exp.Expression) -> list[str]:
    """Extract referenced column names from an SQL string or sqlglot Expression."""
    if isinstance(sql_or_stmt, str):
        try:
            stmt = sqlglot.parse_one(sql_or_stmt.strip().rstrip(";"), read="duckdb")
        except Exception:
            return []
    else:
        stmt = sql_or_stmt

    if not isinstance(stmt, exp.Expression):
        return []

    cte_names = {
        cte.alias_or_name.lower()
        for cte in stmt.find_all(exp.CTE)
        if cte.alias_or_name
    }
    aliases = {a.alias.lower() for a in stmt.find_all(exp.Alias) if a.alias}
    cols: list[str] = []
    for c in stmt.find_all(exp.Column):
        name = c.name
        if not name or name == "*":
            continue
        if name.lower() in aliases:
            continue
        if c.table and c.table.lower() in cte_names:
            continue
        if name not in cols:
            cols.append(name)
    return cols


def repair_columns(
    sql_cols_used: Sequence[str],
    valid_cols: Sequence[str],
) -> dict[str, str]:
    """Identify invalid columns and auto-match against schema using cutoff=0.75."""
    valid_cols_list = list(valid_cols)
    valid_cols_lower = {c.lower() for c in valid_cols_list}
    fixes = {}
    for c in sql_cols_used:
        if c not in valid_cols_list and c.lower() not in valid_cols_lower:
            m = difflib.get_close_matches(c, valid_cols_list, n=1, cutoff=0.75)
            if not m:
                m_lower = difflib.get_close_matches(
                    c.lower(),
                    [v.lower() for v in valid_cols_list],
                    n=1,
                    cutoff=0.75,
                )
                if m_lower:
                    matched_lower = m_lower[0]
                    for orig in valid_cols_list:
                        if orig.lower() == matched_lower:
                            m = [orig]
                            break
            if m:
                fixes[c] = m[0]
    return fixes   # empty means no close match, so stop retrying


def _is_column_requested_in_query(col_name: str, query: str) -> bool:
    """Check if the column or its constituent words were mentioned in the user query."""
    q_lower = query.lower()
    c_lower = col_name.lower()
    if c_lower in q_lower:
        return True
    tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", c_lower) if len(t) >= 4]
    for t in tokens:
        if t in q_lower:
            return True
        if len(t) >= 5 and (t[:5] in q_lower or any(word.startswith(t[:5]) for word in q_lower.split())):
            return True
    return False


async def prune_schema(
    user_query: str,
    schemas: Sequence[DatasetSchema],
    client: OllamaClient,
) -> str:
    """Step 1: Pick relevant tables and columns via LLM JSON mode."""
    if not schemas:
        raise SQLAgentExecutionError("No datasets are currently loaded in DuckDB.")

    schema_catalog = [
        {
            "table": s.table_name,
            "columns": [{"name": c.name, "type": c.type} for c in s.columns],
            "low_cardinality_values": s.low_cardinality_values,
        }
        for s in schemas
    ]

    # If only one small table exists, format directly without extra LLM overhead
    if len(schemas) == 1 and len(schemas[0].columns) <= 12:
        return json.dumps(schema_catalog, indent=2)

    prompt = (
        "You are a database schema pruner. Given the user question and the available table schemas, "
        "identify which tables and columns are necessary to answer the question.\n"
        "Return a JSON object with this structure:\n"
        '{"relevant_tables": [{"table": "table_name", "columns": ["col1", "col2"]}]}\n\n'
        f"Available Schemas:\n{json.dumps(schema_catalog, indent=2)}\n\n"
        f"User Question: {user_query}"
    )

    try:
        response = await client.chat(
            [ChatMessage(role="user", content=prompt)],
            json_mode=True,
        )
        parsed = json.loads(response.content)
        relevant = parsed.get("relevant_tables", [])
        if relevant:
            return json.dumps(relevant, indent=2)
    except Exception:
        pass

    return json.dumps(schema_catalog, indent=2)


async def generate_sql(
    user_query: str,
    pruned_schema_info: str,
    client: OllamaClient,
    error_feedback: str | None = None,
    invalid_columns: list[str] | None = None,
) -> SQLGenerationResult:
    """Step 2: Generate DuckDB SQL using a few-shot prompt in JSON mode."""
    few_shot_examples = (
        "Example 1:\n"
        "Question: What is total revenue by region?\n"
        'Output: {"answerable": true, "sql": "SELECT region, SUM(total_revenue) AS total_revenue FROM sales GROUP BY region ORDER BY total_revenue DESC", "reasoning": "Aggregate total_revenue grouped by region and sorted descending."}\n\n'
        "Example 2:\n"
        "Question: How many orders in North region used PayPal?\n"
        'Output: {"answerable": true, "sql": "SELECT COUNT(*) AS paypal_order_count FROM sales WHERE region = \'North\' AND payment_method = \'PayPal\'", "reasoning": "Count rows filtering on region and payment_method."}\n\n'
        "Example 3:\n"
        "Question: What is the estimated delivery time for North region orders?\n"
        'Output: {"answerable": false, "missing": "delivery_estimate"}'
    )

    system_message = (
        "You are an expert DuckDB SQL data engineer. Generate a valid, executable read-only SELECT query.\n"
        "Rules:\n"
        "- Generate ONLY a single SELECT statement when answerable.\n"
        "- Never use DDL or DML (no DROP, INSERT, DELETE, UPDATE, ALTER, CREATE).\n"
        "- Never call file or system functions (no read_csv, read_parquet, etc.).\n"
        "- Query ONLY from tables listed in Schema Context (e.g. 'sales'). NEVER query from non-existent tables like 'orders' or 'customers'.\n"
        "- Use exact table and column names matching the provided schema.\n"
        "- If the question needs data not in the listed columns, set answerable to false. Never guess column names.\n"
        "- When filtering text columns, match exact case or use ILIKE (e.g. product_category ILIKE 'Electronics', region ILIKE 'North') to ensure case-insensitive matching.\n"
        "- Output strictly in JSON:\n"
        '  {"answerable": true, "sql": "...", "reasoning": "..."}\n'
        "  or\n"
        '  {"answerable": false, "missing": "column_or_metric_name"}'
    )

    user_content = (
        f"{few_shot_examples}\n\n"
        f"Schema Context:\n{pruned_schema_info}\n\n"
        f"Question: {user_query}"
    )

    if error_feedback:
        user_content += (
            f"\n\nCRITICAL FIX: Your previous query failed with this error:\n{error_feedback}\n"
            "Please diagnose the schema, ensure table and column names strictly match only the available fields in Schema Context, and generate a corrected SQL query."
        )

    if invalid_columns and (not error_feedback or "These columns do not exist and must not be used again" not in error_feedback):
        user_content += f"\nThese columns do not exist and must not be used again: {invalid_columns}."

    response = await client.chat(
        [
            ChatMessage(role="system", content=system_message),
            ChatMessage(role="user", content=user_content),
        ],
        json_mode=True,
        options={"temperature": 0.0},
    )

    content = response.content.strip()
    try:
        parsed = json.loads(content)
        answerable = parsed.get("answerable")
        if answerable is False or (answerable is None and "missing" in parsed and "sql" not in parsed):
            missing_val = parsed.get("missing", "requested_fields")
            return SQLGenerationResult(
                answerable=False,
                missing=str(missing_val),
                reasoning=parsed.get("reasoning", ""),
            )

        sql = parsed.get("sql", "").strip()
        reasoning = parsed.get("reasoning", "")
    except json.JSONDecodeError:
        # Fallback regex extraction if json_mode output was loosely formatted
        if re.search(r'"answerable"\s*:\s*false', content, re.IGNORECASE):
            missing_match = re.search(r'"missing"\s*:\s*"([^"]+)"', content)
            missing = missing_match.group(1) if missing_match else "requested_fields"
            return SQLGenerationResult(answerable=False, missing=missing)

        match = re.search(r'"sql"\s*:\s*"([^"]+)"', content)
        if match:
            sql = match.group(1).replace(r"\"", '"')
            reasoning = "Extracted from raw model response."
        else:
            sql = content
            reasoning = "Unstructured response."

    # Strip any accidental markdown formatting
    sql = re.sub(r"^```sql\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"^```\s*", "", sql)
    sql = re.sub(r"\s*```$", "", sql).strip()

    return SQLGenerationResult(answerable=True, sql=sql, reasoning=reasoning)


async def explain_results(
    user_query: str,
    sql: str,
    rows: list[dict[str, Any]],
    client: OllamaClient,
) -> str:
    """Step 6: Generate a concise plain-language answer explaining the query results."""
    if not rows:
        return "The query completed successfully, but returned no matching records."

    sample_context = rows[:10]
    prompt = (
        "You are a concise data analyst. Given the user's question, the SQL query executed, and the results, "
        "provide a factual 1-3 sentence plain-language summary answering the question directly.\n"
        "Do not output markdown tables or repeat the SQL query.\n\n"
        f"User Question: {user_query}\n"
        f"Executed SQL: {sql}\n"
        f"Query Results (first 10 rows): {json.dumps(sample_context, default=str)}\n"
        f"Total Rows: {len(rows)}"
    )

    try:
        response = await client.chat([ChatMessage(role="user", content=prompt)])
        return response.content.strip()
    except Exception:
        return f"Query returned {len(rows)} record(s) matching your request."


def _is_date_field(field_name: str, values: list[Any]) -> bool:
    """Check if a field name or sampled values represent dates or timestamps."""
    lower_name = field_name.lower()
    if any(k in lower_name for k in ("date", "month", "year", "day", "time")):
        return True
    date_regex = re.compile(r"^\d{4}-\d{2}(-\d{2})?")
    for v in values[:5]:
        if v is not None and date_regex.match(str(v)):
            return True
    return False


def _is_numeric_field(field_name: str, values: list[Any]) -> bool:
    """Check if a field is numeric and not an identifier."""
    lower_name = field_name.lower()
    if lower_name.endswith("_id") or lower_name == "id":
        return False
    valid_numbers = [v for v in values if v is not None]
    if not valid_numbers:
        return False
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in valid_numbers)


async def infer_chart_spec(
    user_query: str,
    columns: list[str],
    rows: list[dict[str, Any]],
    client: OllamaClient,
) -> ChartSpec:
    """Step 7: Choose chart type and axes via rule-based heuristics first, LLM if ambiguous."""
    if not rows or len(columns) <= 1 or len(rows) == 1:
        return ChartSpec(chart_type="table", title="Tabular View")

    date_cols: list[str] = []
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in columns:
        vals = [r.get(col) for r in rows]
        if _is_date_field(col, vals):
            date_cols.append(col)
        elif _is_numeric_field(col, vals):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    # Heuristic 1: Temporal trend -> line chart
    if date_cols and numeric_cols:
        return ChartSpec(
            chart_type="line",
            x_field=date_cols[0],
            y_fields=numeric_cols[:3],
            title=f"{', '.join(numeric_cols[:3])} over Time",
        )

    # Heuristic 2: Composition / share words -> pie chart
    query_lower = user_query.lower()
    is_share_query = any(w in query_lower for w in ("share", "proportion", "percentage", "distribution", "breakdown"))
    if is_share_query and categorical_cols and numeric_cols:
        unique_cats = len({r.get(categorical_cols[0]) for r in rows})
        if unique_cats <= 7:
            return ChartSpec(
                chart_type="pie",
                x_field=categorical_cols[0],
                y_fields=[numeric_cols[0]],
                title=f"Distribution of {numeric_cols[0]} by {categorical_cols[0]}",
            )

    # Heuristic 3: 1 categorical + 1+ numeric metrics -> bar chart
    if categorical_cols and numeric_cols:
        unique_cats = len({r.get(categorical_cols[0]) for r in rows})
        if unique_cats <= 30:
            return ChartSpec(
                chart_type="bar",
                x_field=categorical_cols[0],
                y_fields=numeric_cols[:3],
                title=f"{', '.join(numeric_cols[:3])} by {categorical_cols[0]}",
            )

    # Ambiguous shape -> consult LLM with fallback to table
    prompt = (
        "Given the query, columns, and sample data, recommend the best chart visualization.\n"
        "Options: 'bar', 'line', 'pie', 'table'.\n"
        "Return strictly JSON: {\"chart_type\": \"...\", \"x_field\": \"...\", \"y_fields\": [\"...\"], \"title\": \"...\"}\n\n"
        f"Query: {user_query}\n"
        f"Columns: {columns}\n"
        f"Sample Row: {json.dumps(rows[0], default=str)}"
    )

    try:
        response = await client.chat([ChatMessage(role="user", content=prompt)], json_mode=True)
        parsed = json.loads(response.content)
        ctype = parsed.get("chart_type", "table").lower()
        if ctype in ("bar", "line", "pie", "table"):
            return ChartSpec(
                chart_type=ctype,
                x_field=parsed.get("x_field"),
                y_fields=parsed.get("y_fields", []),
                title=parsed.get("title", f"{ctype.title()} Chart"),
            )
    except Exception:
        pass

    return ChartSpec(chart_type="table", title="Tabular View")


async def run_sql_agent(
    user_query: str,
    datastore: Datastore | None = None,
    client: OllamaClient | None = None,
) -> SQLAgentResult:
    """Execute SQL Agent workflow: prune -> generate -> validate -> execute (retry max 3) -> explain -> chart."""
    store = datastore or Datastore()
    llm = client or OllamaClient()

    # Step 1: Prune schema
    all_datasets = store.list_datasets()
    if not all_datasets:
        raise SQLAgentExecutionError("No datasets found in datastore. Please upload a dataset first.")

    schemas: list[DatasetSchema] = [store.get_schema(d.table_name) for d in all_datasets]
    pruned_schema_info = await prune_schema(user_query, schemas, llm)

    valid_cols: list[str] = [c.name for s in schemas for c in s.columns]
    valid_cols_lower: set[str] = {c.lower() for c in valid_cols}
    invalid_columns_seen: list[str] = []

    # Retry loop: max 3 retries (total up to 4 attempts)
    max_retries = 3
    attempts = 0
    error_feedback: str | None = None
    last_sql = ""
    validated_sql = ""
    lineage = Lineage()
    rows: list[dict[str, Any]] = []

    while attempts <= max_retries:
        attempts += 1
        try:
            # Step 2: Generate SQL
            gen_res = await generate_sql(
                user_query,
                pruned_schema_info,
                llm,
                error_feedback=error_feedback,
                invalid_columns=invalid_columns_seen if invalid_columns_seen else None,
            )
            if not gen_res.answerable:
                return SQLAgentResult(
                    status="not_in_data",
                    missing=gen_res.missing,
                    explanation=f"Requested data '{gen_res.missing}' is not available in the database tables.",
                    attempts=attempts,
                )

            raw_sql = gen_res.sql
            last_sql = raw_sql

            # Step 2b: Repair columns if close matches exist
            sql_cols = extract_referenced_columns(raw_sql)
            fixes = repair_columns(sql_cols, valid_cols)
            if fixes:
                for old_col, new_col in fixes.items():
                    raw_sql = re.sub(rf"\b{re.escape(old_col)}\b", new_col, raw_sql, flags=re.IGNORECASE)
                last_sql = raw_sql
                sql_cols = extract_referenced_columns(raw_sql)

            # Check if there are remaining invalid columns with no close match
            unmatched_cols = [
                c for c in sql_cols
                if c not in valid_cols and c.lower() not in valid_cols_lower
            ]
            if unmatched_cols:
                first_unmatched = unmatched_cols[0]
                if (
                    attempts > 1
                    or first_unmatched in invalid_columns_seen
                    or _is_column_requested_in_query(first_unmatched, user_query)
                ):
                    return SQLAgentResult(
                        status="not_in_data",
                        missing=first_unmatched,
                        explanation=f"Requested data '{first_unmatched}' is not available in the database tables.",
                        attempts=attempts,
                    )

                for c in unmatched_cols:
                    if c not in invalid_columns_seen:
                        invalid_columns_seen.append(c)

                sample_table = schemas[0].table_name if schemas else "sales"
                available_columns = ", ".join(repr(c) for c in valid_cols[:15])
                raise SQLValidationError(
                    f"Column '{first_unmatched}' does not exist in table '{sample_table}'. "
                    f"Available columns are: {available_columns}. Only use columns that exist in the schema."
                )

            # Step 3: Validate SQL with sqlglot and verify against schema
            validated_sql, lineage = validate_and_format_sql(raw_sql, schemas=schemas)

            # Step 4: Execute on read-only DuckDB connection
            rows = store.execute_query(validated_sql)
            break
        except Exception as exc:
            m_err = re.search(r"Column '([^']+)' does not exist", str(exc))
            if m_err:
                col_name = m_err.group(1)
                if col_name not in invalid_columns_seen:
                    invalid_columns_seen.append(col_name)

            all_cols = []
            for s in schemas:
                all_cols.extend([f"{s.table_name}.{c.name}" for c in s.columns])
            avail_cols_str = ", ".join(all_cols[:20])
            error_feedback = (
                f"{type(exc).__name__}: {str(exc)}\n"
                f"Valid available columns in database: {avail_cols_str}. Only use existing columns."
            )
            if invalid_columns_seen:
                error_feedback += f"\nThese columns do not exist and must not be used again: {invalid_columns_seen}."

            if attempts > max_retries:
                raise SQLAgentExecutionError(
                    f"SQL Agent failed after {attempts} attempts. Last error: {error_feedback}. SQL: {last_sql}"
                ) from exc

    # Step 6: Explain results
    columns = list(rows[0].keys()) if rows else []
    explanation = await explain_results(user_query, validated_sql, rows, llm)

    # Step 7: Chart specification
    chart_spec = await infer_chart_spec(user_query, columns, rows, llm)

    return SQLAgentResult(
        status="success",
        sql=validated_sql,
        columns=columns,
        rows=rows,
        explanation=explanation,
        chart_spec=chart_spec,
        lineage=lineage,
        attempts=attempts,
    )
