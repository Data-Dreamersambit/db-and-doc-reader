"""Unit and integration tests for query router, concurrent agents, and SSE chat streaming.

Trade-off: In-process ASGI SSE streaming tests typed events without running background web servers.
Sliding window tests verify strict 6-turn SQLite retention bounds.
"""
from pathlib import Path
import json
import httpx
import pytest

from backend.app.agents.doc_agent import DocAgentResult
from backend.app.agents.router import (
    classify_query,
    format_sse,
    keyword_fallback_classifier,
    synthesize_both,
)
from backend.app.agents.sql_agent import ChartSpec, Lineage, SQLAgentResult
from backend.app.llm.client import OllamaClient
from backend.app.main import (
    app,
    get_datastore,
    get_doc_store,
    get_ollama_client,
    get_session_manager,
)
from backend.app.observability.session import SessionManager
from backend.app.rag.chunker import DocumentChunk
from backend.app.rag.retriever import ChunkMatch
from backend.app.rag.store import SQLiteDocStore
from backend.app.tools.datastore import Datastore


def test_format_sse() -> None:
    """Verify SSE line formatting convention."""
    msg = format_sse("route", {"route": "sql"})
    assert msg == 'event: route\ndata: {"route": "sql"}\n\n'


def test_keyword_fallback_classifier() -> None:
    """Test heuristic keyword fallback for all 4 routing paths."""
    assert keyword_fallback_classifier("What is the total revenue by region?").route == "sql"
    assert keyword_fallback_classifier("Explain the company travel policy document").route == "docs"
    assert keyword_fallback_classifier("Compare quarterly sales against the bonus policy guidelines").route == "both"
    assert keyword_fallback_classifier("Hello! Good morning").route == "chitchat"


@pytest.mark.asyncio
async def test_classify_query_llm() -> None:
    """Test LLM-based query routing in JSON mode."""
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "model": "qwen2.5:3b",
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"route": "sql", "reasoning": "Involves aggregation over orders"}),
                },
                "done": True,
            },
        )

    client = OllamaClient(transport=httpx.MockTransport(mock_handler))
    res = await classify_query("Sum of orders by month", client)
    assert res.route == "sql"
    assert "aggregation" in res.reasoning


def test_session_manager_sliding_window(tmp_path: Path) -> None:
    """Verify session manager retains strictly the latest 6 turns (12 messages)."""
    db_file = tmp_path / "sessions.sqlite"
    sm = SessionManager(db_path=db_file)
    sid = "sess_001"

    # Add 8 turns (16 messages)
    for i in range(1, 9):
        sm.add_message(sid, "user", f"User question {i}")
        sm.add_message(sid, "assistant", f"Assistant answer {i}")

    history = sm.get_history(sid)
    assert len(history) == 12  # 6 turns * 2
    # Verify the oldest messages (1 and 2) were pruned
    assert history[0].content == "User question 3"
    assert history[-1].content == "Assistant answer 8"


@pytest.mark.asyncio
async def test_synthesize_both() -> None:
    """Test synthesis prompt combining SQL metrics and document policies."""
    async def mock_synth_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt_content = str(body["messages"])
        assert "MANDATORY REQUIREMENT" in prompt_content
        return httpx.Response(
            status_code=200,
            json={
                "model": "qwen2.5:3b",
                "message": {
                    "role": "assistant",
                    "content": "Total sales in North were $1,200 [Data: sales], adhering to discount cap [policy.pdf p.2].",
                },
                "done": True,
            },
        )

    client = OllamaClient(transport=httpx.MockTransport(mock_synth_handler))

    sql_res = SQLAgentResult(
        sql="SELECT SUM(revenue) FROM sales",
        columns=["revenue"],
        rows=[{"revenue": 1200}],
        explanation="Total sales in North were $1200.",
        chart_spec=ChartSpec(chart_type="table"),
        lineage=Lineage(tables=["sales"], columns=["revenue"]),
        attempts=1,
    )
    doc_res = DocAgentResult(
        answer="The discount policy specifies a maximum cap of 15% [policy.pdf p.2].",
        citations=["[policy.pdf p.2]"],
        chunks=[
            ChunkMatch(
                chunk_id="c1",
                doc_id="d1",
                doc_name="policy.pdf",
                page_number=2,
                text="Discount policy cap 15%",
                cosine_score=0.88,
                bm25_score=1.2,
                rrf_score=0.03,
            )
        ],
        best_score=0.88,
        refused=False,
    )

    synth_text = await synthesize_both("Check sales and policy", sql_res, doc_res, client)
    assert "[Data: sales]" in synth_text
    assert "[policy.pdf p.2]" in synth_text


@pytest.mark.asyncio
async def test_chat_sse_chitchat(tmp_path: Path) -> None:
    """Test POST /api/chat SSE streaming with chitchat route."""
    sm = SessionManager(db_path=tmp_path / "chat_sess.sqlite")

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        url_path = request.url.path
        body = json.loads(request.content)
        if body.get("stream"):
            # Stream tokens
            lines = [
                json.dumps({"message": {"content": "Hello! "}, "done": False}) + "\n",
                json.dumps({"message": {"content": "How can I help you?"}, "done": True}) + "\n",
            ]
            return httpx.Response(200, content="".join(lines).encode("utf-8"))
        # Non-streaming classification
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5:3b",
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"route": "chitchat", "reasoning": "Greeting"}),
                },
                "done": True,
            },
        )

    client = OllamaClient(transport=httpx.MockTransport(mock_handler))
    app.dependency_overrides[get_ollama_client] = lambda: client
    app.dependency_overrides[get_session_manager] = lambda: sm

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            res = await ac.post("/api/chat", json={"query": "Hello there", "session_id": "test_s1"})
            assert res.status_code == 200
            assert "text/event-stream" in res.headers["content-type"]

            content = res.text
            assert "event: route" in content
            assert "event: step" in content
            assert "event: token" in content
            assert "event: final" in content
            assert "How can I help you?" in content

            # Verify saved turns in SQLite
            history = sm.get_history("test_s1")
            assert len(history) == 2
            assert history[0].role == "user"
            assert history[1].role == "assistant"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_chat_sse_sql_route(tmp_path: Path) -> None:
    """Test POST /api/chat SSE streaming for sql route with typed events."""
    db_duck = tmp_path / "duck.duckdb"
    store = Datastore(db_path=db_duck)
    store.load_csv(Path("sample_data/sales.csv").read_bytes(), "sales.csv")

    sm = SessionManager(db_path=tmp_path / "sql_sess.sqlite")

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt_str = str(body.get("messages", []))

        # Classification
        if "intent classification router" in prompt_str:
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5:3b",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"route": "sql", "reasoning": "Asks for total revenue"}),
                    },
                    "done": True,
                },
            )

        # SQL schema prune
        if "relevant_tables" in prompt_str:
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5:3b",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"relevant_tables": [{"table": "sales", "columns": ["total_revenue"]}]}),
                    },
                    "done": True,
                },
            )

        # SQL generation
        if "DuckDB SQL data engineer" in prompt_str:
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5:3b",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "sql": "SELECT SUM(total_revenue) AS total_revenue FROM sales",
                            "reasoning": "Compute sum of total_revenue",
                        }),
                    },
                    "done": True,
                },
            )

        # Explanation
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5:3b",
                "message": {
                    "role": "assistant",
                    "content": "Total revenue is $5,431.27 across all completed orders.",
                },
                "done": True,
            },
        )

    client = OllamaClient(transport=httpx.MockTransport(mock_handler))
    app.dependency_overrides[get_ollama_client] = lambda: client
    app.dependency_overrides[get_datastore] = lambda: store
    app.dependency_overrides[get_session_manager] = lambda: sm

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            res = await ac.post("/api/chat", json={"query": "Total sales revenue", "session_id": "test_s2"})
            assert res.status_code == 200

            content = res.text
            assert "event: route" in content
            assert "event: sql" in content
            assert "event: rows" in content
            assert "event: token" in content
            assert "event: final" in content
            assert "total_revenue" in content
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_both_route_partial_sql_failure_synthesizes_gracefully(tmp_path: Path) -> None:
    """Verify that when SQL agent fails on an invalid column in a mixed question, the doc answer is still preserved."""
    from backend.app.agents.doc_agent import DocAgentResult
    from backend.app.agents.router import synthesize_both
    from backend.app.rag.retriever import ChunkMatch

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5:3b",
                "message": {
                    "role": "assistant",
                    "content": "Shipping delivery policy requires 3-5 days [policy.pdf p.1]. The sales database does not contain delivery estimates.",
                },
                "done": True,
            },
        )

    client = OllamaClient(transport=httpx.MockTransport(mock_handler))

    doc_res = DocAgentResult(
        answer="Standard shipping requires 3 to 5 business days [policy.pdf p.1].",
        citations=["[policy.pdf p.1]"],
        chunks=[
            ChunkMatch(
                chunk_id="c1",
                doc_id="d1",
                doc_name="policy.pdf",
                page_number=1,
                text="Standard shipping requires 3 to 5 business days.",
                cosine_score=0.85,
                bm25_score=1.5,
                rrf_score=0.03,
            )
        ],
        best_score=0.85,
        refused=False,
    )

    synth = await synthesize_both(
        query="What is the average delivery estimate by region, and what does policy say about shipping?",
        sql_result=None,
        doc_result=doc_res,
        client=client,
        sql_error="SQLAgentExecutionError: Column 'delivery_estimate' does not exist in table 'sales'.",
    )

    assert "[policy.pdf p.1]" in synth
    assert "delivery" in synth.lower()


@pytest.mark.asyncio
async def test_classify_query_both_decomposition(tmp_path: Path) -> None:
    """Test router decomposes 'both' questions into sql_question and doc_question with column_list."""
    from backend.app.agents.router import get_column_list
    captured_prompt = ""

    db_duck = tmp_path / "duck_cols.duckdb"
    store = Datastore(db_path=db_duck)
    store.load_csv(Path("sample_data/sales.csv").read_bytes(), "sales.csv")

    cols = get_column_list(store)
    assert "total_revenue" in cols
    assert "region" in cols

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_prompt
        body = json.loads(request.content)
        captured_prompt = str(body["messages"])
        return httpx.Response(
            status_code=200,
            json={
                "model": "qwen2.5:3b",
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "route": "both",
                        "sql_question": "Average order value and order count by region for completed orders",
                        "doc_question": "What is the estimated delivery time per region?",
                    }),
                },
                "done": True,
            },
        )

    client = OllamaClient(transport=httpx.MockTransport(mock_handler))
    res = await classify_query(
        "Average order value and order count by region, plus delivery times",
        client,
        datastore=store,
    )

    assert res.route == "both"
    assert res.sql_question == "Average order value and order count by region for completed orders"
    assert res.doc_question == "What is the estimated delivery time per region?"
    assert 'If the route is "both", split the question. sql_question must be answerable' in captured_prompt
    assert "total_revenue" in captured_prompt


@pytest.mark.asyncio
async def test_chat_sse_both_decomposed_agents(tmp_path: Path) -> None:
    """Verify stream_chat_events dispatches sql_question to SQL agent and doc_question to doc agent."""
    from backend.app.agents.router import stream_chat_events
    from backend.app.rag.chunker import DocumentChunk

    db_duck = tmp_path / "both_duck.duckdb"
    store = Datastore(db_path=db_duck)
    store.load_csv(Path("sample_data/sales.csv").read_bytes(), "sales.csv")

    doc_store = SQLiteDocStore(db_path=tmp_path / "both_docs.sqlite")
    doc_store.save_document(
        doc_id="d1",
        filename="shipping.pdf",
        page_count=1,
        chunks=[
            DocumentChunk(
                chunk_id="c1",
                doc_id="d1",
                doc_name="shipping.pdf",
                page_number=1,
                text="Delivery time is 3-5 business days across all regions.",
                token_count=10,
            )
        ],
        embeddings=[[0.1] * 768],
    )

    sm = SessionManager(db_path=tmp_path / "both_sess.sqlite")

    dispatched_prompts: list[str] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        url_path = request.url.path
        body = json.loads(request.content)
        prompt_str = str(body.get("messages", []))
        dispatched_prompts.append(prompt_str)

        if "intent classification router" in prompt_str:
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5:3b",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "route": "both",
                            "sql_question": "Average order value and order count by region for completed orders",
                            "doc_question": "What is the estimated delivery time per region?",
                        }),
                    },
                    "done": True,
                },
            )

        # SQL schema prune
        if "relevant_tables" in prompt_str:
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5:3b",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"relevant_tables": [{"table": "sales", "columns": ["region", "total_revenue"]}]}),
                    },
                    "done": True,
                },
            )

        # SQL generation
        if "DuckDB SQL data engineer" in prompt_str:
            # Verify the SQL prompt received the decomposed sub-question!
            assert "Average order value and order count by region for completed orders" in prompt_str
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5:3b",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "sql": "SELECT region, AVG(total_revenue) AS avg_aov, COUNT(*) AS order_cnt FROM sales GROUP BY region",
                            "reasoning": "Compute AOV and count by region",
                        }),
                    },
                    "done": True,
                },
            )

        # SQL explanation
        if "data analyst explaining" in prompt_str:
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5:3b",
                    "message": {
                        "role": "assistant",
                        "content": "North region had average order value of $150 across 6 orders.",
                    },
                    "done": True,
                },
            )

        # Embedding call
        if "/api/embed" in url_path:
            return httpx.Response(200, json={"embeddings": [[0.1] * 768]})

        # Doc agent synthesis
        if "cited RAG assistant" in prompt_str:
            # Verify the Doc agent received the decomposed doc sub-question!
            assert "What is the estimated delivery time per region?" in prompt_str
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5:3b",
                    "message": {
                        "role": "assistant",
                        "content": "Estimated delivery time is 3-5 business days [shipping.pdf p.1].",
                    },
                    "done": True,
                },
            )

        # Synthesizer
        if "senior data synthesis analyst" in prompt_str:
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5:3b",
                    "message": {
                        "role": "assistant",
                        "content": "Orders averaged $150 [Data: sales] with delivery taking 3-5 days [shipping.pdf p.1].",
                    },
                    "done": True,
                },
            )

        return httpx.Response(200, json={"model": "qwen2.5:3b", "message": {"role": "assistant", "content": "ok"}})

    client = OllamaClient(transport=httpx.MockTransport(mock_handler))

    events: list[str] = []
    async for chunk in stream_chat_events(
        query="Combined question: Give me order stats and delivery times",
        session_id="both_test_sess",
        client=client,
        datastore=store,
        doc_store=doc_store,
        session_mgr=sm,
    ):
        events.append(chunk)

    full_output = "".join(events)
    assert 'event: route' in full_output
    assert '"route": "both"' in full_output
    assert 'Average order value and order count by region for completed orders' in full_output
    assert 'What is the estimated delivery time per region?' in full_output
    assert 'event: sql' in full_output
    assert 'event: hits' in full_output
    assert 'event: final' in full_output
    assert 'Orders averaged $150' in full_output


@pytest.mark.asyncio
async def test_both_route_unanswerable_sql_relies_on_documents(tmp_path: Path) -> None:
    """Verify that in 'both' mode, if SQL is unanswerable, synthesizer relies on documents and states numbers aren't in dataset."""
    from backend.app.agents.router import stream_chat_events
    from backend.app.rag.chunker import DocumentChunk

    db_duck = tmp_path / "both_unans_duck.duckdb"
    store = Datastore(db_path=db_duck)
    store.load_csv(Path("sample_data/sales.csv").read_bytes(), "sales.csv")

    doc_store = SQLiteDocStore(db_path=tmp_path / "both_unans_docs.sqlite")
    doc_store.save_document(
        doc_id="d1",
        filename="shipping.pdf",
        page_count=1,
        chunks=[
            DocumentChunk(
                chunk_id="c1",
                doc_id="d1",
                doc_name="shipping.pdf",
                page_number=1,
                text="Delivery time is 3-5 business days across all regions.",
                token_count=10,
            )
        ],
        embeddings=[[0.1] * 768],
    )

    sm = SessionManager(db_path=tmp_path / "both_unans_sess.sqlite")

    synth_prompt_captured = ""

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal synth_prompt_captured
        url_path = request.url.path
        body = json.loads(request.content)
        prompt_str = str(body.get("messages", []))

        # Router classification
        if "intent classification router" in prompt_str:
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5:3b",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "route": "both",
                            "sql_question": "What is the delivery estimate per region?",
                            "doc_question": "What does policy say about shipping timelines?",
                        }),
                    },
                    "done": True,
                },
            )

        # SQL schema pruner
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

        # SQL generator says UNANSWERABLE!
        if "DuckDB SQL data engineer" in prompt_str:
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

        # Embedding call
        if "/api/embed" in url_path:
            return httpx.Response(200, json={"embeddings": [[0.1] * 768]})

        # Doc agent synthesis
        if "cited RAG assistant" in prompt_str:
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5:3b",
                    "message": {
                        "role": "assistant",
                        "content": "Standard delivery takes 3-5 days [shipping.pdf p.1].",
                    },
                    "done": True,
                },
            )

        # Synthesizer
        if "senior data synthesis analyst" in prompt_str:
            synth_prompt_captured = prompt_str
            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5:3b",
                    "message": {
                        "role": "assistant",
                        "content": "Delivery estimates are not in the dataset. According to policy, shipping takes 3-5 days [shipping.pdf p.1].",
                    },
                    "done": True,
                },
            )

        return httpx.Response(200, json={"model": "qwen2.5:3b", "message": {"role": "assistant", "content": "ok"}})

    client = OllamaClient(transport=httpx.MockTransport(mock_handler))

    events: list[str] = []
    async for chunk in stream_chat_events(
        query="What is the delivery estimate and policy?",
        session_id="both_unans_test_sess",
        client=client,
        datastore=store,
        doc_store=doc_store,
        session_mgr=sm,
    ):
        events.append(chunk)

    full_output = "".join(events)
    assert 'event: route' in full_output
    # No SQL was executed because it exited early with not_in_data
    assert 'event: sql' not in full_output
    assert 'event: hits' in full_output
    assert 'event: final' in full_output
    assert "Delivery estimates are not in the dataset" in full_output

    # Verify synthesizer was prompted with structured data not in dataset instruction
    assert "Structured Data Not Available" in synth_prompt_captured
    assert "delivery_estimate" in synth_prompt_captured
    assert "state that the numbers aren't in the dataset" in synth_prompt_captured



