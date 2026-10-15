"""Query router, concurrent agent orchestrator, and Server-Sent Events stream generator.

Trade-off: JSON routing with keyword fallback prevents agent stalls when small 3B models misformat.
SSE streaming delivers sub-second first-token latency and granular step progress to the frontend.
"""
import asyncio
from collections.abc import AsyncIterator
import json
import re
from typing import Any, Literal
from pydantic import BaseModel, Field

from backend.app.agents.doc_agent import DocAgentResult, run_doc_agent
from backend.app.agents.sql_agent import SQLAgentResult, run_sql_agent
from backend.app.llm.client import ChatMessage, OllamaClient
from backend.app.observability.session import SessionManager
from backend.app.rag.store import SQLiteDocStore
from backend.app.tools.datastore import Datastore

RouteType = Literal["sql", "docs", "both", "chitchat"]


class ClassificationResult(BaseModel):
    """Route classification decision with model reasoning."""

    route: RouteType = Field(..., description="Target agent route: sql | docs | both | chitchat")
    reasoning: str = Field(default="", description="Classification explanation")
    sql_question: str | None = Field(
        default=None,
        description="Decomposed sub-question for SQL agent when route is 'both'",
    )
    doc_question: str | None = Field(
        default=None,
        description="Decomposed sub-question for Document agent when route is 'both'",
    )


def get_column_list(datastore: Datastore | None) -> list[str]:
    """Retrieve all column names from datasets currently loaded in the datastore."""
    if not datastore:
        return []
    columns: list[str] = []
    try:
        datasets = datastore.list_datasets()
        for d in datasets:
            schema = datastore.get_schema(d.table_name)
            for c in schema.columns:
                name = c.name if len(datasets) == 1 else f"{d.table_name}.{c.name}"
                if name not in columns:
                    columns.append(name)
    except Exception:
        pass
    return columns


def format_sse(event: str, data: dict[str, Any]) -> str:
    """Format dictionary payload into valid SSE event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def keyword_fallback_classifier(query: str) -> ClassificationResult:
    """Keyword-based heuristic fallback if LLM classification fails."""
    q = query.lower()
    sql_keywords = {
        "sales", "revenue", "order", "orders", "total", "count", "sum", "avg",
        "average", "dataset", "table", "rows", "columns", "highest", "lowest",
        "max", "min", "top", "trend", "discount", "customer", "customers", "price",
        "units", "quantity", "profit",
    }
    doc_keywords = {
        "policy", "policies", "doc", "docs", "document", "documents", "pdf",
        "manual", "guide", "guideline", "guidelines", "handbook", "procedure",
        "rule", "rules", "clause", "terms", "condition", "conditions", "overview",
    }
    chitchat_keywords = {
        "hello", "hi", "hey", "greetings", "good morning", "good evening",
        "how are you", "who are you", "what can you do", "help", "thanks",
        "thank you", "bye", "goodbye",
    }

    words = set(re.findall(r"\w+", q))
    has_sql = bool(words & sql_keywords)
    has_doc = bool(words & doc_keywords)

    if has_sql and has_doc:
        return ClassificationResult(
            route="both",
            reasoning="Fallback identified both SQL and document keywords.",
        )
    if has_sql:
        return ClassificationResult(
            route="sql",
            reasoning="Fallback identified quantitative SQL keywords.",
        )
    if has_doc:
        return ClassificationResult(
            route="docs",
            reasoning="Fallback identified qualitative document keywords.",
        )
    if any(q.startswith(cw) for cw in chitchat_keywords) or (words & chitchat_keywords):
        return ClassificationResult(
            route="chitchat",
            reasoning="Fallback identified conversational chitchat keywords.",
        )

    return ClassificationResult(
        route="chitchat",
        reasoning="Fallback defaulted to chitchat for unstructured greeting.",
    )


async def classify_query(
    query: str,
    client: OllamaClient,
    history: list[ChatMessage] | None = None,
    column_list: list[str] | str | None = None,
    datastore: Datastore | None = None,
) -> ClassificationResult:
    """Classify the incoming question as sql | docs | both | chitchat using LLM in JSON mode."""
    if column_list is not None:
        if isinstance(column_list, list):
            cols_str = ", ".join(column_list)
        else:
            cols_str = str(column_list)
    else:
        cols = get_column_list(datastore)
        cols_str = ", ".join(cols) if cols else "available database columns"

    system_prompt = (
        "You are an intent classification router for an analytical assistant. "
        "Classify the user's prompt into exactly one of these 4 categories:\n"
        "- 'sql': questions querying structured datasets, numbers, counts, sums, revenue, metrics, or table data.\n"
        "- 'docs': questions querying written documents, manuals, policies, procedures, or PDFs.\n"
        "- 'both': questions requiring combining quantitative table numbers AND qualitative document policies.\n"
        "- 'chitchat': greetings, pleasantries, questions about yourself, or general conversation.\n\n"
        f"If the route is \"both\", split the question. sql_question must be answerable\n"
        f"using ONLY these columns: {cols_str}. Anything about policies, timelines,\n"
        "reasons, or explanations goes in doc_question. Return JSON only.\n\n"
        'Respond ONLY with valid JSON: {"route": "sql" | "docs" | "both" | "chitchat", "reasoning": "...", "sql_question": "...", "doc_question": "..."}'
    )

    messages = [ChatMessage(role="system", content=system_prompt)]
    if history:
        messages.extend(history[-4:])
    messages.append(ChatMessage(role="user", content=query))

    try:
        response = await client.chat(messages, json_mode=True)
        parsed = json.loads(response.content.strip())
        route = parsed.get("route", "").lower().strip()
        if route in ("sql", "docs", "both", "chitchat"):
            sql_q = parsed.get("sql_question")
            doc_q = parsed.get("doc_question")
            sql_question = sql_q.strip() if isinstance(sql_q, str) and sql_q.strip() else None
            doc_question = doc_q.strip() if isinstance(doc_q, str) and doc_q.strip() else None
            return ClassificationResult(
                route=route,  # type: ignore[arg-type]
                reasoning=parsed.get("reasoning", "Classified by LLM."),
                sql_question=sql_question,
                doc_question=doc_question,
            )
    except Exception:
        pass

    return keyword_fallback_classifier(query)


async def synthesize_both(
    query: str,
    sql_result: SQLAgentResult | None,
    doc_result: DocAgentResult | None,
    client: OllamaClient,
    sql_error: str | None = None,
    doc_error: str | None = None,
    sql_question: str | None = None,
    doc_question: str | None = None,
) -> str:
    """Synthesize results from concurrent SQL and Document agents, explicitly labeling source claims."""
    system_prompt = (
        "You are a senior data synthesis analyst. Answer the user's question by combining both the "
        "structured data findings and the document findings.\n"
        "MANDATORY REQUIREMENTS:\n"
        "- Explicitly label the provenance of every factual claim:\n"
        "  * For numbers, metrics, or findings from the database, cite them as [Data: {table}] or [Data].\n"
        "  * For facts or policies from documents, cite them using their exact citation [doc p.N].\n"
        "- If one source (e.g. database table or documents) does not contain the requested fields or encountered an error, "
        "clearly state that limitation while answering fully using the available source.\n"
        "- Never fabricate data that was reported as unavailable or missing from the table.\n"
        "- Present a unified, coherent synthesis."
    )

    if sql_result and sql_result.status == "success":
        data_summary = (
            f"SQL Query: {sql_result.sql}\n"
            f"Data Rows: {json.dumps(sql_result.rows[:5], default=str)}\n"
            f"Data Explanation: {sql_result.explanation}"
        )
    elif sql_result and sql_result.status == "not_in_data":
        missing_clause = f" ('{sql_result.missing}')" if sql_result.missing else ""
        data_summary = (
            f"Structured Data Not Available: The requested numbers/metrics{missing_clause} are not present in the dataset.\n"
            "MANDATORY: Rely on the documents for the answer and state that the numbers aren't in the dataset."
        )
    elif sql_error:
        data_summary = (
            f"Structured Data Query Unsuccessful: {sql_error}\n"
            "Note: The requested quantitative data or columns are not available in the database tables."
        )
    else:
        data_summary = "Structured Data: Not available in the dataset."

    if doc_result:
        doc_summary = (
            f"Document Answer: {doc_result.answer}\n"
            f"Document Citations: {doc_result.citations}"
        )
    elif doc_error:
        doc_summary = (
            f"Document Retrieval Unsuccessful: {doc_error}\n"
            "Note: Relevant document excerpts could not be retrieved."
        )
    else:
        doc_summary = "Document Knowledge: Not available."

    sub_q_section = ""
    if sql_question or doc_question:
        sub_q_section = (
            f"Decomposed Sub-questions:\n"
            f"- Structured Data Question: {sql_question or query}\n"
            f"- Document Knowledge Question: {doc_question or query}\n\n"
        )

    user_content = (
        f"User Question: {query}\n\n"
        f"{sub_q_section}"
        f"Structured Data Results:\n{data_summary}\n\n"
        f"Document Knowledge Results:\n{doc_summary}"
    )

    response = await client.chat(
        [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_content),
        ]
    )
    return response.content.strip()


async def stream_chat_events(
    query: str,
    session_id: str,
    client: OllamaClient | None = None,
    datastore: Datastore | None = None,
    doc_store: SQLiteDocStore | None = None,
    session_mgr: SessionManager | None = None,
) -> AsyncIterator[str]:
    """Execute routed workflows and yield typed Server-Sent Events."""
    llm = client or OllamaClient()
    dstore = datastore or Datastore()
    docs = doc_store or SQLiteDocStore()
    sm = session_mgr or SessionManager()

    # Retrieve history and record user message
    history = sm.get_history(session_id)
    sm.add_message(session_id, "user", query)

    # 1. Route classification
    yield format_sse("step", {"step": "Classifying query intent", "status": "running"})
    classification = await classify_query(query, llm, history, datastore=dstore)
    yield format_sse("route", classification.model_dump())
    yield format_sse("step", {"step": "Classifying query intent", "status": "done"})

    final_answer = ""

    try:
        if classification.route == "chitchat":
            yield format_sse("step", {"step": "Generating response", "status": "running"})
            chat_messages = [
                ChatMessage(
                    role="system",
                    content="You are AgenticInsight, a friendly local-first AI assistant running on a Mac.",
                )
            ]
            chat_messages.extend(history)
            chat_messages.append(ChatMessage(role="user", content=query))

            tokens: list[str] = []
            async for tok in llm.stream(chat_messages):
                tokens.append(tok)
                yield format_sse("token", {"delta": tok})

            final_answer = "".join(tokens)
            yield format_sse("step", {"step": "Generating response", "status": "done"})
            yield format_sse("final", {"answer": final_answer, "route": "chitchat"})

        elif classification.route == "sql":
            yield format_sse("step", {"step": "Analyzing database schemas and executing query", "status": "running"})
            try:
                sql_res = await run_sql_agent(query, datastore=dstore, client=llm)

                if sql_res.status == "not_in_data":
                    doc_res = None
                    try:
                        doc_res = await run_doc_agent(query, store=docs, client=llm)
                    except Exception:
                        pass

                    if doc_res and not doc_res.refused and doc_res.chunks:
                        yield format_sse("hits", {"chunks": [c.model_dump() for c in doc_res.chunks]})
                        missing_text = f" ('{sql_res.missing}')" if sql_res.missing else ""
                        fallback_answer = (
                            f"{doc_res.answer}\n\n"
                            f"*Note: The requested data{missing_text} is not in the database table. "
                            "The information above was retrieved from your documents.*"
                        )
                        for word in fallback_answer.split(" "):
                            yield format_sse("token", {"delta": word + " "})
                            await asyncio.sleep(0.01)
                        final_answer = fallback_answer
                        yield format_sse("step", {"step": "Analyzing database schemas and executing query", "status": "done"})
                        yield format_sse(
                            "final",
                            {
                                "answer": final_answer,
                                "route": "docs",
                                "citations": doc_res.citations,
                                "status": "not_in_data",
                                "missing": sql_res.missing,
                            },
                        )
                    else:
                        missing_text = f" '{sql_res.missing}'" if sql_res.missing else ""
                        not_found_msg = f"The requested data{missing_text} is not available in the database dataset."
                        for word in not_found_msg.split(" "):
                            yield format_sse("token", {"delta": word + " "})
                            await asyncio.sleep(0.01)
                        final_answer = not_found_msg
                        yield format_sse("step", {"step": "Analyzing database schemas and executing query", "status": "done"})
                        yield format_sse(
                            "final",
                            {
                                "answer": final_answer,
                                "route": "sql",
                                "status": "not_in_data",
                                "missing": sql_res.missing,
                            },
                        )
                else:
                    yield format_sse("sql", {"sql": sql_res.sql, "lineage": sql_res.lineage.model_dump()})
                    yield format_sse(
                        "rows",
                        {
                            "columns": sql_res.columns,
                            "rows": sql_res.rows,
                            "chart_spec": sql_res.chart_spec.model_dump(),
                        },
                    )

                    # Stream explanation tokens
                    for word in sql_res.explanation.split(" "):
                        yield format_sse("token", {"delta": word + " "})
                        await asyncio.sleep(0.01)

                    final_answer = sql_res.explanation
                    yield format_sse("step", {"step": "Analyzing database schemas and executing query", "status": "done"})
                    yield format_sse(
                        "final",
                        {
                            "answer": final_answer,
                            "route": "sql",
                            "sql": sql_res.sql,
                            "columns": sql_res.columns,
                            "rows": sql_res.rows,
                            "chart_spec": sql_res.chart_spec.model_dump(),
                            "lineage": sql_res.lineage.model_dump(),
                        },
                    )
            except Exception as sql_err:
                # If SQL fails, check if the question can be answered by loaded documents (e.g. policy mixture or misclassified)
                doc_res = None
                try:
                    doc_res = await run_doc_agent(query, store=docs, client=llm)
                except Exception:
                    pass

                if doc_res and not doc_res.refused and doc_res.chunks:
                    yield format_sse("hits", {"chunks": [c.model_dump() for c in doc_res.chunks]})
                    fallback_answer = (
                        f"{doc_res.answer}\n\n"
                        f"*Note: Database query could not be completed ({str(sql_err).splitlines()[0]}). "
                        "The information above was retrieved from your documents.*"
                    )
                    for word in fallback_answer.split(" "):
                        yield format_sse("token", {"delta": word + " "})
                        await asyncio.sleep(0.01)
                    final_answer = fallback_answer
                    yield format_sse("step", {"step": "Analyzing database schemas and executing query", "status": "done"})
                    yield format_sse(
                        "final",
                        {
                            "answer": final_answer,
                            "route": "docs",
                            "citations": doc_res.citations,
                        },
                    )
                else:
                    raise sql_err

        elif classification.route == "docs":
            yield format_sse("step", {"step": "Searching documents and synthesizing answer", "status": "running"})
            doc_res = await run_doc_agent(query, store=docs, client=llm)

            yield format_sse(
                "hits",
                {"chunks": [c.model_dump() for c in doc_res.chunks]},
            )

            for word in doc_res.answer.split(" "):
                yield format_sse("token", {"delta": word + " "})
                await asyncio.sleep(0.01)

            final_answer = doc_res.answer
            yield format_sse("step", {"step": "Searching documents and synthesizing answer", "status": "done"})
            yield format_sse(
                "final",
                {
                    "answer": final_answer,
                    "route": "docs",
                    "citations": doc_res.citations,
                    "best_score": doc_res.best_score,
                    "refused": doc_res.refused,
                },
            )

        elif classification.route == "both":
            yield format_sse(
                "step",
                {"step": "Running SQL Agent and Document Agent concurrently", "status": "running"},
            )

            # Concurrent execution with decomposed sub-questions
            sql_query = classification.sql_question or query
            doc_query = classification.doc_question or query

            sql_task = run_sql_agent(sql_query, datastore=dstore, client=llm)
            doc_task = run_doc_agent(doc_query, store=docs, client=llm)
            results = await asyncio.gather(sql_task, doc_task, return_exceptions=True)

            raw_sql_res, raw_doc_res = results[0], results[1]
            sql_res = raw_sql_res if isinstance(raw_sql_res, SQLAgentResult) else None
            sql_error = str(raw_sql_res) if isinstance(raw_sql_res, Exception) else None
            doc_res = raw_doc_res if isinstance(raw_doc_res, DocAgentResult) else None
            doc_error = str(raw_doc_res) if isinstance(raw_doc_res, Exception) else None

            # Only raise if both agents failed completely
            if sql_res is None and doc_res is None:
                raise RuntimeError(
                    f"Both SQL and Document agents failed: SQL error: {sql_error}; Doc error: {doc_error}"
                )

            if sql_res and sql_res.status == "success":
                yield format_sse("sql", {"sql": sql_res.sql, "lineage": sql_res.lineage.model_dump()})
                yield format_sse(
                    "rows",
                    {
                        "columns": sql_res.columns,
                        "rows": sql_res.rows,
                        "chart_spec": sql_res.chart_spec.model_dump(),
                    },
                )

            if doc_res:
                yield format_sse(
                    "hits",
                    {"chunks": [c.model_dump() for c in doc_res.chunks]},
                )

            yield format_sse(
                "step",
                {"step": "Running SQL Agent and Document Agent concurrently", "status": "done"},
            )

            yield format_sse("step", {"step": "Synthesizing combined answer", "status": "running"})
            synthesized = await synthesize_both(
                query=query,
                sql_result=sql_res,
                doc_result=doc_res,
                client=llm,
                sql_error=sql_error,
                doc_error=doc_error,
                sql_question=classification.sql_question,
                doc_question=classification.doc_question,
            )

            for word in synthesized.split(" "):
                yield format_sse("token", {"delta": word + " "})
                await asyncio.sleep(0.01)

            final_answer = synthesized
            yield format_sse("step", {"step": "Synthesizing combined answer", "status": "done"})
            yield format_sse(
                "final",
                {
                    "answer": final_answer,
                    "route": "both",
                    "sql": sql_res.sql if sql_res and sql_res.status == "success" else None,
                    "columns": sql_res.columns if sql_res and sql_res.status == "success" else [],
                    "rows": sql_res.rows if sql_res and sql_res.status == "success" else [],
                    "chart_spec": sql_res.chart_spec.model_dump() if sql_res and sql_res.status == "success" else None,
                    "lineage": sql_res.lineage.model_dump() if sql_res and sql_res.status == "success" else None,
                    "citations": doc_res.citations if doc_res else [],
                    "sql_question": classification.sql_question,
                    "doc_question": classification.doc_question,
                    "sql_status": sql_res.status if sql_res else "error",
                    "sql_missing": sql_res.missing if sql_res else None,
                },
            )

        # Save assistant answer to session history
        if final_answer:
            sm.add_message(session_id, "assistant", final_answer)

    except asyncio.CancelledError:
        # Client disconnected during processing
        raise
    except Exception as exc:
        yield format_sse("error", {"message": str(exc)})
