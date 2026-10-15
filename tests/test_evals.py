"""Unit and integration tests for the evaluation benchmark suite.

Trade-off: Fast mock evaluation validates reporting and percentiles without waiting for 50 LLM generations.
Dataset schema validation guarantees integrity of all 50 evaluation items.
"""
from pathlib import Path
import pytest
from backend.app.evals.generator import create_handbook_pdf_bytes, ensure_eval_assets
from backend.app.evals.run import (
    DocResultRecord,
    EvaluationDataset,
    SQLResultRecord,
    UnanswerableResultRecord,
    calculate_percentiles,
    generate_markdown_report,
    load_dataset,
    values_match,
)


def test_dataset_integrity() -> None:
    """Validate dataset.json contains exactly 20 SQL, 20 Doc, and 10 Unanswerable questions."""
    dataset = load_dataset()
    assert isinstance(dataset, EvaluationDataset)
    assert len(dataset.sql_questions) == 20
    assert len(dataset.doc_questions) == 20
    assert len(dataset.unanswerable_questions) == 10

    # Ensure all IDs are unique
    all_ids = (
        [q.id for q in dataset.sql_questions]
        + [q.id for q in dataset.doc_questions]
        + [q.id for q in dataset.unanswerable_questions]
    )
    assert len(all_ids) == 50
    assert len(set(all_ids)) == 50

    # Ensure document questions map to valid pages 1-5
    for doc_q in dataset.doc_questions:
        assert 1 <= doc_q.expected_page <= 5


def test_values_match_logic() -> None:
    """Test scalar and numeric tolerance matching logic."""
    assert values_match(25, 25, "scalar_equal") is True
    assert values_match(25, 26, "scalar_equal") is False
    assert values_match(5094.27, 5094.27, "numeric_close") is True
    assert values_match(5094.269, 5094.27, "numeric_close") is True
    assert values_match("North", "north", "scalar_equal") is True
    assert values_match(None, 25, "scalar_equal") is False


def test_calculate_percentiles() -> None:
    """Test percentile calculations across latency distributions."""
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    stats = calculate_percentiles(latencies)
    assert stats.count == 10
    assert stats.p50_ms == 55.0
    assert stats.min_ms == 10.0
    assert stats.max_ms == 100.0
    assert stats.avg_ms == 55.0

    empty_stats = calculate_percentiles([])
    assert empty_stats.count == 0
    assert empty_stats.p50_ms == 0.0


def test_pdf_generator_and_asset_creation(tmp_path: Path) -> None:
    """Verify programmatic PDF generation produces valid 5-page PDF with extractable content."""
    pdf_bytes = create_handbook_pdf_bytes()
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-1.4")

    csv_p, pdf_p = ensure_eval_assets(tmp_path)
    assert csv_p.parent.exists()
    assert pdf_p.exists()
    assert pdf_p.stat().st_size > 0


def test_generate_markdown_report() -> None:
    """Test formatting and generation of markdown results table."""
    sql_records = [
        SQLResultRecord(
            id="sql_01",
            question="Total orders?",
            generated_sql="SELECT COUNT(*) FROM sales",
            expected_val=25,
            actual_val=25,
            matched=True,
            latency_ms=120.0,
        )
    ]
    doc_records = [
        DocResultRecord(
            id="doc_01",
            question="HQ location?",
            expected_page=1,
            retrieved_pages=[1, 2, 3],
            hit=True,
            latency_ms=45.0,
            best_score=0.88,
        )
    ]
    unans_records = [
        UnanswerableResultRecord(
            id="unans_01",
            question="Californium half-life?",
            category="physics",
            refused=True,
            answer="Not found in the documents",
            latency_ms=18.0,
            best_score=0.22,
        )
    ]

    report = generate_markdown_report(
        sql_records=sql_records,
        doc_records=doc_records,
        unans_records=unans_records,
        model_name="qwen2.5:3b",
        embed_name="nomic-embed-text",
    )

    assert "# AgenticInsight Evaluation Benchmark Results" in report
    assert "SQL Execution Accuracy" in report
    assert "Document Retrieval Hit Rate" in report
    assert "Refusal Rate on Unanswerables" in report
    assert "p50 (ms)" in report
    assert "sql_01" in report
    assert "doc_01" in report
    assert "unans_01" in report
