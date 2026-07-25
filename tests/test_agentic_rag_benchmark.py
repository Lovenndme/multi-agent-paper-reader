"""Deterministic scorer tests for the Agentic RAG benchmark harness."""

from tools.benchmark_agentic_rag import aggregate_scores, retrieval_metrics


def test_retrieval_metrics_use_rank_and_deduplicate_predictions() -> None:
    metrics = retrieval_metrics(
        ["E999", "E001", "E001", "T001"],
        {"E001", "T001"},
        k=3,
    )
    assert metrics["recall@3"] == 1.0
    assert metrics["precision@3"] == 2 / 3
    assert metrics["mrr"] == 0.5
    assert 0 < metrics["ndcg@3"] < 1


def test_aggregate_scores_keeps_baseline_scope_explicit() -> None:
    case = {
        "case_id": "case-1",
        "paper_sha256": "a" * 64,
        "query": "What is the main result?",
        "gold_evidence": [
            {
                "evidence_id": "T001",
                "source_anchor": "p5:quote-sha256:abc",
            }
        ],
        "gold_facts": ["Accuracy is 91.2%"],
        "runs": {
            "A1": {
                "retrieved_evidence_ids": ["E001"],
                "supported_facts": [],
                "tool_calls": 0,
                "successful_tool_calls": 0,
                "latency_ms": 10,
            },
            "A4": {
                "retrieved_evidence_ids": ["T001"],
                "supported_facts": ["Accuracy is 91.2%"],
                "tool_calls": 2,
                "successful_tool_calls": 2,
                "repair_success": True,
                "latency_ms": 30,
            },
        },
    }
    report = aggregate_scores([case], k=10)
    assert report["A1"]["recall@10"] == 0
    assert report["A4"]["recall@10"] == 1
    assert report["A4"]["grounded_fact_f1"] == 1
    assert report["A4"]["tool_success_rate"] == 1
