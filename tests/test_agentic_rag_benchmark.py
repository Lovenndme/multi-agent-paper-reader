"""Deterministic scorer and runner tests for the Agentic RAG benchmark harness."""

from types import SimpleNamespace
from unittest.mock import patch

from core.agentic_types import AgenticRagConfig
from core.evidence import EvidenceSnippet
from tools.benchmark_agentic_rag import (
    aggregate_scores,
    anchored_retrieval_metrics,
    merge_cases,
    retrieval_metrics,
)
from tools.run_agentic_rag_benchmark import (
    hydrate_gold,
    prewarm_retrieval_baselines,
    serialize_retrieved,
)
from tools.run_pipeline_benchmark import summarize_pipeline_state


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
            "AQ": {
                "retrieved_evidence_ids": ["T001"],
                "supported_facts": [],
                "tool_calls": 0,
                "successful_tool_calls": 0,
                "latency_ms": 12,
            },
            "A4": {
                "retrieved_evidence_ids": ["T001"],
                "supported_facts": ["Accuracy is 91.2%"],
                "tool_calls": 2,
                "successful_tool_calls": 2,
                "repair_success": True,
                "adaptive_triggered": True,
                "latency_ms": 30,
            },
        },
    }
    report = aggregate_scores([case], k=10)
    assert report["A1"]["recall@10"] == 0
    assert report["AQ"]["recall@10"] == 1
    assert report["A4"]["recall@10"] == 1
    assert report["A4"]["grounded_fact_f1"] == 1
    assert report["A4"]["tool_success_rate"] == 1
    assert report["A4"]["adaptive_trigger_rate"] == 1


def test_source_anchored_metrics_deduplicate_equivalent_snippets() -> None:
    gold = {"pp.4-4:quote-sha256:abc"}
    metrics = anchored_retrieval_metrics(
        [
            {
                "evidence_id": "E009",
                "source_anchor": "pp.4-4:text-sha256:same",
                "matched_gold_anchors": list(gold),
            },
            {
                "evidence_id": "E011",
                "source_anchor": "pp.4-4:text-sha256:same",
                "matched_gold_anchors": list(gold),
            },
        ],
        gold,
        k=16,
    )
    assert metrics["recall@16"] == 1
    assert metrics["precision@16"] == 1
    assert metrics["mrr"] == 1


def test_source_anchored_metrics_treat_alternative_anchors_as_one_fact() -> None:
    metrics = anchored_retrieval_metrics(
        [
            {
                "source_anchor": "retrieved-a",
                "matched_gold_anchors": ["quote-a2"],
            }
        ],
        {
            "quote-a1": "fact-a",
            "quote-a2": "fact-a",
        },
        k=5,
    )

    assert metrics["recall@5"] == 1
    assert metrics["precision@5"] == 1


def test_runner_hydrates_quotes_but_does_not_persist_raw_text() -> None:
    snippets = (
        EvidenceSnippet(
            "E001",
            "Method",
            3,
            3,
            "The model divides each attention logit by the square root of d-k.",
        ),
        EvidenceSnippet(
            "E002",
            "Duplicate section",
            3,
            3,
            "The model divides each attention logit by the square root of d-k.",
        ),
    )
    gold = hydrate_gold(
        [
            {
                "page": 4,
                "quote": (
                    "divides each attention logit by the square root of d-k"
                ),
            }
        ],
        snippets,
    )
    assert gold[0]["equivalent_evidence_ids"] == ["E001", "E002"]
    assert "quote" not in gold[0]

    retrieved = serialize_retrieved(snippets, gold)
    assert len(retrieved) == 1
    assert retrieved[0]["matched_gold_anchors"] == [gold[0]["source_anchor"]]


def test_runner_hydrates_alternative_quotes_under_one_fact() -> None:
    snippets = (
        EvidenceSnippet(
            "E001",
            "Method",
            0,
            0,
            "The primary explanation describes the same supported claim in detail.",
        ),
        EvidenceSnippet(
            "E002",
            "Method detail",
            1,
            1,
            "An alternative passage independently supports the same claim in detail.",
        ),
    )

    gold = hydrate_gold(
        [
            {
                "fact_id": "method-claim",
                "page": 1,
                "quote": "primary explanation describes the same supported claim",
                "alternatives": [
                    {
                        "page": 2,
                        "quote": "alternative passage independently supports the same claim",
                    }
                ],
            }
        ],
        snippets,
    )

    assert len(gold) == 2
    assert {item["fact_id"] for item in gold} == {"method-claim"}


def test_runner_prewarms_only_models_used_by_requested_baselines() -> None:
    snippets = (
        EvidenceSnippet("E001", "Method", 0, 0, "method evidence"),
    )
    with (
        patch(
            "tools.run_agentic_rag_benchmark.semantic_scores"
        ) as dense,
        patch(
            "tools.run_agentic_rag_benchmark.cross_encoder_scores"
        ) as reranker,
    ):
        prewarm_retrieval_baselines(snippets, ("AQL", "AQR"))

    dense.assert_called_once()
    reranker.assert_called_once()


def test_merge_cases_combines_disjoint_baseline_runs() -> None:
    base = {
        "case_id": "case-1",
        "paper_sha256": "a" * 64,
        "query": "query",
        "gold_evidence": [
            {"evidence_id": "E001", "source_anchor": "p1:quote-sha256:abc"}
        ],
    }
    merged = merge_cases(
        [
            [{**base, "runs": {"A0": {"retrieved_evidence_ids": []}}}],
            [{**base, "runs": {"A3": {"retrieved_evidence_ids": ["E001"]}}}],
        ]
    )
    assert set(merged[0]["runs"]) == {"A0", "A3"}


def test_pipeline_summary_records_adaptive_cost_without_model_prose() -> None:
    output = SimpleNamespace(
        evidence=[
            SimpleNamespace(id="E001"),
            SimpleNamespace(id="E999"),
        ]
    )
    state = {
        "method_output": output,
        "experiment_output": SimpleNamespace(evidence=[]),
        "critic_output": SimpleNamespace(evidence=[]),
        "summary_output": SimpleNamespace(evidence=[]),
        "evidence_index": [
            EvidenceSnippet("E001", "Method", 0, 0, "method evidence")
        ],
        "evidence_supervisor": SimpleNamespace(
            sufficient=False,
            coverage_score=67,
            warnings=["one gap"],
        ),
        "repair_count": 1,
        "retrieval_traces": [
            {
                "strategy": "adaptive_static",
                "steps": [],
                "adaptive_triggered": False,
                "fallback_used": False,
            },
            {
                "strategy": "native",
                "steps": [{"tool": "paper_search"}],
                "adaptive_triggered": True,
                "fallback_used": False,
            },
        ],
    }
    summary = summarize_pipeline_state(
        state,
        label="B2",
        latency_ms=123.4,
        config=AgenticRagConfig(mode="adaptive", planner_mode="fast"),
    )

    assert summary["adaptive_trigger_rate"] == 0.5
    assert summary["planner_steps"] == 1
    assert summary["valid_citation_rate"] == 0.5
    assert summary["planner_mode"] == "fast"
    assert "method evidence" not in repr(summary)
