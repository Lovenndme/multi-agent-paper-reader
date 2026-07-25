"""Tests for LangGraph as the single Agent dependency workflow."""

from __future__ import annotations

from types import SimpleNamespace
from collections import defaultdict
from threading import Lock
from unittest.mock import patch

from core.agent_harness import AgentRunContext
from core.analysis_orchestrator import build_demo_outputs
from core.analysis_progress import AnalysisProgressTracker
from core.evidence import EvidenceSnippet
from core.graph import evidence_node, run_pipeline_with_state
from core.pdf_parser import ParsedPaper, Section
from core.schemas import (
    CriticOutput,
    EvidenceSupervisorOutput,
    EvidenceRepairTask,
    ExperimentOutput,
    MethodOutput,
    SummaryOutput,
)


def test_graph_reuses_prebuilt_evidence_and_forwards_run_context() -> None:
    paper = ParsedPaper(
        title="Graph Paper",
        full_text="method experiment",
        sections=[Section("Abstract", "method experiment", 0, 0)],
    )
    snippets = [
        EvidenceSnippet(
            id="E001",
            section="Abstract",
            page_start=0,
            page_end=0,
            text="method experiment",
        )
    ]
    raw = build_demo_outputs(paper, snippets)
    outputs = {
        "method": MethodOutput.model_validate(raw["method_output"]),
        "experiment": ExperimentOutput.model_validate(raw["experiment_output"]),
        "critic": CriticOutput.model_validate(raw["critic_output"]),
        "summary": SummaryOutput.model_validate(raw["summary_output"]),
        "evidence_supervisor": EvidenceSupervisorOutput(
            sufficient=True,
            coverage_score=100,
            summary="证据覆盖充分。",
        ),
    }
    contexts: list[AgentRunContext] = []

    class FakeHarness:
        def run(self, spec, context=None, *, input_data=None):
            contexts.append(context)
            return SimpleNamespace(output=outputs[spec.agent_id])

    context = AgentRunContext(
        tracker=AnalysisProgressTracker(),
        stream=True,
    )
    with (
        patch("core.graph.get_agent_harness", return_value=FakeHarness()),
        patch(
            "core.graph.build_evidence_index",
            side_effect=AssertionError("prebuilt evidence should be reused"),
        ),
    ):
        result = run_pipeline_with_state(
            paper,
            evidence_index=snippets,
            agent_context=context,
        )

    assert result["evidence_index"] == snippets
    assert result["summary_output"].one_sentence_summary
    assert len(contexts) == 5
    assert all(run_context.stream for run_context in contexts)
    assert all(run_context.paper is paper for run_context in contexts)
    assert all(run_context.snippets == snippets for run_context in contexts)


def test_graph_preserves_an_explicit_empty_evidence_index() -> None:
    paper = ParsedPaper(
        title="Empty Evidence Paper",
        full_text="",
        sections=[],
    )

    with patch(
        "core.graph.build_evidence_index",
        side_effect=AssertionError("explicit empty evidence should be preserved"),
    ):
        assert evidence_node(
            {
                "parsed_paper": paper,
                "evidence_index": [],
            }
        )["evidence_index"] == []


def test_graph_runs_one_targeted_repair_before_summary() -> None:
    paper = ParsedPaper(
        title="Repair Paper",
        full_text="method experiment limitation",
        sections=[Section("Abstract", "method experiment limitation", 0, 0)],
    )
    snippets = [
        EvidenceSnippet(
            id="E001",
            section="Abstract",
            page_start=0,
            page_end=0,
            text="method experiment limitation",
        )
    ]
    raw = build_demo_outputs(paper, snippets)
    valid_method = MethodOutput.model_validate(raw["method_output"])
    missing_method = valid_method.model_copy(update={"evidence": []})
    outputs = {
        "experiment": ExperimentOutput.model_validate(raw["experiment_output"]),
        "critic": CriticOutput.model_validate(raw["critic_output"]),
        "summary": SummaryOutput.model_validate(raw["summary_output"]),
    }
    calls: defaultdict[str, int] = defaultdict(int)
    lock = Lock()

    class RepairHarness:
        def run(self, spec, context=None, *, input_data=None):
            with lock:
                calls[spec.agent_id] += 1
                count = calls[spec.agent_id]
            if spec.agent_id == "method":
                return SimpleNamespace(
                    output=missing_method if count == 1 else valid_method,
                    retrieval=None,
                )
            if spec.agent_id == "evidence_supervisor":
                return SimpleNamespace(
                    output=EvidenceSupervisorOutput(
                        sufficient=count > 1,
                        coverage_score=100 if count > 1 else 67,
                        summary="覆盖检查。",
                        repair_tasks=(
                            []
                            if count > 1
                            else [
                                EvidenceRepairTask(
                                    agent="method",
                                    missing_facets=["architecture"],
                                    suggested_queries=["method architecture"],
                                    reason="方法证据缺失。",
                                )
                            ]
                        ),
                    ),
                    retrieval=None,
                )
            return SimpleNamespace(output=outputs[spec.agent_id], retrieval=None)

    with patch("core.graph.get_agent_harness", return_value=RepairHarness()):
        result = run_pipeline_with_state(
            paper,
            evidence_index=snippets,
            agent_context=AgentRunContext(),
        )

    assert calls["method"] == 2
    assert calls["evidence_supervisor"] == 2
    assert calls["summary"] == 1
    assert result["repair_count"] == 1
    assert result["evidence_supervisor"].sufficient is True
