"""Tests for LangGraph as the single Agent dependency workflow."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from core.agent_harness import AgentRunContext
from core.analysis_orchestrator import build_demo_outputs
from core.analysis_progress import AnalysisProgressTracker
from core.evidence import EvidenceSnippet
from core.graph import evidence_node, run_pipeline_with_state
from core.pdf_parser import ParsedPaper, Section
from core.schemas import (
    CriticOutput,
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
    assert len(contexts) == 4
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
