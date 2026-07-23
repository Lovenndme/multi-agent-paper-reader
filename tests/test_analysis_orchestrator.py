"""Tests for the complete paper-analysis task Orchestrator."""

from __future__ import annotations

from typing import Any

import pytest

from agents.critic_agent import CRITIC_AGENT_SPEC
from agents.experiment_agent import EXPERIMENT_AGENT_SPEC
from agents.method_agent import METHOD_AGENT_SPEC
from agents.summary_agent import SUMMARY_AGENT_SPEC
from core.analysis_events import (
    AnalysisOrchestratorError,
    AnalysisRequest,
    AnalysisStage,
)
from core.analysis_orchestrator import PaperAnalysisOrchestrator, build_demo_outputs
from core.evidence import build_evidence_index
from core.graph import GraphStageError
from core.pdf_parser import ParsedPaper, Section
from core.schemas import (
    AnalysisAssessment,
    CriticOutput,
    ExperimentOutput,
    MethodOutput,
    SummaryOutput,
)
from core.vision import VisionEnrichmentResult


def _paper() -> ParsedPaper:
    return ParsedPaper(
        title="Orchestrated Paper",
        full_text="A method and experimental result.",
        sections=[
            Section("Abstract", "A method and experimental result.", 0, 0),
            Section("Method", "The proposed architecture.", 1, 1),
        ],
    )


def _workflow(
    paper: ParsedPaper,
    *,
    evidence_index,
    agent_context,
) -> dict[str, Any]:
    raw = build_demo_outputs(paper, evidence_index)
    outputs = {
        "method": MethodOutput.model_validate(raw["method_output"]),
        "experiment": ExperimentOutput.model_validate(raw["experiment_output"]),
        "critic": CriticOutput.model_validate(raw["critic_output"]),
        "summary": SummaryOutput.model_validate(raw["summary_output"]),
    }
    for spec in (
        METHOD_AGENT_SPEC,
        EXPERIMENT_AGENT_SPEC,
        CRITIC_AGENT_SPEC,
        SUMMARY_AGENT_SPEC,
    ):
        started = agent_context.tracker.start_agent(spec.agent_id, spec.start_summary)
        agent_context.emit("agent_started", started)
        completed = agent_context.tracker.complete_agent(
            spec.agent_id,
            spec.complete_summary,
        )
        agent_context.emit(
            "agent_complete",
            {
                **completed,
                "output_key": spec.output_key,
                "output": outputs[spec.agent_id].model_dump(),
            },
        )
    return {
        "evidence_index": evidence_index,
        "method_output": outputs["method"],
        "experiment_output": outputs["experiment"],
        "critic_output": outputs["critic"],
        "summary_output": outputs["summary"],
        "assessment": AnalysisAssessment.model_validate(raw["assessment"]),
    }


def _orchestrator(**overrides) -> PaperAnalysisOrchestrator:
    defaults = {
        "parser": lambda _path: _paper(),
        "workflow_runner": _workflow,
        "vision_enricher": lambda _path, _paper: VisionEnrichmentResult(
            total_figures=0
        ),
        "session_store": lambda _snippets, _result: "analysis-1",
        "history_saver": lambda **_kwargs: "history-1",
        "manifest_builder": lambda paper: {"paper": {"title": paper.title}},
        "runtime_payload_builder": lambda: {
            "text_provider": "test",
            "text_model": "test-model",
        },
        "llm_configured": lambda: True,
        "provider_id": lambda: "openai",
        "configuration_message": lambda: "model is not configured",
    }
    defaults.update(overrides)
    return PaperAnalysisOrchestrator(**defaults)


def test_live_stream_owns_complete_task_and_public_event_contract() -> None:
    events = list(
        _orchestrator().stream(
            AnalysisRequest(
                filename="paper.pdf",
                pdf_data=b"%PDF-1.7 orchestrator",
            )
        )
    )
    payloads = [event.as_dict() for event in events]
    event_types = [event["type"] for event in payloads]

    assert event_types[0] == "analysis_started"
    assert event_types.count("agent_started") == 4
    assert event_types.count("agent_complete") == 4
    assert event_types[-1] == "complete"
    complete = payloads[-1]
    assert complete["mode"] == "live"
    assert complete["analysis_id"] == "analysis-1"
    assert complete["history_id"] == "history-1"
    assert complete["analysis_process"]["status"] == "completed"
    assert complete["analysis_process"]["agents"]["summary"]["status"] == "complete"
    assert complete["evidence_count"] > 0
    assert "evidence_index" not in complete
    assert "evidence" not in complete["method_output"]


def test_run_consumes_the_same_event_pipeline() -> None:
    result = _orchestrator().run(
        AnalysisRequest(
            filename="paper.pdf",
            pdf_data=b"%PDF-1.7 orchestrator",
        )
    )

    assert result.payload["mode"] == "live"
    assert result.payload["analysis_id"] == "analysis-1"
    assert result.payload["history_id"] == "history-1"


def test_parse_failure_becomes_typed_error_event_and_exception() -> None:
    orchestrator = _orchestrator(
        parser=lambda _path: (_ for _ in ()).throw(ValueError("broken PDF"))
    )
    request = AnalysisRequest(
        filename="paper.pdf",
        pdf_data=b"%PDF broken",
        demo=True,
    )

    events = [event.as_dict() for event in orchestrator.stream(request)]
    error = events[-1]

    assert error["type"] == "error"
    assert error["stage"] == AnalysisStage.PARSING.value
    assert error["category"] == "parse"
    assert error["analysis_process"]["status"] == "failed"
    with pytest.raises(AnalysisOrchestratorError) as captured:
        orchestrator.run(request)
    assert captured.value.stage == AnalysisStage.PARSING
    assert captured.value.category == "parse"


def test_vision_and_history_failures_are_soft_failures() -> None:
    orchestrator = _orchestrator(
        vision_enricher=lambda _path, _paper: (_ for _ in ()).throw(
            RuntimeError("vision unavailable")
        ),
        history_saver=lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("disk unavailable")
        ),
    )

    events = [
        event.as_dict()
        for event in orchestrator.stream(
            AnalysisRequest(
                filename="paper.pdf",
                pdf_data=b"%PDF-1.7 orchestrator",
            )
        )
    ]
    event_types = [event["type"] for event in events]
    complete = events[-1]

    assert "vision_error" in event_types
    assert "history_error" in event_types
    assert complete["type"] == "complete"
    assert complete["history_id"] is None
    assert "disk unavailable" in complete["history_warning"]


def test_chat_session_failure_does_not_discard_analysis() -> None:
    orchestrator = _orchestrator(
        session_store=lambda _snippets, _result: (_ for _ in ()).throw(
            RuntimeError("session unavailable")
        )
    )

    events = [
        event.as_dict()
        for event in orchestrator.stream(
            AnalysisRequest(
                filename="paper.pdf",
                pdf_data=b"%PDF-1.7 orchestrator",
            )
        )
    ]
    complete = events[-1]

    assert any(event["type"] == "session_error" for event in events)
    assert complete["type"] == "complete"
    assert complete["analysis_id"] is None
    assert complete["history_id"] == "history-1"
    assert "session unavailable" in complete["session_warning"]


def test_evidence_failure_is_attributed_to_evidence_stage() -> None:
    orchestrator = _orchestrator(
        evidence_builder=lambda _paper: (_ for _ in ()).throw(
            RuntimeError("index unavailable")
        )
    )

    events = [
        event.as_dict()
        for event in orchestrator.stream(
            AnalysisRequest(
                filename="paper.pdf",
                pdf_data=b"%PDF-1.7 orchestrator",
                demo=True,
            )
        )
    ]

    assert events[-1]["type"] == "error"
    assert events[-1]["stage"] == AnalysisStage.EVIDENCE.value
    assert events[-1]["category"] == "evidence"


def test_graph_assessment_failure_is_attributed_to_assessment_stage() -> None:
    orchestrator = _orchestrator(
        workflow_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            GraphStageError("assessment", RuntimeError("assessment unavailable"))
        )
    )

    events = [
        event.as_dict()
        for event in orchestrator.stream(
            AnalysisRequest(
                filename="paper.pdf",
                pdf_data=b"%PDF-1.7 orchestrator",
            )
        )
    ]

    assert events[-1]["type"] == "error"
    assert events[-1]["stage"] == AnalysisStage.ASSESSMENT.value
    assert events[-1]["category"] == "assessment"
    assert events[-1]["message"] == "assessment unavailable"


def test_configuration_is_rejected_before_execution() -> None:
    orchestrator = _orchestrator(llm_configured=lambda: False)
    request = AnalysisRequest(filename="paper.pdf", pdf_data=b"%PDF")

    with pytest.raises(AnalysisOrchestratorError) as captured:
        orchestrator.validate(request)

    assert captured.value.stage == AnalysisStage.PREPARING
    assert captured.value.category == "configuration"
    assert captured.value.message == "model is not configured"


def test_rejects_filename_that_can_escape_temporary_directory() -> None:
    orchestrator = _orchestrator()

    with pytest.raises(AnalysisOrchestratorError) as captured:
        orchestrator.validate(
            AnalysisRequest(
                filename="../paper.pdf",
                pdf_data=b"%PDF",
                demo=True,
            )
        )

    assert captured.value.category == "request"
