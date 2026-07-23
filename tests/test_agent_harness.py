"""Tests for retrieval, lifecycle, and failure behavior in the Agent Harness."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from core.agent_harness import (
    AgentHarness,
    AgentHarnessError,
    AgentRunContext,
    AgentSpec,
)
from core.agent_runtime import AgentRuntimeRequest, AgentRuntimeResult
from core.analysis_progress import AnalysisProgressTracker
from core.evidence import EvidenceSnippet
from core.pdf_parser import ParsedPaper


class HarnessOutput(BaseModel):
    answer: str


def _messages(value: Any) -> list[HumanMessage]:
    return [HumanMessage(content=str(value))]


TEST_SPEC = AgentSpec(
    agent_id="method",
    output_key="method_output",
    output_schema=HarnessOutput,
    build_messages=_messages,
    start_summary="方法开始",
    complete_summary="方法完成",
    failed_summary="方法失败",
    retrieval_profile="method",
    max_snippets=1,
    max_evidence_chars=1_000,
)


class RecordingRuntime:
    def __init__(self) -> None:
        self.request: AgentRuntimeRequest[HarnessOutput] | None = None

    def execute(
        self,
        request: AgentRuntimeRequest[HarnessOutput],
    ) -> AgentRuntimeResult[HarnessOutput]:
        self.request = request
        if request.callbacks.on_progress:
            request.callbacks.on_progress("正在核对 E001", "reasoning-1")
            request.callbacks.on_progress(" 的方法定义", "reasoning-1")
        if request.callbacks.on_activity:
            request.callbacks.on_activity("已读取 F001", "tool-1")
        return AgentRuntimeResult(
            output=HarnessOutput(answer="done"),
            runtime_id="runtime-test",
            provider="test",
            model="test-model",
            mode="test-mode",
            streamed=request.stream,
            duration_ms=5,
        )


class FailingRuntime:
    def execute(self, request: AgentRuntimeRequest[HarnessOutput]):
        raise TimeoutError("provider timed out")


def test_harness_selects_evidence_and_emits_public_lifecycle() -> None:
    runtime = RecordingRuntime()
    tracker = AnalysisProgressTracker()
    events: list[tuple[str, dict[str, Any]]] = []
    snippets = [
        EvidenceSnippet("E001", "Method", 0, 0, "method architecture", "text"),
        EvidenceSnippet("F001", "Figure", 1, 1, "architecture diagram", "figure"),
    ]

    with patch("core.evidence.semantic_scores", return_value=[0.1, 0.9]):
        result = AgentHarness(runtime).run(
            TEST_SPEC,
            AgentRunContext(
                paper=ParsedPaper(title="Paper", full_text="paper"),
                snippets=snippets,
                tracker=tracker,
                emit=lambda event_type, payload: events.append((event_type, payload)),
                stream=True,
            ),
        )

    assert result.output.answer == "done"
    assert result.selected_evidence_ids == ("F001",)
    assert runtime.request is not None
    assert "[F001 | figure" in runtime.request.messages[0].content
    assert runtime.request.stream is True
    assert runtime.request.retries == TEST_SPEC.stream_retries
    assert [event_type for event_type, _ in events] == [
        "agent_started",
        "agent_progress",
        "agent_progress",
        "agent_progress",
        "agent_complete",
    ]
    visible_progress = [
        payload["text"]
        for event_type, payload in events
        if event_type == "agent_progress"
    ]
    assert all("E001" not in text and "F001" not in text for text in visible_progress)
    complete = events[-1][1]
    assert complete["output_key"] == "method_output"
    assert complete["output"] == {"answer": "done"}
    assert tracker.snapshot()["agents"]["method"]["status"] == "complete"


def test_harness_preserves_explicit_input_without_retrieval_context() -> None:
    runtime = RecordingRuntime()
    AgentHarness(runtime).run(TEST_SPEC, input_data="preselected paper text")

    assert runtime.request is not None
    assert runtime.request.messages[0].content == "preselected paper text"


def test_harness_classifies_failure_and_updates_tracker() -> None:
    tracker = AnalysisProgressTracker()

    with pytest.raises(AgentHarnessError) as captured:
        AgentHarness(FailingRuntime()).run(
            TEST_SPEC,
            AgentRunContext(tracker=tracker),
            input_data="paper",
        )

    assert captured.value.agent_id == "method"
    assert captured.value.category == "timeout"
    assert captured.value.failure_payload["summary"] == "方法失败"
    assert tracker.snapshot()["agents"]["method"]["status"] == "failed"
