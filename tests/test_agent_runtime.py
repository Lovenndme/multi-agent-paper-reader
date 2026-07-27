"""Tests for the provider-neutral Agent Runtime."""

from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from core.agent_runtime import (
    AgentRuntimeCallbacks,
    AgentRuntimeRequest,
    StructuredOutputAgentRuntime,
)


class RuntimeOutput(BaseModel):
    answer: str


def test_runtime_routes_non_streaming_structured_request() -> None:
    expected = RuntimeOutput(answer="ok")
    with (
        patch("core.agent_runtime.text_provider_id", return_value="openai"),
        patch("core.agent_runtime.selected_text_model", return_value="gpt-test"),
        patch("core.agent_runtime.selected_text_mode", return_value="balanced"),
        patch(
            "core.agent_runtime.invoke_structured_with_retry",
            return_value=expected,
        ) as invoke,
    ):
        result = StructuredOutputAgentRuntime().execute(
            AgentRuntimeRequest(
                schema=RuntimeOutput,
                messages=[HumanMessage(content="analyze")],
                retries=4,
                delay=0.25,
                tool_context_path=Path("/tmp/tool-context.json"),
            )
        )

    assert result.output == expected
    assert result.provider == "openai"
    assert result.model == "gpt-test"
    assert result.mode == "balanced"
    assert result.streamed is False
    assert result.runtime_id
    invoke.assert_called_once()
    assert invoke.call_args.kwargs["retries"] == 4
    assert invoke.call_args.kwargs["delay"] == 0.25
    assert invoke.call_args.kwargs["tool_context_path"] == Path("/tmp/tool-context.json")


def test_runtime_routes_stream_callbacks() -> None:
    callbacks = AgentRuntimeCallbacks(
        on_token=Mock(),
        on_progress=Mock(),
        on_activity=Mock(),
    )
    expected = RuntimeOutput(answer="streamed")
    with (
        patch("core.agent_runtime.text_provider_id", return_value="codex"),
        patch("core.agent_runtime.selected_text_model", return_value="gpt-5.6-codex"),
        patch("core.agent_runtime.selected_text_mode", return_value="high"),
        patch(
            "core.agent_runtime.stream_structured_with_retry",
            return_value=expected,
        ) as stream,
    ):
        result = StructuredOutputAgentRuntime().execute(
            AgentRuntimeRequest(
                schema=RuntimeOutput,
                messages=[HumanMessage(content="analyze")],
                stream=True,
                retries=2,
                callbacks=callbacks,
            )
        )

    assert result.output == expected
    assert result.streamed is True
    assert stream.call_args.kwargs["on_token"] is callbacks.on_token
    assert stream.call_args.kwargs["on_progress"] is callbacks.on_progress
    assert stream.call_args.kwargs["on_activity"] is callbacks.on_activity
    assert stream.call_args.kwargs["retries"] == 2


def test_runtime_uses_frozen_route_for_client_and_metadata() -> None:
    expected = RuntimeOutput(answer="frozen")
    frozen_llm = object()
    with (
        patch(
            "core.agent_runtime.get_llm_for_route",
            return_value=frozen_llm,
        ) as get_route,
        patch(
            "core.agent_runtime.invoke_structured_with_retry",
            return_value=expected,
        ) as invoke,
        patch(
            "core.agent_runtime.text_provider_id",
            side_effect=AssertionError("global provider must not replace snapshot"),
        ),
    ):
        result = StructuredOutputAgentRuntime().execute(
            AgentRuntimeRequest(
                schema=RuntimeOutput,
                messages=[HumanMessage(content="analyze")],
                provider_id="qwen",
                model="qwen3.7-max",
                mode="thinking",
            )
        )

    get_route.assert_called_once_with("qwen", "qwen3.7-max", "thinking")
    assert invoke.call_args.kwargs["llm"] is frozen_llm
    assert (result.provider, result.model, result.mode) == (
        "qwen",
        "qwen3.7-max",
        "thinking",
    )
