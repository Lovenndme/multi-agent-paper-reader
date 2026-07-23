"""Provider-neutral runtime for structured paper-reading agents."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, Protocol, TypeVar
from uuid import uuid4

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from core.model_providers import selected_text_model, selected_text_mode, text_provider_id
from utils.llm import invoke_structured_with_retry, stream_structured_with_retry


SchemaT = TypeVar("SchemaT", bound=BaseModel)
ProgressCallback = Callable[[str, str], None]


@dataclass(frozen=True)
class AgentRuntimeCallbacks:
    """Optional callbacks exposed by a provider while an agent is running."""

    on_token: Callable[[str], None] | None = None
    on_progress: ProgressCallback | None = None
    on_activity: ProgressCallback | None = None


@dataclass(frozen=True)
class AgentRuntimeRequest(Generic[SchemaT]):
    """One provider-neutral structured model invocation."""

    schema: type[SchemaT]
    messages: Sequence[BaseMessage]
    stream: bool = False
    retries: int = 3
    delay: float = 2.0
    tool_context_path: str | Path | None = None
    callbacks: AgentRuntimeCallbacks = field(default_factory=AgentRuntimeCallbacks)


@dataclass(frozen=True)
class AgentRuntimeResult(Generic[SchemaT]):
    """Validated output plus reproducibility metadata for one invocation."""

    output: SchemaT
    runtime_id: str
    provider: str
    model: str
    mode: str
    streamed: bool
    duration_ms: int


class AgentRuntime(Protocol):
    """Execution boundary used by the Agent Harness."""

    def execute(self, request: AgentRuntimeRequest[SchemaT]) -> AgentRuntimeResult[SchemaT]:
        """Execute one structured agent request."""


class StructuredOutputAgentRuntime:
    """Run structured agents through the project's existing provider adapters."""

    def execute(self, request: AgentRuntimeRequest[SchemaT]) -> AgentRuntimeResult[SchemaT]:
        started = time.monotonic()
        provider = text_provider_id()
        model = selected_text_model()
        mode = selected_text_mode()

        if request.stream:
            output = stream_structured_with_retry(
                request.schema,
                request.messages,
                on_token=request.callbacks.on_token,
                on_progress=request.callbacks.on_progress,
                on_activity=request.callbacks.on_activity,
                retries=request.retries,
                delay=request.delay,
                tool_context_path=request.tool_context_path,
            )
        else:
            output = invoke_structured_with_retry(
                request.schema,
                request.messages,
                retries=request.retries,
                delay=request.delay,
                tool_context_path=request.tool_context_path,
            )

        return AgentRuntimeResult(
            output=request.schema.model_validate(output),
            runtime_id=uuid4().hex,
            provider=provider,
            model=model,
            mode=mode,
            streamed=request.stream,
            duration_ms=max(0, round((time.monotonic() - started) * 1000)),
        )


_DEFAULT_RUNTIME = StructuredOutputAgentRuntime()


def get_agent_runtime() -> AgentRuntime:
    """Return the process-wide stateless Agent Runtime."""

    return _DEFAULT_RUNTIME
