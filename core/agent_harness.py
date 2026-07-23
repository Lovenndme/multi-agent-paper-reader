"""Lifecycle, retrieval, progress, and validation wrapper for paper agents."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ValidationError

from core.agent_runtime import (
    AgentRuntime,
    AgentRuntimeCallbacks,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    get_agent_runtime,
)
from core.analysis_progress import AnalysisProgressTracker
from core.evidence import EvidenceSnippet, format_evidence_context, select_evidence_snippets
from core.pdf_parser import ParsedPaper
from core.public_analysis import public_agent_output, sanitize_visible_text


SchemaT = TypeVar("SchemaT", bound=BaseModel)
AgentEventEmitter = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class AgentSpec(Generic[SchemaT]):
    """Declarative contract for one agent."""

    agent_id: str
    output_key: str
    output_schema: type[SchemaT]
    build_messages: Callable[[Any], Sequence[BaseMessage]]
    start_summary: str
    complete_summary: str
    failed_summary: str
    retrieval_profile: str | None = None
    max_snippets: int = 10
    max_evidence_chars: int = 18_000
    invoke_retries: int = 3
    stream_retries: int = 1
    retry_delay: float = 2.0


@dataclass(frozen=True)
class AgentRunContext:
    """Per-run resources supplied to a stateless Harness."""

    paper: ParsedPaper | None = None
    snippets: Sequence[EvidenceSnippet] = ()
    tool_context_path: str | Path | None = None
    tracker: AnalysisProgressTracker | None = None
    emit: AgentEventEmitter | None = None
    stream: bool = False
    include_output_event: bool = True
    callbacks: AgentRuntimeCallbacks = field(default_factory=AgentRuntimeCallbacks)


@dataclass(frozen=True)
class AgentRunResult(Generic[SchemaT]):
    """One completed Harness run."""

    output: SchemaT
    selected_evidence_ids: tuple[str, ...]
    runtime: AgentRuntimeResult[SchemaT]


class AgentHarnessError(RuntimeError):
    """Typed failure raised after Harness lifecycle bookkeeping completes."""

    def __init__(
        self,
        agent_id: str,
        category: str,
        cause: Exception,
        failure_payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{agent_id} {category} failure: {cause}")
        self.agent_id = agent_id
        self.category = category
        self.cause = cause
        self.failure_payload = failure_payload or {}


_UNSET = object()


class AgentHarness:
    """Prepare and execute an AgentSpec through a shared Agent Runtime."""

    def __init__(self, runtime: AgentRuntime | None = None) -> None:
        self._runtime = runtime or get_agent_runtime()

    def run(
        self,
        spec: AgentSpec[SchemaT],
        context: AgentRunContext | None = None,
        *,
        input_data: Any = _UNSET,
    ) -> AgentRunResult[SchemaT]:
        context = context or AgentRunContext()
        selected: list[EvidenceSnippet] = []

        try:
            prepared_input = input_data
            if spec.retrieval_profile and context.snippets:
                selected = select_evidence_snippets(
                    list(context.snippets),
                    spec.retrieval_profile,
                    max_chars=spec.max_evidence_chars,
                    max_snippets=spec.max_snippets,
                )
                prepared_input = format_evidence_context(selected)
                if not prepared_input and context.paper is not None:
                    prepared_input = context.paper.get_sections_for_agent(spec.retrieval_profile)
            elif (
                spec.retrieval_profile
                and prepared_input is _UNSET
                and context.paper is not None
            ):
                prepared_input = context.paper.get_sections_for_agent(spec.retrieval_profile)
            if prepared_input is _UNSET:
                raise ValueError(f"{spec.agent_id} requires input_data.")

            self._start(spec, context)
            callbacks = self._runtime_callbacks(spec, context)
            runtime_result = self._runtime.execute(
                AgentRuntimeRequest(
                    schema=spec.output_schema,
                    messages=spec.build_messages(prepared_input),
                    stream=context.stream,
                    retries=spec.stream_retries if context.stream else spec.invoke_retries,
                    delay=spec.retry_delay,
                    tool_context_path=context.tool_context_path,
                    callbacks=callbacks,
                )
            )
            output = spec.output_schema.model_validate(runtime_result.output)
            self._complete(spec, context, output)
            return AgentRunResult(
                output=output,
                selected_evidence_ids=tuple(snippet.id for snippet in selected),
                runtime=runtime_result,
            )
        except AgentHarnessError:
            raise
        except Exception as exc:
            failure_payload = self._fail(spec, context)
            raise AgentHarnessError(
                spec.agent_id,
                _classify_failure(exc),
                exc,
                failure_payload,
            ) from exc

    def _runtime_callbacks(
        self,
        spec: AgentSpec[SchemaT],
        context: AgentRunContext,
    ) -> AgentRuntimeCallbacks:
        progress_buffers: dict[str, str] = {}

        def on_progress(delta: str, progress_id: str) -> None:
            if context.callbacks.on_progress:
                context.callbacks.on_progress(delta, progress_id)
            combined = f"{progress_buffers.get(progress_id, '')}{delta}"
            progress_buffers[progress_id] = combined
            self._progress(
                spec,
                context,
                combined,
                progress_id,
                source="native_reasoning_summary",
            )

        def on_activity(summary: str, progress_id: str) -> None:
            if context.callbacks.on_activity:
                context.callbacks.on_activity(summary, progress_id)
            self._progress(
                spec,
                context,
                summary,
                progress_id,
                source="tool_activity",
            )

        return AgentRuntimeCallbacks(
            on_token=context.callbacks.on_token,
            on_progress=on_progress,
            on_activity=on_activity,
        )

    @staticmethod
    def _start(spec: AgentSpec[SchemaT], context: AgentRunContext) -> None:
        if context.tracker is None:
            return
        payload = context.tracker.start_agent(spec.agent_id, spec.start_summary)
        if context.emit:
            context.emit("agent_started", payload)

    @staticmethod
    def _progress(
        spec: AgentSpec[SchemaT],
        context: AgentRunContext,
        summary: str,
        progress_id: str,
        *,
        source: str,
    ) -> None:
        if context.tracker is None:
            return
        visible_summary = sanitize_visible_text(summary)
        if not visible_summary:
            return
        payload = context.tracker.progress(
            spec.agent_id,
            visible_summary,
            source=source,
            progress_id=progress_id,
        )
        if context.emit:
            context.emit("agent_progress", payload)

    @staticmethod
    def _complete(
        spec: AgentSpec[SchemaT],
        context: AgentRunContext,
        output: SchemaT,
    ) -> None:
        if context.tracker is None:
            return
        payload = context.tracker.complete_agent(spec.agent_id, spec.complete_summary)
        if context.include_output_event:
            payload = {
                **payload,
                "output_key": spec.output_key,
                "output": public_agent_output(output.model_dump()),
            }
        if context.emit:
            context.emit("agent_complete", payload)

    @staticmethod
    def _fail(spec: AgentSpec[SchemaT], context: AgentRunContext) -> dict[str, Any]:
        if context.tracker is None:
            return {}
        return context.tracker.fail_agent(spec.agent_id, spec.failed_summary)


def _classify_failure(exc: Exception) -> str:
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ValidationError):
        return "schema"
    text = f"{type(exc).__name__} {exc}".lower()
    if "rate limit" in text or "ratelimit" in text or "429" in text:
        return "rate_limit"
    if "schema" in text or "validation" in text or "could not be parsed" in text:
        return "schema"
    if "tool" in text:
        return "tool"
    return "runtime"


_DEFAULT_HARNESS = AgentHarness()


def get_agent_harness() -> AgentHarness:
    """Return the process-wide stateless Agent Harness."""

    return _DEFAULT_HARNESS
