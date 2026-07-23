"""ComparisonAgent: aligns and contrasts multiple evidence-grounded paper analyses."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage

from core.agent_harness import AgentRunContext, AgentSpec, get_agent_harness
from core.agent_runtime import AgentRuntimeCallbacks
from core.comparison import ComparisonCreateRequest, ComparisonSource, format_comparison_sources
from core.schemas import ComparisonOutput


_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "comparison.txt"


@dataclass(frozen=True)
class ComparisonAgentInput:
    sources: list[ComparisonSource]
    request: ComparisonCreateRequest


def build_comparison_messages(
    input_data: ComparisonAgentInput | list[ComparisonSource],
    request: ComparisonCreateRequest | None = None,
) -> list[HumanMessage]:
    if isinstance(input_data, ComparisonAgentInput):
        payload = input_data
    else:
        if request is None:
            raise ValueError("ComparisonAgent requires a comparison request.")
        payload = ComparisonAgentInput(sources=input_data, request=request)
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = (
        template.replace("{focus}", payload.request.focus)
        .replace("{custom_focus}", payload.request.custom_focus or "无")
        .replace("{paper_count}", str(len(payload.sources)))
        .replace(
            "{paper_sources}",
            format_comparison_sources(payload.sources, payload.request),
        )
    )
    return [HumanMessage(content=prompt)]


COMPARISON_AGENT_SPEC = AgentSpec(
    agent_id="comparison",
    output_key="comparison",
    output_schema=ComparisonOutput,
    build_messages=build_comparison_messages,
    start_summary="已载入待比较论文及其证据，正在对齐研究问题、方法与实验结论。",
    complete_summary="多论文比较已完成，差异、共识与证据引用已整理。",
    failed_summary="多论文比较失败，无法生成可靠结果。",
)


def run_comparison_agent(
    sources: list[ComparisonSource],
    request: ComparisonCreateRequest,
) -> ComparisonOutput:
    return get_agent_harness().run(
        COMPARISON_AGENT_SPEC,
        input_data=ComparisonAgentInput(sources=sources, request=request),
    ).output


def stream_comparison_agent(
    sources: list[ComparisonSource],
    request: ComparisonCreateRequest,
    on_token: Callable[[str], None],
) -> ComparisonOutput:
    return get_agent_harness().run(
        COMPARISON_AGENT_SPEC,
        AgentRunContext(
            stream=True,
            callbacks=AgentRuntimeCallbacks(on_token=on_token),
        ),
        input_data=ComparisonAgentInput(sources=sources, request=request),
    ).output
