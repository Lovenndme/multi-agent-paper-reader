"""ComparisonAgent: aligns and contrasts multiple evidence-grounded paper analyses."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from langchain_core.messages import HumanMessage

from core.comparison import ComparisonCreateRequest, ComparisonSource, format_comparison_sources
from core.schemas import ComparisonOutput
from utils.llm import invoke_structured_with_retry, stream_structured_with_retry


_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "comparison.txt"


def run_comparison_agent(
    sources: list[ComparisonSource],
    request: ComparisonCreateRequest,
) -> ComparisonOutput:
    return invoke_structured_with_retry(
        ComparisonOutput,
        build_comparison_messages(sources, request),
    )


def stream_comparison_agent(
    sources: list[ComparisonSource],
    request: ComparisonCreateRequest,
    on_token: Callable[[str], None],
) -> ComparisonOutput:
    return stream_structured_with_retry(
        ComparisonOutput,
        build_comparison_messages(sources, request),
        on_token=on_token,
    )


def build_comparison_messages(
    sources: list[ComparisonSource],
    request: ComparisonCreateRequest,
) -> list[HumanMessage]:
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = (
        template.replace("{focus}", request.focus)
        .replace("{custom_focus}", request.custom_focus or "无")
        .replace("{paper_count}", str(len(sources)))
        .replace("{paper_sources}", format_comparison_sources(sources, request))
    )
    return [HumanMessage(content=prompt)]
