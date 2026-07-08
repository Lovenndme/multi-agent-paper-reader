"""SummaryAgent: integrates outputs from all three parallel agents into reading notes."""

import json
from pathlib import Path
from collections.abc import Callable

from langchain_core.messages import HumanMessage

from core.schemas import CriticOutput, ExperimentOutput, MethodOutput, SummaryOutput
from utils.llm import invoke_structured_with_retry, stream_structured_with_retry

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "summary.txt"


def run_summary_agent(
    paper_title: str,
    method_output: MethodOutput,
    experiment_output: ExperimentOutput,
    critic_output: CriticOutput,
) -> SummaryOutput:
    """Synthesize outputs from the three parallel agents into a structured reading note."""
    return invoke_structured_with_retry(
        SummaryOutput,
        build_summary_messages(paper_title, method_output, experiment_output, critic_output),
    )


def stream_summary_agent(
    paper_title: str,
    method_output: MethodOutput,
    experiment_output: ExperimentOutput,
    critic_output: CriticOutput,
    on_token: Callable[[str], None],
) -> SummaryOutput:
    """Stream SummaryAgent JSON tokens and return parsed structured output."""
    return stream_structured_with_retry(
        SummaryOutput,
        build_summary_messages(paper_title, method_output, experiment_output, critic_output),
        on_token=on_token,
    )


def build_summary_messages(
    paper_title: str,
    method_output: MethodOutput,
    experiment_output: ExperimentOutput,
    critic_output: CriticOutput,
) -> list[HumanMessage]:
    """Build SummaryAgent messages from upstream structured outputs."""
    prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = (
        prompt_template
        .replace("{paper_title}", paper_title)
        .replace("{method_output}", method_output.model_dump_json(indent=2))
        .replace("{experiment_output}", experiment_output.model_dump_json(indent=2))
        .replace("{critic_output}", critic_output.model_dump_json(indent=2))
    )
    return [HumanMessage(content=prompt)]
