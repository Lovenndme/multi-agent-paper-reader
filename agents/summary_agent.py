"""SummaryAgent: integrates outputs from all three parallel agents into reading notes."""

import json
from pathlib import Path

from langchain_core.messages import HumanMessage

from core.schemas import CriticOutput, ExperimentOutput, MethodOutput, SummaryOutput
from utils.llm import get_llm, invoke_with_retry

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "summary.txt"


def run_summary_agent(
    paper_title: str,
    method_output: MethodOutput,
    experiment_output: ExperimentOutput,
    critic_output: CriticOutput,
) -> SummaryOutput:
    """Synthesize outputs from the three parallel agents into a structured reading note."""
    prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = (
        prompt_template
        .replace("{paper_title}", paper_title)
        .replace("{method_output}", method_output.model_dump_json(indent=2))
        .replace("{experiment_output}", experiment_output.model_dump_json(indent=2))
        .replace("{critic_output}", critic_output.model_dump_json(indent=2))
    )

    structured_llm = get_llm().with_structured_output(SummaryOutput)
    return invoke_with_retry(structured_llm, [HumanMessage(content=prompt)])
