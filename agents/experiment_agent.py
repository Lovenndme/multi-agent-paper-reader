"""ExperimentAgent: extracts datasets, metrics, and experimental results."""

from pathlib import Path
from collections.abc import Callable

from langchain_core.messages import HumanMessage

from core.schemas import ExperimentOutput
from utils.llm import invoke_structured_with_retry, stream_structured_with_retry

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "experiment.txt"


def run_experiment_agent(paper_text: str, *, tool_context_path: str | Path | None = None) -> ExperimentOutput:
    """Analyze experiment-related sections of a paper and return structured output."""
    return invoke_structured_with_retry(
        ExperimentOutput,
        build_experiment_messages(paper_text),
        tool_context_path=tool_context_path,
    )


def stream_experiment_agent(
    paper_text: str,
    on_token: Callable[[str], None],
    *,
    tool_context_path: str | Path | None = None,
) -> ExperimentOutput:
    """Stream ExperimentAgent JSON tokens and return parsed structured output."""
    return stream_structured_with_retry(
        ExperimentOutput,
        build_experiment_messages(paper_text),
        on_token=on_token,
        tool_context_path=tool_context_path,
    )


def build_experiment_messages(paper_text: str) -> list[HumanMessage]:
    """Build ExperimentAgent messages from evidence-grounded context."""
    prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = prompt_template.replace("{paper_text}", paper_text)
    return [HumanMessage(content=prompt)]
