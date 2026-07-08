"""MethodAgent: extracts research method and technical innovations."""

from pathlib import Path
from collections.abc import Callable

from langchain_core.messages import HumanMessage

from core.schemas import MethodOutput
from utils.llm import invoke_structured_with_retry, stream_structured_with_retry

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "method.txt"


def run_method_agent(paper_text: str) -> MethodOutput:
    """Analyze method-related sections of a paper and return structured output."""
    return invoke_structured_with_retry(MethodOutput, build_method_messages(paper_text))


def stream_method_agent(paper_text: str, on_token: Callable[[str], None]) -> MethodOutput:
    """Stream MethodAgent JSON tokens and return parsed structured output."""
    return stream_structured_with_retry(
        MethodOutput,
        build_method_messages(paper_text),
        on_token=on_token,
    )


def build_method_messages(paper_text: str) -> list[HumanMessage]:
    """Build MethodAgent messages from evidence-grounded context."""
    prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = prompt_template.replace("{paper_text}", paper_text)
    return [HumanMessage(content=prompt)]
