"""CriticAgent: provides critical review, identifies strengths and limitations."""

from pathlib import Path
from collections.abc import Callable

from langchain_core.messages import HumanMessage

from core.schemas import CriticOutput
from utils.llm import invoke_structured_with_retry, stream_structured_with_retry

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "critic.txt"


def run_critic_agent(paper_text: str, *, tool_context_path: str | Path | None = None) -> CriticOutput:
    """Critically review the paper and return structured output."""
    return invoke_structured_with_retry(
        CriticOutput,
        build_critic_messages(paper_text),
        tool_context_path=tool_context_path,
    )


def stream_critic_agent(
    paper_text: str,
    on_token: Callable[[str], None] | None = None,
    *,
    on_progress: Callable[[str, str], None] | None = None,
    on_activity: Callable[[str, str], None] | None = None,
    tool_context_path: str | Path | None = None,
) -> CriticOutput:
    """Return structured output while optionally streaming public progress."""
    return stream_structured_with_retry(
        CriticOutput,
        build_critic_messages(paper_text),
        on_token=on_token,
        on_progress=on_progress,
        on_activity=on_activity,
        tool_context_path=tool_context_path,
    )


def build_critic_messages(paper_text: str) -> list[HumanMessage]:
    """Build CriticAgent messages from evidence-grounded context."""
    prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = prompt_template.replace("{paper_text}", paper_text)
    return [HumanMessage(content=prompt)]
