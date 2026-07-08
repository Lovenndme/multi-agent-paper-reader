"""CriticAgent: provides critical review, identifies strengths and limitations."""

from pathlib import Path
from collections.abc import Callable

from langchain_core.messages import HumanMessage

from core.schemas import CriticOutput
from utils.llm import invoke_structured_with_retry, stream_structured_with_retry

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "critic.txt"


def run_critic_agent(paper_text: str) -> CriticOutput:
    """Critically review the paper and return structured output."""
    return invoke_structured_with_retry(CriticOutput, build_critic_messages(paper_text))


def stream_critic_agent(paper_text: str, on_token: Callable[[str], None]) -> CriticOutput:
    """Stream CriticAgent JSON tokens and return parsed structured output."""
    return stream_structured_with_retry(
        CriticOutput,
        build_critic_messages(paper_text),
        on_token=on_token,
    )


def build_critic_messages(paper_text: str) -> list[HumanMessage]:
    """Build CriticAgent messages from evidence-grounded context."""
    prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = prompt_template.replace("{paper_text}", paper_text)
    return [HumanMessage(content=prompt)]
