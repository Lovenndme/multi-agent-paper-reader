"""CriticAgent: provides critical review, identifies strengths and limitations."""

from pathlib import Path

from langchain_core.messages import HumanMessage

from core.schemas import CriticOutput
from utils.llm import get_llm, invoke_with_retry

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "critic.txt"


def run_critic_agent(paper_text: str) -> CriticOutput:
    """Critically review the paper and return structured output."""
    prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = prompt_template.replace("{paper_text}", paper_text)

    structured_llm = get_llm().with_structured_output(CriticOutput)
    return invoke_with_retry(structured_llm, [HumanMessage(content=prompt)])
