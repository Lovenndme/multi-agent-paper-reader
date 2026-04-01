"""MethodAgent: extracts research method and technical innovations."""

import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from core.schemas import MethodOutput
from utils.llm import get_llm

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "method.txt"


def run_method_agent(paper_text: str) -> MethodOutput:
    """Analyze method-related sections of a paper and return structured output."""
    prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = prompt_template.replace("{paper_text}", paper_text)

    llm = get_llm()
    structured_llm = llm.with_structured_output(MethodOutput)

    result = structured_llm.invoke([HumanMessage(content=prompt)])
    return result
