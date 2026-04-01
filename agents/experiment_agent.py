"""ExperimentAgent: extracts datasets, metrics, and experimental results."""

from pathlib import Path

from langchain_core.messages import HumanMessage

from core.schemas import ExperimentOutput
from utils.llm import get_llm

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "experiment.txt"


def run_experiment_agent(paper_text: str) -> ExperimentOutput:
    """Analyze experiment-related sections of a paper and return structured output."""
    prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = prompt_template.replace("{paper_text}", paper_text)

    llm = get_llm()
    structured_llm = llm.with_structured_output(ExperimentOutput)

    result = structured_llm.invoke([HumanMessage(content=prompt)])
    return result
