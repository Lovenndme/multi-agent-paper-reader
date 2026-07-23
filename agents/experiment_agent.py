"""ExperimentAgent: extracts datasets, metrics, and experimental results."""

from pathlib import Path
from collections.abc import Callable

from langchain_core.messages import HumanMessage

from core.agent_harness import AgentRunContext, AgentSpec, get_agent_harness
from core.agent_runtime import AgentRuntimeCallbacks
from core.schemas import ExperimentOutput

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "experiment.txt"


def build_experiment_messages(paper_text: str) -> list[HumanMessage]:
    """Build ExperimentAgent messages from evidence-grounded context."""
    prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = prompt_template.replace("{paper_text}", paper_text)
    return [HumanMessage(content=prompt)]


EXPERIMENT_AGENT_SPEC = AgentSpec(
    agent_id="experiment",
    output_key="experiment_output",
    output_schema=ExperimentOutput,
    build_messages=build_experiment_messages,
    start_summary="已选取实验章节、表格和指标证据，正在核对数据集、基线与主要结果。",
    complete_summary="实验分析已完成，数据集、指标、基线和结果已核对。",
    failed_summary="实验分析失败，无法生成可靠结果。",
    retrieval_profile="experiment",
)


def run_experiment_agent(paper_text: str, *, tool_context_path: str | Path | None = None) -> ExperimentOutput:
    """Analyze experiment-related sections of a paper and return structured output."""
    return get_agent_harness().run(
        EXPERIMENT_AGENT_SPEC,
        AgentRunContext(tool_context_path=tool_context_path),
        input_data=paper_text,
    ).output


def stream_experiment_agent(
    paper_text: str,
    on_token: Callable[[str], None] | None = None,
    *,
    on_progress: Callable[[str, str], None] | None = None,
    on_activity: Callable[[str, str], None] | None = None,
    tool_context_path: str | Path | None = None,
) -> ExperimentOutput:
    """Return structured output while optionally streaming public progress."""
    return get_agent_harness().run(
        EXPERIMENT_AGENT_SPEC,
        AgentRunContext(
            tool_context_path=tool_context_path,
            stream=True,
            callbacks=AgentRuntimeCallbacks(
                on_token=on_token,
                on_progress=on_progress,
                on_activity=on_activity,
            ),
        ),
        input_data=paper_text,
    ).output
