"""SummaryAgent: integrates outputs from all three parallel agents into reading notes."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage

from core.agent_harness import AgentRunContext, AgentSpec, get_agent_harness
from core.agent_runtime import AgentRuntimeCallbacks
from core.schemas import CriticOutput, ExperimentOutput, MethodOutput, SummaryOutput

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "summary.txt"


@dataclass(frozen=True)
class SummaryAgentInput:
    paper_title: str
    method_output: MethodOutput
    experiment_output: ExperimentOutput
    critic_output: CriticOutput


def build_summary_messages(
    input_data: SummaryAgentInput | str,
    method_output: MethodOutput | None = None,
    experiment_output: ExperimentOutput | None = None,
    critic_output: CriticOutput | None = None,
) -> list[HumanMessage]:
    """Build SummaryAgent messages from upstream structured outputs."""
    if isinstance(input_data, SummaryAgentInput):
        payload = input_data
    else:
        if method_output is None or experiment_output is None or critic_output is None:
            raise ValueError("SummaryAgent requires all upstream agent outputs.")
        payload = SummaryAgentInput(
            paper_title=input_data,
            method_output=method_output,
            experiment_output=experiment_output,
            critic_output=critic_output,
        )
    prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = (
        prompt_template
        .replace("{paper_title}", payload.paper_title)
        .replace("{method_output}", payload.method_output.model_dump_json(indent=2))
        .replace("{experiment_output}", payload.experiment_output.model_dump_json(indent=2))
        .replace("{critic_output}", payload.critic_output.model_dump_json(indent=2))
    )
    return [HumanMessage(content=prompt)]


SUMMARY_AGENT_SPEC = AgentSpec(
    agent_id="summary",
    output_key="summary_output",
    output_schema=SummaryOutput,
    build_messages=build_summary_messages,
    start_summary="已收到三个专业 Agent 的结构化结论，正在综合冲突与不确定性并生成最终笔记。",
    complete_summary="最终研读笔记已完成，已保留上游结论中的不确定性与冲突。",
    failed_summary="总结 Agent 失败，无法生成可靠的最终笔记。",
)


def run_summary_agent(
    paper_title: str,
    method_output: MethodOutput,
    experiment_output: ExperimentOutput,
    critic_output: CriticOutput,
    *,
    tool_context_path: str | Path | None = None,
) -> SummaryOutput:
    """Synthesize outputs from the three parallel agents into a structured reading note."""
    return get_agent_harness().run(
        SUMMARY_AGENT_SPEC,
        AgentRunContext(tool_context_path=tool_context_path),
        input_data=SummaryAgentInput(
            paper_title=paper_title,
            method_output=method_output,
            experiment_output=experiment_output,
            critic_output=critic_output,
        ),
    ).output


def stream_summary_agent(
    paper_title: str,
    method_output: MethodOutput,
    experiment_output: ExperimentOutput,
    critic_output: CriticOutput,
    on_token: Callable[[str], None] | None = None,
    *,
    on_progress: Callable[[str, str], None] | None = None,
    on_activity: Callable[[str, str], None] | None = None,
    tool_context_path: str | Path | None = None,
) -> SummaryOutput:
    """Return structured output while optionally streaming public progress."""
    return get_agent_harness().run(
        SUMMARY_AGENT_SPEC,
        AgentRunContext(
            tool_context_path=tool_context_path,
            stream=True,
            callbacks=AgentRuntimeCallbacks(
                on_token=on_token,
                on_progress=on_progress,
                on_activity=on_activity,
            ),
        ),
        input_data=SummaryAgentInput(
            paper_title=paper_title,
            method_output=method_output,
            experiment_output=experiment_output,
            critic_output=critic_output,
        ),
    ).output
