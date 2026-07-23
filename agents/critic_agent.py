"""CriticAgent: provides critical review, identifies strengths and limitations."""

from pathlib import Path
from collections.abc import Callable

from langchain_core.messages import HumanMessage

from core.agent_harness import AgentRunContext, AgentSpec, get_agent_harness
from core.agent_runtime import AgentRuntimeCallbacks
from core.schemas import CriticOutput

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "critic.txt"


def build_critic_messages(paper_text: str) -> list[HumanMessage]:
    """Build CriticAgent messages from evidence-grounded context."""
    prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = prompt_template.replace("{paper_text}", paper_text)
    return [HumanMessage(content=prompt)]


CRITIC_AGENT_SPEC = AgentSpec(
    agent_id="critic",
    output_key="critic_output",
    output_schema=CriticOutput,
    build_messages=build_critic_messages,
    start_summary="已汇集论文主张和支撑证据，正在评估创新性、优点、局限与证据覆盖。",
    complete_summary="批判性评审已完成，创新性、优点、局限和改进方向已整理。",
    failed_summary="批判性评审失败，无法生成可靠结果。",
    retrieval_profile="critic",
)


def run_critic_agent(paper_text: str, *, tool_context_path: str | Path | None = None) -> CriticOutput:
    """Critically review the paper and return structured output."""
    return get_agent_harness().run(
        CRITIC_AGENT_SPEC,
        AgentRunContext(tool_context_path=tool_context_path),
        input_data=paper_text,
    ).output


def stream_critic_agent(
    paper_text: str,
    on_token: Callable[[str], None] | None = None,
    *,
    on_progress: Callable[[str, str], None] | None = None,
    on_activity: Callable[[str, str], None] | None = None,
    tool_context_path: str | Path | None = None,
) -> CriticOutput:
    """Return structured output while optionally streaming public progress."""
    return get_agent_harness().run(
        CRITIC_AGENT_SPEC,
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
