"""MethodAgent: extracts research method and technical innovations."""

from pathlib import Path
from collections.abc import Callable

from langchain_core.messages import HumanMessage

from core.agent_harness import AgentRunContext, AgentSpec, get_agent_harness
from core.agent_runtime import AgentRuntimeCallbacks
from core.schemas import MethodOutput

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "method.txt"


def build_method_messages(paper_text: str) -> list[HumanMessage]:
    """Build MethodAgent messages from evidence-grounded context."""
    prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = prompt_template.replace("{paper_text}", paper_text)
    return [HumanMessage(content=prompt)]


METHOD_AGENT_SPEC = AgentSpec(
    agent_id="method",
    output_key="method_output",
    output_schema=MethodOutput,
    build_messages=build_method_messages,
    start_summary="已选取方法相关章节与证据，正在识别研究问题、方法组件和创新点。",
    complete_summary="方法分析已完成，研究问题、关键组件和创新点已整理。",
    failed_summary="方法分析失败，无法生成可靠结果。",
    retrieval_profile="method",
)


def run_method_agent(paper_text: str, *, tool_context_path: str | Path | None = None) -> MethodOutput:
    """Analyze method-related sections of a paper and return structured output."""
    return get_agent_harness().run(
        METHOD_AGENT_SPEC,
        AgentRunContext(tool_context_path=tool_context_path),
        input_data=paper_text,
    ).output


def stream_method_agent(
    paper_text: str,
    on_token: Callable[[str], None] | None = None,
    *,
    on_progress: Callable[[str, str], None] | None = None,
    on_activity: Callable[[str, str], None] | None = None,
    tool_context_path: str | Path | None = None,
) -> MethodOutput:
    """Return structured output while optionally streaming public progress."""
    return get_agent_harness().run(
        METHOD_AGENT_SPEC,
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
