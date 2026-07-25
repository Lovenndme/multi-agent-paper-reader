"""Evidence Supervisor: grades specialist coverage and requests bounded repairs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage

from core.agent_harness import AgentRunContext, AgentSpec, get_agent_harness
from core.schemas import (
    CriticOutput,
    EvidenceSupervisorOutput,
    ExperimentOutput,
    MethodOutput,
)


_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "evidence_supervisor.txt"


@dataclass(frozen=True)
class EvidenceSupervisorInput:
    paper_title: str
    method_output: MethodOutput
    experiment_output: ExperimentOutput
    critic_output: CriticOutput
    cited_evidence_context: str
    repair_round: int = 0


def build_evidence_supervisor_messages(
    input_data: EvidenceSupervisorInput,
) -> list[HumanMessage]:
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = (
        template.replace("{paper_title}", input_data.paper_title)
        .replace("{repair_round}", str(input_data.repair_round))
        .replace("{method_output}", input_data.method_output.model_dump_json(indent=2))
        .replace(
            "{experiment_output}",
            input_data.experiment_output.model_dump_json(indent=2),
        )
        .replace("{critic_output}", input_data.critic_output.model_dump_json(indent=2))
        .replace(
            "{cited_evidence_context}",
            input_data.cited_evidence_context or "No valid cited evidence was found.",
        )
    )
    return [HumanMessage(content=prompt)]


EVIDENCE_SUPERVISOR_SPEC = AgentSpec(
    agent_id="evidence_supervisor",
    output_key="evidence_supervisor",
    output_schema=EvidenceSupervisorOutput,
    build_messages=build_evidence_supervisor_messages,
    start_summary="三个专业 Agent 已完成，正在检查关键结论的证据覆盖与相互冲突。",
    complete_summary="证据覆盖检查已完成，必要的定向补检索任务已确定。",
    failed_summary="证据覆盖模型检查失败，已退回确定性证据校验。",
)


def run_evidence_supervisor(
    input_data: EvidenceSupervisorInput,
    context: AgentRunContext | None = None,
) -> EvidenceSupervisorOutput:
    return get_agent_harness().run(
        EVIDENCE_SUPERVISOR_SPEC,
        context,
        input_data=input_data,
    ).output
