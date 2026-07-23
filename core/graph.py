"""LangGraph workflow: fan-out to three parallel agents, fan-in to summary agent."""

from dataclasses import replace
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agents.critic_agent import CRITIC_AGENT_SPEC
from agents.experiment_agent import EXPERIMENT_AGENT_SPEC
from agents.method_agent import METHOD_AGENT_SPEC
from agents.summary_agent import SUMMARY_AGENT_SPEC, SummaryAgentInput
from core.agent_harness import AgentRunContext, get_agent_harness
from core.assessment import build_analysis_assessment
from core.evidence import EvidenceSnippet, build_evidence_index
from core.pdf_parser import ParsedPaper
from core.schemas import AnalysisAssessment, CriticOutput, ExperimentOutput, MethodOutput, SummaryOutput


class PaperState(TypedDict, total=False):
    """Shared state flowing through the graph."""

    # Input
    parsed_paper: ParsedPaper
    evidence_index: list[EvidenceSnippet]
    agent_context: AgentRunContext

    # Parallel agent outputs
    method_output: MethodOutput
    experiment_output: ExperimentOutput
    critic_output: CriticOutput

    # Final output
    summary_output: SummaryOutput
    assessment: AnalysisAssessment


class GraphStageError(RuntimeError):
    """Identify a non-Agent graph stage without hiding the original cause."""

    def __init__(self, stage: str, cause: Exception) -> None:
        super().__init__(f"{stage} stage failed: {cause}")
        self.stage = stage
        self.cause = cause


# --- Node functions ---

def evidence_node(state: PaperState) -> dict:
    if "evidence_index" in state:
        return {"evidence_index": state["evidence_index"]}
    paper = state["parsed_paper"]
    return {"evidence_index": build_evidence_index(paper)}


def _run_context(state: PaperState) -> AgentRunContext:
    return replace(
        state.get("agent_context") or AgentRunContext(),
        paper=state["parsed_paper"],
        snippets=state["evidence_index"],
    )


def method_node(state: PaperState) -> dict:
    result = get_agent_harness().run(
        METHOD_AGENT_SPEC,
        _run_context(state),
    )
    return {"method_output": result.output}


def experiment_node(state: PaperState) -> dict:
    result = get_agent_harness().run(
        EXPERIMENT_AGENT_SPEC,
        _run_context(state),
    )
    return {"experiment_output": result.output}


def critic_node(state: PaperState) -> dict:
    result = get_agent_harness().run(
        CRITIC_AGENT_SPEC,
        _run_context(state),
    )
    return {"critic_output": result.output}


def summary_node(state: PaperState) -> dict:
    paper = state["parsed_paper"]
    result = get_agent_harness().run(
        SUMMARY_AGENT_SPEC,
        _run_context(state),
        input_data=SummaryAgentInput(
            paper_title=paper.title,
            method_output=state["method_output"],
            experiment_output=state["experiment_output"],
            critic_output=state["critic_output"],
        ),
    )
    return {"summary_output": result.output}


def assessment_node(state: PaperState) -> dict:
    """Calculate transparent novelty and reliability scores after all agents finish."""
    try:
        assessment = build_analysis_assessment(
            state["parsed_paper"],
            state["evidence_index"],
            state["method_output"],
            state["experiment_output"],
            state["critic_output"],
            state["summary_output"],
        )
    except Exception as exc:
        raise GraphStageError("assessment", exc) from exc
    return {"assessment": assessment}


# --- Build the graph ---

def build_graph() -> StateGraph:
    graph = StateGraph(PaperState)

    graph.add_node("evidence", evidence_node)
    graph.add_node("method", method_node)
    graph.add_node("experiment", experiment_node)
    graph.add_node("critic", critic_node)
    graph.add_node("summary", summary_node)
    graph.add_node("assessment", assessment_node)

    # Evidence-first fan-out: START → evidence index → three parallel agents
    graph.add_edge(START, "evidence")
    graph.add_edge("evidence", "method")
    graph.add_edge("evidence", "experiment")
    graph.add_edge("evidence", "critic")

    # Fan-in: all three → summary
    graph.add_edge("method", "summary")
    graph.add_edge("experiment", "summary")
    graph.add_edge("critic", "summary")

    graph.add_edge("summary", "assessment")
    graph.add_edge("assessment", END)

    return graph.compile()


def run_pipeline(parsed_paper: ParsedPaper) -> SummaryOutput:
    """Run the full multi-agent pipeline on a parsed paper."""
    final_state = run_pipeline_with_state(parsed_paper)
    return final_state["summary_output"]


def run_pipeline_with_state(
    parsed_paper: ParsedPaper,
    *,
    evidence_index: list[EvidenceSnippet] | None = None,
    agent_context: AgentRunContext | None = None,
) -> PaperState:
    """Run the full pipeline and return intermediate agent outputs too."""
    app = build_graph()
    initial_state: PaperState = {"parsed_paper": parsed_paper}
    if evidence_index is not None:
        initial_state["evidence_index"] = evidence_index
    if agent_context is not None:
        initial_state["agent_context"] = agent_context
    final_state = app.invoke(initial_state)
    return final_state
