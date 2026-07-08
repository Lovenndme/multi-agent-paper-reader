"""LangGraph workflow: fan-out to three parallel agents, fan-in to summary agent."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agents.critic_agent import run_critic_agent
from agents.experiment_agent import run_experiment_agent
from agents.method_agent import run_method_agent
from agents.summary_agent import run_summary_agent
from core.evidence import EvidenceSnippet, build_evidence_index, evidence_context_for_agent
from core.pdf_parser import ParsedPaper
from core.schemas import CriticOutput, ExperimentOutput, MethodOutput, SummaryOutput


class PaperState(TypedDict, total=False):
    """Shared state flowing through the graph."""

    # Input
    parsed_paper: ParsedPaper
    evidence_index: list[EvidenceSnippet]

    # Parallel agent outputs
    method_output: MethodOutput
    experiment_output: ExperimentOutput
    critic_output: CriticOutput

    # Final output
    summary_output: SummaryOutput


# --- Node functions ---

def evidence_node(state: PaperState) -> dict:
    paper = state["parsed_paper"]
    return {"evidence_index": build_evidence_index(paper)}


def method_node(state: PaperState) -> dict:
    paper = state["parsed_paper"]
    text = evidence_context_for_agent(state["evidence_index"], "method") or paper.get_sections_for_agent("method")
    return {"method_output": run_method_agent(text)}


def experiment_node(state: PaperState) -> dict:
    paper = state["parsed_paper"]
    text = evidence_context_for_agent(state["evidence_index"], "experiment") or paper.get_sections_for_agent("experiment")
    return {"experiment_output": run_experiment_agent(text)}


def critic_node(state: PaperState) -> dict:
    paper = state["parsed_paper"]
    text = evidence_context_for_agent(state["evidence_index"], "critic") or paper.get_sections_for_agent("critic")
    return {"critic_output": run_critic_agent(text)}


def summary_node(state: PaperState) -> dict:
    paper = state["parsed_paper"]
    result = run_summary_agent(
        paper_title=paper.title,
        method_output=state["method_output"],
        experiment_output=state["experiment_output"],
        critic_output=state["critic_output"],
    )
    return {"summary_output": result}


# --- Build the graph ---

def build_graph() -> StateGraph:
    graph = StateGraph(PaperState)

    graph.add_node("evidence", evidence_node)
    graph.add_node("method", method_node)
    graph.add_node("experiment", experiment_node)
    graph.add_node("critic", critic_node)
    graph.add_node("summary", summary_node)

    # Evidence-first fan-out: START → evidence index → three parallel agents
    graph.add_edge(START, "evidence")
    graph.add_edge("evidence", "method")
    graph.add_edge("evidence", "experiment")
    graph.add_edge("evidence", "critic")

    # Fan-in: all three → summary
    graph.add_edge("method", "summary")
    graph.add_edge("experiment", "summary")
    graph.add_edge("critic", "summary")

    graph.add_edge("summary", END)

    return graph.compile()


def run_pipeline(parsed_paper: ParsedPaper) -> SummaryOutput:
    """Run the full multi-agent pipeline on a parsed paper."""
    final_state = run_pipeline_with_state(parsed_paper)
    return final_state["summary_output"]


def run_pipeline_with_state(parsed_paper: ParsedPaper) -> PaperState:
    """Run the full pipeline and return intermediate agent outputs too."""
    app = build_graph()
    final_state = app.invoke({"parsed_paper": parsed_paper})
    return final_state
