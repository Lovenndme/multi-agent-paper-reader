#!/usr/bin/env python3
"""Run one reproducible, result-only whole-paper RAG pipeline benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import patch

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.agent_harness import AgentRunContext
from core.agentic_types import AgenticRagConfig, AgenticRunBudget
from core.graph import run_pipeline_with_state
from core.history import load_paper_analysis, retained_paper_pdf_path
from core.model_providers import (
    selected_text_model,
    selected_text_mode,
    text_provider_id,
)
from core.pdf_parser import parse_pdf
from core.settings import PROJECT_VERSION


def summarize_pipeline_state(
    state: dict[str, Any],
    *,
    label: str,
    latency_ms: float,
    config: AgenticRagConfig,
) -> dict[str, Any]:
    """Return bounded metrics without persisting model prose or paper text."""
    outputs = [
        state["method_output"],
        state["experiment_output"],
        state["critic_output"],
        state["summary_output"],
    ]
    citation_ids = [
        str(getattr(item, "id", "") or "").upper()
        for output in outputs
        for item in (getattr(output, "evidence", None) or [])
        if str(getattr(item, "id", "") or "")
    ]
    valid_ids = {
        str(getattr(snippet, "id", "") or "").upper()
        for snippet in state["evidence_index"]
    }
    traces = list(state.get("retrieval_traces") or [])
    planner_steps = sum(
        len(trace.get("steps") or [])
        for trace in traces
        if isinstance(trace, dict)
    )
    adaptive_values = [
        bool(trace["adaptive_triggered"])
        for trace in traces
        if isinstance(trace, dict) and trace.get("adaptive_triggered") is not None
    ]
    strategies = Counter(
        str(trace.get("strategy") or "unknown")
        for trace in traces
        if isinstance(trace, dict)
    )
    supervisor = state["evidence_supervisor"]
    return {
        "label": label,
        "rag_mode": config.mode,
        "tool_strategy": config.tool_strategy,
        "planner_model": config.planner_model or None,
        "planner_mode": config.planner_mode or None,
        "latency_ms": round(max(0.0, latency_ms), 3),
        "schema_success": True,
        "citations": len(citation_ids),
        "distinct_citations": len(set(citation_ids)),
        "valid_citation_rate": (
            sum(item in valid_ids for item in citation_ids) / len(citation_ids)
            if citation_ids
            else 1.0
        ),
        "repair_count": int(state.get("repair_count", 0)),
        "supervisor_sufficient": bool(supervisor.sufficient),
        "supervisor_coverage_score": int(supervisor.coverage_score),
        "supervisor_warnings": len(supervisor.warnings),
        "retrieval_traces": len(traces),
        "planner_steps": planner_steps,
        "fallback_traces": sum(
            bool(trace.get("fallback_used"))
            for trace in traces
            if isinstance(trace, dict)
        ),
        "adaptive_triggered_traces": sum(adaptive_values),
        "adaptive_skipped_traces": len(adaptive_values) - sum(adaptive_values),
        "adaptive_trigger_rate": (
            sum(adaptive_values) / len(adaptive_values)
            if adaptive_values
            else None
        ),
        "strategies": dict(sorted(strategies.items())),
    }


def execute_pipeline_benchmark(
    history_id: str,
    *,
    label: str,
    rag_mode: str,
    tool_strategy: str,
    planner_model: str | None = None,
    planner_mode: str | None = None,
) -> dict[str, Any]:
    loaded = load_paper_analysis(history_id)
    if loaded is None:
        raise ValueError(f"Unknown paper_history_id: {history_id}")
    pdf_path = retained_paper_pdf_path(history_id)
    if pdf_path is None:
        raise ValueError(f"Retained PDF is unavailable for {history_id}")

    pdf_bytes = pdf_path.read_bytes()
    parse_started = time.perf_counter()
    paper = parse_pdf(pdf_path)
    parse_latency_ms = (time.perf_counter() - parse_started) * 1000
    updates = {
        "RAG_MODE": rag_mode,
        "AGENTIC_TOOL_STRATEGY": tool_strategy,
        "AGENTIC_RAG_CHECKPOINTS": "false",
    }
    if planner_model is not None:
        updates["AGENTIC_RAG_PLANNER_MODEL"] = planner_model
    if planner_mode is not None:
        updates["AGENTIC_RAG_PLANNER_MODE"] = planner_mode

    with patch.dict(os.environ, updates):
        provider_id = text_provider_id()
        model = selected_text_model(provider_id)
        model_mode = selected_text_mode(provider_id, model)
        config = AgenticRagConfig.from_env()
        budget = AgenticRunBudget.from_env()
        context = AgentRunContext(
            paper=paper,
            snippets=tuple(loaded["snippets"]),
            retrieval_budget=budget,
            agentic_config=config,
            retrieval_provider_id=provider_id,
            retrieval_model=model,
            retrieval_model_mode=model_mode,
        )
        started = time.perf_counter()
        state = run_pipeline_with_state(
            paper,
            evidence_index=list(loaded["snippets"]),
            agent_context=context,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        run = summarize_pipeline_state(
            state,
            label=label,
            latency_ms=latency_ms,
            config=config,
        )

    return {
        "scope": "single-paper whole-pipeline pilot; not answer accuracy",
        "project_version": PROJECT_VERSION,
        "paper": str(loaded["history"]["title"]),
        "paper_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
        "parse_latency_ms": round(parse_latency_ms, 3),
        "provider": provider_id,
        "model": model,
        "model_mode": model_mode,
        "run": run,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one result-only whole-paper RAG pipeline benchmark."
    )
    parser.add_argument("paper_history_id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="B2")
    parser.add_argument(
        "--rag-mode",
        choices=("hybrid", "adaptive", "agentic"),
        default="adaptive",
    )
    parser.add_argument(
        "--tool-strategy",
        choices=("auto", "native", "structured"),
        default="native",
    )
    parser.add_argument("--planner-model")
    parser.add_argument("--planner-mode")
    args = parser.parse_args()

    load_dotenv(".env", override=False)
    report = execute_pipeline_benchmark(
        args.paper_history_id,
        label=args.label,
        rag_mode=args.rag_mode,
        tool_strategy=args.tool_strategy,
        planner_model=args.planner_model,
        planner_mode=args.planner_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
