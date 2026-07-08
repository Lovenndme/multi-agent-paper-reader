"""Run A/B paper analysis with and without vision evidence."""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.critic_agent import run_critic_agent
from agents.experiment_agent import run_experiment_agent
from agents.method_agent import run_method_agent
from agents.summary_agent import run_summary_agent
from core.evidence import build_evidence_index, evidence_context_for_agent, evidence_payload
from core.pdf_parser import parse_pdf
from core.vision import enrich_paper_figures_with_vision

OUT_DIR = ROOT / "vision_ab_outputs"


def _log(message: str) -> None:
    print(message, flush=True)


def _safe_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _summary_metrics(result: dict[str, Any]) -> dict[str, Any]:
    outputs = [
        result.get("method_output") or {},
        result.get("experiment_output") or {},
        result.get("critic_output") or {},
        result.get("summary_output") or {},
    ]
    evidence_ids: list[str] = []
    for output in outputs:
        for item in output.get("evidence") or []:
            evidence_id = str(item.get("id", ""))
            if evidence_id:
                evidence_ids.append(evidence_id)
    evidence_index = result.get("evidence_index") or []
    return {
        "duration_seconds": round(result.get("duration_seconds", 0), 2),
        "index_total": len(evidence_index),
        "index_text": sum(str(item.get("id", "")).startswith("E") for item in evidence_index),
        "index_table": sum(str(item.get("id", "")).startswith("T") for item in evidence_index),
        "index_figure": sum(str(item.get("id", "")).startswith("F") for item in evidence_index),
        "output_evidence_count": len(evidence_ids),
        "output_text_evidence": sum(eid.startswith("E") for eid in evidence_ids),
        "output_table_evidence": sum(eid.startswith("T") for eid in evidence_ids),
        "output_figure_evidence": sum(eid.startswith("F") for eid in evidence_ids),
        "unique_output_evidence": sorted(set(evidence_ids)),
    }


def run_variant(pdf_path: Path, name: str, *, use_vision: bool) -> dict[str, Any]:
    started = time.perf_counter()
    _log(f"[{name}] parsing PDF")
    paper = parse_pdf(pdf_path)

    vision_result = None
    if use_vision:
        _log(f"[{name}] enriching {len(paper.figures)} figure candidates with vision")
        vision_result = enrich_paper_figures_with_vision(pdf_path, paper)
        _log(
            f"[{name}] vision enriched={vision_result.enriched} "
            f"attempted={vision_result.attempted} skipped={vision_result.skipped} "
            f"errors={len(vision_result.errors)}"
        )
    else:
        _log(f"[{name}] vision disabled")

    snippets = build_evidence_index(paper)
    method_context = evidence_context_for_agent(snippets, "method") or paper.get_sections_for_agent("method")
    experiment_context = evidence_context_for_agent(snippets, "experiment") or paper.get_sections_for_agent("experiment")
    critic_context = evidence_context_for_agent(snippets, "critic") or paper.get_sections_for_agent("critic")

    _log(f"[{name}] running MethodAgent + ExperimentAgent + CriticAgent in parallel")
    agent_jobs = {
        "method": (run_method_agent, method_context),
        "experiment": (run_experiment_agent, experiment_context),
        "critic": (run_critic_agent, critic_context),
    }
    agent_outputs: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(agent_fn, context): agent_name
            for agent_name, (agent_fn, context) in agent_jobs.items()
        }
        for future in as_completed(futures):
            agent_name = futures[future]
            agent_outputs[agent_name] = future.result()
            _log(f"[{name}] {agent_name} complete")

    method = agent_outputs["method"]
    experiment = agent_outputs["experiment"]
    critic = agent_outputs["critic"]
    _log(f"[{name}] running SummaryAgent")
    summary = run_summary_agent(
        paper_title=paper.title,
        method_output=method,
        experiment_output=experiment,
        critic_output=critic,
    )

    duration = time.perf_counter() - started
    payload = {
        "variant": name,
        "use_vision": use_vision,
        "duration_seconds": duration,
        "paper": {
            "title": paper.title,
            "sections": len(paper.sections),
            "tables": len(paper.tables),
            "figures": len(paper.figures),
        },
        "vision_result": None
        if vision_result is None
        else {
            "total_figures": vision_result.total_figures,
            "attempted": vision_result.attempted,
            "enriched": vision_result.enriched,
            "skipped": vision_result.skipped,
            "errors": vision_result.errors,
        },
        "evidence_index": evidence_payload(snippets),
        "method_output": method.model_dump(),
        "experiment_output": experiment.model_dump(),
        "critic_output": critic.model_dump(),
        "summary_output": summary.model_dump(),
    }
    payload["metrics"] = _summary_metrics(payload)
    _safe_dump(OUT_DIR / f"{name}.json", payload)
    _log(f"[{name}] complete in {duration:.2f}s")
    return payload


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    matches = list(Path(r"C:\Users\lenovo\Desktop").glob("*/39.pdf"))
    if not matches:
        raise FileNotFoundError("Could not find Desktop/*/39.pdf")
    pdf_path = matches[0]
    _log(f"PDF={pdf_path}")

    os.environ["ENABLE_VISION_SUMMARY"] = "false"
    without_vision = run_variant(pdf_path, "without_vision", use_vision=False)

    os.environ["ENABLE_VISION_SUMMARY"] = "true"
    with_vision = run_variant(pdf_path, "with_vision", use_vision=True)

    comparison = {
        "pdf": str(pdf_path),
        "without_vision_metrics": without_vision["metrics"],
        "with_vision_metrics": with_vision["metrics"],
        "without_vision_summary": without_vision["summary_output"],
        "with_vision_summary": with_vision["summary_output"],
        "without_vision_experiment": without_vision["experiment_output"],
        "with_vision_experiment": with_vision["experiment_output"],
        "without_vision_method": without_vision["method_output"],
        "with_vision_method": with_vision["method_output"],
    }
    _safe_dump(OUT_DIR / "comparison.json", comparison)
    _log("WROTE comparison.json")


if __name__ == "__main__":
    main()
