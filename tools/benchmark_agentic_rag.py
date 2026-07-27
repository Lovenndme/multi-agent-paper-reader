#!/usr/bin/env python3
"""Score layered RAG baselines from source-anchored benchmark JSONL.

The benchmark deliberately separates retrieval quality from answer grounding.
It does not call a model itself: a run records predictions from each baseline under the
same paper/query gold set, then this script computes reproducible metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BASELINES = ("A0", "A1", "AQL", "AQ", "AQR", "A2", "A3", "A4")


def score_case(case: dict[str, Any], *, k: int = 10) -> dict[str, dict[str, float]]:
    """Score every available baseline for one validated benchmark case."""
    _validate_case(case)
    gold_ids = {str(item["evidence_id"]).upper() for item in case["gold_evidence"]}
    gold_anchor_facts = {
        str(item["source_anchor"]): str(
            item.get("fact_id") or item["source_anchor"]
        )
        for item in case["gold_evidence"]
    }
    gold_facts = {_normalize_fact(item) for item in case.get("gold_facts", []) if item}
    scores: dict[str, dict[str, float]] = {}
    for baseline, run in case["runs"].items():
        if baseline not in BASELINES or not isinstance(run, dict):
            continue
        retrieved = run.get("retrieved_evidence")
        if isinstance(retrieved, list):
            retrieval = anchored_retrieval_metrics(
                retrieved,
                gold_anchor_facts,
                k=k,
            )
        else:
            predicted = [
                str(item).upper()
                for item in run.get("retrieved_evidence_ids", [])
            ]
            retrieval = retrieval_metrics(predicted, gold_ids, k=k)
        supported_facts = {
            _normalize_fact(item)
            for item in run.get("supported_facts", [])
            if item
        }
        scores[baseline] = {
            **retrieval,
            "tool_success_rate": _ratio(
                run.get("successful_tool_calls", 0),
                run.get("tool_calls", 0),
                empty=1.0,
            ),
            "tool_calls": float(max(0, int(run.get("tool_calls", 0)))),
            "latency_ms": max(0.0, float(run.get("latency_ms", 0))),
            "evidence_items": float(
                max(
                    0,
                    int(
                        run.get(
                            "evidence_items",
                            len(retrieved)
                            if isinstance(retrieved, list)
                            else len(run.get("retrieved_evidence_ids", [])),
                        )
                    ),
                )
            ),
            "evidence_chars": float(max(0, int(run.get("evidence_chars", 0)))),
            "fallback_used": float(bool(run.get("fallback_used", False))),
            "error_rate": float(bool(run.get("error"))),
        }
        if run.get("adaptive_triggered") is not None:
            scores[baseline]["adaptive_trigger_rate"] = float(
                bool(run["adaptive_triggered"])
            )
        if "repair_success" in run:
            scores[baseline]["repair_success"] = float(
                bool(run.get("repair_success", False))
            )
        if gold_facts:
            factual = set_f1(supported_facts, gold_facts)
            scores[baseline].update(
                {
                    "grounded_fact_precision": factual["precision"],
                    "grounded_fact_recall": factual["recall"],
                    "grounded_fact_f1": factual["f1"],
                }
            )
    return scores


def retrieval_metrics(
    predicted: list[str],
    gold: set[str],
    *,
    k: int = 10,
) -> dict[str, float]:
    """Return Recall@k, Precision@k, reciprocal rank, and binary nDCG@k."""
    bounded_k = max(1, k)
    ranked = list(dict.fromkeys(predicted))[:bounded_k]
    hits = [1 if item in gold else 0 for item in ranked]
    hit_count = sum(hits)
    recall = _ratio(hit_count, len(gold), empty=1.0)
    precision = _ratio(hit_count, len(ranked), empty=0.0)
    first_hit = next((index + 1 for index, hit in enumerate(hits) if hit), None)
    reciprocal_rank = 1.0 / first_hit if first_hit else 0.0
    dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
    ideal_hits = min(len(gold), bounded_k)
    ideal_dcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return {
        f"recall@{bounded_k}": recall,
        f"precision@{bounded_k}": precision,
        "mrr": reciprocal_rank,
        f"ndcg@{bounded_k}": _ratio(dcg, ideal_dcg, empty=1.0),
    }


def anchored_retrieval_metrics(
    retrieved: list[Any],
    gold_anchors: set[str] | dict[str, str],
    *,
    k: int = 10,
) -> dict[str, float]:
    """Score ranked snippets by stable source anchors instead of ephemeral E/T/F IDs."""
    bounded_k = max(1, k)
    ranked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in retrieved:
        if not isinstance(item, dict):
            continue
        identity = str(
            item.get("source_anchor")
            or item.get("evidence_id")
            or f"rank-{len(ranked)}"
        )
        if identity in seen:
            continue
        seen.add(identity)
        ranked.append(item)
        if len(ranked) >= bounded_k:
            break

    anchor_facts = (
        gold_anchors
        if isinstance(gold_anchors, dict)
        else {anchor: anchor for anchor in gold_anchors}
    )
    gold_facts = set(anchor_facts.values())
    matched: set[str] = set()
    hits: list[int] = []
    for item in ranked:
        item_matches = {
            anchor_facts[str(anchor)]
            for anchor in item.get("matched_gold_anchors", [])
            if str(anchor) in anchor_facts
        }
        new_matches = item_matches - matched
        hits.append(1 if new_matches else 0)
        matched.update(new_matches)

    recall = _ratio(len(matched), len(gold_facts), empty=1.0)
    precision = _ratio(sum(hits), len(ranked), empty=0.0)
    first_hit = next((index + 1 for index, hit in enumerate(hits) if hit), None)
    reciprocal_rank = 1.0 / first_hit if first_hit else 0.0
    dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
    ideal_hits = min(len(gold_facts), bounded_k)
    ideal_dcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return {
        f"recall@{bounded_k}": recall,
        f"precision@{bounded_k}": precision,
        "mrr": reciprocal_rank,
        f"ndcg@{bounded_k}": _ratio(dcg, ideal_dcg, empty=1.0),
    }


def set_f1(predicted: set[str], gold: set[str]) -> dict[str, float]:
    true_positive = len(predicted & gold)
    precision = _ratio(true_positive, len(predicted), empty=1.0 if not gold else 0.0)
    recall = _ratio(true_positive, len(gold), empty=1.0)
    f1 = _ratio(2 * precision * recall, precision + recall, empty=0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def aggregate_scores(
    cases: Iterable[dict[str, Any]],
    *,
    k: int = 10,
) -> dict[str, dict[str, float]]:
    """Macro-average metrics without turning fixture results into system claims."""
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    count = 0
    for case in cases:
        count += 1
        for baseline, metrics in score_case(case, k=k).items():
            for metric, value in metrics.items():
                values[baseline][metric].append(value)
    if count == 0:
        raise ValueError("Benchmark input contains no cases.")
    return {
        baseline: {
            "cases": float(len(next(iter(metrics.values()), []))),
            **{
                metric: statistics.fmean(items)
                for metric, items in sorted(metrics.items())
                if items
            },
            **(
                {
                    "latency_p50_ms": statistics.median(metrics["latency_ms"]),
                    "latency_p95_ms": _percentile(metrics["latency_ms"], 0.95),
                }
                if metrics.get("latency_ms")
                else {}
            ),
        }
        for baseline, metrics in sorted(values.items())
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"Line {line_number} must contain one JSON object.")
        _validate_case(item)
        cases.append(item)
    return cases


def merge_cases(groups: Iterable[Iterable[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Merge disjoint baseline run files for the same case/repeat identities."""
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    for cases in groups:
        for case in cases:
            _validate_case(case)
            key = (str(case["case_id"]), int(case.get("repeat", 1)))
            if key not in merged:
                merged[key] = {**case, "runs": dict(case["runs"])}
                continue
            current = merged[key]
            for field in ("paper_sha256", "query", "gold_evidence"):
                if current.get(field) != case.get(field):
                    raise ValueError(
                        f"Cannot merge {key}: field {field} does not match."
                    )
            overlap = set(current["runs"]) & set(case["runs"])
            if overlap:
                raise ValueError(
                    f"Cannot merge {key}: duplicate baselines {sorted(overlap)}."
                )
            current["runs"].update(case["runs"])
    return list(merged.values())


def _validate_case(case: dict[str, Any]) -> None:
    if not str(case.get("case_id") or "").strip():
        raise ValueError("Each benchmark case requires case_id.")
    digest = str(case.get("paper_sha256") or "").lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("paper_sha256 must be a 64-character hexadecimal digest.")
    if not str(case.get("query") or "").strip():
        raise ValueError("Each benchmark case requires query.")
    gold = case.get("gold_evidence")
    if not isinstance(gold, list) or not gold:
        raise ValueError("Each case requires non-empty gold_evidence.")
    for item in gold:
        if not isinstance(item, dict):
            raise ValueError("gold_evidence items must be objects.")
        if not str(item.get("evidence_id") or "").strip():
            raise ValueError("Each gold evidence item requires evidence_id.")
        if not str(item.get("source_anchor") or "").strip():
            raise ValueError(
                "Each gold evidence item requires a stable source_anchor "
                "(page plus quote hash or equivalent)."
            )
    runs = case.get("runs")
    if not isinstance(runs, dict) or not runs:
        raise ValueError("Each case requires at least one supported baseline run.")


def _normalize_fact(value: Any) -> str:
    return " ".join(str(value).lower().split())


def _ratio(numerator: float, denominator: float, *, empty: float) -> float:
    return numerator / denominator if denominator else empty


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * min(1.0, max(0.0, quantile))
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score source-anchored Agentic RAG benchmark runs."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="+",
        help="One or more benchmark JSONL run files",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--k", type=int, default=16)
    args = parser.parse_args()
    cases = merge_cases(load_jsonl(path) for path in args.input)
    report = {
        "input": [str(path) for path in args.input],
        "k": max(1, args.k),
        "baselines": {
            "A0": "lexical role-static retrieval",
            "A1": "FastEmbed role-static retrieval",
            "AQL": "legacy parent-chunk query-aware retrieval",
            "AQ": "hierarchical BM25 + Dense RRF retrieval",
            "AQR": "hierarchical retrieval + local cross-encoder",
            "A2": "structured-action Agentic RAG",
            "A3": "native-tool Agentic RAG",
            "A4": "adaptive native-tool Agentic RAG",
        },
        "metrics": aggregate_scores(cases, k=args.k),
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
