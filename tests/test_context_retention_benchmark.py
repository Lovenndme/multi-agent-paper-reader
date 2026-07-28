"""Tests for the frozen long-context retention benchmark helpers."""

from __future__ import annotations

import json
from pathlib import Path

from core.chat import estimate_chat_tokens
from tools.context_retention_benchmark import (
    aggregate_results,
    build_analysis_text,
    build_history,
    build_question,
    fact_map,
    load_manifest,
    manifest_sha256,
    parse_prediction,
    score_prediction,
)


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "benchmarks" / "context-retention-v1.json"


def test_frozen_manifest_builds_declared_candidate_token_volume() -> None:
    manifest = load_manifest(MANIFEST)
    history = build_history(manifest)
    analysis = build_analysis_text(manifest)

    assert len(history) == manifest["history_message_count"]
    assert all(len(turn.content) <= 8_000 for turn in history)
    assert sum(estimate_chat_tokens(turn.content) for turn in history) == 110_000
    assert estimate_chat_tokens(analysis) == 20_019
    assert len(manifest_sha256(manifest)) == 64


def test_question_and_exact_answer_scoring_are_deterministic() -> None:
    manifest = load_manifest(MANIFEST)
    question, gold = build_question(manifest, "G01")
    assert "M01" in question
    response = json.dumps(gold, ensure_ascii=False)
    prediction = parse_prediction(response, gold)
    score = score_prediction(prediction, gold, fact_map(manifest))

    assert score["accuracy"] == 1.0
    assert score["missing_rate"] == 0.0
    assert score["unsupported_answer_rate"] == 0.0


def test_aggregate_results_reports_primary_ab_deltas() -> None:
    manifest = load_manifest(MANIFEST)
    facts = fact_map(manifest)
    _, gold = build_question(manifest, "G01")
    perfect = score_prediction(gold, gold, facts)
    empty_prediction = {fact_id: "" for fact_id in gold}
    empty = score_prediction(empty_prediction, gold, facts)
    rows = []
    for baseline, score, tokens in (
        ("A0_full_context", perfect, 130_000),
        ("A1_trim_only_8k", empty, 7_000),
        ("A2_managed_8k", perfect, 7_500),
    ):
        rows.append(
            {
                "baseline": baseline,
                "group_id": "G01",
                "repeat": 0,
                "latency_ms": 10,
                "estimated_input_tokens": tokens,
                "usage": {"input_tokens": tokens},
                "score": score,
                "error": None,
            }
        )

    report = aggregate_results(rows, manifest)
    assert report["baselines"]["A0_full_context"]["accuracy"] == 1.0
    assert report["comparisons"]["managed_vs_trim_accuracy_delta_pp"] == 100.0
    assert report["comparisons"]["managed_estimated_input_reduction_vs_full_pct"] > 90
