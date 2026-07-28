#!/usr/bin/env python3
"""Pure helpers for the frozen long-context retention benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from core.chat import ChatHistoryTurn, estimate_chat_tokens


BASELINES = ("A0_full_context", "A1_trim_only_8k", "A2_managed_8k")
CHANNELS = ("memory", "evidence", "recent")


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("benchmark_id") != "context-retention-v1":
        raise ValueError("Unsupported context-retention benchmark id.")
    facts = manifest.get("facts")
    groups = manifest.get("groups")
    if not isinstance(facts, list) or not facts:
        raise ValueError("Manifest must contain facts.")
    if not isinstance(groups, list) or not groups:
        raise ValueError("Manifest must contain question groups.")

    fact_ids: set[str] = set()
    for fact in facts:
        if not isinstance(fact, dict):
            raise ValueError("Every fact must be an object.")
        fact_id = str(fact.get("id") or "")
        if not fact_id or fact_id in fact_ids:
            raise ValueError(f"Duplicate or empty fact id: {fact_id!r}")
        fact_ids.add(fact_id)
        if fact.get("channel") not in CHANNELS:
            raise ValueError(f"Unsupported channel for {fact_id}.")
        if not all(str(fact.get(key) or "").strip() for key in ("statement", "question", "answer")):
            raise ValueError(f"Fact {fact_id} is missing statement, question, or answer.")
        if fact["channel"] in {"memory", "recent"}:
            index = fact.get("message_index")
            if not isinstance(index, int) or not 0 <= index < int(manifest["history_message_count"]):
                raise ValueError(f"Fact {fact_id} has an invalid message index.")
        if fact["channel"] == "memory" and fact.get("category") not in {
            "user",
            "feedback",
            "project",
            "reference",
        }:
            raise ValueError(f"Fact {fact_id} has an invalid memory category.")
        if fact["channel"] == "evidence" and not str(fact.get("evidence_id") or "").strip():
            raise ValueError(f"Fact {fact_id} is missing an evidence id.")

    grouped: list[str] = []
    group_ids: set[str] = set()
    for group in groups:
        group_id = str(group.get("id") or "")
        if not group_id or group_id in group_ids:
            raise ValueError(f"Duplicate or empty group id: {group_id!r}")
        group_ids.add(group_id)
        ids = group.get("fact_ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"Group {group_id} has no fact ids.")
        if any(str(fact_id) not in fact_ids for fact_id in ids):
            raise ValueError(f"Group {group_id} references an unknown fact.")
        grouped.extend(str(fact_id) for fact_id in ids)
    if sorted(grouped) != sorted(fact_ids):
        raise ValueError("Every frozen fact must appear in exactly one question group.")


def manifest_sha256(manifest: dict[str, Any]) -> str:
    payload = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fact_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(fact["id"]): fact for fact in manifest["facts"]}


def build_history(manifest: dict[str, Any]) -> list[ChatHistoryTurn]:
    facts_by_message: dict[int, list[str]] = defaultdict(list)
    for fact in manifest["facts"]:
        if fact["channel"] in {"memory", "recent"}:
            facts_by_message[int(fact["message_index"])].append(str(fact["statement"]))

    message_count = int(manifest["history_message_count"])
    target_total = int(manifest["target_history_tokens"])
    base_target, remainder = divmod(target_total, message_count)
    history: list[ChatHistoryTurn] = []
    for index in range(message_count):
        target = base_target + (1 if index < remainder else 0)
        prefix_parts = facts_by_message.get(index, [])
        prefix = "\n".join(
            [
                *prefix_parts,
                (
                    f"临时背景记录 H{index + 1:02d}：以下内容仅用于上下文长度压力测试，"
                    "不包含任何问题答案，也不得写入长期记忆。"
                ),
            ]
        ) + "\n"
        filler = "上下文占位。"
        content = pad_to_token_target(prefix, target, filler)
        history.append(
            ChatHistoryTurn(
                role="user" if index % 2 == 0 else "assistant",
                content=content,
            )
        )
    return history


def build_analysis_text(manifest: dict[str, Any]) -> str:
    evidence_statements = [
        f"[{fact['evidence_id']}] {fact['statement']}"
        for fact in manifest["facts"]
        if fact["channel"] == "evidence"
    ]
    prefix = "\n".join(evidence_statements) + "\n"
    filler = (
        "分析占位段：该段只用于构造长分析上下文，不包含额外答案，"
        "所有可判分的论文事实均已在本段开头声明。"
    )
    return pad_to_token_target(
        prefix,
        int(manifest["target_analysis_tokens"]),
        filler,
    )


def pad_to_token_target(prefix: str, target_tokens: int, filler_unit: str) -> str:
    """Pad deterministic text to the largest token count not exceeding the target."""
    if target_tokens <= 0:
        return ""
    prefix_tokens = estimate_chat_tokens(prefix)
    if prefix_tokens > target_tokens:
        raise ValueError("Frozen fact prefix exceeds its token target.")
    unit = f"\n{filler_unit}"
    unit_tokens = max(1, estimate_chat_tokens(unit))
    repeats = max(1, math.ceil((target_tokens - prefix_tokens) / unit_tokens) + 2)
    candidate = prefix + unit * repeats
    while estimate_chat_tokens(candidate) < target_tokens:
        candidate += unit

    low, high = len(prefix), len(candidate)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_chat_tokens(candidate[:middle]) <= target_tokens:
            low = middle
        else:
            high = middle - 1
    fitted = candidate[:low]
    return fitted


def build_question(manifest: dict[str, Any], group_id: str) -> tuple[str, dict[str, str]]:
    facts = fact_map(manifest)
    group = next((item for item in manifest["groups"] if item["id"] == group_id), None)
    if group is None:
        raise KeyError(f"Unknown question group: {group_id}")
    selected = [facts[str(fact_id)] for fact_id in group["fact_ids"]]
    gold = {str(fact["id"]): str(fact["answer"]) for fact in selected}
    questions = "\n".join(
        f"- {fact['id']}: {fact['question']}" for fact in selected
    )
    keys = ", ".join(gold)
    prompt = (
        "请仅根据当前提供的对话、长期记忆或论文原文证据回答下列事实问题。"
        "不得猜测，也不要解释。返回一个且仅返回一个合法JSON对象；"
        f"键必须严格为 {keys}，值只填写对应答案字符串，缺失时填写空字符串。\n"
        f"{questions}"
    )
    return prompt, gold


def parse_prediction(text: str, expected_ids: Iterable[str]) -> dict[str, str]:
    expected = [str(item) for item in expected_ids]
    candidates = [text.strip()]
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        candidates.append(fence.group(1).strip())
    extracted = _extract_json_object(text)
    if extracted:
        candidates.append(extracted)
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except Exception:
            continue
        if isinstance(value, dict):
            return {
                fact_id: _prediction_value(value.get(fact_id, ""))
                for fact_id in expected
            }
    return {fact_id: "" for fact_id in expected}


def score_prediction(
    prediction: dict[str, str],
    gold: dict[str, str],
    facts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    items: dict[str, dict[str, Any]] = {}
    for fact_id, answer in gold.items():
        predicted = str(prediction.get(fact_id) or "")
        correct = normalize_answer(predicted) == normalize_answer(answer)
        items[fact_id] = {
            "predicted": predicted,
            "gold": answer,
            "correct": correct,
            "channel": facts[fact_id]["channel"],
            "numeric": bool(facts[fact_id].get("numeric", False)),
        }
    correct_count = sum(bool(item["correct"]) for item in items.values())
    numeric_items = [item for item in items.values() if item["numeric"]]
    unsupported = sum(
        bool(item["predicted"]) and not bool(item["correct"])
        for item in items.values()
    )
    return {
        "items": items,
        "correct": correct_count,
        "total": len(items),
        "accuracy": _ratio(correct_count, len(items)),
        "numeric_exact_match": _ratio(
            sum(bool(item["correct"]) for item in numeric_items),
            len(numeric_items),
        ),
        "missing_rate": _ratio(
            sum(not bool(item["predicted"]) for item in items.values()),
            len(items),
        ),
        "unsupported_answer_rate": _ratio(unsupported, len(items)),
    }


def aggregate_results(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    facts = fact_map(manifest)
    output: dict[str, Any] = {}
    for baseline in BASELINES:
        baseline_rows = [
            row for row in rows
            if row.get("baseline") == baseline and not row.get("error")
        ]
        item_rows = [
            item
            for row in baseline_rows
            for item in (row.get("score") or {}).get("items", {}).values()
        ]
        latencies = [float(row["latency_ms"]) for row in baseline_rows]
        estimated_inputs = [
            float(row.get("estimated_input_tokens") or 0)
            for row in baseline_rows
        ]
        actual_inputs = [
            float((row.get("usage") or {}).get("input_tokens") or 0)
            for row in baseline_rows
            if (row.get("usage") or {}).get("input_tokens") is not None
        ]
        cache_reads = [
            float((row.get("usage") or {}).get("cache_read_tokens") or 0)
            for row in baseline_rows
        ]
        cold_latencies = [
            float(row["latency_ms"])
            for row in baseline_rows
            if int(row.get("repeat") or 0) == 0
        ]
        warm_latencies = [
            float(row["latency_ms"])
            for row in baseline_rows
            if int(row.get("repeat") or 0) > 0
        ]
        by_channel: dict[str, float] = {}
        for channel in CHANNELS:
            channel_items = [
                item for item in item_rows if item.get("channel") == channel
            ]
            by_channel[channel] = _ratio(
                sum(bool(item.get("correct")) for item in channel_items),
                len(channel_items),
            )
        numeric_items = [item for item in item_rows if item.get("numeric")]
        output[baseline] = {
            "calls": len(baseline_rows),
            "fact_observations": len(item_rows),
            "accuracy": _ratio(
                sum(bool(item.get("correct")) for item in item_rows),
                len(item_rows),
            ),
            "numeric_exact_match": _ratio(
                sum(bool(item.get("correct")) for item in numeric_items),
                len(numeric_items),
            ),
            "missing_rate": _ratio(
                sum(not bool(item.get("predicted")) for item in item_rows),
                len(item_rows),
            ),
            "unsupported_answer_rate": _ratio(
                sum(
                    bool(item.get("predicted")) and not bool(item.get("correct"))
                    for item in item_rows
                ),
                len(item_rows),
            ),
            "accuracy_by_channel": by_channel,
            "latency_p50_ms": _percentile(latencies, 0.50),
            "latency_p95_ms": _percentile(latencies, 0.95),
            "cold_latency_p50_ms": _percentile(cold_latencies, 0.50),
            "warm_latency_p50_ms": _percentile(warm_latencies, 0.50),
            "estimated_input_tokens_mean": (
                statistics.fmean(estimated_inputs) if estimated_inputs else 0.0
            ),
            "actual_input_tokens_mean": (
                statistics.fmean(actual_inputs) if actual_inputs else None
            ),
            "cache_read_tokens_mean": (
                statistics.fmean(cache_reads) if cache_reads else 0.0
            ),
            "errors": sum(
                row.get("baseline") == baseline and bool(row.get("error"))
                for row in rows
            ),
        }

    full = output["A0_full_context"]
    managed = output["A2_managed_8k"]
    trim_only = output["A1_trim_only_8k"]
    return {
        "benchmark_id": manifest["benchmark_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "facts": len(facts),
        "groups": len(manifest["groups"]),
        "repeats": max(
            (int(row.get("repeat") or 0) for row in rows),
            default=-1,
        ) + 1,
        "baselines": output,
        "comparisons": {
            "managed_vs_full_accuracy_delta_pp": round(
                100 * (managed["accuracy"] - full["accuracy"]),
                4,
            ),
            "managed_vs_trim_accuracy_delta_pp": round(
                100 * (managed["accuracy"] - trim_only["accuracy"]),
                4,
            ),
            "managed_estimated_input_reduction_vs_full_pct": round(
                100
                * (
                    1
                    - _ratio(
                        managed["estimated_input_tokens_mean"],
                        full["estimated_input_tokens_mean"],
                    )
                ),
                4,
            ),
        },
    }


def normalize_answer(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    normalized = normalized.replace("％", "%")
    normalized = re.sub(r"^[\"'`]+|[\"'`]+$", "", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.rstrip("。.,;；")
    return normalized


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _prediction_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value).strip()


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
