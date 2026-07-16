"""Benchmark visual QA accuracy at 120 DPI, 144 DPI, and native crop quality.

The benchmark keeps crop coordinates and prompts identical across profiles so
that image resolution is the only intended variable. It is deliberately a
credentialed, opt-in tool: responses are produced by the locally configured
vision route and the report never includes credentials or authentication data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=False)

from core.codex_sdk import CodexSDKError, get_codex_sdk_service
from core.model_providers import (
    selected_text_mode,
    selected_vision_model,
    vision_provider_id,
)
from core.pdf_rendering import MAX_RENDER_DIMENSION, MAX_RENDER_PIXELS


PROFILES = ("120dpi", "144dpi", "native")
PROFILE_DPI = {"120dpi": 120, "144dpi": 144}
MATERIAL_MIN_FIELDS = 2
MATERIAL_MIN_PERCENTAGE_POINTS = 5.0


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    filename: str
    kind: str
    questions: dict[str, str]
    expected: dict[str, Any]


CASES = (
    BenchmarkCase(
        id="figure_1",
        filename="page-02_figure-1.png",
        kind="figure",
        questions={
            "panel_titles": "List the titles of panels (a), (b), and (c), in that order.",
            "panel_b_segment_count": "How many blue segmented-series boxes X^i are shown in panel (b)?",
            "panel_c_segmental_chunks": "List the three blue word chunks in the Segmental Processing box, top to bottom.",
        },
        expected={
            "panel_titles": [
                "Previous fMRI-to-text Framework",
                "Our fMRI-to-text Framework",
                "Language comprehension mechanism",
            ],
            "panel_b_segment_count": 3,
            "panel_c_segmental_chunks": ["It was a", "rainy day", "when Emma"],
        },
    ),
    BenchmarkCase(
        id="figure_2",
        filename="page-04_figure-2.png",
        kind="figure",
        questions={
            "phase1_mask_layer": "What mask-layer label appears in Phase 1 of Stage A?",
            "phase2_mask_layer": "What mask-layer label appears in Phase 2 of Stage A?",
            "wrap_up_text_encoder": "Which text encoder is used in the gray Wrap-up inset in Stage B?",
            "main_text_decoder": "What is the large decoder block in Stage B called?",
        },
        expected={
            "phase1_mask_layer": "Random Mask Layer",
            "phase2_mask_layer": "Text-guided Mask Layer",
            "wrap_up_text_encoder": "BERT Encoder",
            "main_text_decoder": "BART Decoder",
        },
    ),
    BenchmarkCase(
        id="figure_3",
        filename="page-05_figure-3.png",
        kind="figure",
        questions={
            "left_encoder": "What encoder is named in the leftmost stage?",
            "grouped_text_box_count": "How many blue Grouped Text boxes are shown?",
            "attention_labels": "List the three TR Frame Attention labels from top to bottom.",
        },
        expected={
            "left_encoder": "Pretrained BERT",
            "grouped_text_box_count": 3,
            "attention_labels": ["TR1", "TR2", "TR3"],
        },
    ),
    BenchmarkCase(
        id="figure_4",
        filename="page-07_figure-4.png",
        kind="figure",
        questions={
            "metrics": "List the three y-axis labels from left to right.",
            "bleu_peak_tr": "At which input fMRI length (TR) does BLEU-1 reach its highest point?",
            "rouge_peak_tr": "At which input fMRI length (TR) does ROUGE-R reach its highest point?",
            "bertscore_peak_tr": "At which input fMRI length (TR) does BERTScore-R reach its highest point?",
        },
        expected={
            "metrics": ["BLEU-1", "ROUGE-R", "BERTScore-R"],
            "bleu_peak_tr": 20,
            "rouge_peak_tr": 30,
            "bertscore_peak_tr": 20,
        },
    ),
    BenchmarkCase(
        id="figure_5",
        filename="page-08_figure-5.png",
        kind="figure",
        questions={
            "legend_labels": "List every legend label from top to bottom.",
            "highest_bleu_at_60": "Which legend series is highest for BLEU-1 at 60 TR?",
            "highest_rouge_at_60": "Which legend series is highest for ROUGE-R at 60 TR?",
            "highest_bertscore_at_60": "Which legend series is highest for BERTScore-R at 60 TR?",
        },
        expected={
            "legend_labels": ["w/o MLP", "MLP32", "MLP64", "MLP128", "MLP256"],
            "highest_bleu_at_60": "MLP32",
            "highest_rouge_at_60": "MLP32",
            "highest_bertscore_at_60": "MLP128",
        },
    ),
    BenchmarkCase(
        id="table_1",
        filename="page-08_table-1.png",
        kind="table",
        questions={
            "cogreader_20tr_bleu4": "CogReader (ours), 20TR: BLEU-4.",
            "cogreader_40tr_bertscore_r": "CogReader (ours), 40TR: BERTScore-R.",
            "predft_60tr_rouge_r": "PREDFT, 60TR: ROUGE-R.",
            "cogreader_60tr_bleu1": "CogReader (ours), 60TR: BLEU-1.",
        },
        expected={
            "cogreader_20tr_bleu4": 2.6,
            "cogreader_40tr_bertscore_r": 51.1,
            "predft_60tr_rouge_r": 20.5,
            "cogreader_60tr_bleu1": 36.2,
        },
    ),
    BenchmarkCase(
        id="table_2",
        filename="page-09_table-2.png",
        kind="table",
        questions={
            "target_figure_name": "What name did the Target text give the figure?",
            "target_latin_phrase": "What Latin catchphrase appears in the Target text?",
            "ours_emergence_location": "From what location does the Ours text say Pie Man emerged?",
            "unicorn_creature": "What creatures are mentioned in the UniCoRN text?",
        },
        expected={
            "target_figure_name": "Pie Man",
            "target_latin_phrase": "Ego sum non an bestia",
            "ours_emergence_location": "the late night library drop",
            "unicorn_creature": "dragons",
        },
    ),
    BenchmarkCase(
        id="table_3",
        filename="page-09_table-3.png",
        kind="table",
        questions={
            "all_disabled_bleu1": "Row with all three components disabled: BLEU-1.",
            "sequential_only_bleu1": "Row with only Sequential Decoding enabled: BLEU-1.",
            "pretraining_and_masking_only_bertscore_r": "Row with Sequential Decoding disabled and the other two enabled: BERTScore-R.",
            "all_enabled_bleu4": "Row with all three components enabled: BLEU-4.",
        },
        expected={
            "all_disabled_bleu1": 17.7,
            "sequential_only_bleu1": 32.5,
            "pretraining_and_masking_only_bertscore_r": 47.2,
            "all_enabled_bleu4": 12.1,
        },
    ),
    BenchmarkCase(
        id="table_4",
        filename="page-10_table-4.png",
        kind="table",
        questions={
            "ours_10tr_bertscore_r": "Ours, 10TR: BERTScore-R.",
            "unicorn_30tr_bleu2": "UniCoRN, 30TR: BLEU-2.",
            "ours_40tr_rouge_p": "Ours, 40TR: ROUGE-P.",
            "ours_50tr_bertscore_p": "Ours, 50TR: BERTScore-P.",
        },
        expected={
            "ours_10tr_bertscore_r": 41.8,
            "unicorn_30tr_bleu2": 2.8,
            "ours_40tr_rouge_p": 27.0,
            "ours_50tr_bertscore_p": 47.7,
        },
    ),
    BenchmarkCase(
        id="table_5",
        filename="page-10_table-5.png",
        kind="table",
        questions={
            "noise_train_noise_test_bleu1": "Noise train / Noise test: BLEU-1.",
            "noise_train_fmri_test_bertscore_r": "Noise train / fMRI test: BERTScore-R.",
            "fmri_train_noise_test_rouge_p": "fMRI train / Noise test: ROUGE-P.",
            "fmri_train_fmri_test_bleu4": "fMRI train / fMRI test: BLEU-4.",
        },
        expected={
            "noise_train_noise_test_bleu1": 27.5,
            "noise_train_fmri_test_bertscore_r": 48.3,
            "fmri_train_noise_test_rouge_p": 23.1,
            "fmri_train_fmri_test_bleu4": 12.1,
        },
    ),
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _schema_for(case: BenchmarkCase) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for key, expected in case.expected.items():
        if isinstance(expected, list):
            properties[key] = {
                "type": "array",
                "items": {"type": "string"},
                "minItems": len(expected),
                "maxItems": len(expected),
            }
        elif isinstance(expected, int):
            properties[key] = {"type": "integer"}
        elif isinstance(expected, float):
            properties[key] = {"type": "number"}
        else:
            properties[key] = {"type": "string"}
    return {
        "type": "object",
        "properties": properties,
        "required": list(case.expected),
        "additionalProperties": False,
    }


def _prompt_for(case: BenchmarkCase) -> str:
    questions = "\n".join(f"- {key}: {question}" for key, question in case.questions.items())
    return (
        "This is a blind visual-reading benchmark. Inspect only the attached scientific "
        "figure or table crop. Do not use web search, tools, filenames, prior paper knowledge, "
        "or guessed values. Copy labels and numeric cells from the image. Return only the JSON "
        "object required by the response schema.\n\nFields:\n"
        + questions
    )


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = text.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
    return "".join(character for character in text if character.isalnum())


def field_matches(actual: Any, expected: Any) -> bool:
    """Compare benchmark fields while ignoring harmless label punctuation/case."""
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            field_matches(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=0.05)
        except (TypeError, ValueError):
            return False
    return _normalize_text(actual) == _normalize_text(expected)


def score_answer(case: BenchmarkCase, answer: dict[str, Any]) -> dict[str, Any]:
    fields = {
        key: {
            "expected": expected,
            "actual": answer.get(key),
            "correct": field_matches(answer.get(key), expected),
        }
        for key, expected in case.expected.items()
    }
    correct = sum(bool(item["correct"]) for item in fields.values())
    return {"correct": correct, "total": len(fields), "fields": fields}


def _render_assets(pdf_path: Path, index_path: Path, output_dir: Path) -> dict[str, Any]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    indexed = {str(item["file"]): item for item in index["items"]}
    manifest: dict[str, Any] = {}
    document = fitz.open(pdf_path)
    try:
        for case in CASES:
            item = indexed[case.filename]
            page = document[int(item["page"]) - 1]
            clip = fitz.Rect(item["crop"])
            for profile in PROFILES:
                dpi = int(item["render_dpi"]) if profile == "native" else PROFILE_DPI[profile]
                matrix = fitz.Matrix(dpi / 72, dpi / 72)
                pixel_rect = (clip * matrix).irect
                width, height = pixel_rect.width, pixel_rect.height
                if (
                    width > MAX_RENDER_DIMENSION
                    or height > MAX_RENDER_DIMENSION
                    or width * height > MAX_RENDER_PIXELS
                ):
                    raise ValueError(
                        f"{case.id}/{profile} requires {width}x{height}; safe benchmark limit exceeded."
                    )
                destination = output_dir / "assets" / profile / case.filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                pixmap = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
                pixmap.save(destination)
                manifest[f"{profile}:{case.id}"] = {
                    "path": str(destination),
                    "dpi": dpi,
                    "width_px": pixmap.width,
                    "height_px": pixmap.height,
                    "pixels": pixmap.width * pixmap.height,
                    "bytes": destination.stat().st_size,
                }
    finally:
        document.close()
    return manifest


def _run_one(
    case: BenchmarkCase,
    profile: str,
    repeat: int,
    asset: dict[str, Any],
    *,
    model: str,
    effort: str,
) -> dict[str, Any]:
    image_bytes = Path(asset["path"]).read_bytes()
    started = time.perf_counter()
    attempts = 0
    last_error: Exception | None = None
    while attempts < 2:
        attempts += 1
        try:
            result = get_codex_sdk_service().run_text(
                _prompt_for(case),
                model=model,
                effort=effort or None,
                output_schema=_schema_for(case),
                image_bytes=image_bytes,
                timeout=240,
            )
            answer = json.loads(result.text)
            return {
                "key": f"r{repeat}:{profile}:{case.id}",
                "repeat": repeat,
                "profile": profile,
                "case": case.id,
                "kind": case.kind,
                "model": result.model,
                "effort": result.effort,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "attempts": attempts,
                "web_search_used": result.web_search_used,
                "tools_used": list(result.tools_used),
                "asset": asset,
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "answer": answer,
                "score": score_answer(case, answer),
                "status": "passed",
            }
        except (CodexSDKError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempts < 2:
                time.sleep(2)
    return {
        "key": f"r{repeat}:{profile}:{case.id}",
        "repeat": repeat,
        "profile": profile,
        "case": case.id,
        "kind": case.kind,
        "model": model,
        "effort": effort,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "attempts": attempts,
        "asset": asset,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "status": "failed",
        "error_type": type(last_error).__name__ if last_error else "UnknownError",
    }


def _percent(correct: int, total: int) -> float:
    return round(100.0 * correct / total, 2) if total else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _value_signature(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_value_signature(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), 6)
    return _normalize_text(value)


def _majority_metrics(passed: list[dict[str, Any]], *, kind: str | None = None) -> dict[str, Any]:
    case_index = {case.id: case for case in CASES}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in passed:
        case = case_index[str(record["case"])]
        if kind is not None and case.kind != kind:
            continue
        for field_name, field_score in record["score"]["fields"].items():
            groups.setdefault((case.id, field_name), []).append(field_score)

    correct = 0
    consistent = 0
    for field_scores in groups.values():
        threshold = len(field_scores) // 2 + 1
        correct += sum(bool(item["correct"]) for item in field_scores) >= threshold
        signatures = {_value_signature(item.get("actual")) for item in field_scores}
        consistent += len(signatures) == 1
    total = len(groups)
    return {
        "correct": correct,
        "total": total,
        "accuracy_percent": _percent(correct, total),
        "repeat_consistent_fields": consistent,
        "repeat_consistency_percent": _percent(consistent, total),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for profile in PROFILES:
        records = [item for item in results if item.get("profile") == profile]
        passed = [item for item in records if item.get("status") == "passed"]
        observed_correct = sum(int(item["score"]["correct"]) for item in passed)
        observed_total = sum(int(item["score"]["total"]) for item in passed)
        majority = _majority_metrics(passed)
        latencies = [float(item["elapsed_seconds"]) for item in passed]
        kinds: dict[str, Any] = {}
        for kind in ("figure", "table"):
            kinds[kind] = _majority_metrics(passed, kind=kind)
        profiles[profile] = {
            "requests": len(records),
            "passed_requests": len(passed),
            "failed_requests": len(records) - len(passed),
            "correct": majority["correct"],
            "total": majority["total"],
            "accuracy_percent": majority["accuracy_percent"],
            "repeat_consistent_fields": majority["repeat_consistent_fields"],
            "repeat_consistency_percent": majority["repeat_consistency_percent"],
            "observations": {
                "correct": observed_correct,
                "total": observed_total,
                "accuracy_percent": _percent(observed_correct, observed_total),
            },
            "by_kind": kinds,
            "latency_seconds": {
                "mean": round(statistics.mean(latencies), 3) if latencies else 0.0,
                "median": round(statistics.median(latencies), 3) if latencies else 0.0,
                "p95": round(_percentile(latencies, 0.95), 3),
                "total": round(sum(latencies), 3),
            },
            "image_bytes_total": sum(int(item["asset"]["bytes"]) for item in records),
            "image_pixels_total": sum(int(item["asset"]["pixels"]) for item in records),
            "web_search_requests": sum(bool(item.get("web_search_used")) for item in passed),
            "tool_using_requests": sum(bool(item.get("tools_used")) for item in passed),
        }

    baseline = profiles["144dpi"]
    native = profiles["native"]
    comparisons: dict[str, Any] = {}
    for profile in ("120dpi", "144dpi"):
        base = profiles[profile]
        comparisons[f"native_vs_{profile}"] = {
            "accuracy_delta_percentage_points": round(
                native["accuracy_percent"] - base["accuracy_percent"], 2
            ),
            "correct_field_delta": native["correct"] - base["correct"],
            "mean_latency_ratio": (
                round(
                    native["latency_seconds"]["mean"]
                    / base["latency_seconds"]["mean"],
                    3,
                )
                if base["latency_seconds"]["mean"]
                else 0.0
            ),
            "image_bytes_ratio": (
                round(native["image_bytes_total"] / base["image_bytes_total"], 3)
                if base["image_bytes_total"]
                else 0.0
            ),
            "image_pixels_ratio": (
                round(native["image_pixels_total"] / base["image_pixels_total"], 3)
                if base["image_pixels_total"]
                else 0.0
            ),
        }

    decisions: dict[str, Any] = {}
    for kind in ("figure", "table"):
        before = baseline["by_kind"][kind]
        after = native["by_kind"][kind]
        field_delta = after["correct"] - before["correct"]
        pp_delta = round(after["accuracy_percent"] - before["accuracy_percent"], 2)
        material = (
            field_delta >= MATERIAL_MIN_FIELDS
            and pp_delta >= MATERIAL_MIN_PERCENTAGE_POINTS
        )
        decisions[kind] = {
            "native_is_materially_better": material,
            "correct_field_delta": field_delta,
            "accuracy_delta_percentage_points": pp_delta,
            "recommended_model_profile": "native" if material else "144dpi",
        }
    recommendations = {item["recommended_model_profile"] for item in decisions.values()}
    overall_policy = (
        recommendations.pop()
        if len(recommendations) == 1
        else "hybrid-by-visual-kind"
    )
    return {
        "profiles": profiles,
        "comparisons": comparisons,
        "decision_rule": {
            "baseline": "144dpi",
            "minimum_additional_majority_correct_fields_per_kind": MATERIAL_MIN_FIELDS,
            "minimum_accuracy_gain_percentage_points_per_kind": MATERIAL_MIN_PERCENTAGE_POINTS,
        },
        "decisions": decisions,
        "recommended_model_policy": overall_policy,
        "native_export_policy": "always-retain-when-render-limits-allow",
    }


def _write_report(path: Path, metadata: dict[str, Any], summary: dict[str, Any]) -> None:
    profiles = summary["profiles"]
    native_comparison = summary["comparisons"]["native_vs_144dpi"]
    lines = [
        "# 视觉分辨率 A/B 基准",
        "",
        f"- 模型：`{metadata['model']}`",
        f"- 推理强度：`{metadata['effort']}`",
        f"- 题集：{metadata['cases']} 张视觉区域，{metadata['fields']} 个客观字段，"
        f"每档重复 {metadata['repeats']} 次并以字段多数票计分",
        f"- 变量控制：三档使用相同裁剪框、相同提示词和相同输出 Schema；调用顺序固定随机化。",
        "",
        "| 输入档位 | 多数票正确字段 | 准确率 | Figure | Table | 重复一致率 | 平均延迟 | 图像总字节 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        item = profiles[profile]
        lines.append(
            f"| {profile} | {item['correct']}/{item['total']} | "
            f"{item['accuracy_percent']:.2f}% | "
            f"{item['by_kind']['figure']['accuracy_percent']:.2f}% | "
            f"{item['by_kind']['table']['accuracy_percent']:.2f}% | "
            f"{item['repeat_consistency_percent']:.2f}% | "
            f"{item['latency_seconds']['mean']:.2f}s | {item['image_bytes_total']:,} |"
        )
    lines.extend(
        [
            "",
            "## 判定",
            "",
            f"原生分辨率相对 144 DPI：准确率变化 "
            f"{native_comparison['accuracy_delta_percentage_points']:+.2f} 个百分点，"
            f"正确字段变化 {native_comparison['correct_field_delta']:+d}。",
            "",
            f"最终模型输入策略：`{summary['recommended_model_policy']}`。",
            "无论模型输入采用哪一档，只要不超过预分配安全限制，均保留原生 PNG 渲染能力。",
            "",
            "按视觉类型：",
            "",
        ]
    )
    for kind, decision in summary["decisions"].items():
        lines.append(
            f"- {kind}: `{decision['recommended_model_profile']}`，"
            f"相对 144 DPI {decision['accuracy_delta_percentage_points']:+.2f} 个百分点，"
            f"{decision['correct_field_delta']:+d} 个字段。"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output" / "pdf" / "vision-resolution-benchmark",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=560144)
    args = parser.parse_args()

    if vision_provider_id() != "codex":
        raise RuntimeError(
            "This reproducible benchmark currently requires the local Codex vision route."
        )
    model = selected_vision_model()
    effort = selected_text_mode()
    status = get_codex_sdk_service().status(force=True)
    if not status.get("runtime_ready") or not status.get("authenticated"):
        raise RuntimeError("Local Codex subscription is not ready for a credentialed benchmark.")

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = _render_assets(args.pdf.resolve(), args.index.resolve(), output_dir)
    results_path = output_dir / "results.json"
    if results_path.exists():
        stored = json.loads(results_path.read_text(encoding="utf-8"))
        results = list(stored.get("results", []))
    else:
        results = []
    case_index = {case.id: case for case in CASES}
    for record in results:
        if record.get("status") == "passed" and isinstance(record.get("answer"), dict):
            case = case_index.get(str(record.get("case")))
            if case is not None:
                record["score"] = score_answer(case, record["answer"])
    repeats = max(1, min(args.repeats, 5))
    completed = {
        str(item.get("key"))
        for item in results
        if item.get("status") == "passed"
    }
    jobs = [
        (case, profile, repeat)
        for case in CASES
        for profile in PROFILES
        for repeat in range(1, repeats + 1)
        if f"r{repeat}:{profile}:{case.id}" not in completed
    ]
    random.Random(args.seed).shuffle(jobs)
    metadata = {
        "provider": "codex",
        "model": model,
        "effort": effort,
        "cases": len(CASES),
        "fields": sum(len(case.expected) for case in CASES),
        "profiles": list(PROFILES),
        "repeats": repeats,
        "seed": args.seed,
        "source_pdf": str(args.pdf.resolve()),
    }
    print(
        f"benchmark model={model} effort={effort} jobs={len(jobs)} "
        f"already_complete={len(completed)}",
        flush=True,
    )
    try:
        with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 4))) as executor:
            future_map = {
                executor.submit(
                    _run_one,
                    case,
                    profile,
                    repeat,
                    assets[f"{profile}:{case.id}"],
                    model=model,
                    effort=effort,
                ): (case, profile, repeat)
                for case, profile, repeat in jobs
            }
            for index, future in enumerate(as_completed(future_map), start=1):
                case, profile, repeat = future_map[future]
                record = future.result()
                results = [item for item in results if item.get("key") != record["key"]]
                results.append(record)
                results.sort(key=lambda item: str(item.get("key")))
                _atomic_json(results_path, {"metadata": metadata, "results": results})
                score = record.get("score") or {}
                print(
                    f"[{index}/{len(jobs)}] r{repeat}/{profile}/{case.id} "
                    f"status={record['status']} score={score.get('correct', '-')}/"
                    f"{score.get('total', '-')} elapsed={record['elapsed_seconds']}s",
                    flush=True,
                )
    finally:
        get_codex_sdk_service().close()

    if (
        any(item.get("status") != "passed" for item in results)
        or len(results) != len(CASES) * len(PROFILES) * repeats
    ):
        print("Benchmark is incomplete; rerun the same command to resume.", flush=True)
        return 2

    _atomic_json(results_path, {"metadata": metadata, "results": results})
    summary = summarize(results)
    _atomic_json(output_dir / "summary.json", {"metadata": metadata, "summary": summary})
    _write_report(output_dir / "report.md", metadata, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
