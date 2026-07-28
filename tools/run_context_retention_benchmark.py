#!/usr/bin/env python3
"""Run the frozen long-context A/B benchmark against the configured text model."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.chat import (  # noqa: E402
    PaperChatRequest,
    build_chat_prompt,
    clear_analysis_sessions,
    estimate_chat_tokens,
    store_analysis_session,
)
from core.chat_memory import (  # noqa: E402
    add_conversation_message,
    create_conversation,
    refresh_conversation_memory,
)
from core.evidence import EvidenceSnippet  # noqa: E402
from core.history import save_paper_analysis  # noqa: E402
from core.langmem_store import list_langmem_memories, reset_langmem_store  # noqa: E402
from core.model_providers import (  # noqa: E402
    selected_text_mode,
    selected_text_model,
    text_provider_id,
)
from tools.context_retention_benchmark import (  # noqa: E402
    BASELINES,
    aggregate_results,
    build_analysis_text,
    build_history,
    build_question,
    fact_map,
    load_manifest,
    manifest_sha256,
    normalize_answer,
    parse_prediction,
    score_prediction,
)
from utils.llm import get_chat_llm_for_route, invoke_with_retry  # noqa: E402


DEFAULT_MANIFEST = ROOT / "benchmarks" / "context-retention-v1.json"
DEFAULT_OUTPUT_DIR = ROOT / ".paper-reader" / "benchmarks" / "context-retention-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--provider", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--mode", default="fast")
    parser.add_argument("--repeats", type=int, default=0)
    parser.add_argument(
        "--baselines",
        default=",".join(BASELINES),
        help=f"Comma-separated subset of: {', '.join(BASELINES)}",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    provider = args.provider.strip() or text_provider_id()
    model = args.model.strip() or selected_text_model(provider)
    mode = args.mode.strip() or selected_text_mode(provider, model)
    repeats = args.repeats or int(manifest["default_repeats"])
    if repeats <= 0 or repeats > 10:
        raise ValueError("Repeats must be between 1 and 10.")
    baselines = tuple(
        item.strip() for item in args.baselines.split(",") if item.strip()
    )
    unknown = sorted(set(baselines) - set(BASELINES))
    if unknown:
        raise ValueError(f"Unknown baselines: {', '.join(unknown)}")
    if not baselines:
        raise ValueError("At least one baseline is required.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    report_path = output_dir / "report.json"
    if args.overwrite:
        results_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)

    existing_rows = _load_jsonl(results_path)
    completed = {
        str(row.get("run_key"))
        for row in existing_rows
        if row.get("manifest_sha256") == manifest_sha256(manifest)
        and row.get("provider") == provider
        and row.get("model") == model
        and row.get("mode") == mode
        and not row.get("error")
    }

    history = build_history(manifest)
    analysis_text = build_analysis_text(manifest)
    candidate_tokens = {
        "history": sum(estimate_chat_tokens(turn.content) for turn in history),
        "analysis": estimate_chat_tokens(analysis_text),
    }
    candidate_tokens["total"] = candidate_tokens["history"] + candidate_tokens["analysis"]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "benchmark_id": manifest["benchmark_id"],
                    "manifest_sha256": manifest_sha256(manifest),
                    "provider": provider,
                    "model": model,
                    "mode": mode,
                    "facts": len(manifest["facts"]),
                    "groups": len(manifest["groups"]),
                    "repeats": repeats,
                    "baselines": baselines,
                    "candidate_tokens": candidate_tokens,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    with _benchmark_environment(int(manifest["input_budget_tokens"])) as data_dir:
        reset_langmem_store()
        clear_analysis_sessions()
        runtime = _prepare_runtime(
            manifest,
            history,
            analysis_text,
            provider=provider,
            model=model,
            mode=mode,
        )
        llm = get_chat_llm_for_route(provider, model, mode).bind(max_tokens=600)
        new_rows: list[dict[str, Any]] = []
        try:
            for group in manifest["groups"]:
                group_id = str(group["id"])
                question, gold = build_question(manifest, group_id)
                prompts = _build_prompts(
                    manifest,
                    runtime,
                    history,
                    question,
                    provider=provider,
                    model=model,
                    mode=mode,
                )
                for repeat in range(repeats):
                    for baseline in baselines:
                        run_key = f"{baseline}:{group_id}:{repeat}"
                        if run_key in completed:
                            continue
                        row = _run_one(
                            llm,
                            prompts[baseline],
                            manifest=manifest,
                            baseline=baseline,
                            group_id=group_id,
                            repeat=repeat,
                            gold=gold,
                            provider=provider,
                            model=model,
                            mode=mode,
                        )
                        _append_jsonl(results_path, row)
                        new_rows.append(row)
                        completed.add(run_key)
                        status = (
                            f"{row['score']['correct']}/{row['score']['total']}"
                            if not row.get("error")
                            else f"ERROR: {row['error']}"
                        )
                        print(
                            f"[{len(completed):02d}] {run_key} "
                            f"accuracy={status} latency={row['latency_ms']:.1f}ms",
                            flush=True,
                        )
        finally:
            clear_analysis_sessions()
            reset_langmem_store()

        rows = _matching_rows(
            _load_jsonl(results_path),
            manifest_hash=manifest_sha256(manifest),
            provider=provider,
            model=model,
            mode=mode,
        )
        report = aggregate_results(rows, manifest)
        report.update(
            {
                "provider": provider,
                "model": model,
                "mode": mode,
                "candidate_tokens": candidate_tokens,
                "input_budget_tokens": int(manifest["input_budget_tokens"]),
                "memory_extraction": runtime["memory_extraction"],
                "data_directory_is_temporary": True,
                "result_rows": len(rows),
                "new_rows": len(new_rows),
            }
        )
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))


def _prepare_runtime(
    manifest: dict[str, Any],
    history: list[Any],
    analysis_text: str,
    *,
    provider: str,
    model: str,
    mode: str,
) -> dict[str, Any]:
    facts = fact_map(manifest)
    snippets = [
        EvidenceSnippet(
            id=str(fact["evidence_id"]),
            section=str(fact.get("section") or "Evidence"),
            page_start=index,
            page_end=index,
            text=str(fact["statement"]),
            kind="table" if str(fact["evidence_id"]).startswith("T") else "text",
        )
        for index, fact in enumerate(
            item for item in manifest["facts"] if item["channel"] == "evidence"
        )
    ]
    pdf_data = b"%PDF-1.4 frozen context retention benchmark"
    context = {
        "mode": "live",
        "paper": {
            "title": "Frozen Context Retention Evaluation Paper",
            "filename": "context-retention-evaluation.pdf",
            "pages": len(snippets),
            "sections_count": len(snippets),
            "size_bytes": len(pdf_data),
        },
        "summary_output": {"notes": analysis_text},
    }
    history_id = save_paper_analysis(
        pdf_data=pdf_data,
        result=context,
        snippets=snippets,
    )
    conversation = create_conversation(history_id, title="Context Retention Benchmark")
    conversation_id = str(conversation["id"])
    for turn in history:
        add_conversation_message(
            conversation_id,
            role=turn.role,
            content=turn.content,
        )

    started = time.perf_counter()
    processed_messages = refresh_conversation_memory(
        conversation_id,
        text_provider=provider,
        text_model=model,
        text_mode=mode,
    )
    memory_latency_ms = (time.perf_counter() - started) * 1000
    records = list_langmem_memories(history_id, limit=20, min_score=-1)
    memory_facts = [
        fact for fact in facts.values() if fact["channel"] == "memory"
    ]
    extracted: dict[str, bool] = {}
    record_text = normalize_answer(
        "\n".join(
            " ".join(
                str(record.get(key) or "")
                for key in ("topic", "description", "content")
            )
            for record in records
        )
    )
    for fact in memory_facts:
        extracted[str(fact["id"])] = (
            normalize_answer(str(fact["answer"])) in record_text
        )

    analysis_id = store_analysis_session(snippets, context)
    return {
        "history_id": history_id,
        "conversation_id": conversation_id,
        "analysis_id": analysis_id,
        "context": context,
        "analysis_text": analysis_text,
        "memory_extraction": {
            "processed_messages": processed_messages,
            "latency_ms": round(memory_latency_ms, 3),
            "records": len(records),
            "gold_facts": len(memory_facts),
            "facts_extracted": sum(extracted.values()),
            "recall": (
                sum(extracted.values()) / len(memory_facts)
                if memory_facts
                else 0.0
            ),
            "per_fact": extracted,
        },
    }


def _build_prompts(
    manifest: dict[str, Any],
    runtime: dict[str, Any],
    history: list[Any],
    question: str,
    *,
    provider: str,
    model: str,
    mode: str,
) -> dict[str, tuple[BaseMessage, ...]]:
    trimmed = build_chat_prompt(
        PaperChatRequest(
            question=question,
            history=history,
            context=runtime["context"],
            text_provider=provider,
            text_model=model,
            text_mode=mode,
        )
    )
    managed = build_chat_prompt(
        PaperChatRequest(
            question=question,
            analysis_id=runtime["analysis_id"],
            history_id=runtime["history_id"],
            conversation_id=runtime["conversation_id"],
            context=runtime["context"],
            text_provider=provider,
            text_model=model,
            text_mode=mode,
        )
    )

    managed_system = str(managed.messages[0].content)
    full_system = (
        f"{managed_system}\n\n"
        "<unabridged_analysis_context>\n"
        f"{runtime['analysis_text']}\n"
        "</unabridged_analysis_context>"
    )
    full_messages: list[BaseMessage] = [SystemMessage(content=full_system)]
    full_messages.extend(
        HumanMessage(content=turn.content)
        if turn.role == "user"
        else AIMessage(content=turn.content)
        for turn in history
    )
    full_messages.append(HumanMessage(content=question))
    prompts = {
        "A0_full_context": tuple(full_messages),
        "A1_trim_only_8k": tuple(trimmed.messages),
        "A2_managed_8k": tuple(managed.messages),
    }
    for baseline, messages in prompts.items():
        measured = sum(estimate_chat_tokens(str(message.content)) for message in messages)
        if baseline != "A0_full_context" and measured > int(manifest["input_budget_tokens"]):
            raise RuntimeError(
                f"{baseline} exceeded the frozen input budget: {measured} tokens."
            )
    return prompts


def _run_one(
    llm: Any,
    messages: Sequence[BaseMessage],
    *,
    manifest: dict[str, Any],
    baseline: str,
    group_id: str,
    repeat: int,
    gold: dict[str, str],
    provider: str,
    model: str,
    mode: str,
) -> dict[str, Any]:
    estimated_tokens = sum(
        estimate_chat_tokens(str(message.content)) for message in messages
    )
    started = time.perf_counter()
    error: str | None = None
    text = ""
    usage: dict[str, Any] = {}
    upstream_model: str | None = None
    try:
        response = invoke_with_retry(llm, list(messages), retries=2, delay=1.5)
        text = _content_to_text(getattr(response, "content", response)).strip()
        raw_usage = dict(getattr(response, "usage_metadata", {}) or {})
        usage = {
            key: raw_usage.get(key)
            for key in ("input_tokens", "output_tokens", "total_tokens")
            if raw_usage.get(key) is not None
        }
        details = raw_usage.get("input_token_details")
        if isinstance(details, dict) and details.get("cache_read") is not None:
            usage["cache_read_tokens"] = details["cache_read"]
        metadata = dict(getattr(response, "response_metadata", {}) or {})
        upstream_model = str(
            metadata.get("model_name") or metadata.get("model") or ""
        ) or None
    except Exception as exc:  # keep the remaining frozen run resumable
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter() - started) * 1000
    prediction = parse_prediction(text, gold)
    score = score_prediction(prediction, gold, fact_map(manifest))
    return {
        "run_key": f"{baseline}:{group_id}:{repeat}",
        "manifest_sha256": manifest_sha256(manifest),
        "baseline": baseline,
        "group_id": group_id,
        "repeat": repeat,
        "provider": provider,
        "model": model,
        "mode": mode,
        "upstream_model": upstream_model,
        "estimated_input_tokens": estimated_tokens,
        "message_count": len(messages),
        "section_tokens": _section_tokens(messages),
        "usage": usage,
        "latency_ms": round(latency_ms, 3),
        "prediction": prediction,
        "score": score,
        "error": error,
    }


def _section_tokens(messages: Sequence[BaseMessage]) -> dict[str, int]:
    if not messages:
        return {}
    system = str(messages[0].content)
    output: dict[str, int] = {}
    for tag in (
        "paper_evidence",
        "analysis_context",
        "memory_index",
        "session_memory",
        "recalled_topic_memory",
        "recalled_conversation",
        "external_sources",
        "unabridged_analysis_context",
    ):
        match = re.search(
            fr"<{tag}>\n(.*?)\n</{tag}>",
            system,
            flags=re.DOTALL,
        )
        if match:
            output[tag] = estimate_chat_tokens(match.group(1))
    return output


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def _matching_rows(
    rows: list[dict[str, Any]],
    *,
    manifest_hash: str,
    provider: str,
    model: str,
    mode: str,
) -> list[dict[str, Any]]:
    """Keep the latest matching row per stable run key."""
    matched: dict[str, dict[str, Any]] = {}
    for row in rows:
        if (
            row.get("manifest_sha256") != manifest_hash
            or row.get("provider") != provider
            or row.get("model") != model
            or row.get("mode") != mode
        ):
            continue
        matched[str(row.get("run_key"))] = row
    return list(matched.values())


@contextmanager
def _benchmark_environment(input_budget: int) -> Iterator[Path]:
    prior = {
        key: os.environ.get(key)
        for key in (
            "PAPER_READER_DATA_DIR",
            "CHAT_INPUT_TOKEN_BUDGET",
            "CHAT_TEMPERATURE",
        )
    }
    with tempfile.TemporaryDirectory(prefix="paper-reader-context-benchmark-") as tmp:
        data_dir = Path(tmp).resolve()
        os.environ["PAPER_READER_DATA_DIR"] = str(data_dir)
        os.environ["CHAT_INPUT_TOKEN_BUDGET"] = str(input_budget)
        os.environ["CHAT_TEMPERATURE"] = "0"
        try:
            yield data_dir
        finally:
            for key, value in prior.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    main()
