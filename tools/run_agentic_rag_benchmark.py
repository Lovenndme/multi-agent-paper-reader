#!/usr/bin/env python3
"""Execute source-anchored Agentic RAG retrieval baselines.

The runner reads local paper history, so benchmark PDFs and gold quotations can
remain under ``.paper-reader/``. It records only bounded hashes, ranks, counts,
and public retrieval traces in its output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import patch

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.agentic_runtime import AgenticRetrievalRequest, AgenticRetrievalRuntime
from core.agentic_types import AgenticRunBudget
from core.evidence import (
    AGENT_SEMANTIC_QUERIES,
    AGENT_TERMS,
    EvidenceSnippet,
    select_evidence_snippets,
)
from core.history import load_paper_analysis, retained_paper_pdf_path
from core.model_providers import (
    selected_text_model,
    selected_text_mode,
    text_provider_id,
)
from core.paper_tools import PaperToolRegistry, search_paper_evidence
from core.semantic_search import (
    DEFAULT_EMBEDDING_MODEL,
    cross_encoder_scores,
    semantic_scores,
)
from tools.benchmark_agentic_rag import aggregate_scores


BASELINE_LABELS = {
    "A0": "lexical role-static retrieval",
    "A1": "FastEmbed role-static retrieval",
    "AQL": "legacy parent-chunk query-aware retrieval",
    "AQ": "hierarchical BM25 + Dense RRF retrieval",
    "AQR": "hierarchical retrieval + local cross-encoder",
    "A2": "structured-action Agentic RAG",
    "A3": "native-tool Agentic RAG",
    "A4": "adaptive native-tool Agentic RAG",
}
AGENTS = frozenset({"method", "experiment", "critic"})


def execute_case(
    case: dict[str, Any],
    *,
    baselines: Sequence[str],
    repeat: int = 1,
) -> dict[str, Any]:
    """Execute requested baselines for one validated local-history case."""
    history_id = str(case.get("paper_history_id") or "").strip()
    loaded = load_paper_analysis(history_id)
    if loaded is None:
        raise ValueError(f"Unknown paper_history_id: {history_id}")
    pdf_path = retained_paper_pdf_path(history_id)
    if pdf_path is None:
        raise ValueError(f"Retained PDF is unavailable for {history_id}")
    paper_digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    expected_digest = str(case.get("paper_sha256") or "").lower()
    if paper_digest != expected_digest:
        raise ValueError(
            f"PDF SHA-256 mismatch for {history_id}: expected {expected_digest}, "
            f"found {paper_digest}"
        )

    agent = str(case.get("agent") or "").strip().lower()
    if agent not in AGENTS:
        raise ValueError(f"agent must be one of {sorted(AGENTS)}")
    query = str(case.get("query") or "").strip()
    if not query:
        raise ValueError("Each case requires a query.")
    snippets = tuple(loaded["snippets"])
    gold = hydrate_gold(case.get("gold_evidence"), snippets)

    output = {
        "case_id": str(case.get("case_id") or "").strip(),
        "paper_history_id": history_id,
        "paper_title": str(loaded["history"]["title"]),
        "paper_sha256": paper_digest,
        "query": query,
        "agent": agent,
        "repeat": repeat,
        "gold_evidence": gold,
        "runs": {},
    }
    if not output["case_id"]:
        raise ValueError("Each case requires case_id.")

    semantic_seed = select_evidence_snippets(
        list(snippets),
        agent,
        max_chars=18_000,
        max_snippets=10,
    )
    prewarm_retrieval_baselines(snippets, baselines)
    for baseline in baselines:
        if baseline == "A0":
            started = time.perf_counter()
            selected = legacy_role_static_search(
                snippets,
                agent,
                semantic=False,
            )
            output["runs"][baseline] = build_run_record(
                selected,
                gold,
                latency_ms=(time.perf_counter() - started) * 1000,
                effective_strategy="lexical",
            )
        elif baseline == "A1":
            started = time.perf_counter()
            selected = legacy_role_static_search(
                snippets,
                agent,
                semantic=True,
            )
            output["runs"][baseline] = build_run_record(
                selected,
                gold,
                latency_ms=(time.perf_counter() - started) * 1000,
                effective_strategy="hybrid",
            )
        elif baseline == "AQ":
            started = time.perf_counter()
            selected = search_paper_evidence(
                snippets,
                query,
                kind="any",
                limit=10,
                rerank=False,
            )
            output["runs"][baseline] = build_run_record(
                selected,
                gold,
                latency_ms=(time.perf_counter() - started) * 1000,
                effective_strategy="query-aware-hybrid",
            )
        elif baseline == "AQL":
            started = time.perf_counter()
            selected = legacy_query_aware_search(snippets, query, limit=10)
            output["runs"][baseline] = build_run_record(
                selected,
                gold,
                latency_ms=(time.perf_counter() - started) * 1000,
                effective_strategy="legacy-query-aware-parent",
            )
        elif baseline == "AQR":
            started = time.perf_counter()
            selected = search_paper_evidence(
                snippets,
                query,
                kind="any",
                limit=10,
                rerank=True,
            )
            output["runs"][baseline] = build_run_record(
                selected,
                gold,
                latency_ms=(time.perf_counter() - started) * 1000,
                effective_strategy="hierarchical-cross-encoder",
            )
        elif baseline in {"A2", "A3", "A4"}:
            output["runs"][baseline] = run_agentic_baseline(
                baseline,
                agent=agent,
                query=query,
                snippets=snippets,
                seed=semantic_seed,
                gold=gold,
            )
        else:
            raise ValueError(f"Unsupported executable baseline: {baseline}")
    return output


def prewarm_retrieval_baselines(
    snippets: Sequence[EvidenceSnippet],
    baselines: Sequence[str],
) -> None:
    """Exclude one-time local model/index initialization from timed queries."""
    parent_documents = [
        f"{snippet.section}\n{snippet.text}"
        for snippet in snippets
    ]
    if {"A1", "AQL"} & set(baselines):
        semantic_scores("benchmark parent prewarm", parent_documents)
    if "AQR" in baselines:
        cross_encoder_scores(
            "benchmark reranker prewarm",
            parent_documents[:24],
        )


def legacy_query_aware_search(
    snippets: Sequence[EvidenceSnippet],
    query: str,
    *,
    limit: int,
) -> list[EvidenceSnippet]:
    """Reproduce the pre-hierarchical AQ implementation as a fair baseline."""
    documents = [f"{snippet.section}\n{snippet.text}" for snippet in snippets]
    similarities = semantic_scores(query, documents)
    terms = {
        token
        for token in re.findall(r"[a-z][a-z0-9_.-]{1,}", query.casefold())
        if token
        not in {
            "a",
            "an",
            "and",
            "are",
            "as",
            "by",
            "did",
            "do",
            "does",
            "for",
            "from",
            "how",
            "in",
            "is",
            "of",
            "on",
            "or",
            "paper",
            "that",
            "the",
            "this",
            "to",
            "what",
            "which",
            "why",
            "with",
        }
    }
    scored: list[tuple[float, int, EvidenceSnippet]] = []
    lowered_query = query.casefold()
    for index, snippet in enumerate(snippets):
        section = snippet.section.casefold()
        text = snippet.text.casefold()
        score = similarities[index] * 20.0 if similarities is not None else 0.0
        for term in terms:
            if term in section:
                score += 3.0
            score += min(text.count(term), 4) * 0.8
        if snippet.kind == "table" and any(
            marker in lowered_query
            for marker in (
                "table",
                "result",
                "metric",
                "ablation",
                "表",
                "结果",
                "指标",
                "消融",
            )
        ):
            score += 5.0
        if snippet.kind == "figure" and any(
            marker in lowered_query
            for marker in ("figure", "architecture", "pipeline", "图", "架构", "流程")
        ):
            score += 4.0
        scored.append((score, index, snippet))
    scored.sort(key=lambda item: (-item[0], item[1]))

    selected: list[EvidenceSnippet] = []
    seen_sections: dict[str, int] = {}
    for score, _, snippet in scored:
        if similarities is None and score <= 0 and selected:
            continue
        section_key = snippet.section.casefold()
        if seen_sections.get(section_key, 0) >= 3:
            continue
        selected.append(snippet)
        seen_sections[section_key] = seen_sections.get(section_key, 0) + 1
        if len(selected) >= max(1, min(limit, 10)):
            break
    return selected


def legacy_role_static_search(
    snippets: Sequence[EvidenceSnippet],
    agent: str,
    *,
    semantic: bool,
) -> list[EvidenceSnippet]:
    """Reproduce the original role-static baseline independently of AQ changes."""
    terms = AGENT_TERMS.get(agent, set())
    similarities = (
        semantic_scores(
            AGENT_SEMANTIC_QUERIES.get(agent, agent),
            [f"{snippet.section}\n{snippet.text}" for snippet in snippets],
        )
        if semantic
        else None
    )
    scored: list[tuple[float, int, EvidenceSnippet]] = []
    for index, snippet in enumerate(snippets):
        if similarities is None:
            haystack = f"{snippet.section} {snippet.text[:900]}".casefold()
            score = float(sum(1 for term in terms if term in haystack))
            if "abstract" in haystack:
                score += 1.0
        else:
            score = similarities[index] * 20.0
        if agent == "experiment" and snippet.kind == "table":
            score += 7.0
        if agent == "method" and snippet.kind == "figure":
            score += 3.0
        if agent == "critic" and snippet.kind in {"table", "figure"}:
            score += 2.0
        if snippet.id.startswith(("T", "F")) and snippet.text:
            score += 1.0
        scored.append((score, index, snippet))
    scored.sort(key=lambda item: (-item[0], item[1]))

    chosen: list[EvidenceSnippet] = []
    total_chars = 0
    for score, _, snippet in scored:
        if similarities is None and score <= 0 and chosen:
            continue
        if chosen and total_chars + len(snippet.text) > 18_000:
            continue
        chosen.append(snippet)
        total_chars += len(snippet.text)
        if len(chosen) >= 10:
            break
    return sorted(chosen, key=lambda snippet: (snippet.page_start, snippet.id))


def run_agentic_baseline(
    baseline: str,
    *,
    agent: str,
    query: str,
    snippets: Sequence[EvidenceSnippet],
    seed: Sequence[EvidenceSnippet],
    gold: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    strategy = "structured" if baseline == "A2" else "native"
    rag_mode = "adaptive" if baseline == "A4" else "agentic"
    runtime = AgenticRetrievalRuntime()
    started = time.perf_counter()
    try:
        with patch.dict(
            os.environ,
            {
                "RAG_MODE": rag_mode,
                "AGENTIC_TOOL_STRATEGY": strategy,
                "AGENTIC_RAG_CHECKPOINTS": "false",
            },
        ):
            result = runtime.retrieve(
                AgenticRetrievalRequest(
                    agent_id=agent,
                    objective=query,
                    registry=PaperToolRegistry.create(snippets),
                    seed_snippets=tuple(seed),
                    budget=AgenticRunBudget.from_env(),
                    provider_id=text_provider_id(),
                )
            )
    except Exception as exc:  # noqa: BLE001 - benchmark failures remain in denominator
        return build_run_record(
            (),
            gold,
            latency_ms=(time.perf_counter() - started) * 1000,
            effective_strategy=strategy,
            error=f"{type(exc).__name__}: {str(exc)[:240]}",
        )

    tool_steps = [
        step
        for step in result.steps
        if step.action.tool != "finish_retrieval"
    ]
    return build_run_record(
        result.snippets,
        gold,
        latency_ms=(time.perf_counter() - started) * 1000,
        effective_strategy=result.strategy,
        tool_calls=len(tool_steps),
        successful_tool_calls=sum(step.error is None for step in tool_steps),
        planner_calls=len(result.steps),
        fallback_used=result.fallback_used,
        stop_reason=result.stop_reason,
        adaptive_triggered=result.adaptive_triggered,
        adaptive_reason=result.adaptive_reason,
        public_steps=[step.public_payload() for step in result.steps],
    )


def build_run_record(
    snippets: Sequence[EvidenceSnippet],
    gold: Sequence[dict[str, Any]],
    *,
    latency_ms: float,
    effective_strategy: str,
    tool_calls: int = 0,
    successful_tool_calls: int = 0,
    planner_calls: int = 0,
    fallback_used: bool = False,
    stop_reason: str = "",
    adaptive_triggered: bool | None = None,
    adaptive_reason: str | None = None,
    public_steps: Sequence[dict[str, Any]] = (),
    error: str | None = None,
) -> dict[str, Any]:
    retrieved = serialize_retrieved(snippets, gold)
    return {
        "retrieved_evidence_ids": [item["evidence_id"] for item in retrieved],
        "retrieved_evidence": retrieved,
        "evidence_items": len(retrieved),
        "evidence_chars": sum(len(snippet.text) for snippet in snippets),
        "tool_calls": tool_calls,
        "successful_tool_calls": successful_tool_calls,
        "planner_calls": planner_calls,
        "latency_ms": round(max(0.0, latency_ms), 3),
        "effective_strategy": effective_strategy,
        "fallback_used": fallback_used,
        "stop_reason": stop_reason,
        "adaptive_triggered": adaptive_triggered,
        "adaptive_reason": adaptive_reason,
        "public_steps": list(public_steps),
        "error": error,
    }


def hydrate_gold(
    raw_gold: Any,
    snippets: Sequence[EvidenceSnippet],
) -> list[dict[str, Any]]:
    """Validate quote/page anchors against the current parsed evidence index."""
    if not isinstance(raw_gold, list) or not raw_gold:
        raise ValueError("Each case requires non-empty gold_evidence.")
    hydrated: list[dict[str, Any]] = []
    for fact_index, item in enumerate(raw_gold, start=1):
        if not isinstance(item, dict):
            raise ValueError("gold_evidence items must be objects.")
        alternatives = item.get("alternatives", [])
        if not isinstance(alternatives, list):
            raise ValueError("gold_evidence alternatives must be a list.")
        fact_id = str(item.get("fact_id") or f"fact-{fact_index}").strip()
        anchors = [
            {
                "quote": item.get("quote"),
                "page_start": item.get("page_start") or item.get("page"),
                "page_end": item.get("page_end"),
            },
            *alternatives,
        ]
        for anchor in anchors:
            if not isinstance(anchor, dict):
                raise ValueError("Gold alternatives must be objects.")
            hydrated.append(
                _hydrate_gold_anchor(
                    anchor,
                    snippets,
                    fact_id=fact_id,
                )
            )
    return hydrated


def _hydrate_gold_anchor(
    item: dict[str, Any],
    snippets: Sequence[EvidenceSnippet],
    *,
    fact_id: str,
) -> dict[str, Any]:
    quote = normalize_text(item.get("quote"))
    if len(quote) < 24:
        raise ValueError("Each gold quote must contain at least 24 normalized characters.")
    page_start = int(item.get("page_start") or item.get("page") or 0)
    page_end = int(item.get("page_end") or page_start)
    if page_start < 1 or page_end < page_start:
        raise ValueError("Gold pages are one-based and page_end must be >= page_start.")
    matches = [
        snippet
        for snippet in snippets
        if page_ranges_overlap(
            page_start,
            page_end,
            snippet.page_start + 1,
            snippet.page_end + 1,
        )
        and quote.casefold() in normalize_text(snippet.text).casefold()
    ]
    if not matches:
        raise ValueError(
            f"Gold quote was not found on pp.{page_start}-{page_end}: {quote[:80]}"
        )
    anchor = quote_anchor(page_start, page_end, quote)
    return {
        "fact_id": fact_id,
        "evidence_id": matches[0].id,
        "equivalent_evidence_ids": sorted({snippet.id for snippet in matches}),
        "source_anchor": anchor,
        "page_start": page_start,
        "page_end": page_end,
        "quote_sha256": hashlib.sha256(quote.encode("utf-8")).hexdigest(),
    }


def serialize_retrieved(
    snippets: Sequence[EvidenceSnippet],
    gold: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, snippet in enumerate(snippets, start=1):
        anchor = snippet_anchor(snippet)
        if anchor in seen:
            continue
        seen.add(anchor)
        matched = [
            str(item["source_anchor"])
            for item in gold
            if gold_matches_snippet(item, snippet)
        ]
        output.append(
            {
                "rank": rank,
                "evidence_id": snippet.id,
                "source_anchor": anchor,
                "kind": snippet.kind,
                "page_start": snippet.page_start + 1,
                "page_end": snippet.page_end + 1,
                "matched_gold_anchors": matched,
            }
        )
    return output


def gold_matches_snippet(item: dict[str, Any], snippet: EvidenceSnippet) -> bool:
    quote_hash = str(item.get("quote_sha256") or "")
    if not quote_hash:
        return False
    if not page_ranges_overlap(
        int(item["page_start"]),
        int(item["page_end"]),
        snippet.page_start + 1,
        snippet.page_end + 1,
    ):
        return False
    for quote in item.get("_quotes", []):
        if normalize_text(quote).casefold() in normalize_text(snippet.text).casefold():
            return True
    # hydrate_gold intentionally removes raw quotations from persisted run output.
    equivalent = {str(value) for value in item.get("equivalent_evidence_ids", [])}
    return snippet.id in equivalent


def quote_anchor(page_start: int, page_end: int, quote: str) -> str:
    digest = hashlib.sha256(normalize_text(quote).encode("utf-8")).hexdigest()
    return f"pp.{page_start}-{page_end}:quote-sha256:{digest}"


def snippet_anchor(snippet: EvidenceSnippet) -> str:
    normalized = normalize_text(snippet.text)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return (
        f"pp.{snippet.page_start + 1}-{snippet.page_end + 1}:"
        f"{snippet.kind}-sha256:{digest}"
    )


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def page_ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) <= min(a_end, b_end)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"Line {line_number} must contain an object.")
        cases.append(item)
    if not cases:
        raise ValueError("Benchmark manifest contains no cases.")
    return cases


def environment_int(name: str, default: int) -> int:
    """Read a benchmark configuration value without making reporting fragile."""
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute local source-anchored Agentic RAG retrieval baselines."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--baselines",
        default="A0,A1,AQL,AQ,AQR,A2,A3,A4",
        help="Comma-separated subset of A0,A1,AQL,AQ,AQR,A2,A3,A4.",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--k", type=int, default=16)
    args = parser.parse_args()

    load_dotenv(".env")
    baselines = tuple(
        dict.fromkeys(
            value.strip().upper()
            for value in args.baselines.split(",")
            if value.strip()
        )
    )
    unsupported = set(baselines) - set(BASELINE_LABELS)
    if unsupported:
        raise ValueError(f"Unsupported baselines: {sorted(unsupported)}")
    repeat_count = max(1, args.repeat)
    outputs = [
        execute_case(case, baselines=baselines, repeat=repeat)
        for repeat in range(1, repeat_count + 1)
        for case in load_manifest(args.manifest)
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
            for item in outputs
        ),
        encoding="utf-8",
    )
    report_path = args.report or args.output.with_suffix(".summary.json")
    report = {
        "scope": "retrieval pilot; not whole-system answer accuracy",
        "latency_scope": (
            "warm-process retrieval latency; one-time FastEmbed model loading is "
            "excluded after seed prewarm"
        ),
        "manifest": str(args.manifest),
        "output": str(args.output),
        "cases": len(outputs),
        "repeats": repeat_count,
        "k": max(1, args.k),
        "provider": text_provider_id(),
        "model": selected_text_model(),
        "mode": selected_text_mode(),
        "retrieval_config": {
            "embedding_model": os.environ.get(
                "EMBEDDING_MODEL",
                DEFAULT_EMBEDDING_MODEL,
            ),
            "embedding_batch_size": environment_int(
                "PAPER_READER_EMBEDDING_BATCH_SIZE",
                64,
            ),
            "subchunk_tokens": environment_int(
                "PAPER_READER_RETRIEVAL_SUBCHUNK_TOKENS",
                220,
            ),
            "subchunk_overlap": environment_int(
                "PAPER_READER_RETRIEVAL_SUBCHUNK_OVERLAP",
                40,
            ),
            "candidate_pool": environment_int(
                "PAPER_READER_RETRIEVAL_CANDIDATES",
                24,
            ),
            "reranker_model": os.environ.get(
                "PAPER_READER_RERANKER_MODEL",
                "Xenova/ms-marco-MiniLM-L-6-v2",
            ),
            "production_reranker_enabled": os.environ.get(
                "PAPER_READER_RERANKER_ENABLED",
                "true",
            ),
        },
        "baselines": {key: BASELINE_LABELS[key] for key in baselines},
        "metrics": aggregate_scores(outputs, k=max(1, args.k)),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
