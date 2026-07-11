"""Evidence-grounded comparison helpers for multiple saved papers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from core.evidence import EvidenceSnippet, select_evidence_snippets
from core.history import load_paper_analysis
from core.schemas import (
    ComparisonAssessment,
    ComparisonCell,
    ComparisonDimension,
    ComparisonOutput,
    ComparisonPaper,
)


ComparisonFocus = Literal["comprehensive", "method", "experiment", "critique", "custom"]
FOCUS_LABELS: dict[str, str] = {
    "comprehensive": "综合对比",
    "method": "方法与架构",
    "experiment": "实验结果",
    "critique": "创新性与局限",
    "custom": "自定义问题",
}


class ComparisonCreateRequest(BaseModel):
    """Create an evidence-grounded comparison from saved paper analyses."""

    history_ids: list[str] = Field(min_length=2, max_length=4)
    focus: ComparisonFocus = "comprehensive"
    custom_focus: str | None = Field(default=None, max_length=1_000)

    @model_validator(mode="after")
    def validate_selection(self) -> "ComparisonCreateRequest":
        cleaned = [history_id.strip() for history_id in self.history_ids if history_id.strip()]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("同一篇论文不能重复加入对比。")
        if len(cleaned) < 2:
            raise ValueError("请至少选择两篇论文。")
        if self.focus == "custom" and not (self.custom_focus or "").strip():
            raise ValueError("自定义对比需要填写研究问题。")
        self.history_ids = cleaned
        self.custom_focus = (self.custom_focus or "").strip() or None
        return self


@dataclass(frozen=True)
class ComparisonSource:
    """One authoritative saved-paper source used by the comparison agent."""

    history_id: str
    label: str
    history: dict[str, Any]
    result: dict[str, Any]
    snippets: tuple[EvidenceSnippet, ...]

    @property
    def title(self) -> str:
        paper = self.result.get("paper")
        if isinstance(paper, dict) and paper.get("title"):
            return str(paper["title"])
        return str(self.history.get("title") or self.history.get("filename") or self.label)

    @property
    def filename(self) -> str:
        paper = self.result.get("paper")
        if isinstance(paper, dict) and paper.get("filename"):
            return str(paper["filename"])
        return str(self.history.get("filename") or "paper.pdf")


def load_comparison_sources(history_ids: list[str]) -> list[ComparisonSource]:
    """Load complete saved analyses and assign stable P1..P4 labels."""
    sources: list[ComparisonSource] = []
    for index, history_id in enumerate(history_ids, start=1):
        stored = load_paper_analysis(history_id)
        if stored is None:
            raise KeyError(f"未找到已保存论文：{history_id}")
        sources.append(
            ComparisonSource(
                history_id=history_id,
                label=f"P{index}",
                history=stored["history"],
                result=stored["result"],
                snippets=tuple(stored["snippets"]),
            )
        )
    return sources


def comparison_papers(sources: list[ComparisonSource]) -> list[ComparisonPaper]:
    return [
        ComparisonPaper(
            history_id=source.history_id,
            label=source.label,
            title=source.title,
            filename=source.filename,
        )
        for source in sources
    ]


def format_comparison_sources(
    sources: list[ComparisonSource],
    request: ComparisonCreateRequest,
) -> str:
    """Build a bounded, balanced source package for the comparison model."""
    blocks: list[str] = []
    for source in sources:
        selected = select_comparison_evidence(
            list(source.snippets),
            request.focus,
            request.custom_focus,
        )
        evidence = "\n\n".join(
            "\n".join(
                [
                    f"[{source.label}:{snippet.id} | {snippet.kind} | {snippet.section} | {snippet.page_label}]",
                    snippet.text[:1_800],
                ]
            )
            for snippet in selected
        ) or "未保存可用原文证据。"
        blocks.append(
            "\n".join(
                [
                    f'<paper label="{source.label}" history_id="{source.history_id}">',
                    f"<title>{source.title}</title>",
                    "<analysis>",
                    json.dumps(_compact_result(source.result), ensure_ascii=False, separators=(",", ":")),
                    "</analysis>",
                    "<evidence>",
                    evidence,
                    "</evidence>",
                    "</paper>",
                ]
            )
        )
    return "\n\n".join(blocks)


def select_comparison_evidence(
    snippets: list[EvidenceSnippet],
    focus: str,
    custom_focus: str | None = None,
    *,
    max_snippets: int = 10,
) -> list[EvidenceSnippet]:
    """Select a balanced evidence slice for one paper and one comparison focus."""
    if not snippets:
        return []
    if focus == "custom":
        selected = _select_query_evidence(snippets, custom_focus or "", max_snippets)
        if selected:
            return selected

    agent_type = {
        "method": "method",
        "experiment": "experiment",
        "critique": "critic",
    }.get(focus)
    if agent_type:
        return select_evidence_snippets(
            snippets,
            agent_type,
            max_chars=18_000,
            max_snippets=max_snippets,
        )

    chosen: dict[str, EvidenceSnippet] = {}
    for role in ("method", "experiment", "critic"):
        for snippet in select_evidence_snippets(
            snippets,
            role,
            max_chars=7_500,
            max_snippets=4,
        ):
            chosen.setdefault(snippet.id, snippet)
    return sorted(chosen.values(), key=lambda item: (item.page_start, item.id))[:max_snippets]


def select_query_evidence(
    snippets: list[EvidenceSnippet],
    query: str,
    *,
    max_snippets: int = 6,
) -> list[EvidenceSnippet]:
    """Public query retrieval used by cross-paper follow-up chat."""
    selected = _select_query_evidence(snippets, query, max_snippets)
    if selected:
        return selected
    return select_comparison_evidence(snippets, "comprehensive", max_snippets=max_snippets)


def sanitize_comparison_output(
    output: ComparisonOutput,
    sources: list[ComparisonSource],
    request: ComparisonCreateRequest,
) -> ComparisonOutput:
    """Replace model-controlled identities and remove invalid evidence references."""
    papers = comparison_papers(sources)
    labels = [paper.label for paper in papers]
    valid_refs = {
        f"{source.label}:{snippet.id}"
        for source in sources
        for snippet in source.snippets
    }
    dimensions: list[ComparisonDimension] = []
    for dimension in output.dimensions[:16]:
        cells_by_label = {cell.paper_label: cell for cell in dimension.cells if cell.paper_label in labels}
        cells: list[ComparisonCell] = []
        for label in labels:
            original = cells_by_label.get(label)
            if original is None:
                cells.append(
                    ComparisonCell(
                        paper_label=label,
                        summary="现有分析与证据不足，无法确认该维度。",
                        evidence_ids=[],
                    )
                )
                continue
            refs = [
                ref.upper()
                for ref in original.evidence_ids
                if ref.upper() in valid_refs and ref.upper().startswith(f"{label}:")
            ]
            cells.append(
                ComparisonCell(
                    paper_label=label,
                    summary=original.summary.strip(),
                    evidence_ids=list(dict.fromkeys(refs))[:6],
                )
            )
        warning = (dimension.warning or "").strip() or None
        if dimension.comparability != "direct" and not warning:
            warning = "论文的任务、数据集、指标或实验条件存在差异，需要结合条件理解。"
        dimensions.append(dimension.model_copy(update={"cells": cells, "warning": warning}))

    warnings = list(dict.fromkeys(item.strip() for item in output.warnings if item.strip()))
    missing_evidence = sum(not cell.evidence_ids for dimension in dimensions for cell in dimension.cells)
    if missing_evidence:
        warnings.append(f"有 {missing_evidence} 个论文维度缺少可核验的原文证据。")
    return output.model_copy(
        update={
            "title": output.title.strip()[:120] or "多论文对比",
            "focus": FOCUS_LABELS.get(request.focus, request.focus),
            "papers": papers,
            "dimensions": dimensions,
            "warnings": list(dict.fromkeys(warnings)),
        }
    )


def build_comparison_assessment(output: ComparisonOutput) -> ComparisonAssessment:
    cells = [cell for dimension in output.dimensions for cell in dimension.cells]
    total_claims = len(cells)
    referenced_claims = sum(bool(cell.evidence_ids) for cell in cells)
    evidence_coverage = round(referenced_claims / total_claims * 100) if total_claims else 0
    expected_labels = {paper.label for paper in output.papers}
    covered_dimensions = sum(
        {
            cell.paper_label
            for cell in dimension.cells
            if cell.summary.strip()
            and not cell.summary.startswith("现有分析与证据不足")
        }
        == expected_labels
        for dimension in output.dimensions
    )
    paper_coverage = (
        round(covered_dimensions / len(output.dimensions) * 100)
        if output.dimensions
        else 0
    )
    completeness = 100 if output.executive_summary and output.dimensions else 0
    score = round(evidence_coverage * 0.65 + paper_coverage * 0.25 + completeness * 0.10)
    label = "证据充分" if score >= 85 else "证据基本充分" if score >= 65 else "证据有限"
    warnings = list(output.warnings)
    non_direct = sum(dimension.comparability != "direct" for dimension in output.dimensions)
    if non_direct:
        warnings.append(f"{non_direct} 个对比维度需要条件说明，不能直接按数值排序。")
    return ComparisonAssessment(
        score=score,
        label=label,
        evidence_coverage=evidence_coverage,
        paper_coverage=paper_coverage,
        referenced_claims=referenced_claims,
        total_claims=total_claims,
        warnings=list(dict.fromkeys(warnings)),
    )


def build_comparison_evidence_catalog(
    output: ComparisonOutput,
    sources: list[ComparisonSource],
) -> list[dict[str, Any]]:
    """Return previews for evidence IDs that survived backend validation."""
    referenced = {
        evidence_id
        for dimension in output.dimensions
        for cell in dimension.cells
        for evidence_id in cell.evidence_ids
    }
    catalog: list[dict[str, Any]] = []
    for source in sources:
        for snippet in source.snippets:
            prefixed_id = f"{source.label}:{snippet.id}"
            if prefixed_id not in referenced:
                continue
            catalog.append(
                {
                    "id": prefixed_id,
                    "paper_label": source.label,
                    "paper_title": source.title,
                    "evidence_id": snippet.id,
                    "section": snippet.section,
                    "page_label": snippet.page_label,
                    "kind": snippet.kind,
                    "preview": snippet.text[:900],
                }
            )
    return catalog


def demo_comparison_output(
    sources: list[ComparisonSource],
    request: ComparisonCreateRequest,
) -> ComparisonOutput:
    """Return deterministic comparison output for API and UI tests."""
    papers = comparison_papers(sources)
    definitions = [
        ("research_problem", "研究问题", "overview"),
        ("method", "核心方法", "method"),
        ("experiments", "实验设计与结果", "experiment"),
        ("limitations", "局限与适用边界", "critique"),
    ]
    dimensions: list[ComparisonDimension] = []
    for key, title, category in definitions:
        cells = []
        for source in sources:
            refs = [f"{source.label}:{snippet.id}" for snippet in source.snippets[:1]]
            cells.append(
                ComparisonCell(
                    paper_label=source.label,
                    summary=_demo_cell_summary(source.result, key),
                    evidence_ids=refs,
                )
            )
        dimensions.append(
            ComparisonDimension(
                key=key,
                title=title,
                category=category,
                description=f"比较各论文的{title}。",
                cells=cells,
                synthesis=f"Demo 模式已完成{len(sources)}篇论文的{title}字段对齐。",
                comparability="conditional" if key == "experiments" else "direct",
                warning="实验数据集和指标可能不同，正式分析时需要逐项核验。" if key == "experiments" else None,
            )
        )
    return ComparisonOutput(
        title="多论文对比演示",
        focus=FOCUS_LABELS[request.focus],
        executive_summary=f"已读取 {len(sources)} 篇历史论文，并验证结构化对比与证据引用链路。",
        papers=papers,
        common_ground=["所有论文均已完成结构化分析并保留原文证据。"],
        key_differences=["不同论文的方法、任务和实验设置需要分别核验。"],
        dimensions=dimensions,
        research_gaps=["正式模式会基于各论文证据识别共同空白与尚未解决的问题。"],
        recommendations=["使用 Live 模式获得由 GLM 生成的证据化对比结论。"],
        warnings=["当前是确定性 Demo 对比，不代表模型对论文内容的真实判断。"],
    )


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in ("paper", "method_output", "experiment_output", "critic_output", "summary_output", "assessment"):
        value = result.get(key)
        if not isinstance(value, dict):
            continue
        copied = dict(value)
        copied.pop("evidence", None)
        if key == "paper":
            copied.pop("sections", None)
            copied.pop("metadata", None)
        compact[key] = copied
    return compact


def _select_query_evidence(
    snippets: list[EvidenceSnippet],
    query: str,
    max_snippets: int,
) -> list[EvidenceSnippet]:
    terms = _query_terms(query)
    if not terms:
        return []
    scored: list[tuple[float, int, EvidenceSnippet]] = []
    for index, snippet in enumerate(snippets):
        haystack = f"{snippet.section} {snippet.text[:1_600]}".lower()
        score = sum((3.0 if len(term) > 2 else 1.0) * min(haystack.count(term), 4) for term in terms)
        if snippet.kind == "table" and any(term in query.lower() for term in ("实验", "结果", "指标", "性能")):
            score += 8
        if score > 0:
            scored.append((score, index, snippet))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return sorted(
        [item[2] for item in scored[:max_snippets]],
        key=lambda item: (item.page_start, item.id),
    )


def _query_terms(text: str) -> set[str]:
    lowered = text.lower()
    terms = {
        token
        for token in re.findall(r"[a-z][a-z0-9_-]{1,}", lowered)
        if token not in {"what", "which", "with", "that", "this", "from", "about"}
    }
    for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        terms.add(sequence)
        terms.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return {term for term in terms if len(term) >= 2}


def _demo_cell_summary(result: dict[str, Any], key: str) -> str:
    method = result.get("method_output") if isinstance(result.get("method_output"), dict) else {}
    experiment = result.get("experiment_output") if isinstance(result.get("experiment_output"), dict) else {}
    critic = result.get("critic_output") if isinstance(result.get("critic_output"), dict) else {}
    summary = result.get("summary_output") if isinstance(result.get("summary_output"), dict) else {}
    value = {
        "research_problem": method.get("research_problem") or summary.get("one_sentence_summary"),
        "method": method.get("proposed_method") or summary.get("method_highlights"),
        "experiments": experiment.get("main_results") or summary.get("experiment_highlights"),
        "limitations": critic.get("limitations") or summary.get("limitations_and_future_work"),
    }.get(key)
    if isinstance(value, list):
        return "；".join(str(item) for item in value[:3])
    return str(value or "当前历史分析未提供该字段。")
