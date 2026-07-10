"""Deterministic novelty aggregation and analysis reliability scoring."""

from __future__ import annotations

from core.evidence import EvidenceSnippet, select_evidence_snippets
from core.pdf_parser import ParsedPaper
from core.schemas import (
    AnalysisAssessment,
    CriticOutput,
    ExperimentOutput,
    MethodOutput,
    NoveltyAssessment,
    NoveltyDimensionScore,
    ReliabilityAssessment,
    ReliabilityBreakdown,
    SummaryOutput,
)


NOVELTY_WEIGHTS = {
    "problem_originality": 0.15,
    "method_originality": 0.40,
    "prior_work_difference": 0.30,
    "generality": 0.15,
}

SECTION_GROUP_TERMS = {
    "abstract_intro": ("abstract", "introduction", "摘要", "引言"),
    "related_work": (
        "related work",
        "prior work",
        "previous work",
        "existing method",
        "literature",
        "相关工作",
        "已有工作",
        "现有方法",
    ),
    "method": (
        "method",
        "methodology",
        "model",
        "approach",
        "framework",
        "architecture",
        "方法",
        "模型",
        "架构",
    ),
    "experiment": (
        "experiment",
        "evaluation",
        "result",
        "benchmark",
        "dataset",
        "实验",
        "评估",
        "结果",
        "数据集",
    ),
    "discussion": (
        "discussion",
        "limitation",
        "conclusion",
        "future work",
        "analysis",
        "讨论",
        "局限",
        "结论",
        "未来工作",
    ),
}


def build_analysis_assessment(
    paper: ParsedPaper,
    snippets: list[EvidenceSnippet],
    method: MethodOutput,
    experiment: ExperimentOutput,
    critic: CriticOutput,
    summary: SummaryOutput | None = None,
    *,
    demo: bool = False,
) -> AnalysisAssessment:
    """Build a transparent assessment from model judgments and pipeline evidence."""
    novelty = _build_novelty_assessment(critic, snippets)
    dimension_keys = {
        dimension.dimension
        for dimension in critic.novelty_dimensions
        if dimension.dimension in NOVELTY_WEIGHTS
    }
    reliability = _build_reliability_assessment(
        paper,
        snippets,
        method,
        experiment,
        critic,
        summary,
        novelty_dimensions_complete=dimension_keys == set(NOVELTY_WEIGHTS),
        demo=demo,
    )
    return AnalysisAssessment(novelty=novelty, reliability=reliability)


def _build_novelty_assessment(
    critic: CriticOutput,
    snippets: list[EvidenceSnippet],
) -> NoveltyAssessment:
    warnings: list[str] = []
    dimensions_by_key: dict[str, NoveltyDimensionScore] = {}
    for dimension in critic.novelty_dimensions:
        if dimension.dimension in NOVELTY_WEIGHTS and dimension.dimension not in dimensions_by_key:
            dimensions_by_key[dimension.dimension] = dimension

    complete = set(dimensions_by_key) == set(NOVELTY_WEIGHTS)
    ordered_dimensions = [
        dimensions_by_key[key]
        for key in NOVELTY_WEIGHTS
        if key in dimensions_by_key
    ]
    if complete:
        score = round(
            sum(
                dimensions_by_key[key].score * weight
                for key, weight in NOVELTY_WEIGHTS.items()
            ),
            1,
        )
    else:
        score = float(critic.novelty_score)
        warnings.append("模型未返回完整的四维创新性评分，当前总分使用兼容评分。")

    valid_ids = {snippet.id for snippet in snippets}
    invalid_ids = sorted(
        {
            evidence_id
            for dimension in ordered_dimensions
            for evidence_id in dimension.evidence_ids
            if evidence_id not in valid_ids
        }
    )
    if invalid_ids:
        warnings.append(f"创新性维度包含无效证据 ID：{', '.join(invalid_ids)}。")

    return NoveltyAssessment(
        score=score,
        label=_novelty_label(score),
        dimensions=ordered_dimensions,
        warnings=warnings,
    )


def _build_reliability_assessment(
    paper: ParsedPaper,
    snippets: list[EvidenceSnippet],
    method: MethodOutput,
    experiment: ExperimentOutput,
    critic: CriticOutput,
    summary: SummaryOutput | None,
    *,
    novelty_dimensions_complete: bool,
    demo: bool,
) -> ReliabilityAssessment:
    warnings: list[str] = []
    parsed_groups = _groups_from_sections(paper)
    critic_snippets = select_evidence_snippets(
        snippets,
        "critic",
        max_chars=18000,
        max_snippets=10,
    )
    covered_groups = _groups_from_snippets(critic_snippets)

    text_length = len(paper.full_text.strip())
    parsing = _tier_score(text_length, ((5000, 5), (1000, 3), (1, 1)))
    parsing += _tier_score(len(paper.sections), ((5, 5), (3, 3), (1, 1)))
    parsing += round(10 * len(parsed_groups) / len(SECTION_GROUP_TERMS))

    coverage = _tier_score(len(critic_snippets), ((6, 10), (3, 6), (1, 2)))
    distinct_sections = len({snippet.section for snippet in critic_snippets})
    coverage += _tier_score(distinct_sections, ((4, 10), (2, 6), (1, 2)))
    coverage += round(15 * len(covered_groups) / len(SECTION_GROUP_TERMS))

    valid_ids = {snippet.id for snippet in snippets}
    output_evidence = [*method.evidence, *experiment.evidence, *critic.evidence]
    if summary is not None:
        output_evidence.extend(summary.evidence)
    referenced_ids = {
        item.id for item in output_evidence if item.id
    } | {
        evidence_id
        for dimension in critic.novelty_dimensions
        for evidence_id in dimension.evidence_ids
        if evidence_id
    }
    valid_references = referenced_ids & valid_ids
    citation_ratio = len(valid_references) / len(referenced_ids) if referenced_ids else 0
    citations = round(10 * citation_ratio)
    citations += _tier_score(len(valid_references), ((5, 15), (3, 10), (1, 3)))
    complete_evidence_items = sum(
        1
        for item in output_evidence
        if item.id in valid_ids and item.quote.strip() and item.note.strip()
    )
    evidence_item_ratio = (
        complete_evidence_items / len(output_evidence) if output_evidence else 0
    )
    citations += round(5 * evidence_item_ratio)

    critic_fields = (
        critic.novelty_justification,
        critic.strengths,
        critic.limitations,
        critic.potential_improvements,
    )
    output_integrity = sum(1 for value in critic_fields if value)
    if method.proposed_method.strip() and method.key_components:
        output_integrity += 2
    if experiment.main_results.strip() and experiment.metrics:
        output_integrity += 2
    if novelty_dimensions_complete:
        output_integrity += 3
    if summary is not None:
        summary_fields = (
            summary.one_sentence_summary,
            summary.core_contributions,
            summary.method_highlights,
            summary.experiment_highlights,
        )
        output_integrity += sum(1 for value in summary_fields if value)

    breakdown = ReliabilityBreakdown(
        parsing=min(parsing, 20),
        coverage=min(coverage, 35),
        citations=min(citations, 30),
        output_integrity=min(output_integrity, 15),
    )
    raw_score = sum(breakdown.model_dump().values())
    cap = 100

    if "related_work" not in covered_groups:
        cap = min(cap, 69)
        warnings.append("相关工作证据覆盖不足，创新性判断可靠度已限制为中或低。")
    if len(valid_references) < 3:
        cap = min(cap, 59)
        warnings.append("有效证据引用少于 3 条，分析可靠度已限制为低。")
    if len(paper.sections) < 3 or text_length < 1000:
        cap = min(cap, 59)
        warnings.append("论文解析内容不足，分析可靠度已限制为低。")
    if not novelty_dimensions_complete:
        cap = min(cap, 79)
        warnings.append("创新性分维度结果不完整，无法给出高可靠度。")
    if demo:
        cap = min(cap, 39)
        warnings.append("Demo 模式未运行真实模型评审，结果仅用于界面验证。")

    score = min(raw_score, cap)
    level, label = _reliability_level(score)
    return ReliabilityAssessment(
        score=score,
        raw_score=raw_score,
        score_cap=cap,
        level=level,
        label=label,
        breakdown=breakdown,
        warnings=warnings,
    )


def _groups_from_sections(paper: ParsedPaper) -> set[str]:
    groups: set[str] = set()
    for section in paper.sections:
        groups.update(_groups_for_text(section.title, section.content))
    return groups


def _groups_from_snippets(snippets: list[EvidenceSnippet]) -> set[str]:
    groups: set[str] = set()
    for snippet in snippets:
        groups.update(_groups_for_text(snippet.section, snippet.text))
    return groups


def _groups_for_text(section: str, text: str) -> set[str]:
    haystack = f"{section} {text[:1200]}".lower()
    return {
        group
        for group, terms in SECTION_GROUP_TERMS.items()
        if any(term in haystack for term in terms)
    }


def _tier_score(value: int, tiers: tuple[tuple[int, int], ...]) -> int:
    for threshold, score in tiers:
        if value >= threshold:
            return score
    return 0


def _novelty_label(score: float) -> str:
    if score >= 4.2:
        return "高度创新"
    if score >= 3.5:
        return "创新性较高"
    if score >= 2.5:
        return "中等创新"
    if score >= 1.5:
        return "增量改进"
    return "创新性有限"


def _reliability_level(score: int) -> tuple[str, str]:
    if score >= 80:
        return "high", "高"
    if score >= 60:
        return "medium", "中"
    return "low", "低"
