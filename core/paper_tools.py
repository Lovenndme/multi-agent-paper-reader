"""Read-only, provider-neutral paper tools used by Agentic RAG."""

from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from core.agentic_types import RetrievalAction, ToolObservation
from core.evidence import EvidenceSnippet, format_evidence_context
from core.pdf_parser import ParsedPaper, _normalize_title
from core.hybrid_retrieval import RetrievalRanking, rank_evidence


class PaperToolError(ValueError):
    """A safe, user-actionable paper-tool failure."""


class SearchPaperInput(BaseModel):
    query: str = Field(min_length=1, max_length=800)
    kind: str = Field(default="any", pattern="^(any|text|table|figure)$")
    limit: int = Field(default=6, ge=1, le=10)
    public_summary: str = Field(min_length=1, max_length=240)


class ReadSectionInput(BaseModel):
    section: str = Field(min_length=1, max_length=240)
    limit: int = Field(default=6, ge=1, le=10)
    public_summary: str = Field(min_length=1, max_length=240)


class ReadPageInput(BaseModel):
    page: int = Field(ge=1, le=10_000)
    limit: int = Field(default=6, ge=1, le=10)
    public_summary: str = Field(min_length=1, max_length=240)


class ReadIndexedInput(BaseModel):
    index: int | None = Field(default=None, ge=1, le=10_000)
    evidence_id: str | None = Field(default=None, max_length=32)
    public_summary: str = Field(min_length=1, max_length=240)


class OverviewInput(BaseModel):
    public_summary: str = Field(min_length=1, max_length=240)


class CalculateInput(BaseModel):
    expression: str = Field(min_length=1, max_length=256)
    public_summary: str = Field(min_length=1, max_length=240)


class FinishInput(BaseModel):
    public_summary: str = Field(min_length=1, max_length=240)


@dataclass(frozen=True)
class PaperToolRegistry:
    """Capability-scoped tool registry over one in-memory paper evidence index."""

    snippets: tuple[EvidenceSnippet, ...]
    paper: ParsedPaper | None = None
    title: str = ""

    @classmethod
    def create(
        cls,
        snippets: Sequence[EvidenceSnippet],
        paper: ParsedPaper | None = None,
        *,
        title: str = "",
    ) -> "PaperToolRegistry":
        return cls(tuple(snippets), paper, title or (paper.title if paper else ""))

    def execute(self, action: RetrievalAction) -> ToolObservation:
        handlers = {
            "paper_search": self._search,
            "paper_overview": self._overview,
            "paper_read_section": self._read_section,
            "paper_read_page": self._read_page,
            "paper_read_table": self._read_table,
            "paper_read_figure": self._read_figure,
            "calculate": self._calculate,
        }
        if action.tool == "finish_retrieval":
            return ToolObservation(
                tool=action.tool,
                summary="检索已结束。",
                content=action.public_summary,
            )
        handler = handlers.get(action.tool)
        if handler is None:
            raise PaperToolError("不支持的论文检索工具。")
        return handler(action)

    def native_tools(self) -> list[StructuredTool]:
        """Return schemas that LangChain can bind to OpenAI/Anthropic-style models."""

        def paper_search(
            query: str,
            kind: str = "any",
            limit: int = 6,
            public_summary: str = "正在检索论文证据。",
        ) -> str:
            return self.execute(
                RetrievalAction(
                    tool="paper_search",
                    query=query,
                    kind=kind,
                    limit=limit,
                    public_summary=public_summary,
                )
            ).content

        def paper_overview(
            public_summary: str = "正在查看论文结构。",
        ) -> str:
            return self.execute(
                RetrievalAction(
                    tool="paper_overview",
                    public_summary=public_summary,
                )
            ).content

        def paper_read_section(
            section: str,
            limit: int = 6,
            public_summary: str = "正在读取相关章节。",
        ) -> str:
            return self.execute(
                RetrievalAction(
                    tool="paper_read_section",
                    section=section,
                    limit=limit,
                    public_summary=public_summary,
                )
            ).content

        def paper_read_page(
            page: int,
            limit: int = 6,
            public_summary: str = "正在读取指定页面。",
        ) -> str:
            return self.execute(
                RetrievalAction(
                    tool="paper_read_page",
                    page=page,
                    limit=limit,
                    public_summary=public_summary,
                )
            ).content

        def paper_read_table(
            index: int | None = None,
            evidence_id: str | None = None,
            public_summary: str = "正在核对论文表格。",
        ) -> str:
            return self.execute(
                RetrievalAction(
                    tool="paper_read_table",
                    index=index,
                    evidence_id=evidence_id,
                    public_summary=public_summary,
                )
            ).content

        def paper_read_figure(
            index: int | None = None,
            evidence_id: str | None = None,
            public_summary: str = "正在核对论文图像摘要。",
        ) -> str:
            return self.execute(
                RetrievalAction(
                    tool="paper_read_figure",
                    index=index,
                    evidence_id=evidence_id,
                    public_summary=public_summary,
                )
            ).content

        def calculate(
            expression: str,
            public_summary: str = "正在核对数值计算。",
        ) -> str:
            return self.execute(
                RetrievalAction(
                    tool="calculate",
                    expression=expression,
                    public_summary=public_summary,
                )
            ).content

        def finish_retrieval(
            public_summary: str = "现有证据已覆盖任务需求。",
        ) -> str:
            return public_summary

        definitions = (
            (
                paper_search,
                "Search the complete paper evidence index with a fresh semantic and lexical query.",
                SearchPaperInput,
            ),
            (
                paper_overview,
                "Inspect the paper title, sections, page span, and available tables/figures.",
                OverviewInput,
            ),
            (
                paper_read_section,
                "Read evidence from a named paper section.",
                ReadSectionInput,
            ),
            (
                paper_read_page,
                "Read evidence from a one-based paper page number.",
                ReadPageInput,
            ),
            (
                paper_read_table,
                "Read one indexed table evidence item.",
                ReadIndexedInput,
            ),
            (
                paper_read_figure,
                "Read one indexed figure caption or visual summary.",
                ReadIndexedInput,
            ),
            (
                calculate,
                "Evaluate a small arithmetic expression through a strict AST whitelist.",
                CalculateInput,
            ),
            (
                finish_retrieval,
                "Finish retrieval once the available evidence is sufficient.",
                FinishInput,
            ),
        )
        return [
            StructuredTool.from_function(
                function,
                name=function.__name__,
                description=description,
                args_schema=args_schema,
            )
            for function, description, args_schema in definitions
        ]

    def _search(self, action: RetrievalAction) -> ToolObservation:
        selected = search_paper_evidence(
            self.snippets,
            action.query or "",
            kind=action.kind,
            limit=action.limit,
        )
        return self._evidence_observation(
            action.tool,
            selected,
            f"检索到 {len(selected)} 个相关证据片段。",
        )

    def _overview(self, action: RetrievalAction) -> ToolObservation:
        sections: list[tuple[str, int, int]] = []
        if self.paper is not None:
            sections = [
                (section.title, section.page_start + 1, section.page_end + 1)
                for section in self.paper.sections[:80]
            ]
        else:
            seen: set[str] = set()
            for snippet in self.snippets:
                if snippet.section in seen:
                    continue
                seen.add(snippet.section)
                sections.append(
                    (snippet.section, snippet.page_start + 1, snippet.page_end + 1)
                )
        tables = sum(snippet.kind == "table" for snippet in self.snippets)
        figures = sum(snippet.kind == "figure" for snippet in self.snippets)
        lines = [
            f"Title: {self.title or 'Unknown'}",
            f"Evidence: {len(self.snippets)}; tables: {tables}; figures: {figures}",
            "Sections:",
            *[
                f"- {title} (p.{start})" if start == end else f"- {title} (pp.{start}-{end})"
                for title, start, end in sections
            ],
        ]
        return ToolObservation(
            tool=action.tool,
            summary=f"已读取论文结构，共 {len(sections)} 个章节。",
            content="\n".join(lines)[:8_000],
            metadata={"section_count": len(sections), "tables": tables, "figures": figures},
        )

    def _read_section(self, action: RetrievalAction) -> ToolObservation:
        query = _normalize_title(action.section or "")
        exact = [
            snippet
            for snippet in self.snippets
            if query in _normalize_title(snippet.section)
            or _normalize_title(snippet.section) in query
        ]
        selected = exact[: action.limit]
        if not selected:
            selected = search_paper_evidence(
                self.snippets,
                action.section or "",
                limit=action.limit,
            )
        return self._evidence_observation(
            action.tool,
            selected,
            f"已读取 {len(selected)} 个章节证据片段。",
        )

    def _read_page(self, action: RetrievalAction) -> ToolObservation:
        zero_based = int(action.page or 1) - 1
        selected = [
            snippet
            for snippet in self.snippets
            if snippet.page_start <= zero_based <= snippet.page_end
        ][: action.limit]
        return self._evidence_observation(
            action.tool,
            selected,
            f"第 {action.page} 页包含 {len(selected)} 个证据片段。",
        )

    def _read_table(self, action: RetrievalAction) -> ToolObservation:
        return self._read_indexed(action, prefix="T", kind="table")

    def _read_figure(self, action: RetrievalAction) -> ToolObservation:
        return self._read_indexed(action, prefix="F", kind="figure")

    def _read_indexed(
        self,
        action: RetrievalAction,
        *,
        prefix: str,
        kind: str,
    ) -> ToolObservation:
        evidence_id = (action.evidence_id or "").upper().strip()
        if not evidence_id and action.index is not None:
            evidence_id = f"{prefix}{action.index:03d}"
        selected = [
            snippet
            for snippet in self.snippets
            if snippet.id.upper() == evidence_id and snippet.kind == kind
        ]
        if not selected:
            raise PaperToolError(f"证据 {evidence_id or prefix} 不存在。")
        return self._evidence_observation(
            action.tool,
            selected,
            f"已读取 {evidence_id}。",
        )

    def _calculate(self, action: RetrievalAction) -> ToolObservation:
        expression = (action.expression or "").strip()
        value = safe_calculate(expression)
        return ToolObservation(
            tool=action.tool,
            summary="数值计算已完成。",
            content=f"{expression} = {value:g}",
            metadata={"value": value},
        )

    @staticmethod
    def _evidence_observation(
        tool: str,
        snippets: Sequence[EvidenceSnippet],
        summary: str,
    ) -> ToolObservation:
        selected = tuple(snippets)
        return ToolObservation(
            tool=tool,
            summary=summary,
            content=format_evidence_context(selected),
            snippets=selected,
            metadata={"evidence_ids": [snippet.id for snippet in selected]},
        )


def search_paper_evidence(
    snippets: Sequence[EvidenceSnippet],
    query: str,
    *,
    kind: str = "any",
    limit: int = 6,
    rerank: bool | None = None,
) -> list[EvidenceSnippet]:
    """Run fresh hybrid retrieval for an arbitrary model-selected query."""
    ranking = search_paper_evidence_ranking(
        snippets,
        query,
        kind=kind,
        rerank=rerank,
    )
    bounded = max(1, min(int(limit), 10))
    selected: list[EvidenceSnippet] = []
    seen_sections: dict[str, int] = {}
    for hit in ranking.hits:
        snippet = hit.snippet
        section_key = snippet.section.casefold()
        if seen_sections.get(section_key, 0) >= 3:
            continue
        selected.append(snippet)
        seen_sections[section_key] = seen_sections.get(section_key, 0) + 1
        if len(selected) >= bounded:
            break
    return selected


def search_paper_evidence_ranking(
    snippets: Sequence[EvidenceSnippet],
    query: str,
    *,
    kind: str = "any",
    rerank: bool | None = None,
) -> RetrievalRanking:
    """Return full hierarchical retrieval ranks and confidence diagnostics."""
    clean_query = " ".join(str(query).split())[:800]
    if not clean_query:
        raise PaperToolError("检索词不能为空。")
    candidates = [
        snippet
        for snippet in snippets
        if kind == "any" or snippet.kind == kind
    ]
    if not candidates:
        return rank_evidence((), clean_query, rerank=rerank)
    return rank_evidence(candidates, clean_query, rerank=rerank)


def prefixed_snippets(
    groups: Iterable[tuple[str, Sequence[EvidenceSnippet]]],
) -> tuple[EvidenceSnippet, ...]:
    """Create a collision-free, source-balanced multi-paper evidence view."""
    materialized = [
        (label, tuple(snippets))
        for label, snippets in groups
    ]
    output: list[EvidenceSnippet] = []
    max_group_size = max(
        (len(snippets) for _, snippets in materialized),
        default=0,
    )
    for index in range(max_group_size):
        for label, snippets in materialized:
            if index >= len(snippets):
                continue
            snippet = snippets[index]
            output.append(
                EvidenceSnippet(
                    id=f"{label}:{snippet.id}",
                    section=f"{label} · {snippet.section}",
                    page_start=snippet.page_start,
                    page_end=snippet.page_end,
                    text=snippet.text,
                    kind=snippet.kind,
                )
            )
    return tuple(output)


def safe_calculate(expression: str) -> float:
    """Evaluate bounded arithmetic without eval or access to Python objects."""
    if not expression or len(expression) > 256:
        raise PaperToolError("计算表达式不能为空或过长。")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise PaperToolError("计算表达式格式无效。") from exc
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise PaperToolError("计算表达式过于复杂。")
    value = _evaluate_node(tree.body)
    if not math.isfinite(value) or abs(value) > 1e100:
        raise PaperToolError("计算结果超过安全范围。")
    return value


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MATH_FUNCTIONS = {
    "abs": abs,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
}


def _evaluate_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool):
            raise PaperToolError("布尔值不能用于计算。")
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in {"pi", "e"}:
        return float(getattr(math, node.id))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return float(_UNARY_OPERATORS[type(node.op)](_evaluate_node(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise PaperToolError("指数超过安全范围。")
        try:
            return float(_BINARY_OPERATORS[type(node.op)](left, right))
        except (ArithmeticError, OverflowError, ValueError) as exc:
            raise PaperToolError("计算失败或结果越界。") from exc
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _MATH_FUNCTIONS
        and not node.keywords
        and 1 <= len(node.args) <= 2
    ):
        try:
            return float(_MATH_FUNCTIONS[node.func.id](*[_evaluate_node(arg) for arg in node.args]))
        except (ArithmeticError, OverflowError, ValueError, TypeError) as exc:
            raise PaperToolError("数学函数参数无效。") from exc
    raise PaperToolError("表达式包含不允许的语法。")


def _query_terms(text: str) -> set[str]:
    lowered = text.lower()
    terms = {
        token
        for token in re.findall(r"[a-z][a-z0-9_-]{1,}", lowered)
        if token
        not in {
            "what",
            "which",
            "with",
            "that",
            "this",
            "from",
            "about",
            "paper",
            "please",
        }
    }
    for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        terms.add(sequence)
        terms.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return terms
