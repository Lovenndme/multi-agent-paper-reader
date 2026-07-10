"""Evidence indexing helpers for accuracy-first paper analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from core.pdf_parser import FigureBlock, ParsedPaper, Section, TableBlock, _normalize_title


@dataclass(frozen=True)
class EvidenceSnippet:
    """A bounded, traceable slice of original paper text."""

    id: str
    section: str
    page_start: int
    page_end: int
    text: str
    kind: str = "text"

    @property
    def page_label(self) -> str:
        start = self.page_start + 1
        end = self.page_end + 1
        return f"p.{start}" if start == end else f"pp.{start}-{end}"


AGENT_TERMS = {
    "method": {
        "abstract", "introduction", "method", "methodology", "model", "approach",
        "framework", "architecture", "algorithm", "training", "loss", "objective",
        "proposed", "implementation", "模型", "方法", "架构", "算法",
    },
    "experiment": {
        "abstract", "experiment", "experiments", "evaluation", "result", "results",
        "dataset", "datasets", "metric", "metrics", "baseline", "ablation", "table",
        "benchmark", "实验", "评估", "结果", "数据集", "指标", "消融",
    },
    "critic": {
        "abstract", "introduction", "related work", "discussion", "limitation",
        "limitations", "conclusion", "future work", "analysis", "method",
        "methodology", "model", "approach", "framework", "experiment",
        "experiments", "evaluation", "result", "results", "实验", "方法",
        "模型", "讨论", "局限", "结论", "未来工作", "相关工作",
    },
}


def build_evidence_index(
    paper: ParsedPaper,
    *,
    chunk_chars: int = 2400,
    overlap_chars: int = 260,
) -> list[EvidenceSnippet]:
    """Split parsed sections into bounded original-text evidence snippets."""
    snippets: list[EvidenceSnippet] = []
    source_sections = paper.sections or [
        Section("Full Paper", paper.full_text, 0, 0),
    ]

    for section in source_sections:
        cleaned = _compact_text(section.content)
        if not cleaned:
            continue
        for chunk in _chunk_text(cleaned, chunk_chars, overlap_chars):
            snippets.append(
                EvidenceSnippet(
                    id=f"E{len(snippets) + 1:03d}",
                    section=section.title,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    text=chunk,
                    kind="text",
                )
            )

    if not snippets and paper.full_text.strip():
        snippets.append(
            EvidenceSnippet(
                id="E001",
                section="Full Paper",
                page_start=0,
                page_end=0,
                text=_compact_text(paper.full_text)[:chunk_chars],
                kind="text",
            )
        )

    for index, table in enumerate(paper.tables, start=1):
        table_text = _format_table_block(table)
        if not table_text:
            continue
        snippets.append(
            EvidenceSnippet(
                id=f"T{index:03d}",
                section=table.caption or "Table",
                page_start=table.page,
                page_end=table.page,
                text=table_text,
                kind="table",
            )
        )

    for index, figure in enumerate(paper.figures, start=1):
        figure_text = _format_figure_block(figure)
        if not figure_text:
            continue
        snippets.append(
            EvidenceSnippet(
                id=f"F{index:03d}",
                section=figure.caption or "Figure / Visual region",
                page_start=figure.page,
                page_end=figure.page,
                text=figure_text,
                kind="figure",
            )
        )
    return snippets


def evidence_payload(snippets: Iterable[EvidenceSnippet]) -> list[dict[str, object]]:
    """Return a compact API-safe evidence index preview."""
    return [
        {
            "id": snippet.id,
            "section": snippet.section,
            "page_start": snippet.page_start,
            "page_end": snippet.page_end,
            "page_label": snippet.page_label,
            "kind": snippet.kind,
            "chars": len(snippet.text),
            "preview": snippet.text[:220],
        }
        for snippet in snippets
    ]


def evidence_context_for_agent(
    snippets: list[EvidenceSnippet],
    agent_type: str,
    *,
    max_chars: int = 18000,
    max_snippets: int = 10,
) -> str:
    """Select relevant original snippets and format them with stable evidence IDs."""
    selected = select_evidence_snippets(
        snippets,
        agent_type,
        max_chars=max_chars,
        max_snippets=max_snippets,
    )
    if not selected:
        return ""

    blocks = []
    for snippet in selected:
        blocks.append(
            "\n".join(
                [
                    f"[{snippet.id} | {snippet.kind} | {snippet.section} | {snippet.page_label}]",
                    snippet.text,
                ]
            )
        )
    return "\n\n".join(blocks)


def select_evidence_snippets(
    snippets: list[EvidenceSnippet],
    agent_type: str,
    *,
    max_chars: int,
    max_snippets: int,
) -> list[EvidenceSnippet]:
    """Rank snippets by section and keyword relevance while preserving source text."""
    terms = AGENT_TERMS.get(agent_type, set())
    scored: list[tuple[int, int, EvidenceSnippet]] = []
    for index, snippet in enumerate(snippets):
        haystack = f"{_normalize_title(snippet.section)} {snippet.text[:900].lower()}"
        score = sum(1 for term in terms if term in haystack)
        if "abstract" in haystack:
            score += 1
        if agent_type == "experiment" and snippet.kind == "table":
            score += 7
        if agent_type == "method" and snippet.kind == "figure":
            score += 3
        if agent_type == "critic" and snippet.kind in {"table", "figure"}:
            score += 2
        if snippet.id.startswith(("T", "F")) and snippet.text:
            score += 1
        scored.append((score, index, snippet))

    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen: list[EvidenceSnippet] = []
    total_chars = 0
    for score, _, snippet in scored:
        if score <= 0 and chosen:
            continue
        next_size = len(snippet.text)
        if chosen and total_chars + next_size > max_chars:
            continue
        chosen.append(snippet)
        total_chars += next_size
        if len(chosen) >= max_snippets:
            break

    if not chosen:
        for snippet in snippets[:max_snippets]:
            next_size = len(snippet.text)
            if chosen and total_chars + next_size > max_chars:
                break
            chosen.append(snippet)
            total_chars += next_size

    return sorted(chosen, key=lambda snippet: (snippet.page_start, snippet.id))


def _chunk_text(text: str, chunk_chars: int, overlap_chars: int) -> Iterable[str]:
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)


def _compact_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _format_table_block(table: TableBlock, *, max_rows: int = 35, max_cols: int = 10) -> str:
    pieces = []
    if table.caption:
        pieces.append(f"Caption: {table.caption}")
    if not table.rows:
        return "\n".join(pieces)

    rows = [row[:max_cols] for row in table.rows[:max_rows]]
    width = max((len(row) for row in rows), default=0)
    if width == 0:
        return "\n".join(pieces)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    separator = ["---"] * width
    body = normalized[1:]
    markdown_rows = [header, separator, *body]
    table_markdown = "\n".join(
        "| " + " | ".join(_escape_table_cell(cell) for cell in row) + " |"
        for row in markdown_rows
    )
    pieces.append(table_markdown)
    if len(table.rows) > max_rows or any(len(row) > max_cols for row in table.rows):
        pieces.append("Note: table was truncated for model context; use visible rows only.")
    return "\n".join(pieces).strip()


def _format_figure_block(figure: FigureBlock) -> str:
    pieces = []
    if figure.caption:
        pieces.append(f"Caption: {figure.caption}")
    if figure.visual_summary:
        pieces.append(f"Vision summary: {figure.visual_summary}")
    if figure.bbox:
        x0, y0, x1, y1 = figure.bbox
        pieces.append(
            "Visual region detected on page "
            f"{figure.page + 1}: bbox=({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f})."
        )
        if not figure.visual_summary:
            pieces.append("Pixel-level interpretation was not generated for this region.")
    return "\n".join(pieces).strip()


def _escape_table_cell(cell: str) -> str:
    return str(cell).replace("|", "\\|").strip()
