"""PDF parsing and section splitting utilities."""

import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Section → Agent keyword mapping  (English + Chinese)
# ---------------------------------------------------------------------------

METHOD_SECTIONS = {
    # English
    "abstract", "introduction", "related work", "background",
    "method", "methodology", "model", "approach", "framework",
    "system", "architecture", "proposed",
    # Chinese
    "摘要", "引言", "相关工作", "背景", "方法", "模型", "框架",
    "系统", "架构", "提出", "技术", "算法",
}
EXPERIMENT_SECTIONS = {
    # English
    "abstract", "experiment", "experiments", "experimental setup",
    "experimental results", "experimental evaluation",
    "result", "results", "discussion", "analysis", "evaluation",
    # Chinese
    "摘要", "实验", "评估", "结果", "分析", "讨论", "对比",
}
CRITIC_SECTIONS = {
    # English
    "abstract", "introduction", "related work", "conclusion",
    "conclusions", "concluding remarks", "limitation", "limitations",
    "future work", "discussion",
    # Chinese
    "摘要", "引言", "相关工作", "结论", "局限", "未来工作", "讨论", "展望",
}

# ---------------------------------------------------------------------------
# Regex patterns for section headers (English + Chinese)
# ---------------------------------------------------------------------------

_EN_PATTERNS = [
    r"^abstract$",
    r"^\d+\.?\s+introduction$",
    r"^\d+\.?\s+related\s+work$",
    r"^\d+\.?\s+background$",
    r"^\d+\.?\s+method(?:ology)?$",
    r"^\d+\.?\s+(?:proposed\s+)?(?:model|approach|framework|system|architecture)$",
    r"^\d+\.?\s+experiment(?:s|al\s+(?:setup|results|evaluation))?$",
    r"^\d+\.?\s+result(?:s)?$",
    r"^\d+\.?\s+(?:discussion|analysis|evaluation)$",
    r"^\d+\.?\s+(?:conclusion|conclusions|concluding\s+remarks)$",
    r"^\d+\.?\s+(?:limitation(?:s)?|future\s+work)$",
    r"^\d+\.?\s+(?:acknowledgment(?:s)?|acknowledgement(?:s)?)$",
    r"^\d+\.?\s+reference(?:s)?$",
    r"^appendix.*$",
]

_ZH_PATTERNS = [
    r"^摘\s*要$",
    r"^[一二三四五六七八九十\d]+[\.、\s]+[\u4e00-\u9fff\w\s]{1,20}$",  # 1. 引言 / 一、方法
    r"^\d+\s+[\u4e00-\u9fff][\u4e00-\u9fff\w\s]{0,15}$",              # 1 引言
    r"^[\u4e00-\u9fff]{2,10}$",                                         # 纯中文短标题
]

_ALL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _EN_PATTERNS] + \
                [re.compile(p) for p in _ZH_PATTERNS]

_FONT_HEADER_KEYWORDS = {
    "abstract", "introduction", "related work", "background", "method",
    "methodology", "model", "approach", "framework", "architecture",
    "experiment", "experiments", "evaluation", "result", "results",
    "discussion", "analysis", "conclusion", "limitations", "references",
    "dataset", "datasets", "matching", "acquisition", "preprocessing",
    "classification", "training", "algorithm",
}


@dataclass
class Section:
    title: str
    content: str
    page_start: int
    page_end: int


@dataclass
class TableBlock:
    page: int
    rows: List[List[str]]
    caption: str = ""


@dataclass
class FigureBlock:
    page: int
    caption: str = ""
    image_index: int = 0
    bbox: Optional[Tuple[float, float, float, float]] = None
    visual_summary: str = ""


@dataclass
class ParsedPaper:
    title: str
    full_text: str
    sections: List[Section] = field(default_factory=list)
    tables: List[TableBlock] = field(default_factory=list)
    figures: List[FigureBlock] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)

    def get_sections_for_agent(self, agent_type: str) -> str:
        """Return concatenated section text relevant to a given agent type."""
        if agent_type == "method":
            relevant = METHOD_SECTIONS
        elif agent_type == "experiment":
            relevant = EXPERIMENT_SECTIONS
        elif agent_type == "critic":
            relevant = CRITIC_SECTIONS
        else:
            return self.full_text

        parts = []
        for sec in self.sections:
            normalized = _normalize_title(sec.title)
            if any(kw in normalized for kw in relevant):
                parts.append(f"## {sec.title}\n{sec.content}")

        # Fall back to full text when section matching fails
        if not parts:
            return self.full_text

        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_title(title: str) -> str:
    """Lowercase, strip leading numbering."""
    t = title.strip()
    t = re.sub(r"^[\d一二三四五六七八九十]+(?:[\.\d]*)[\.、\s]+", "", t)
    return t.lower()


def _matches_header_pattern(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return False
    for pat in _ALL_PATTERNS:
        if pat.match(stripped.lower() if pat.flags & re.IGNORECASE else stripped):
            return True
    return False


def _looks_like_font_header(line: str) -> bool:
    """Reject title fragments, equations, and symbol-heavy text from font heuristics."""
    stripped = re.sub(r"\s+", " ", line).strip()
    if not stripped or len(stripped) > 72:
        return False
    if stripped.startswith(("(", "[", "{")):
        return False
    if "\ufffd" in stripped or "�" in stripped:
        return False
    if re.fullmatch(r"[\W_]+", stripped):
        return False

    letters = re.findall(r"[A-Za-z\u4e00-\u9fff]", stripped)
    if len(letters) < 2:
        return False

    symbols = re.findall(r"[^A-Za-z0-9\u4e00-\u9fff\s.\-:/&]", stripped)
    if len(symbols) / max(len(stripped), 1) > 0.16:
        return False

    if _matches_header_pattern(stripped):
        return True

    normalized = _normalize_title(stripped)
    if any(keyword in normalized for keyword in _FONT_HEADER_KEYWORDS):
        return True

    if re.search(r"[\u4e00-\u9fff]", stripped) and len(stripped) <= 24:
        return True

    words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", stripped)
    if 1 <= len(words) <= 5:
        return all(word.isupper() or word[:1].isupper() for word in words)

    return False


def _extract_blocks_with_fontsize(doc: fitz.Document) -> List[Tuple[int, str, float]]:
    """Return list of (page_num, line_text, font_size) for every text span."""
    rows = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block.get("type") != 0:  # 0 = text block
                continue
            for line in block.get("lines", []):
                line_text = "".join(s["text"] for s in line.get("spans", []))
                sizes = [s["size"] for s in line.get("spans", []) if s["text"].strip()]
                if not sizes:
                    continue
                avg_size = sum(sizes) / len(sizes)
                rows.append((page_num, line_text, avg_size))
    return rows


def _detect_body_fontsize(rows: List[Tuple[int, str, float]]) -> float:
    """Estimate the dominant body font size via median."""
    sizes = [sz for _, txt, sz in rows if txt.strip()]
    if not sizes:
        return 10.0
    return statistics.median(sizes)


def _split_by_fontsize(
    rows: List[Tuple[int, str, float]], body_size: float, threshold_ratio: float = 1.15
) -> List[Section]:
    """
    Detect section headers as lines whose font size exceeds body_size * threshold_ratio
    AND whose text is short (plausible header length).
    """
    sections: List[Section] = []
    current_title: Optional[str] = None
    current_lines: List[str] = []
    current_page_start = 0
    current_page = 0

    header_size = body_size * threshold_ratio

    for page_num, text, size in rows:
        stripped = text.strip()
        is_header = (
            size >= header_size
            and stripped
            and _looks_like_font_header(stripped)
            and not stripped[-1] in ".。,，;；:："  # headers don't end with punctuation
        )
        if is_header:
            if current_title and current_lines:
                sections.append(Section(
                    title=current_title,
                    content="\n".join(current_lines).strip(),
                    page_start=current_page_start,
                    page_end=current_page,
                ))
            current_title = stripped
            current_lines = []
            current_page_start = page_num
        else:
            if current_title:
                current_lines.append(text)
        current_page = page_num

    if current_title and current_lines:
        sections.append(Section(
            title=current_title,
            content="\n".join(current_lines).strip(),
            page_start=current_page_start,
            page_end=current_page,
        ))
    return sections


def _split_by_regex(pages_text: List[Tuple[int, str]]) -> List[Section]:
    """Original regex-based splitting, kept as fallback."""
    sections: List[Section] = []
    current_title: Optional[str] = None
    current_lines: List[str] = []
    current_page_start = 0
    current_page = 0

    for page_num, page_text in pages_text:
        for line in page_text.splitlines():
            if _matches_header_pattern(line):
                if current_title and current_lines:
                    sections.append(Section(
                        title=current_title,
                        content="\n".join(current_lines).strip(),
                        page_start=current_page_start,
                        page_end=current_page,
                    ))
                current_title = line.strip()
                current_lines = []
                current_page_start = page_num
            else:
                if current_title:
                    current_lines.append(line)
        current_page = page_num

    if current_title and current_lines:
        sections.append(Section(
            title=current_title,
            content="\n".join(current_lines).strip(),
            page_start=current_page_start,
            page_end=current_page,
        ))
    return sections


def _split_by_toc(toc: List[List[object]], pages_text: List[Tuple[int, str]]) -> List[Section]:
    """Split by embedded PDF outline/bookmarks when present."""
    if not toc or not pages_text:
        return []

    page_count = len(pages_text)
    candidates: List[Tuple[int, str, int]] = []
    seen: set[Tuple[str, int]] = set()
    for row in toc:
        if len(row) < 3:
            continue
        level, title, page_number = row[:3]
        if not isinstance(title, str):
            continue
        title = re.sub(r"\s+", " ", title).strip()
        if not title or len(title) > 100:
            continue
        try:
            page_index = int(page_number) - 1
            level_num = int(level)
        except (TypeError, ValueError):
            continue
        if page_index < 0 or page_index >= page_count or level_num > 3:
            continue
        key = (title.lower(), page_index)
        if key in seen:
            continue
        seen.add(key)
        candidates.append((page_index, title, level_num))

    if len(candidates) < 2:
        return []

    candidates.sort(key=lambda item: item[0])
    sections: List[Section] = []
    for index, (page_start, title, _) in enumerate(candidates):
        next_page = candidates[index + 1][0] if index + 1 < len(candidates) else page_count
        page_end = max(page_start, next_page - 1)
        page_end = min(page_end, page_count - 1)
        content = "\n".join(text for page, text in pages_text if page_start <= page <= page_end).strip()
        if content:
            sections.append(Section(title=title, content=content, page_start=page_start, page_end=page_end))
    return sections


def _extract_caption_lines(page_text: str, kind: str) -> List[str]:
    if kind == "table":
        pattern = re.compile(r"^\s*(?:Table|TABLE|Tab\.|表)\s*[\dIVXivx一二三四五六七八九十]+[.:：\s-]*(.+)?")
    else:
        pattern = re.compile(r"^\s*(?:Figure|FIGURE|Fig\.|图)\s*[\dIVXivx一二三四五六七八九十]+[.:：\s-]*(.+)?")

    captions: List[str] = []
    lines = [line.strip() for line in page_text.splitlines()]
    for index, line in enumerate(lines):
        if not line:
            continue
        if not pattern.match(line):
            continue
        pieces = [line]
        if index + 1 < len(lines):
            next_line = lines[index + 1].strip()
            if next_line and len(next_line) <= 180 and not _matches_header_pattern(next_line):
                pieces.append(next_line)
        captions.append(" ".join(pieces))
    return captions


def _clean_table_rows(rows: List[List[object]]) -> List[List[str]]:
    cleaned: List[List[str]] = []
    for row in rows:
        cells = []
        for cell in row:
            value = "" if cell is None else str(cell)
            value = re.sub(r"\s+", " ", value).strip()
            cells.append(value)
        if any(cells):
            cleaned.append(cells)
    return cleaned


def _extract_tables_and_figures(
    doc: fitz.Document,
    pages_text: List[Tuple[int, str]],
) -> Tuple[List[TableBlock], List[FigureBlock]]:
    tables: List[TableBlock] = []
    figures: List[FigureBlock] = []
    text_by_page = {page: text for page, text in pages_text}

    for page_num, page in enumerate(doc):
        page_text = text_by_page.get(page_num, "")
        table_captions = _extract_caption_lines(page_text, "table")
        figure_captions = _extract_caption_lines(page_text, "figure")

        try:
            found_tables = page.find_tables().tables
        except Exception:  # noqa: BLE001 - table detection is best-effort
            found_tables = []

        for index, table in enumerate(found_tables):
            try:
                rows = _clean_table_rows(table.extract())
            except Exception:  # noqa: BLE001 - skip malformed table objects
                continue
            if len(rows) < 2:
                continue
            caption = table_captions[index] if index < len(table_captions) else ""
            tables.append(TableBlock(page=page_num, rows=rows, caption=caption))

        if not found_tables:
            for caption in table_captions:
                tables.append(TableBlock(page=page_num, rows=[], caption=caption))

        try:
            image_infos = page.get_image_info(xrefs=True)
        except Exception:  # noqa: BLE001 - image metadata is best-effort
            image_infos = []

        page_area = max(page.rect.width * page.rect.height, 1)
        large_images = []
        for info in image_infos:
            bbox = info.get("bbox")
            if not bbox:
                continue
            x0, y0, x1, y1 = bbox
            area = max((x1 - x0) * (y1 - y0), 0)
            if area / page_area >= 0.015:
                large_images.append(tuple(float(v) for v in bbox))

        max_items = max(len(figure_captions), len(large_images))
        for index in range(max_items):
            caption = figure_captions[index] if index < len(figure_captions) else ""
            bbox = large_images[index] if index < len(large_images) else None
            figures.append(FigureBlock(page=page_num, caption=caption, image_index=index + 1, bbox=bbox))

    return tables, figures


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path: str | Path) -> ParsedPaper:
    """Extract text from a PDF and split it into sections.

    Strategy (best-effort, graceful fallback):
    1. Try font-size heuristic (works for most formatted PDFs).
    2. If that yields <3 sections, try regex patterns (English + Chinese).
    3. If still <2 sections, store full text as one section so agents still work.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))

    meta = doc.metadata or {}
    paper_title = meta.get("title", "").strip() or pdf_path.stem
    toc = doc.get_toc(simple=True)

    # Plain text per page (for regex fallback and full_text)
    pages_text: List[Tuple[int, str]] = []
    for page_num, page in enumerate(doc):
        pages_text.append((page_num, page.get_text("text")))

    full_text = "\n".join(t for _, t in pages_text)
    tables, figures = _extract_tables_and_figures(doc, pages_text)

    # --- Strategy 1: embedded outline / bookmarks ---
    sections = _split_by_toc(toc, pages_text)

    # --- Strategy 2: font-size heuristic ---
    rows = _extract_blocks_with_fontsize(doc)
    doc.close()

    if len(sections) < 3:
        body_size = _detect_body_fontsize(rows)
        sections = _split_by_fontsize(rows, body_size)

    # --- Strategy 3: regex fallback ---
    if len(sections) < 3:
        sections = _split_by_regex(pages_text)

    # --- Strategy 4: single-block fallback ---
    if len(sections) < 2:
        sections = [Section(
            title="Full Paper",
            content=full_text,
            page_start=0,
            page_end=len(pages_text) - 1,
        )]

    return ParsedPaper(
        title=paper_title,
        full_text=full_text,
        sections=sections,
        tables=tables,
        figures=figures,
        metadata={k: str(v) for k, v in meta.items() if v},
    )
