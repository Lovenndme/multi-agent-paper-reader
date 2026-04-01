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


@dataclass
class Section:
    title: str
    content: str
    page_start: int
    page_end: int


@dataclass
class ParsedPaper:
    title: str
    full_text: str
    sections: List[Section] = field(default_factory=list)
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
            and len(stripped) <= 60
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

    # Plain text per page (for regex fallback and full_text)
    pages_text: List[Tuple[int, str]] = []
    for page_num, page in enumerate(doc):
        pages_text.append((page_num, page.get_text("text")))

    full_text = "\n".join(t for _, t in pages_text)

    # --- Strategy 1: font-size heuristic ---
    rows = _extract_blocks_with_fontsize(doc)
    doc.close()

    body_size = _detect_body_fontsize(rows)
    sections = _split_by_fontsize(rows, body_size)

    # --- Strategy 2: regex fallback ---
    if len(sections) < 3:
        sections = _split_by_regex(pages_text)

    # --- Strategy 3: single-block fallback ---
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
        metadata={k: str(v) for k, v in meta.items() if v},
    )
