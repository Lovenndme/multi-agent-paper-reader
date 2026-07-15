"""PDF parsing and section splitting utilities."""

import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pymupdf4llm


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
    bbox: Optional[Tuple[float, float, float, float]] = None
    caption_bbox: Optional[Tuple[float, float, float, float]] = None


@dataclass
class FigureBlock:
    page: int
    caption: str = ""
    image_index: int = 0
    bbox: Optional[Tuple[float, float, float, float]] = None
    caption_bbox: Optional[Tuple[float, float, float, float]] = None
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


def _infer_title_from_first_page(doc: fitz.Document) -> str:
    """Infer a missing PDF title from the most prominent first-page text block."""
    if doc.page_count < 1:
        return ""
    page = doc[0]
    page_height = max(float(page.rect.height), 1.0)
    candidates: list[tuple[float, float, str]] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        y0 = float(block.get("bbox", (0, page_height, 0, page_height))[1])
        if y0 > page_height * 0.45:
            continue
        lines: list[str] = []
        sizes: list[float] = []
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                lines.append(text)
            sizes.extend(
                float(span.get("size", 0))
                for span in line.get("spans", [])
                if str(span.get("text", "")).strip()
            )
        candidate = re.sub(r"\s+", " ", " ".join(lines)).strip()
        if not sizes or not 8 <= len(candidate) <= 300:
            continue
        normalized = candidate.lower().strip(" .:-")
        if normalized.startswith(("abstract", "keywords", "arxiv", "doi:")):
            continue
        if len(re.findall(r"[A-Za-z\u4e00-\u9fff]", candidate)) < 5:
            continue
        candidates.append((max(sizes), -y0, candidate))
    return max(candidates, default=(0.0, 0.0, ""))[2]


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


_LAYOUT_BODY_CLASSES = {"section-header", "text", "list-item", "formula", "footnote"}


def _extract_layout_content(
    pdf_path: Path,
) -> Tuple[List[Tuple[int, str]], List[Section], List[TableBlock], List[FigureBlock]]:
    """Use PyMuPDF4LLM Layout to classify body, tables, captions and vector figures."""
    raw = pymupdf4llm.to_json(str(pdf_path))
    payload = json.loads(raw) if isinstance(raw, str) else raw
    layout_pages = payload.get("pages", []) if isinstance(payload, dict) else []
    if not layout_pages:
        raise ValueError("Layout parser returned no pages.")

    pages_text: List[Tuple[int, str]] = []
    tables: List[TableBlock] = []
    figures: List[FigureBlock] = []
    section_events: list[tuple[int, str, str]] = []

    for fallback_page, page_payload in enumerate(layout_pages):
        page_number = int(page_payload.get("page_number") or fallback_page + 1) - 1
        page_width = float(page_payload.get("width") or 1.0)
        page_height = float(page_payload.get("height") or 1.0)
        boxes = [box for box in page_payload.get("boxes", []) if isinstance(box, dict)]
        boxes.sort(key=lambda box: (float(box.get("y0") or 0), float(box.get("x0") or 0)))

        page_lines: list[str] = []
        caption_boxes: list[dict[str, object]] = []
        picture_boxes: list[dict[str, object]] = []
        table_boxes: list[dict[str, object]] = []
        for box in boxes:
            boxclass = str(box.get("boxclass") or "")
            text = _layout_box_text(box)
            is_caption_text = bool(text and (_is_figure_caption(text) or _is_table_caption(text)))
            if is_caption_text:
                caption_boxes.append({"bbox": _layout_bbox(box), "text": text})
            elif boxclass in _LAYOUT_BODY_CLASSES and text:
                page_lines.append(text)
                section_events.append((page_number, boxclass, text))
            if boxclass == "caption" and text and not is_caption_text:
                caption_boxes.append({"bbox": _layout_bbox(box), "text": text})
            elif boxclass == "picture":
                bbox = _layout_bbox(box)
                if bbox and _bbox_area(bbox) / max(page_width * page_height, 1.0) >= 0.015:
                    picture_boxes.append({"bbox": bbox, "text": text})
            elif boxclass == "table" and isinstance(box.get("table"), dict):
                table_boxes.append(box)

        pages_text.append((page_number, "\n".join(page_lines).strip()))

        table_captions = [item for item in caption_boxes if _is_table_caption(str(item["text"]))]
        figure_captions = [item for item in caption_boxes if _is_figure_caption(str(item["text"]))]
        used_table_captions: set[int] = set()
        for table_box in table_boxes:
            table_data = table_box.get("table") or {}
            rows = _clean_table_rows(table_data.get("extract") or [])
            bbox = _layout_bbox(table_box)
            caption, caption_bbox = _match_layout_caption(
                bbox,
                table_captions,
                used_table_captions,
                page_height=page_height,
            )
            if rows or caption:
                tables.append(
                    TableBlock(
                        page=page_number,
                        rows=rows,
                        caption=caption,
                        bbox=bbox,
                        caption_bbox=caption_bbox,
                    )
                )

        grouped_pictures = _group_layout_pictures(
            picture_boxes,
            figure_captions,
            page_height=page_height,
        )
        for image_index, (bbox, caption, caption_bbox) in enumerate(grouped_pictures, start=1):
            figures.append(
                FigureBlock(
                    page=page_number,
                    caption=caption,
                    image_index=image_index,
                    bbox=bbox,
                    caption_bbox=caption_bbox,
                )
            )

    pages_text.sort(key=lambda item: item[0])
    sections = _split_layout_sections(section_events)
    return pages_text, sections, tables, figures


def _split_layout_sections(events: list[tuple[int, str, str]]) -> List[Section]:
    """Build sections from layout-classified headers without figure-label leakage."""
    sections: List[Section] = []
    current_title = "Front matter"
    current_lines: list[str] = []
    current_page_start = events[0][0] if events else 0
    current_page = current_page_start

    for page_number, boxclass, text in events:
        clean_text = re.sub(r"\s+", " ", text).strip()
        is_header = boxclass == "section-header" and 1 < len(clean_text) <= 160
        if is_header:
            if current_lines:
                sections.append(
                    Section(
                        title=current_title,
                        content="\n".join(current_lines).strip(),
                        page_start=current_page_start,
                        page_end=current_page,
                    )
                )
            current_title = clean_text
            current_lines = []
            current_page_start = page_number
        elif clean_text:
            current_lines.append(text)
        current_page = page_number
    if current_lines:
        sections.append(
            Section(
                title=current_title,
                content="\n".join(current_lines).strip(),
                page_start=current_page_start,
                page_end=current_page,
            )
        )
    return sections


def _layout_box_text(box: dict[str, object]) -> str:
    lines: list[str] = []
    for line in box.get("textlines") or []:
        spans = line.get("spans") or []
        pieces: list[str] = []
        previous_x1: float | None = None
        previous_size = 0.0
        for span in spans:
            text = str(span.get("text") or "")
            bbox = span.get("bbox") or [0, 0, 0, 0]
            x0 = float(bbox[0])
            if (
                pieces
                and previous_x1 is not None
                and x0 - previous_x1 > max(0.8, previous_size * 0.12)
                and text
                and not text.startswith((".", ",", ":", ";", ")", "]", "}", "%"))
            ):
                pieces.append(" ")
            pieces.append(text)
            previous_x1 = float(bbox[2])
            previous_size = float(span.get("size") or 0)
        line_text = "".join(pieces).strip()
        if line_text:
            lines.append(line_text)
    return "\n".join(lines).strip()


def _layout_bbox(box: dict[str, object]) -> Optional[Tuple[float, float, float, float]]:
    values = [box.get("x0"), box.get("y0"), box.get("x1"), box.get("y1")]
    if any(value is None for value in values):
        table = box.get("table")
        values = table.get("bbox", []) if isinstance(table, dict) else []
    if len(values) != 4:
        return None
    try:
        bbox = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
    return bbox if bbox[2] > bbox[0] and bbox[3] > bbox[1] else None


def _bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _is_figure_caption(text: str) -> bool:
    return bool(re.match(r"^\s*(?:Figure|FIGURE|Fig\.?|图)\s*[\dIVXivx一二三四五六七八九十]+", text))


def _is_table_caption(text: str) -> bool:
    return bool(re.match(r"^\s*(?:Table|TABLE|Tab\.?|表)\s*[\dIVXivx一二三四五六七八九十]+", text))


def _match_layout_caption(
    bbox: Optional[Tuple[float, float, float, float]],
    captions: list[dict[str, object]],
    used: set[int],
    *,
    page_height: float,
) -> tuple[str, Optional[Tuple[float, float, float, float]]]:
    """Pair by same-page geometry instead of unrelated list positions."""
    if bbox is None:
        return "", None
    candidates: list[tuple[float, int]] = []
    for index, caption in enumerate(captions):
        if index in used or not caption.get("bbox"):
            continue
        cost = _layout_caption_cost(bbox, caption["bbox"], page_height=page_height)
        if cost is not None:
            candidates.append((cost, index))
    if not candidates:
        return "", None
    _, best_index = min(candidates)
    used.add(best_index)
    caption_bbox = captions[best_index].get("bbox")
    if not isinstance(caption_bbox, tuple) or len(caption_bbox) != 4:
        caption_bbox = None
    return str(captions[best_index]["text"]), caption_bbox


def _layout_caption_cost(
    bbox: Tuple[float, float, float, float],
    caption_bbox: object,
    *,
    page_height: float,
    max_vertical_gap_ratio: float = 0.28,
    require_horizontal_overlap: bool = False,
) -> float | None:
    """Score a same-page visual/caption pair, rejecting distant candidates."""
    if not isinstance(caption_bbox, tuple) or len(caption_bbox) != 4:
        return None
    vertical_gap = (
        caption_bbox[1] - bbox[3]
        if caption_bbox[1] >= bbox[3]
        else bbox[1] - caption_bbox[3]
        if caption_bbox[3] <= bbox[1]
        else 0.0
    )
    if vertical_gap > page_height * max_vertical_gap_ratio:
        return None
    overlap = max(0.0, min(bbox[2], caption_bbox[2]) - max(bbox[0], caption_bbox[0]))
    if require_horizontal_overlap and overlap <= 0:
        return None
    width = max(1.0, min(bbox[2] - bbox[0], caption_bbox[2] - caption_bbox[0]))
    above_penalty = 0.0 if caption_bbox[1] >= bbox[1] else page_height * 0.08
    return max(0.0, vertical_gap) + above_penalty + (1.0 - min(overlap / width, 1.0)) * 30.0


def _group_layout_pictures(
    pictures: list[dict[str, object]],
    captions: list[dict[str, object]],
    *,
    page_height: float,
) -> list[
    tuple[
        Tuple[float, float, float, float],
        str,
        Optional[Tuple[float, float, float, float]],
    ]
]:
    """Merge split subpictures that geometrically belong to the same caption.

    Layout commonly emits one ``picture`` box per panel or vector component. Treating
    those boxes as separate figures sends incomplete crops to the vision model. Each
    picture is therefore assigned to its nearest plausible caption, and boxes sharing
    that caption are rendered as their bounded union. Captionless regions remain
    independent so unrelated visuals are not merged by list position.
    """
    grouped: dict[int, list[Tuple[float, float, float, float]]] = {}
    uncaptioned: list[Tuple[float, float, float, float]] = []

    for picture in pictures:
        bbox = picture.get("bbox")
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            continue
        candidates: list[tuple[float, int]] = []
        for caption_index, caption in enumerate(captions):
            cost = _layout_caption_cost(
                bbox,
                caption.get("bbox"),
                page_height=page_height,
                max_vertical_gap_ratio=0.12,
                require_horizontal_overlap=True,
            )
            if cost is not None:
                candidates.append((cost, caption_index))
        if candidates:
            _, caption_index = min(candidates)
            grouped.setdefault(caption_index, []).append(bbox)
        else:
            uncaptioned.append(bbox)

    results: list[
        tuple[
            Tuple[float, float, float, float],
            str,
            Optional[Tuple[float, float, float, float]],
        ]
    ] = []
    for caption_index, boxes in grouped.items():
        union = (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )
        caption_bbox = captions[caption_index].get("bbox")
        if not isinstance(caption_bbox, tuple) or len(caption_bbox) != 4:
            caption_bbox = None
        results.append(
            (
                union,
                str(captions[caption_index].get("text") or ""),
                caption_bbox,
            )
        )
    results.extend((bbox, "", None) for bbox in uncaptioned)
    results.sort(key=lambda item: (item[0][1], item[0][0]))
    return results


def _extract_classic_content(
    doc: fitz.Document,
    toc: List[List[object]],
    *,
    include_visuals: bool,
) -> Tuple[List[Tuple[int, str]], List[Section], List[TableBlock], List[FigureBlock]]:
    """Return the lightweight/fallback PyMuPDF extraction path."""
    pages_text = [
        (page_num, page.get_text("text"))
        for page_num, page in enumerate(doc)
    ]
    sections = _split_by_toc(toc, pages_text)
    if include_visuals:
        tables, figures = _extract_tables_and_figures(doc, pages_text)
    else:
        tables, figures = [], []
    return pages_text, sections, tables, figures


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path: str | Path, *, layout: bool = True) -> ParsedPaper:
    """Extract text from a PDF and split it into sections.

    Strategy (best-effort, graceful fallback):
    1. Use PyMuPDF4LLM Layout for body/table/caption/raster/vector regions.
    2. Fall back to outline, font-size and regex parsing when layout is unavailable.
    3. ``layout=False`` keeps upload preview lightweight and skips rich evidence extraction.
    4. If section detection still fails, preserve the full text as one section.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    try:
        meta = doc.metadata or {}
        metadata_title = re.sub(r"\s+", " ", meta.get("title", "")).strip()
        normalized_metadata_title = metadata_title.lower().strip(" .:-")
        generic_titles = {"", "paper", "document", "untitled", pdf_path.stem.lower()}
        if normalized_metadata_title in generic_titles or normalized_metadata_title.startswith("microsoft word"):
            paper_title = _infer_title_from_first_page(doc) or pdf_path.stem
        else:
            paper_title = metadata_title
        toc = doc.get_toc(simple=True)

        if layout:
            parser_backend = "pymupdf4llm-layout"
            try:
                pages_text, sections, tables, figures = _extract_layout_content(pdf_path)
            except Exception:  # layout is preferred; classic parsing remains a safe fallback
                parser_backend = "pymupdf-classic-fallback"
                pages_text, sections, tables, figures = _extract_classic_content(
                    doc,
                    toc,
                    include_visuals=True,
                )
        else:
            parser_backend = "pymupdf-classic-preview"
            pages_text, sections, tables, figures = _extract_classic_content(
                doc,
                toc,
                include_visuals=False,
            )

        full_text = "\n".join(text for _, text in pages_text)
        rows = _extract_blocks_with_fontsize(doc)
    finally:
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

    clean_metadata = {k: str(v) for k, v in meta.items() if v}
    clean_metadata["parser_backend"] = parser_backend
    return ParsedPaper(
        title=paper_title,
        full_text=full_text,
        sections=sections,
        tables=tables,
        figures=figures,
        metadata=clean_metadata,
    )
