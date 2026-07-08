"""Vision enrichment for PDF figures and visual regions."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import fitz

from core.pdf_parser import FigureBlock, ParsedPaper
from utils.llm import invoke_vision_image_summary, is_vision_configured


VisionSummarizer = Callable[[bytes, str], str]


@dataclass
class VisionEnrichmentResult:
    total_figures: int
    attempted: int = 0
    enriched: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


def enrich_paper_figures_with_vision(
    pdf_path: str | Path,
    paper: ParsedPaper,
    *,
    summarizer: VisionSummarizer | None = None,
) -> VisionEnrichmentResult:
    """Render PDF visual regions and attach concise vision summaries to figures.

    The enrichment is deliberately best-effort: if no vision model is configured,
    or an individual figure fails to render/analyze, text/table analysis still runs.
    """
    result = VisionEnrichmentResult(total_figures=len(paper.figures))
    if not paper.figures:
        return result

    if summarizer is None and not is_vision_configured():
        result.skipped = len(paper.figures)
        return result

    selected_figures = _select_figures_for_vision(paper.figures, max_figures=_vision_max_figures())
    if not selected_figures:
        result.skipped = len(paper.figures)
        return result
    pdf_path = Path(pdf_path)

    def enrich_one(figure: FigureBlock) -> tuple[FigureBlock, str]:
        doc = fitz.open(str(pdf_path))
        try:
            image_bytes = render_figure_png(doc, figure)
        finally:
            doc.close()
        prompt = _vision_prompt_for_figure(paper, figure)
        summary = (summarizer or invoke_vision_image_summary)(image_bytes, prompt)
        return figure, _clean_summary(summary)

    pending = list(selected_figures)
    failures: list[tuple[FigureBlock, Exception]] = []

    first_pass = _run_vision_batch(pending, enrich_one, workers=_vision_worker_count(len(pending)))
    failures = _apply_batch_result(first_pass, result)

    retry_attempt = 0
    while failures and retry_attempt < _vision_rate_limit_retries():
        retryable = [(figure, exc) for figure, exc in failures if _is_rate_limit_error(exc)]
        permanent = [(figure, exc) for figure, exc in failures if not _is_rate_limit_error(exc)]
        failures = []
        for figure, exc in permanent:
            result.errors.append(f"page {figure.page + 1}: {exc}")

        if not retryable:
            break

        retry_attempt += 1
        time.sleep(_vision_retry_delay(retry_attempt))
        retry_figures = [figure for figure, _ in retryable]
        retry_batch = _run_vision_batch(
            retry_figures,
            enrich_one,
            workers=_vision_retry_worker_count(len(retry_figures)),
        )
        failures.extend(_apply_batch_result(retry_batch, result))

    for figure, exc in failures:
        result.errors.append(f"page {figure.page + 1}: {exc}")

    result.skipped += max(0, len(paper.figures) - result.attempted)
    result.skipped += len(result.errors)
    return result


def render_figure_png(
    doc: fitz.Document,
    figure: FigureBlock,
    *,
    dpi: int | None = None,
) -> bytes:
    """Render one figure region or full page to PNG bytes."""
    if figure.page < 0 or figure.page >= doc.page_count:
        raise ValueError(f"Figure page out of range: {figure.page + 1}")

    page = doc[figure.page]
    dpi = dpi or int(os.environ.get("VISION_RENDER_DPI", "144"))
    scale = max(72, dpi) / 72
    matrix = fitz.Matrix(scale, scale)

    clip = _expanded_clip(page, figure.bbox) if figure.bbox else None
    pixmap = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
    return pixmap.tobytes("png")


def _select_figures_for_vision(
    figures: list[FigureBlock],
    *,
    max_figures: int | None,
) -> list[FigureBlock]:
    scored = []
    for index, figure in enumerate(figures):
        caption = figure.caption.lower()
        score = 0
        if figure.bbox:
            score += 2
        if figure.caption:
            score += 2
        if any(term in caption for term in ("architecture", "framework", "model", "pipeline", "method")):
            score += 4
        if any(term in caption for term in ("result", "ablation", "performance", "comparison", "curve")):
            score += 3
        if any(term in caption for term in ("架构", "框架", "模型", "方法", "实验", "结果", "消融")):
            score += 3
        scored.append((score, index, figure))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = scored if max_figures is None else scored[:max_figures]
    return [figure for _, _, figure in selected]


def _vision_max_figures() -> int | None:
    """Return figure limit; 0 or less means all detected visual candidates."""
    value = int(os.environ.get("VISION_MAX_FIGURES", "0"))
    return None if value <= 0 else value


def _vision_worker_count(selected_count: int) -> int:
    """Return concurrency; 0 or less means one worker per selected figure."""
    configured = int(os.environ.get("VISION_MAX_WORKERS", "0"))
    if configured <= 0:
        return max(1, selected_count)
    return max(1, min(configured, selected_count))


def _vision_retry_worker_count(selected_count: int) -> int:
    configured = int(os.environ.get("VISION_RETRY_WORKERS", "1"))
    return max(1, min(configured, selected_count))


def _vision_rate_limit_retries() -> int:
    return max(0, int(os.environ.get("VISION_RATE_LIMIT_RETRIES", "3")))


def _vision_retry_delay(attempt: int) -> float:
    base = float(os.environ.get("VISION_RETRY_DELAY_SECONDS", "2.0"))
    return base * attempt


def _run_vision_batch(
    figures: list[FigureBlock],
    enrich_one: Callable[[FigureBlock], tuple[FigureBlock, str]],
    *,
    workers: int,
) -> list[tuple[FigureBlock, str | None, Exception | None]]:
    results: list[tuple[FigureBlock, str | None, Exception | None]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(figures)))) as executor:
        futures = {executor.submit(enrich_one, figure): figure for figure in figures}
        for future in as_completed(futures):
            original_figure = futures[future]
            try:
                figure, summary = future.result()
                results.append((figure, summary, None))
            except Exception as exc:  # noqa: BLE001 - return per-figure failure
                results.append((original_figure, None, exc))
    return results


def _apply_batch_result(
    batch: list[tuple[FigureBlock, str | None, Exception | None]],
    result: VisionEnrichmentResult,
) -> list[tuple[FigureBlock, Exception]]:
    failures: list[tuple[FigureBlock, Exception]] = []
    for figure, summary, exc in batch:
        if exc is not None:
            failures.append((figure, exc))
            continue
        figure.visual_summary = summary or ""
        if figure.visual_summary:
            result.enriched += 1
        else:
            result.skipped += 1
    result.attempted = max(result.attempted, len(batch))
    return failures


def _is_rate_limit_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in ("429", "1302", "rate limit", "速率限制"))


def _expanded_clip(
    page: fitz.Page,
    bbox: tuple[float, float, float, float] | None,
    *,
    margin: float = 36.0,
) -> fitz.Rect | None:
    if not bbox:
        return None
    rect = fitz.Rect(*bbox)
    rect.x0 = max(page.rect.x0, rect.x0 - margin)
    rect.y0 = max(page.rect.y0, rect.y0 - margin)
    rect.x1 = min(page.rect.x1, rect.x1 + margin)
    rect.y1 = min(page.rect.y1, rect.y1 + margin)
    if rect.is_empty or rect.width < 20 or rect.height < 20:
        return None
    return rect


def _vision_prompt_for_figure(paper: ParsedPaper, figure: FigureBlock) -> str:
    caption = figure.caption or "No explicit caption was extracted."
    return (
        "你是论文视觉内容分析助手。请阅读这张从 PDF 渲染出的论文页面或图像区域，"
        "用简体中文给出紧凑、忠实、可作为证据的视觉摘要。\n\n"
        f"论文标题：{paper.title}\n"
        f"页码：p.{figure.page + 1}\n"
        f"抽取到的图注：{caption}\n\n"
        "请重点识别：\n"
        "1. 这是模型架构图、流程图、实验曲线、柱状图、表格截图，还是其他视觉内容；\n"
        "2. 图中可读出的关键模块、箭头关系、坐标轴、图例、数值趋势或对比结论；\n"
        "3. 这些视觉信息如何支持方法、实验或批判性评审。\n\n"
        "要求：\n"
        "- 不要臆造看不清的数字或标签；看不清就明确说看不清。\n"
        "- 保留模型名、数据集名、指标名和关键数值的原文形式。\n"
        "- 避免操作性或敏感表述，使用中性的学术措辞。\n"
        "- 输出 3 到 6 条短句，不要 Markdown 表格，不要 JSON。"
    )


def _clean_summary(summary: str) -> str:
    summary = " ".join(str(summary or "").split())
    return summary[:1800]
