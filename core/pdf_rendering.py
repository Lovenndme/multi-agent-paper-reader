"""Bounded PDF rendering helpers without model or environment side effects."""

from __future__ import annotations

import math
from dataclasses import dataclass

import fitz

from core.pdf_parser import FigureBlock, TableBlock


MAX_RENDER_PIXELS = 16_000_000
MAX_RENDER_DIMENSION = 8_192
PREVIEW_MAX_DPI = 144
MODEL_PREVIEW_MIN_DPI = 120
PAGE_PREVIEW_MIN_DPI = 96
FIGURE_MIN_DPI = 144
VECTOR_RENDER_DPI = 600
TABLE_RENDER_DPI = 600


class RenderLimitExceeded(ValueError):
    """Raised when lossless-quality rendering would exceed the safety envelope."""


class UnsupportedImageGeometry(ValueError):
    """Raised when an overlapping PDF image has unusable placement geometry."""


@dataclass(frozen=True)
class RenderLimits:
    """Pre-allocation limits shared by preview and native region rendering."""

    max_pixels: int = MAX_RENDER_PIXELS
    max_dimension: int = MAX_RENDER_DIMENSION


@dataclass(frozen=True)
class RenderPlan:
    """An exact render request whose DPI must never be reduced implicitly."""

    clip: tuple[float, float, float, float]
    dpi: int
    width_px: int
    height_px: int
    policy: str


DEFAULT_RENDER_LIMITS = RenderLimits()


def plan_figure_render(
    page: fitz.Page,
    figure: FigureBlock,
    *,
    minimum_dpi: int = FIGURE_MIN_DPI,
    vector_dpi: int = VECTOR_RENDER_DPI,
    limits: RenderLimits = DEFAULT_RENDER_LIMITS,
) -> RenderPlan:
    """Plan a figure crop without downsampling any overlapping source image."""
    if figure.bbox is None:
        raise ValueError("Figure has no verified layout bounding box; refusing full-page fallback.")
    source_dpis = _overlapping_image_dpis(page, fitz.Rect(*figure.bbox))
    minimum = _validated_dpi(minimum_dpi)
    if source_dpis:
        target_dpi = max(minimum, math.ceil(max(source_dpis)))
        policy = "native-raster"
    else:
        target_dpi = max(minimum, _validated_dpi(vector_dpi))
        policy = f"vector-{target_dpi}dpi"
    return _build_plan(
        _figure_clip(page, figure),
        dpi=target_dpi,
        policy=policy,
        limits=limits,
        label="Figure",
    )


def plan_table_render(
    page: fitz.Page,
    table: TableBlock,
    *,
    minimum_dpi: int = TABLE_RENDER_DPI,
    limits: RenderLimits = DEFAULT_RENDER_LIMITS,
) -> RenderPlan:
    """Plan a 600-DPI vector table crop without downsampling embedded images."""
    if table.bbox is None:
        raise ValueError("Table has no verified layout bounding box; refusing full-page fallback.")
    source_dpis = _overlapping_image_dpis(page, fitz.Rect(*table.bbox))
    target_dpi = max(
        _validated_dpi(minimum_dpi),
        math.ceil(max(source_dpis)) if source_dpis else 0,
    )
    return _build_plan(
        _table_clip(page, table),
        dpi=target_dpi,
        policy="native-raster-table" if source_dpis else f"vector-{target_dpi}dpi",
        limits=limits,
        label="Table",
    )


def render_figure_native_png(
    doc: fitz.Document,
    figure: FigureBlock,
    *,
    dpi: int | None = None,
    limits: RenderLimits = DEFAULT_RENDER_LIMITS,
) -> bytes:
    """Render a verified figure at native source density or vector quality.

    ``dpi`` is a minimum, never an override that may reduce source resolution.
    """
    if figure.page < 0 or figure.page >= doc.page_count:
        raise ValueError(f"Figure page out of range: {figure.page + 1}")
    page = doc[figure.page]
    plan = plan_figure_render(
        page,
        figure,
        minimum_dpi=max(FIGURE_MIN_DPI, _validated_dpi(dpi)) if dpi is not None else FIGURE_MIN_DPI,
        limits=limits,
    )
    return render_planned_png(page, plan)


def render_table_native_png(
    doc: fitz.Document,
    table: TableBlock,
    *,
    dpi: int | None = None,
    limits: RenderLimits = DEFAULT_RENDER_LIMITS,
) -> bytes:
    """Render a verified table at 600 DPI or higher without downsampling."""
    if table.page < 0 or table.page >= doc.page_count:
        raise ValueError(f"Table page out of range: {table.page + 1}")
    page = doc[table.page]
    plan = plan_table_render(
        page,
        table,
        minimum_dpi=max(TABLE_RENDER_DPI, _validated_dpi(dpi)) if dpi is not None else TABLE_RENDER_DPI,
        limits=limits,
    )
    return render_planned_png(page, plan)


def render_figure_png(
    doc: fitz.Document,
    figure: FigureBlock,
    *,
    dpi: int | None = None,
    limits: RenderLimits = DEFAULT_RENDER_LIMITS,
) -> bytes:
    """Backward-compatible alias for native/export-quality figure rendering."""
    return render_figure_native_png(doc, figure, dpi=dpi, limits=limits)


def render_table_png(
    doc: fitz.Document,
    table: TableBlock,
    *,
    dpi: int | None = None,
    limits: RenderLimits = DEFAULT_RENDER_LIMITS,
) -> bytes:
    """Backward-compatible alias for native/export-quality table rendering."""
    return render_table_native_png(doc, table, dpi=dpi, limits=limits)


def render_figure_preview_png(
    doc: fitz.Document,
    figure: FigureBlock,
    *,
    dpi: int = PREVIEW_MAX_DPI,
    limits: RenderLimits = DEFAULT_RENDER_LIMITS,
) -> bytes:
    """Render a deliberately bounded model preview of one verified figure."""
    if figure.page < 0 or figure.page >= doc.page_count:
        raise ValueError(f"Figure page out of range: {figure.page + 1}")
    page = doc[figure.page]
    plan = plan_figure_preview_render(page, figure, dpi=dpi, limits=limits)
    return render_planned_png(page, plan)


def render_table_preview_png(
    doc: fitz.Document,
    table: TableBlock,
    *,
    dpi: int = PREVIEW_MAX_DPI,
    limits: RenderLimits = DEFAULT_RENDER_LIMITS,
) -> bytes:
    """Render a deliberately bounded model preview of one verified table."""
    if table.page < 0 or table.page >= doc.page_count:
        raise ValueError(f"Table page out of range: {table.page + 1}")
    page = doc[table.page]
    plan = plan_table_preview_render(page, table, dpi=dpi, limits=limits)
    return render_planned_png(page, plan)


def plan_figure_preview_render(
    page: fitz.Page,
    figure: FigureBlock,
    *,
    dpi: int = PREVIEW_MAX_DPI,
    limits: RenderLimits = DEFAULT_RENDER_LIMITS,
) -> RenderPlan:
    """Plan a bounded 120-144 DPI model preview for one verified figure."""
    if figure.bbox is None:
        raise ValueError("Figure has no verified layout bounding box; refusing full-page fallback.")
    plan = _build_plan(
        _figure_clip(page, figure),
        dpi=_preview_dpi(dpi, minimum=MODEL_PREVIEW_MIN_DPI),
        policy="model-preview",
        limits=limits,
        label="Figure preview",
    )
    return plan


def plan_table_preview_render(
    page: fitz.Page,
    table: TableBlock,
    *,
    dpi: int = PREVIEW_MAX_DPI,
    limits: RenderLimits = DEFAULT_RENDER_LIMITS,
) -> RenderPlan:
    """Plan a bounded 120-144 DPI model preview for one verified table."""
    if table.bbox is None:
        raise ValueError("Table has no verified layout bounding box; refusing full-page fallback.")
    return _build_plan(
        _table_clip(page, table),
        dpi=_preview_dpi(dpi, minimum=MODEL_PREVIEW_MIN_DPI),
        policy="model-preview",
        limits=limits,
        label="Table preview",
    )


def render_page_preview_png(
    doc: fitz.Document,
    page_index: int,
    *,
    dpi: int = 120,
    limits: RenderLimits = DEFAULT_RENDER_LIMITS,
) -> bytes:
    """Render exactly one complete PDF page as a bounded 96-144 DPI preview."""
    if page_index < 0 or page_index >= doc.page_count:
        raise ValueError(f"Page out of range: {page_index + 1}")
    page = doc[page_index]
    plan = _build_plan(
        page.rect,
        dpi=_preview_dpi(dpi, minimum=PAGE_PREVIEW_MIN_DPI),
        policy="page-preview",
        limits=limits,
        label="Page preview",
    )
    return render_planned_png(page, plan)


def render_page_png(doc: fitz.Document, page_index: int, *, dpi: int = 120) -> bytes:
    """Backward-compatible alias for bounded full-page preview rendering."""
    return render_page_preview_png(doc, page_index, dpi=dpi)


def _build_plan(
    clip: fitz.Rect,
    *,
    dpi: int,
    policy: str,
    limits: RenderLimits,
    label: str,
) -> RenderPlan:
    render_dpi = _validated_dpi(dpi)
    matrix = fitz.Matrix(render_dpi / 72, render_dpi / 72)
    pixel_rect = (fitz.Rect(clip) * matrix).irect
    width = max(1, pixel_rect.width)
    height = max(1, pixel_rect.height)
    if (
        width > limits.max_dimension
        or height > limits.max_dimension
        or width * height > limits.max_pixels
    ):
        raise RenderLimitExceeded(
            f"{label} requires {render_dpi} DPI and {width}x{height} pixels to avoid "
            f"downsampling; limit is {limits.max_dimension} per dimension / "
            f"{limits.max_pixels} total pixels. Render refused."
        )
    return RenderPlan(
        clip=tuple(float(value) for value in clip),
        dpi=render_dpi,
        width_px=width,
        height_px=height,
        policy=policy,
    )


def render_planned_png(page: fitz.Page, plan: RenderPlan) -> bytes:
    """Execute an already validated plan without changing its DPI."""
    scale = plan.dpi / 72
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        clip=fitz.Rect(*plan.clip),
        alpha=False,
    )
    return pixmap.tobytes("png")


def _overlapping_image_dpis(page: fitz.Page, visual_bbox: fitz.Rect) -> list[float]:
    """Return effective DPI for every displayed image intersecting a raw visual bbox."""
    dpis: list[float] = []
    for info in page.get_image_info(xrefs=False):
        try:
            image_bbox = fitz.Rect(info["bbox"])
        except (KeyError, TypeError, ValueError) as exc:
            raise UnsupportedImageGeometry("PDF image has an invalid bounding box.") from exc
        if image_bbox.is_empty or (image_bbox & visual_bbox).is_empty:
            continue
        try:
            matrix = fitz.Matrix(info["transform"])
            width_px = int(info["width"])
            height_px = int(info["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise UnsupportedImageGeometry("PDF image has invalid placement metadata.") from exc
        display_width = math.hypot(matrix.a, matrix.b)
        display_height = math.hypot(matrix.c, matrix.d)
        values = (display_width, display_height, float(width_px), float(height_px))
        if (
            width_px <= 0
            or height_px <= 0
            or not all(math.isfinite(value) and value > 0 for value in values)
        ):
            raise UnsupportedImageGeometry("PDF image has unsupported placement geometry.")
        dpis.append(max(72 * width_px / display_width, 72 * height_px / display_height))
    return dpis


def _validated_dpi(value: int | float) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Render DPI must be a finite number.") from exc
    if not math.isfinite(numeric) or numeric < 72:
        raise ValueError("Render DPI must be at least 72.")
    return math.ceil(numeric)


def _preview_dpi(value: int | float, *, minimum: int) -> int:
    return max(minimum, min(_validated_dpi(value), PREVIEW_MAX_DPI))


def _expanded_clip(
    page: fitz.Page,
    bbox: tuple[float, float, float, float],
    *,
    margin: float,
    min_width: float = 20,
    min_height: float = 20,
) -> fitz.Rect | None:
    rect = fitz.Rect(*bbox)
    rect.x0 = max(page.rect.x0, rect.x0 - margin)
    rect.y0 = max(page.rect.y0, rect.y0 - margin)
    rect.x1 = min(page.rect.x1, rect.x1 + margin)
    rect.y1 = min(page.rect.y1, rect.y1 + margin)
    if rect.is_empty or rect.width < min_width or rect.height < min_height:
        return None
    return rect


def _figure_clip(page: fitz.Page, figure: FigureBlock) -> fitz.Rect:
    """Build a clean crop containing the complete visual and matched caption."""
    if figure.bbox is None:
        raise ValueError("Figure has no verified layout bounding box; refusing full-page fallback.")
    picture_margin = 4.0 if figure.caption_bbox is not None else 8.0
    clip = _expanded_clip(page, figure.bbox, margin=picture_margin)
    if clip is None:
        raise ValueError("Figure layout bounding box is too small to render.")
    if figure.caption_bbox is not None:
        caption_clip = _expanded_clip(
            page,
            figure.caption_bbox,
            margin=4.0,
            min_height=2,
        )
        if caption_clip is not None:
            clip.include_rect(caption_clip)
    return clip


def _table_clip(page: fitz.Page, table: TableBlock) -> fitz.Rect:
    """Build a tight table crop that includes the complete matched caption."""
    if table.bbox is None:
        raise ValueError("Table has no verified layout bounding box; refusing full-page fallback.")
    clip = _expanded_clip(page, table.bbox, margin=4.0)
    if clip is None:
        raise ValueError("Table layout bounding box is too small to render.")
    if table.caption_bbox is not None:
        caption_clip = _expanded_clip(
            page,
            table.caption_bbox,
            margin=4.0,
            min_height=2,
        )
        if caption_clip is not None:
            clip.include_rect(caption_clip)
    return clip
