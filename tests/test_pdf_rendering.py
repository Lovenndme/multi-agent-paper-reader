"""Native-density and bounded-preview tests for PDF visual rendering."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import fitz

from core.pdf_parser import FigureBlock, TableBlock
from core.pdf_rendering import (
    RenderLimitExceeded,
    RenderLimits,
    _table_clip,
    plan_figure_render,
    plan_figure_preview_render,
    plan_table_render,
    render_figure_native_png,
    render_figure_preview_png,
    render_table_native_png,
)


class TestPDFRenderingQuality(unittest.TestCase):
    def _embedded_pdf(self, *, width: int = 1201, height: int = 601) -> Path:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "embedded.pdf"
        document = fitz.open()
        page = document.new_page(width=420, height=320)
        pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, width, height), False)
        pixmap.clear_with(245)
        page.insert_image(fitz.Rect(100, 80, 300, 180), pixmap=pixmap)
        page.insert_text((100, 205), "Figure 1: Native density.", fontsize=10)
        document.save(path)
        document.close()
        return path

    def _vector_pdf(self) -> Path:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "vector.pdf"
        document = fitz.open()
        page = document.new_page(width=420, height=320)
        page.draw_rect(fitz.Rect(60, 90, 360, 230), color=(0.1, 0.3, 0.8), width=1)
        page.insert_text((80, 130), "Vector figure", fontsize=12)
        page.insert_text((60, 70), "Table 1: Compact caption.", fontsize=9)
        document.save(path)
        document.close()
        return path

    def test_figure_uses_embedded_source_density_even_with_lower_requested_dpi(self):
        document = fitz.open(self._embedded_pdf())
        try:
            figure = FigureBlock(
                page=0,
                bbox=(100, 80, 300, 180),
                caption="Figure 1: Native density.",
                caption_bbox=(100, 194, 245, 208),
            )
            plan = plan_figure_render(document[0], figure)
            image = render_figure_native_png(document, figure, dpi=96)
            pixmap = fitz.Pixmap(image)
        finally:
            document.close()

        self.assertEqual(plan.dpi, 433)
        self.assertEqual(plan.policy, "native-raster")
        self.assertEqual(pixmap.width, plan.width_px)
        self.assertEqual(pixmap.height, plan.height_px)
        self.assertGreaterEqual(round(200 * plan.dpi / 72), 1201)

    def test_vector_figure_and_table_use_600_dpi_quality_floor(self):
        document = fitz.open(self._vector_pdf())
        try:
            figure = FigureBlock(page=0, bbox=(60, 90, 360, 230))
            table = TableBlock(
                page=0,
                rows=[["Metric", "Score"], ["Accuracy", "0.95"]],
                caption="Table 1: Compact caption.",
                bbox=(60, 90, 360, 230),
                caption_bbox=(60, 60, 185, 72),
            )
            figure_plan = plan_figure_render(document[0], figure)
            table_plan = plan_table_render(document[0], table)
            table_png = render_table_native_png(document, table, dpi=96)
            table_pixmap = fitz.Pixmap(table_png)
        finally:
            document.close()

        self.assertEqual(figure_plan.dpi, 600)
        self.assertEqual(figure_plan.policy, "vector-600dpi")
        self.assertEqual(table_plan.dpi, 600)
        self.assertEqual(table_plan.policy, "vector-600dpi")
        self.assertEqual(table_pixmap.width, table_plan.width_px)
        self.assertEqual(table_pixmap.height, table_plan.height_px)

    def test_model_preview_remains_capped_at_144_dpi(self):
        document = fitz.open(self._embedded_pdf())
        try:
            figure = FigureBlock(page=0, bbox=(100, 80, 300, 180))
            native = fitz.Pixmap(render_figure_native_png(document, figure))
            preview = fitz.Pixmap(render_figure_preview_png(document, figure, dpi=999))
            low_plan = plan_figure_preview_render(document[0], figure, dpi=72)
            low_preview = fitz.Pixmap(render_figure_preview_png(document, figure, dpi=72))
        finally:
            document.close()

        self.assertGreater(native.width, preview.width)
        # The no-caption safety margin expands the 200 pt region by 8 pt on
        # both sides, so a 144 DPI preview is exactly 216 * 2 pixels wide.
        self.assertEqual(preview.width, 432)
        self.assertEqual(low_plan.dpi, 120)
        self.assertEqual(low_preview.width, low_plan.width_px)

    def test_multiple_images_choose_highest_effective_dpi(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "multiple.pdf"
        document = fitz.open()
        page = document.new_page(width=420, height=320)
        low = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 300, 150), False)
        high = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 1000, 500), False)
        low.clear_with(230)
        high.clear_with(210)
        page.insert_image(fitz.Rect(60, 80, 160, 130), pixmap=low)
        page.insert_image(fitz.Rect(200, 80, 300, 130), pixmap=high)
        document.save(path)
        document.close()

        document = fitz.open(path)
        try:
            plan = plan_figure_render(
                document[0],
                FigureBlock(page=0, bbox=(50, 70, 310, 140)),
            )
        finally:
            document.close()

        self.assertEqual(plan.dpi, 720)

    def test_short_single_line_table_caption_is_not_dropped(self):
        document = fitz.open(self._vector_pdf())
        try:
            table = TableBlock(
                page=0,
                rows=[["Metric", "Score"]],
                caption="Table 1: Compact caption.",
                bbox=(60, 90, 360, 230),
                caption_bbox=(60, 60, 185, 69),
            )
            clip = _table_clip(document[0], table)
        finally:
            document.close()

        self.assertLessEqual(clip.y0, 56)

    def test_native_plan_refuses_limits_without_lowering_dpi(self):
        document = fitz.open(self._embedded_pdf())
        try:
            figure = FigureBlock(page=0, bbox=(100, 80, 300, 180))
            with self.assertRaisesRegex(
                RenderLimitExceeded,
                "433 DPI.*avoid downsampling.*Render refused",
            ):
                plan_figure_render(
                    document[0],
                    figure,
                    limits=RenderLimits(max_pixels=10_000, max_dimension=200),
                )
        finally:
            document.close()


if __name__ == "__main__":
    unittest.main()
