"""Tests for best-effort PDF vision enrichment."""

import tempfile
import os
import time
import unittest
from pathlib import Path

import fitz

from core.evidence import build_evidence_index, evidence_context_for_agent
from core.pdf_parser import FigureBlock, ParsedPaper, Section
from core.vision import enrich_paper_figures_with_vision, render_figure_png


class TestVisionEnrichment(unittest.TestCase):
    def _pdf(self) -> Path:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "vision-test.pdf"
        doc = fitz.open()
        page = doc.new_page(width=420, height=300)
        page.insert_text((40, 34), "Figure 1: Model architecture overview.", fontsize=12)
        page.draw_rect(fitz.Rect(60, 70, 360, 240), color=(0.1, 0.4, 0.9), width=2)
        page.insert_text((90, 120), "Retriever -> Generator", fontsize=14)
        doc.save(path)
        doc.close()
        return path

    def test_renders_figure_region_to_png(self):
        path = self._pdf()
        doc = fitz.open(path)
        try:
            image = render_figure_png(
                doc,
                FigureBlock(page=0, caption="Figure 1: Model architecture overview.", bbox=(60, 70, 360, 240)),
                dpi=96,
            )
        finally:
            doc.close()

        self.assertTrue(image.startswith(b"\x89PNG"))
        self.assertGreater(len(image), 1000)

    def test_enriches_figure_and_evidence_context(self):
        path = self._pdf()
        paper = ParsedPaper(
            title="Vision Test",
            full_text="The paper introduces a retriever-generator model.",
            sections=[Section("Method", "The architecture uses retrieval and generation.", 0, 0)],
            figures=[
                FigureBlock(
                    page=0,
                    caption="Figure 1: Model architecture overview.",
                    image_index=1,
                    bbox=(60, 70, 360, 240),
                )
            ],
        )

        result = enrich_paper_figures_with_vision(
            path,
            paper,
            summarizer=lambda image_bytes, prompt: "图中展示 Retriever 到 Generator 的模型流程。",
        )
        snippets = build_evidence_index(paper)
        method_context = evidence_context_for_agent(snippets, "method")

        self.assertEqual(result.enriched, 1)
        self.assertIn("图中展示 Retriever 到 Generator", paper.figures[0].visual_summary)
        self.assertIn("[F001 | figure", method_context)
        self.assertIn("Vision summary", method_context)

    def test_enriches_figures_concurrently_by_default(self):
        old_max_figures = os.environ.get("VISION_MAX_FIGURES")
        old_max_workers = os.environ.get("VISION_MAX_WORKERS")
        os.environ["VISION_MAX_FIGURES"] = "0"
        os.environ["VISION_MAX_WORKERS"] = "0"
        self.addCleanup(lambda: _restore_env("VISION_MAX_FIGURES", old_max_figures))
        self.addCleanup(lambda: _restore_env("VISION_MAX_WORKERS", old_max_workers))
        path = self._pdf()
        paper = ParsedPaper(
            title="Vision Concurrency Test",
            full_text="The paper has several visual regions.",
            sections=[Section("Method", "The architecture uses multiple visual components.", 0, 0)],
            figures=[
                FigureBlock(
                    page=0,
                    caption=f"Figure {index}: Architecture part {index}.",
                    image_index=index,
                    bbox=(60, 70, 360, 240),
                )
                for index in range(1, 4)
            ],
        )

        starts = []

        def slow_summarizer(image_bytes, prompt):
            starts.append(time.perf_counter())
            time.sleep(0.25)
            return "并发视觉摘要"

        started = time.perf_counter()
        result = enrich_paper_figures_with_vision(path, paper, summarizer=slow_summarizer)
        elapsed = time.perf_counter() - started

        self.assertEqual(result.attempted, 3)
        self.assertEqual(result.enriched, 3)
        self.assertLess(elapsed, 0.6)
        self.assertLess(max(starts) - min(starts), 0.25)

    def test_retries_rate_limited_figures_with_smaller_worker_pool(self):
        old_values = {
            name: os.environ.get(name)
            for name in (
                "VISION_MAX_FIGURES",
                "VISION_MAX_WORKERS",
                "VISION_RATE_LIMIT_RETRIES",
                "VISION_RETRY_WORKERS",
                "VISION_RETRY_DELAY_SECONDS",
            )
        }
        os.environ["VISION_MAX_FIGURES"] = "0"
        os.environ["VISION_MAX_WORKERS"] = "0"
        os.environ["VISION_RATE_LIMIT_RETRIES"] = "2"
        os.environ["VISION_RETRY_WORKERS"] = "1"
        os.environ["VISION_RETRY_DELAY_SECONDS"] = "0"
        for name, value in old_values.items():
            self.addCleanup(lambda env_name=name, env_value=value: _restore_env(env_name, env_value))

        path = self._pdf()
        paper = ParsedPaper(
            title="Vision Retry Test",
            full_text="The paper has several visual regions.",
            sections=[Section("Method", "The architecture uses multiple visual components.", 0, 0)],
            figures=[
                FigureBlock(
                    page=0,
                    caption=f"Figure {index}: Architecture part {index}.",
                    image_index=index,
                    bbox=(60, 70, 360, 240),
                )
                for index in range(1, 4)
            ],
        )

        calls = {"count": 0}

        def flaky_summarizer(image_bytes, prompt):
            calls["count"] += 1
            if calls["count"] <= 3:
                raise RuntimeError("Error code: 429 - rate limit 1302")
            return "限流后重试成功"

        result = enrich_paper_figures_with_vision(path, paper, summarizer=flaky_summarizer)

        self.assertEqual(result.attempted, 3)
        self.assertEqual(result.enriched, 3)
        self.assertEqual(result.errors, [])
        self.assertTrue(all(figure.visual_summary for figure in paper.figures))

    def test_refuses_unverified_full_page_fallback(self):
        path = self._pdf()
        doc = fitz.open(path)
        try:
            with self.assertRaisesRegex(ValueError, "no verified layout bounding box"):
                render_figure_png(doc, FigureBlock(page=0, caption="Figure 1"))
        finally:
            doc.close()


def _restore_env(name, value):
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
