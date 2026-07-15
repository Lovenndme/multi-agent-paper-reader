"""Smoke tests for the PDF parser (no LLM calls required)."""

import textwrap
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz

from core.pdf_parser import (
    ParsedPaper,
    Section,
    _looks_like_font_header,
    _matches_header_pattern,
    _normalize_title,
    _extract_layout_content,
    parse_pdf,
)


class TestNormalizeTitle(unittest.TestCase):
    def test_strips_english_numbering(self):
        self.assertEqual(_normalize_title("1. Introduction"), "introduction")
        self.assertEqual(_normalize_title("3.2 Related Work"), "related work")

    def test_strips_chinese_numbering(self):
        self.assertEqual(_normalize_title("1. 引言"), "引言")
        self.assertEqual(_normalize_title("三、方法"), "方法")

    def test_lowercase(self):
        self.assertEqual(_normalize_title("Abstract"), "abstract")


class TestHeaderPattern(unittest.TestCase):
    def test_english_headers(self):
        self.assertTrue(_matches_header_pattern("Abstract"))
        self.assertTrue(_matches_header_pattern("1. Introduction"))
        self.assertTrue(_matches_header_pattern("2 Related Work"))
        self.assertTrue(_matches_header_pattern("5. References"))

    def test_chinese_headers(self):
        self.assertTrue(_matches_header_pattern("摘要"))
        self.assertTrue(_matches_header_pattern("1. 引言"))
        self.assertTrue(_matches_header_pattern("三、方法"))

    def test_rejects_body_text(self):
        self.assertFalse(_matches_header_pattern(
            "This paper proposes a novel framework for fatigue detection."
        ))
        self.assertFalse(_matches_header_pattern(""))
        # Long lines should not match
        self.assertFalse(_matches_header_pattern("A" * 61))


class TestFontHeaderHeuristic(unittest.TestCase):
    def test_accepts_short_real_section_titles(self):
        self.assertTrue(_looks_like_font_header("Model"))
        self.assertTrue(_looks_like_font_header("EEG Matching"))
        self.assertTrue(_looks_like_font_header("2.1 EEG Acquisition"))

    def test_rejects_title_fragments_and_equations(self):
        self.assertFalse(_looks_like_font_header("sciences"))
        self.assertFalse(_looks_like_font_header("(ESML): A Deep Learning Framework on"))
        self.assertFalse(_looks_like_font_header("="))
        self.assertFalse(_looks_like_font_header("mn) �→∑"))


class TestPaperTitleInference(unittest.TestCase):
    def test_uses_prominent_first_page_title_when_metadata_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "renamed-local-file.pdf"
            document = fitz.open()
            page = document.new_page()
            page.insert_text((72, 100), "A Reliable Paper Title", fontsize=20)
            page.insert_text((72, 140), "First Author, Second Author", fontsize=10)
            page.insert_text((72, 200), "Abstract", fontsize=13)
            page.insert_text((72, 225), "This paper contains enough body text for parsing.", fontsize=10)
            document.save(path)
            document.close()

            parsed = parse_pdf(path)

        self.assertEqual(parsed.title, "A Reliable Paper Title")


class TestLayoutExtraction(unittest.TestCase):
    def test_extracts_vector_picture_bbox_and_excludes_internal_labels(self):
        payload = {
            "pages": [
                {
                    "page_number": 1,
                    "width": 600,
                    "height": 800,
                    "boxes": [
                        {
                            "x0": 100,
                            "y0": 80,
                            "x1": 500,
                            "y1": 330,
                            "boxclass": "picture",
                            "textlines": [
                                {"spans": [{"text": "Transformer Block", "bbox": [120, 100, 230, 112], "size": 8}]}
                            ],
                        },
                        {
                            "x0": 100,
                            "y0": 340,
                            "x1": 500,
                            "y1": 370,
                            "boxclass": "text",
                            "textlines": [
                                {"spans": [{"text": "Figure 2 | Vector architecture.", "bbox": [100, 340, 320, 352], "size": 10}]}
                            ],
                        },
                        {
                            "x0": 70,
                            "y0": 390,
                            "x1": 530,
                            "y1": 410,
                            "boxclass": "section-header",
                            "textlines": [
                                {"spans": [{"text": "2. Method", "bbox": [70, 390, 145, 402], "size": 14}]}
                            ],
                        },
                        {
                            "x0": 70,
                            "y0": 420,
                            "x1": 530,
                            "y1": 470,
                            "boxclass": "text",
                            "textlines": [
                                {"spans": [{"text": "The body explains the architecture.", "bbox": [70, 420, 300, 432], "size": 10}]}
                            ],
                        },
                    ],
                }
            ]
        }
        with patch("core.pdf_parser.pymupdf4llm.to_json", return_value=json.dumps(payload)):
            pages, sections, tables, figures = _extract_layout_content(Path("vector.pdf"))

        self.assertEqual(tables, [])
        self.assertEqual(figures[0].bbox, (100.0, 80.0, 500.0, 330.0))
        self.assertIn("Figure 2", figures[0].caption)
        self.assertNotIn("Transformer Block", pages[0][1])
        self.assertNotIn("Figure 2", pages[0][1])
        self.assertIn("body explains", sections[0].content)


class TestParsedPaperSectionRouting(unittest.TestCase):
    def _make_paper(self):
        return ParsedPaper(
            title="Test Paper",
            full_text="full text",
            sections=[
                Section("Abstract", "abstract content", 0, 0),
                Section("1. Introduction", "intro content", 1, 1),
                Section("3. Experiments", "experiment content", 3, 4),
                Section("4. Conclusion", "conclusion content", 5, 5),
            ],
        )

    def test_method_agent_gets_relevant_sections(self):
        paper = self._make_paper()
        text = paper.get_sections_for_agent("method")
        self.assertIn("abstract content", text)
        self.assertIn("intro content", text)
        # Experiments not in method sections
        self.assertNotIn("experiment content", text)

    def test_experiment_agent_gets_relevant_sections(self):
        paper = self._make_paper()
        text = paper.get_sections_for_agent("experiment")
        self.assertIn("experiment content", text)
        self.assertIn("abstract content", text)

    def test_critic_agent_gets_relevant_sections(self):
        paper = self._make_paper()
        text = paper.get_sections_for_agent("critic")
        self.assertIn("conclusion content", text)

    def test_fallback_to_full_text_when_no_match(self):
        paper = ParsedPaper(
            title="T",
            full_text="the full paper text",
            sections=[Section("Unrecognized Section", "some content", 0, 0)],
        )
        self.assertEqual(paper.get_sections_for_agent("method"), "the full paper text")


class TestCLIArgContract(unittest.TestCase):
    """Ensure the CLI positional argument 'pdf' is present (not --pdf)."""

    def test_positional_pdf_arg(self):
        import argparse
        import sys
        # Re-import parse_args to test the contract
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from main import parse_args

        args = parse_args.__wrapped__() if hasattr(parse_args, "__wrapped__") else None
        # Just verify the parser accepts a positional argument, not --pdf
        import io
        import contextlib

        parser_output = io.StringIO()
        with contextlib.suppress(SystemExit):
            with contextlib.redirect_stdout(parser_output):
                # If --pdf were the interface this would fail
                import importlib
                import main as main_mod
                # Check the source uses positional, not --pdf
                src = Path(main_mod.__file__).read_text(encoding="utf-8")
                self.assertIn('parser.add_argument("pdf"', src)
                self.assertNotIn('add_argument("--pdf"', src)


if __name__ == "__main__":
    unittest.main()
