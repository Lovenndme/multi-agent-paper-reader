"""Smoke tests for the PDF parser (no LLM calls required)."""

import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.pdf_parser import (
    ParsedPaper,
    Section,
    _looks_like_font_header,
    _matches_header_pattern,
    _normalize_title,
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
