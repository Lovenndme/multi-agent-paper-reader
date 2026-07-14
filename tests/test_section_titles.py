"""Tests for preserving parsed paper section titles."""

import unittest

from core.section_titles import clean_section_title


class TestSectionTitles(unittest.TestCase):
    def test_preserves_english_title(self):
        self.assertEqual(
            clean_section_title("Harness Engineering Foundations", 0),
            "Harness Engineering Foundations",
        )

    def test_preserves_chinese_title(self):
        self.assertEqual(clean_section_title("研究背景", 0), "研究背景")

    def test_preserves_original_numbering(self):
        self.assertEqual(
            clean_section_title("3.2 Multi-Head Attention", 0),
            "3.2 Multi-Head Attention",
        )

    def test_compacts_parser_whitespace_without_translating(self):
        self.assertEqual(
            clean_section_title("  Related   Work\nAnd Background  ", 1),
            "Related Work And Background",
        )

    def test_uses_numbered_fallback_for_noisy_title(self):
        self.assertEqual(clean_section_title("�", 2), "章节 3")


if __name__ == "__main__":
    unittest.main()
