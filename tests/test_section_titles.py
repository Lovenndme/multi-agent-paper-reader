"""Tests for consistent Chinese section-title display."""

import unittest

from unittest.mock import patch

from core.section_titles import (
    SectionTitleTranslation,
    SectionTitleTranslationBatch,
    clean_section_title,
    section_titles_needing_translation,
    translate_section_titles,
)


class TestSectionTitleTranslation(unittest.TestCase):
    def test_translates_transformer_section_titles(self):
        expected = {
            "Model Architecture": "模型架构",
            "Encoder and Decoder Stacks": "编码器与解码器堆栈",
            "Attention": "注意力机制",
            "Scaled Dot-Product Attention": "缩放点积注意力",
            "Multi-Head Attention": "多头注意力",
            "Applications of Attention in our Model": "注意力在模型中的应用",
            "Position-wise Feed-Forward Networks": "逐位置前馈网络",
            "Embeddings and Softmax": "词嵌入与 Softmax",
            "Positional Encoding": "位置编码",
            "Why Self-Attention": "为什么使用自注意力",
        }

        for index, (source, translated) in enumerate(expected.items()):
            with self.subTest(source=source):
                self.assertEqual(clean_section_title(source, index), translated)

    def test_strips_numbering_before_translation(self):
        self.assertEqual(
            clean_section_title("3.2 Multi-Head Attention", 0),
            "多头注意力",
        )

    def test_preserves_existing_chinese_title(self):
        self.assertEqual(clean_section_title("研究背景", 0), "研究背景")

    def test_live_fallback_never_leaves_an_unknown_english_title(self):
        self.assertEqual(
            clean_section_title("Unknown Bespoke Pipeline", 2, {}),
            "章节 3",
        )

    def test_batch_translates_unknown_custom_titles(self):
        titles = [
            "A Curious Custom Mechanism",
            "方法 Method Overview",
            "Introduction",
            "实验设置",
        ]
        self.assertEqual(
            section_titles_needing_translation(titles),
            ["A Curious Custom Mechanism", "方法 Method Overview"],
        )
        response = SectionTitleTranslationBatch(
            translations=[
                SectionTitleTranslation(
                    source="A Curious Custom Mechanism",
                    translated="一种新颖的自定义机制",
                ),
                SectionTitleTranslation(
                    source="方法 Method Overview",
                    translated="方法概述",
                ),
            ]
        )
        with (
            patch("core.section_titles.get_llm", return_value=object()),
            patch("core.section_titles.invoke_with_retry", return_value=response),
        ):
            translations = translate_section_titles(titles)

        self.assertEqual(
            clean_section_title("A Curious Custom Mechanism", 0, translations),
            "一种新颖的自定义机制",
        )


if __name__ == "__main__":
    unittest.main()
