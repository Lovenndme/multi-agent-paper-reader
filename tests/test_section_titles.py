"""Tests for consistent Chinese section-title display."""

import unittest

from app import _clean_section_title


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
                self.assertEqual(_clean_section_title(source, index), translated)

    def test_strips_numbering_before_translation(self):
        self.assertEqual(
            _clean_section_title("3.2 Multi-Head Attention", 0),
            "多头注意力",
        )

    def test_preserves_existing_chinese_title(self):
        self.assertEqual(_clean_section_title("研究背景", 0), "研究背景")


if __name__ == "__main__":
    unittest.main()
