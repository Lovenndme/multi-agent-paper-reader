"""Tests for provider-free multilingual semantic ranking."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from core.semantic_search import (
    clear_semantic_cache,
    cross_encoder_scores,
    semantic_scores,
)


class _FakeEmbeddingModel:
    def embed(self, texts, *, batch_size=256):
        assert 1 <= batch_size <= 256
        for text in texts:
            lowered = text.lower()
            if "压缩注意力" in text or "compressed attention" in lowered:
                yield np.array([1.0, 0.0, 0.0], dtype=np.float32)
            elif "unrelated" in lowered or "天气" in text:
                yield np.array([0.0, 1.0, 0.0], dtype=np.float32)
            else:
                yield np.array([0.2, 0.2, 1.0], dtype=np.float32)


class TestSemanticSearch(unittest.TestCase):
    def tearDown(self):
        clear_semantic_cache()

    def test_cross_language_semantics_rank_relevant_document_first(self):
        with patch.dict(os.environ, {"PAPER_READER_DISABLE_EMBEDDINGS": "0"}), patch(
            "core.semantic_search._embedding_model",
            return_value=_FakeEmbeddingModel(),
        ):
            scores = semantic_scores(
                "什么是重度压缩注意力？",
                [
                    "Heavily Compressed Attention reduces the KV cache.",
                    "An unrelated paragraph about the weather.",
                ],
            )

        self.assertIsNotNone(scores)
        self.assertGreater(scores[0], scores[1])

    def test_explicit_disable_returns_fallback_signal(self):
        with patch.dict(os.environ, {"PAPER_READER_DISABLE_EMBEDDINGS": "1"}), patch(
            "core.semantic_search._cross_encoder_model"
        ) as reranker:
            self.assertIsNone(semantic_scores("query", ["document"]))
            self.assertIsNone(cross_encoder_scores("query", ["document"]))
        reranker.assert_not_called()

    def test_multilingual_e5_applies_query_and_passage_prefixes(self):
        with patch.dict(
            os.environ,
            {
                "PAPER_READER_DISABLE_EMBEDDINGS": "0",
                "EMBEDDING_MODEL": "intfloat/multilingual-e5-large",
            },
        ), patch(
            "core.semantic_search.embed_texts",
            side_effect=[
                [np.array([1.0, 0.0], dtype=np.float32)],
                [np.array([1.0, 0.0], dtype=np.float32)],
            ],
        ) as embed:
            scores = semantic_scores("query text", ["document text"])

        self.assertEqual(scores, [1.0])
        self.assertEqual(embed.call_args_list[0].args[0], ["query: query text"])
        self.assertEqual(
            embed.call_args_list[1].args[0],
            ["passage: document text"],
        )

    def test_english_only_reranker_is_not_applied_to_chinese_query(self):
        with patch.dict(
            os.environ,
            {
                "PAPER_READER_DISABLE_EMBEDDINGS": "0",
                "PAPER_READER_RERANKER_MODEL": (
                    "Xenova/ms-marco-MiniLM-L-6-v2"
                ),
            },
        ), patch("core.semantic_search._cross_encoder_model") as reranker:
            scores = cross_encoder_scores("论文的主要方法是什么？", ["method"])

        self.assertIsNone(scores)
        reranker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
