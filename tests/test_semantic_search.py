"""Tests for provider-free multilingual semantic ranking."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from core.semantic_search import clear_semantic_cache, semantic_scores


class _FakeEmbeddingModel:
    def embed(self, texts):
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
        with patch.dict(os.environ, {"PAPER_READER_DISABLE_EMBEDDINGS": "1"}):
            self.assertIsNone(semantic_scores("query", ["document"]))


if __name__ == "__main__":
    unittest.main()
