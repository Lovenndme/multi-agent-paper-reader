"""Focused regression tests for hierarchical hybrid paper retrieval."""

from unittest.mock import patch

import tiktoken

from core.evidence import EvidenceSnippet
from core.hybrid_retrieval import (
    bm25_scores,
    normalize_retrieval_text,
    rank_evidence,
    token_windows,
)


def test_retrieval_normalization_repairs_pdf_hyphenation_only_in_search_text() -> None:
    value = "pre- trained pa-\nrameter state-of-the-art"

    assert normalize_retrieval_text(value) == (
        "pretrained parameter state-of-the-art"
    )


def test_token_windows_honor_token_budget_and_overlap() -> None:
    text = " ".join(f"token{index}" for index in range(300))
    windows = token_windows(text, max_tokens=64, overlap_tokens=12)
    encoder = tiktoken.get_encoding("cl100k_base")

    assert len(windows) > 1
    assert all(len(encoder.encode(window)) <= 64 for window in windows)
    assert "token0" in windows[0]
    assert "token299" in windows[-1]


def test_subchunk_dense_hit_returns_traceable_parent() -> None:
    snippets = (
        EvidenceSnippet(
            "E001",
            "Method",
            0,
            0,
            ("background " * 300) + "frozen low rank matrix update",
        ),
        EvidenceSnippet("E002", "Related Work", 1, 1, "unrelated baseline"),
    )

    def semantic(_query: str, documents: list[str]) -> list[float]:
        return [
            0.95 if "frozen low rank matrix update" in document else 0.05
            for document in documents
        ]

    with (
        patch("core.hybrid_retrieval.semantic_scores", side_effect=semantic),
        patch(
            "core.hybrid_retrieval.bm25_scores",
            return_value=[1.0, 0.0],
        ),
    ):
        ranking = rank_evidence(
            snippets,
            "frozen low rank matrix update",
            rerank=False,
        )

    assert ranking.hits[0].snippet.id == "E001"
    assert "frozen low rank matrix update" in ranking.hits[0].best_subchunk


def test_bm25_dehyphenation_matches_pdf_word() -> None:
    scores = bm25_scores(
        "pretrained parameter",
        ["The pre- trained pa-\nrameter is frozen.", "unrelated result"],
    )

    assert scores[0] > scores[1]


def test_cross_encoder_failure_preserves_hybrid_order() -> None:
    snippets = (
        EvidenceSnippet("E001", "Method", 0, 0, "target method"),
        EvidenceSnippet("E002", "Other", 1, 1, "unrelated"),
    )
    with (
        patch(
            "core.hybrid_retrieval.semantic_scores",
            return_value=[0.9, 0.1],
        ),
        patch(
            "core.hybrid_retrieval.bm25_scores",
            return_value=[2.0, 0.0],
        ),
        patch(
            "core.hybrid_retrieval.cross_encoder_scores",
            return_value=None,
        ),
    ):
        ranking = rank_evidence(snippets, "target method", rerank=True)

    assert [hit.snippet.id for hit in ranking.hits] == ["E001", "E002"]
    assert ranking.diagnostics.reranker_used is False


def test_cross_encoder_is_fused_only_over_bounded_candidate_pool() -> None:
    snippets = tuple(
        EvidenceSnippet(f"E{index:03d}", "Method", index, index, f"item {index}")
        for index in range(1, 7)
    )
    with (
        patch(
            "core.hybrid_retrieval.semantic_scores",
            return_value=[0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        ),
        patch(
            "core.hybrid_retrieval.bm25_scores",
            return_value=[6, 5, 4, 3, 2, 1],
        ),
        patch(
            "core.hybrid_retrieval.cross_encoder_scores",
            return_value=[0.1, 0.2, 0.9],
        ) as reranker,
    ):
        ranking = rank_evidence(
            snippets,
            "item",
            candidate_pool=3,
            rerank=True,
        )

    assert reranker.call_args.args[1] and len(reranker.call_args.args[1]) == 3
    assert ranking.diagnostics.reranker_used is True
    assert ranking.hits[0].snippet.id == "E003"
