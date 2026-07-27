"""Deterministic hierarchical retrieval shared by paper tools and agent seeds."""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from statistics import fmean
from typing import Sequence

from core.evidence import EvidenceSnippet
from core.semantic_search import cross_encoder_scores, semantic_scores


_RRF_K = 60
_DEFAULT_SUBCHUNK_TOKENS = 220
_DEFAULT_SUBCHUNK_OVERLAP = 40
_DEFAULT_CANDIDATE_POOL = 24

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "model",
    "of",
    "on",
    "or",
    "paper",
    "report",
    "reported",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "use",
    "used",
    "what",
    "when",
    "which",
    "why",
    "with",
}


@dataclass(frozen=True)
class RetrievalHit:
    """One parent evidence item ranked through its best matching subchunk."""

    snippet: EvidenceSnippet
    score: float
    dense_score: float | None
    bm25_score: float
    dense_rank: int | None
    bm25_rank: int
    rerank_score: float | None
    best_subchunk: str


@dataclass(frozen=True)
class RetrievalDiagnostics:
    """Small, model-free signals used by evaluation and adaptive routing."""

    candidates: int
    dense_available: bool
    reranker_used: bool
    dense_bm25_overlap_at_5: int
    top_score: float
    score_margin: float

    @property
    def low_confidence(self) -> bool:
        if self.candidates <= 1:
            return False
        if not self.dense_available:
            return True
        return self.dense_bm25_overlap_at_5 == 0 and self.score_margin < 0.001


@dataclass(frozen=True)
class RetrievalRanking:
    """Ranked parent evidence plus bounded confidence diagnostics."""

    hits: tuple[RetrievalHit, ...]
    diagnostics: RetrievalDiagnostics


def rank_evidence(
    snippets: Sequence[EvidenceSnippet],
    query: str,
    *,
    profile: str | None = None,
    candidate_pool: int | None = None,
    rerank: bool | None = None,
) -> RetrievalRanking:
    """Rank parent evidence with token windows, BM25, RRF, and optional reranking."""
    clean_query = normalize_retrieval_text(query)[:800]
    if not snippets or not clean_query:
        return RetrievalRanking(
            (),
            RetrievalDiagnostics(
                candidates=len(snippets),
                dense_available=False,
                reranker_used=False,
                dense_bm25_overlap_at_5=0,
                top_score=0.0,
                score_margin=0.0,
            ),
        )

    parents = [
        normalize_retrieval_text(f"{snippet.section}\n{snippet.text}")
        for snippet in snippets
    ]
    subdocuments: list[str] = []
    subdocument_parents: list[int] = []
    for parent_index, (snippet, parent) in enumerate(zip(snippets, parents, strict=True)):
        windows = token_windows(
            parent,
            max_tokens=_bounded_env_int(
                "PAPER_READER_RETRIEVAL_SUBCHUNK_TOKENS",
                _DEFAULT_SUBCHUNK_TOKENS,
                96,
                480,
            ),
            overlap_tokens=_bounded_env_int(
                "PAPER_READER_RETRIEVAL_SUBCHUNK_OVERLAP",
                _DEFAULT_SUBCHUNK_OVERLAP,
                0,
                120,
            ),
        )
        if not windows:
            windows = [parent]
        for window in windows:
            subdocuments.append(
                normalize_retrieval_text(f"{snippet.section}\n{window}")
            )
            subdocument_parents.append(parent_index)

    subdense = semantic_scores(clean_query, subdocuments)
    dense_scores: list[float] | None = None
    best_subchunks = [""] * len(snippets)
    if subdense is not None:
        dense_scores = [-1.0] * len(snippets)
        for parent_index, window, score in zip(
            subdocument_parents,
            subdocuments,
            subdense,
            strict=True,
        ):
            if score > dense_scores[parent_index]:
                dense_scores[parent_index] = score
                best_subchunks[parent_index] = window
    else:
        for parent_index, window in zip(
            subdocument_parents,
            subdocuments,
            strict=True,
        ):
            if not best_subchunks[parent_index]:
                best_subchunks[parent_index] = window

    bm25 = bm25_scores(clean_query, parents)
    bm25_available = any(score > 0 for score in bm25)
    dense_ranks = _ranks(dense_scores) if dense_scores is not None else None
    bm25_ranks = _ranks(bm25)
    explicit_ids = {
        match.upper()
        for match in re.findall(r"(?:[A-Z0-9]+:)?[ETF]\d{3}", clean_query, re.I)
    }
    lowered_query = clean_query.casefold()

    fused: list[float] = []
    for index, snippet in enumerate(snippets):
        score = (
            1.0 / (_RRF_K + bm25_ranks[index])
            if bm25_available
            else 0.0
        )
        if dense_ranks is not None:
            score += 1.0 / (_RRF_K + dense_ranks[index])
        normalized_id = snippet.id.upper()
        if normalized_id in explicit_ids or any(
            item.endswith(f":{normalized_id}") for item in explicit_ids
        ):
            score += 10.0
        score += _type_prior(snippet, lowered_query, profile)
        fused.append(score)

    initial_order = sorted(
        range(len(snippets)),
        key=lambda index: (-fused[index], index),
    )
    bounded_pool = candidate_pool or _bounded_env_int(
        "PAPER_READER_RETRIEVAL_CANDIDATES",
        _DEFAULT_CANDIDATE_POOL,
        5,
        80,
    )
    pool = initial_order[: max(1, min(bounded_pool, len(initial_order)))]

    if rerank is None:
        rerank = _env_bool("PAPER_READER_RERANKER_ENABLED", default=True)
    rerank_values: list[float] | None = None
    rerank_by_parent: dict[int, float] = {}
    if rerank and pool:
        rerank_documents = [
            best_subchunks[index] or parents[index]
            for index in pool
        ]
        rerank_values = cross_encoder_scores(clean_query, rerank_documents)
        if rerank_values is not None:
            rerank_ranks = _ranks(rerank_values)
            for pool_index, parent_index in enumerate(pool):
                # RRF keeps the robust dense/BM25 candidate signal while letting
                # a cross-encoder refine only the bounded candidate set.
                fused[parent_index] += (
                    3.0 / (_RRF_K + rerank_ranks[pool_index])
                )
                rerank_by_parent[parent_index] = rerank_values[pool_index]

    order = sorted(
        range(len(snippets)),
        key=lambda index: (-fused[index], index),
    )
    hits = tuple(
        RetrievalHit(
            snippet=snippets[index],
            score=fused[index],
            dense_score=dense_scores[index] if dense_scores is not None else None,
            bm25_score=bm25[index],
            dense_rank=dense_ranks[index] if dense_ranks is not None else None,
            bm25_rank=bm25_ranks[index],
            rerank_score=rerank_by_parent.get(index),
            best_subchunk=best_subchunks[index] or parents[index],
        )
        for index in order
    )
    dense_top = (
        {
            index
            for index in sorted(
                range(len(snippets)),
                key=lambda item: (dense_ranks[item], item),
            )[:5]
        }
        if dense_ranks is not None
        else set()
    )
    bm25_top = (
        {
            index
            for index in sorted(
                range(len(snippets)),
                key=lambda item: (bm25_ranks[item], item),
            )[:5]
        }
        if bm25_available
        else set()
    )
    top_score = hits[0].score if hits else 0.0
    second_score = hits[1].score if len(hits) > 1 else 0.0
    return RetrievalRanking(
        hits,
        RetrievalDiagnostics(
            candidates=len(snippets),
            dense_available=dense_scores is not None,
            reranker_used=rerank_values is not None,
            dense_bm25_overlap_at_5=len(dense_top & bm25_top),
            top_score=top_score,
            score_margin=max(0.0, top_score - second_score),
        ),
    )


def normalize_retrieval_text(value: str) -> str:
    """Normalize PDF extraction artifacts without changing displayed citations."""
    text = str(value or "").replace("\u00ad", "")
    # PyMuPDF output commonly contains "pre- trained" or "pa- rameter".
    # This normalization is retrieval-only, so legitimate source text remains
    # available unchanged in EvidenceSnippet for citations.
    text = re.sub(r"(?<=[A-Za-z])-\s+(?=[a-z])", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_windows(
    value: str,
    *,
    max_tokens: int = _DEFAULT_SUBCHUNK_TOKENS,
    overlap_tokens: int = _DEFAULT_SUBCHUNK_OVERLAP,
) -> list[str]:
    """Split text by an actual tokenizer, with a deterministic fallback."""
    text = normalize_retrieval_text(value)
    if not text:
        return []
    max_tokens = max(16, max_tokens)
    overlap_tokens = max(0, min(overlap_tokens, max_tokens - 1))
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text, disallowed_special=())
        windows: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(len(tokens), start + max_tokens)
            decoded = encoding.decode(tokens[start:end]).strip()
            if decoded:
                windows.append(decoded)
            if end >= len(tokens):
                break
            start = end - overlap_tokens
        return windows
    except Exception:
        chars = max_tokens * 3
        overlap = overlap_tokens * 3
        return [
            text[start : start + chars].strip()
            for start in range(0, len(text), max(1, chars - overlap))
            if text[start : start + chars].strip()
        ]


def bm25_scores(query: str, documents: Sequence[str]) -> list[float]:
    """Return BM25 scores over normalized English tokens and Chinese bigrams."""
    tokenized = [_tokenize(document) for document in documents]
    query_terms = Counter(_tokenize(query))
    document_count = len(tokenized)
    if not document_count:
        return []
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))
    average_length = fmean(len(tokens) for tokens in tokenized) if tokenized else 1.0
    scores: list[float] = []
    for tokens in tokenized:
        frequencies = Counter(tokens)
        length = max(1, len(tokens))
        score = 0.0
        for term, query_frequency in query_terms.items():
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            inverse_document_frequency = math.log(
                1
                + (document_count - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            denominator = frequency + 1.5 * (
                0.25 + 0.75 * length / max(1.0, average_length)
            )
            score += (
                inverse_document_frequency
                * frequency
                * 2.5
                / denominator
                * min(query_frequency, 2)
            )
        scores.append(score)
    return scores


def _tokenize(value: str) -> list[str]:
    normalized = normalize_retrieval_text(value).casefold()
    tokens = [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_.-]*", normalized)
        if len(token) > 1 and token not in _STOPWORDS
    ]
    for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
        tokens.extend(
            sequence[index : index + 2]
            for index in range(len(sequence) - 1)
        )
    return tokens


def _ranks(values: Sequence[float] | None) -> list[int]:
    if values is None:
        return []
    ordered = sorted(
        range(len(values)),
        key=lambda index: (-values[index], index),
    )
    ranks = [0] * len(values)
    for rank, index in enumerate(ordered, start=1):
        ranks[index] = rank
    return ranks


def _type_prior(
    snippet: EvidenceSnippet,
    query: str,
    profile: str | None,
) -> float:
    score = 0.0
    if snippet.kind == "table" and any(
        marker in query
        for marker in (
            "table",
            "表",
            "result",
            "结果",
            "metric",
            "指标",
            "ablation",
            "消融",
        )
    ):
        score += 0.0003
    if snippet.kind == "figure" and any(
        marker in query
        for marker in (
            "figure",
            "图",
            "architecture",
            "架构",
            "pipeline",
            "流程",
        )
    ):
        score += 0.0003
    if profile == "experiment" and snippet.kind == "table":
        score += 0.0002
    if profile == "method" and snippet.kind == "figure":
        score += 0.0001
    if profile == "critic" and snippet.kind in {"table", "figure"}:
        score += 0.00005
    return score


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _bounded_env_int(
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))
