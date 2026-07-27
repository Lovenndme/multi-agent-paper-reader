"""Local multilingual semantic ranking with a deterministic lexical fallback path."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np
from langchain_core.embeddings import Embeddings


LOGGER = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_VECTOR_CACHE_ITEMS = 8_000

_MODEL_LOCK = threading.RLock()
_MODEL = None
_MODEL_NAME = ""
_VECTOR_CACHE: OrderedDict[str, np.ndarray] = OrderedDict()
_WARNED_FAILURE = False
_RERANKER = None
_RERANKER_NAME = ""
_WARNED_RERANKER_FAILURE = False


def embeddings_enabled() -> bool:
    """Return whether local semantic ranking is enabled for this process."""
    disabled = os.environ.get("PAPER_READER_DISABLE_EMBEDDINGS", "").strip().lower()
    return disabled not in {"1", "true", "yes", "on"}


def semantic_scores(query: str, documents: Iterable[str]) -> list[float] | None:
    """Return cosine similarities, or ``None`` when semantic ranking is unavailable.

    Model loading is lazy so ordinary API startup remains fast. The caller keeps
    its existing lexical scorer as an explicit, testable degradation path.
    """
    texts = [str(document).strip() for document in documents]
    if not embeddings_enabled() or not query.strip() or not texts:
        return None
    try:
        query_text, document_texts = _semantic_inputs(query.strip(), texts)
        query_vector = embed_texts([query_text])[0]
        document_vectors = embed_texts(document_texts)
        return [float(np.dot(query_vector, vector)) for vector in document_vectors]
    except Exception as exc:  # provider-free local model must never break paper reading
        global _WARNED_FAILURE
        if not _WARNED_FAILURE:
            LOGGER.warning("Local embedding model unavailable; using lexical fallback: %s", exc)
            _WARNED_FAILURE = True
        return None


def clear_semantic_cache() -> None:
    """Clear process-local model/vector state; primarily useful for tests."""
    global _MODEL, _MODEL_NAME, _RERANKER, _RERANKER_NAME
    global _WARNED_FAILURE, _WARNED_RERANKER_FAILURE
    with _MODEL_LOCK:
        _MODEL = None
        _MODEL_NAME = ""
        _RERANKER = None
        _RERANKER_NAME = ""
        _VECTOR_CACHE.clear()
        _WARNED_FAILURE = False
        _WARNED_RERANKER_FAILURE = False


def cross_encoder_scores(
    query: str,
    documents: Iterable[str],
) -> list[float] | None:
    """Return local cross-encoder relevance logits, or ``None`` on degradation."""
    texts = [str(document).strip() for document in documents]
    if not embeddings_enabled() or not query.strip() or not texts:
        return None
    try:
        model_name = os.environ.get(
            "PAPER_READER_RERANKER_MODEL",
            "Xenova/ms-marco-MiniLM-L-6-v2",
        ).strip()
        if re.search(r"[\u4e00-\u9fff]", query) and (
            "ms-marco" in model_name.casefold()
            or model_name.casefold().endswith("-en")
        ):
            # The default reranker is intentionally small and English-only.
            # Keep multilingual dense/BM25 ranks for Chinese questions instead
            # of applying an out-of-domain cross-encoder with false confidence.
            return None
        model = _cross_encoder_model(model_name)
        scores = [float(value) for value in model.rerank(query.strip(), texts)]
        if len(scores) != len(texts):
            raise RuntimeError("Reranker returned an unexpected score count.")
        return scores
    except Exception as exc:
        global _WARNED_RERANKER_FAILURE
        if not _WARNED_RERANKER_FAILURE:
            LOGGER.warning(
                "Local cross-encoder unavailable; keeping hybrid rank: %s",
                exc,
            )
            _WARNED_RERANKER_FAILURE = True
        return None


def embed_texts(texts: list[str]) -> list[np.ndarray]:
    """Return normalized local embeddings for LangGraph/LangMem and rankers."""
    model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()
    model = _embedding_model(model_name)
    results: list[np.ndarray | None] = [None] * len(texts)
    missing_texts: list[str] = []
    missing_indices: list[int] = []

    with _MODEL_LOCK:
        for index, text in enumerate(texts):
            cache_key = _cache_key(model_name, text)
            cached = _VECTOR_CACHE.get(cache_key)
            if cached is not None:
                _VECTOR_CACHE.move_to_end(cache_key)
                results[index] = cached
            else:
                missing_texts.append(text)
                missing_indices.append(index)

    if missing_texts:
        vectors = list(
            model.embed(
                missing_texts,
                batch_size=_embedding_batch_size(),
            )
        )
        if len(vectors) != len(missing_texts):
            raise RuntimeError("Embedding model returned an unexpected vector count.")
        with _MODEL_LOCK:
            for index, text, raw_vector in zip(missing_indices, missing_texts, vectors, strict=True):
                vector = np.asarray(raw_vector, dtype=np.float32)
                norm = float(np.linalg.norm(vector))
                if not np.isfinite(norm) or norm <= 0:
                    raise RuntimeError("Embedding model returned an invalid vector.")
                vector = vector / norm
                results[index] = vector
                _VECTOR_CACHE[_cache_key(model_name, text)] = vector
            while len(_VECTOR_CACHE) > MAX_VECTOR_CACHE_ITEMS:
                _VECTOR_CACHE.popitem(last=False)

    return [vector for vector in results if vector is not None]


class LocalFastEmbedEmbeddings(Embeddings):
    """LangChain embedding adapter backed by the project's local FastEmbed model."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in embed_texts(texts)]

    def embed_query(self, text: str) -> list[float]:
        return embed_texts([text])[0].tolist()


def _embedding_model(model_name: str):
    global _MODEL, _MODEL_NAME
    with _MODEL_LOCK:
        if _MODEL is not None and _MODEL_NAME == model_name:
            return _MODEL
        from fastembed import TextEmbedding

        configured_cache = os.environ.get("PAPER_READER_MODEL_DIR")
        cache_dir = (
            Path(configured_cache).expanduser().resolve()
            if configured_cache
            else Path(__file__).resolve().parent.parent / ".paper-reader" / "models"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        _MODEL = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))
        _MODEL_NAME = model_name
        _VECTOR_CACHE.clear()
        return _MODEL


def _cross_encoder_model(model_name: str):
    global _RERANKER, _RERANKER_NAME
    with _MODEL_LOCK:
        if _RERANKER is not None and _RERANKER_NAME == model_name:
            return _RERANKER
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        configured_cache = os.environ.get("PAPER_READER_MODEL_DIR")
        cache_dir = (
            Path(configured_cache).expanduser().resolve()
            if configured_cache
            else Path(__file__).resolve().parent.parent / ".paper-reader" / "models"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        _RERANKER = TextCrossEncoder(
            model_name=model_name,
            cache_dir=str(cache_dir),
        )
        _RERANKER_NAME = model_name
        return _RERANKER


def _cache_key(model_name: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{model_name}:{digest}"


def _embedding_batch_size() -> int:
    """Bound local inference memory without exposing FastEmbed internals."""
    try:
        value = int(os.environ.get("PAPER_READER_EMBEDDING_BATCH_SIZE", "64"))
    except (TypeError, ValueError):
        value = 64
    return max(1, min(value, 256))


def _semantic_inputs(
    query: str,
    documents: list[str],
) -> tuple[str, list[str]]:
    """Apply model-specific retrieval prefixes recommended by model families."""
    model_name = os.environ.get(
        "EMBEDDING_MODEL",
        DEFAULT_EMBEDDING_MODEL,
    ).strip().casefold()
    if "multilingual-e5" in model_name:
        return (
            f"query: {query}",
            [f"passage: {document}" for document in documents],
        )
    if model_name.startswith("baai/bge-") and model_name.endswith("-en-v1.5"):
        return (
            "Represent this sentence for searching relevant passages: " + query,
            documents,
        )
    return query, documents
