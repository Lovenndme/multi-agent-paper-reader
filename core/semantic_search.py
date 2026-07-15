"""Local multilingual semantic ranking with a deterministic lexical fallback path."""

from __future__ import annotations

import hashlib
import logging
import os
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
        query_vector = embed_texts([query.strip()])[0]
        document_vectors = embed_texts(texts)
        return [float(np.dot(query_vector, vector)) for vector in document_vectors]
    except Exception as exc:  # provider-free local model must never break paper reading
        global _WARNED_FAILURE
        if not _WARNED_FAILURE:
            LOGGER.warning("Local embedding model unavailable; using lexical fallback: %s", exc)
            _WARNED_FAILURE = True
        return None


def clear_semantic_cache() -> None:
    """Clear process-local model/vector state; primarily useful for tests."""
    global _MODEL, _MODEL_NAME, _WARNED_FAILURE
    with _MODEL_LOCK:
        _MODEL = None
        _MODEL_NAME = ""
        _VECTOR_CACHE.clear()
        _WARNED_FAILURE = False


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
        vectors = list(model.embed(missing_texts))
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


def _cache_key(model_name: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{model_name}:{digest}"
