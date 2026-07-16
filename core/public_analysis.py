"""Public single-paper analysis mapping.

Internal evidence IDs remain available to persistence, retrieval, assessment,
and comparison code. User-facing single-paper payloads expose only cleaned
analysis text and an aggregate evidence count.
"""

from __future__ import annotations

import copy
import re
from typing import Any


_PUBLIC_OUTPUT_KEYS = (
    "method_output",
    "experiment_output",
    "critic_output",
    "summary_output",
    "assessment",
)

_INTERNAL_EVIDENCE_MARKER = re.compile(
    r"(?<![A-Za-z0-9:])"
    r"(?:\[\s*|\(\s*|（\s*)?"
    r"[ETF]\d{3,}"
    r"(?:\s*[,，、;；]\s*[ETF]\d{3,})*"
    r"(?:\s*\]|\s*\)|\s*）)?"
    r"(?![A-Za-z0-9])",
    flags=re.IGNORECASE,
)

_INTERNAL_ONLY_KEYS = frozenset({"evidence", "evidence_ids", "evidence_index"})


def sanitize_visible_text(text: str) -> str:
    """Remove single-paper internal evidence markers without harming domain terms."""
    cleaned = _INTERNAL_EVIDENCE_MARKER.sub("", text)
    cleaned = re.sub(r"(?:\s*[、，,;；]\s*){2,}", "，", cleaned)
    cleaned = re.sub(r"[ \t]+([，。！？；：,.!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"([（(])\s*([）)])", "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def public_agent_output(output: dict[str, Any]) -> dict[str, Any]:
    """Map one Agent or assessment object to its user-facing representation."""
    return _sanitize_value(copy.deepcopy(output))


def public_analysis_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a cleaned single-paper API payload while preserving internal storage."""
    public = copy.deepcopy(payload)
    evidence_index = public.pop("evidence_index", None)
    if isinstance(evidence_index, list):
        public["evidence_count"] = len(evidence_index)
    for key in _PUBLIC_OUTPUT_KEYS:
        value = public.get(key)
        if isinstance(value, dict):
            public[key] = public_agent_output(value)
    return public


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_visible_text(value)
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _sanitize_value(item)
            for key, item in value.items()
            if key not in _INTERNAL_ONLY_KEYS
        }
    return value
