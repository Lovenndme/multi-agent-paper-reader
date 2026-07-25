"""Provider-neutral contracts for bounded, auditable Agentic RAG."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from core.evidence import EvidenceSnippet


RagMode = Literal["hybrid", "agentic"]
ToolStrategy = Literal["auto", "native", "structured"]
PaperToolName = Literal[
    "paper_search",
    "paper_overview",
    "paper_read_section",
    "paper_read_page",
    "paper_read_table",
    "paper_read_figure",
    "calculate",
    "finish_retrieval",
]


class RetrievalAction(BaseModel):
    """One explicit retrieval action selected by a model."""

    tool: PaperToolName
    query: str | None = Field(default=None, max_length=800)
    section: str | None = Field(default=None, max_length=240)
    page: int | None = Field(default=None, ge=1, le=10_000)
    index: int | None = Field(default=None, ge=1, le=10_000)
    evidence_id: str | None = Field(default=None, max_length=32)
    kind: Literal["any", "text", "table", "figure"] = "any"
    limit: int = Field(default=6, ge=1, le=10)
    expression: str | None = Field(default=None, max_length=256)
    public_summary: str = Field(
        min_length=1,
        max_length=240,
        description=(
            "A concise user-visible action summary. It must not contain private chain of "
            "thought, raw JSON, secrets, or unverified conclusions."
        ),
    )

    @model_validator(mode="after")
    def validate_tool_arguments(self) -> "RetrievalAction":
        required = {
            "paper_search": self.query,
            "paper_read_section": self.section,
            "paper_read_page": self.page,
            "calculate": self.expression,
        }
        value = required.get(self.tool, True)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"{self.tool} is missing its required argument.")
        if self.tool in {"paper_read_table", "paper_read_figure"}:
            if self.index is None and not (self.evidence_id or "").strip():
                raise ValueError(f"{self.tool} requires index or evidence_id.")
        return self


class RetrievalDecision(BaseModel):
    """Structured-action fallback output understood by every provider."""

    action: RetrievalAction


class NativeFinishRetrieval(BaseModel):
    """Schema exposed as a native tool to finish a retrieval loop."""

    public_summary: str = Field(min_length=1, max_length=240)


@dataclass(frozen=True)
class AgenticRunBudget:
    """Hard limits preventing unbounded model/tool loops."""

    max_steps: int = 4
    max_results_per_step: int = 6
    max_evidence_items: int = 16
    max_evidence_chars: int = 24_000
    max_observation_chars: int = 8_000
    planner_retries: int = 2

    @classmethod
    def from_env(cls) -> "AgenticRunBudget":
        return cls(
            max_steps=_bounded_env_int("AGENTIC_RAG_MAX_STEPS", 4, 1, 8),
            max_results_per_step=_bounded_env_int(
                "AGENTIC_RAG_MAX_RESULTS_PER_STEP", 6, 1, 10
            ),
            max_evidence_items=_bounded_env_int(
                "AGENTIC_RAG_MAX_EVIDENCE_ITEMS", 16, 4, 30
            ),
            max_evidence_chars=_bounded_env_int(
                "AGENTIC_RAG_MAX_EVIDENCE_CHARS", 24_000, 4_000, 60_000
            ),
            max_observation_chars=_bounded_env_int(
                "AGENTIC_RAG_MAX_OBSERVATION_CHARS", 8_000, 1_000, 16_000
            ),
            planner_retries=_bounded_env_int(
                "AGENTIC_RAG_PLANNER_RETRIES", 2, 1, 3
            ),
        )


@dataclass(frozen=True)
class ToolObservation:
    """Safe result returned by a paper tool."""

    tool: str
    summary: str
    content: str
    snippets: tuple[EvidenceSnippet, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalStep:
    """One auditable model action and its host-executed outcome."""

    step: int
    strategy: str
    action: RetrievalAction
    observation_summary: str
    evidence_ids: tuple[str, ...] = ()
    error: str | None = None

    def public_payload(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "strategy": self.strategy,
            "tool": self.action.tool,
            "summary": self.action.public_summary,
            "observation": self.observation_summary,
            "evidence_count": len(self.evidence_ids),
            "error": self.error,
        }


@dataclass(frozen=True)
class AgenticRetrievalResult:
    """Bounded evidence selected by one autonomous retrieval loop."""

    snippets: tuple[EvidenceSnippet, ...]
    steps: tuple[RetrievalStep, ...]
    strategy: str
    stop_reason: str
    fallback_used: bool = False

    def public_trace(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "stop_reason": self.stop_reason,
            "fallback_used": self.fallback_used,
            "evidence_ids": [snippet.id for snippet in self.snippets],
            "steps": [step.public_payload() for step in self.steps],
        }


def agentic_rag_mode() -> RagMode:
    value = os.environ.get("RAG_MODE", "agentic").strip().lower()
    return "hybrid" if value == "hybrid" else "agentic"


def agentic_tool_strategy() -> ToolStrategy:
    value = os.environ.get("AGENTIC_TOOL_STRATEGY", "auto").strip().lower()
    return value if value in {"auto", "native", "structured"} else "auto"


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))
