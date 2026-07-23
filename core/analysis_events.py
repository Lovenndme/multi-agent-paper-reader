"""Typed application-level contracts for one complete paper analysis run."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AnalysisStage(str, Enum):
    """Stable stages used for task-level progress and failure reporting."""

    PREPARING = "preparing"
    PARSING = "parsing"
    VISION = "vision"
    EVIDENCE = "evidence"
    SPECIALISTS = "specialists"
    SUMMARY = "summary"
    ASSESSMENT = "assessment"
    PERSISTENCE = "persistence"
    COMPLETED = "completed"


@dataclass(frozen=True)
class AnalysisRequest:
    """Transport-neutral input for one paper analysis."""

    filename: str
    pdf_data: bytes
    demo: bool = False

    @property
    def mode(self) -> str:
        return "demo" if self.demo else "live"


@dataclass(frozen=True)
class AnalysisEvent:
    """One domain event emitted while a paper analysis is running."""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"type": self.type, **self.payload}


@dataclass(frozen=True)
class AnalysisResult:
    """Public final response produced by a completed analysis."""

    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return dict(self.payload)


class AnalysisOrchestratorError(RuntimeError):
    """Task-level failure raised by the non-streaming Orchestrator API."""

    def __init__(
        self,
        message: str,
        *,
        stage: AnalysisStage,
        category: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.category = category
        self.payload = payload or {}
