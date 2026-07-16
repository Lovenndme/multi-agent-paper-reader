"""Thread-safe, user-visible analysis progress tracking.

The tracker stores concise activity summaries and timings only. Raw model
responses, structured JSON tokens, and private reasoning text are never stored.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from threading import RLock
from typing import Any


class AnalysisProgressTracker:
    """Collect one analysis run's public progress trace and elapsed timings."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._started_monotonic = time.monotonic()
        self._started_at = _utc_now()
        self._completed_at: str | None = None
        self._status = "running"
        self._entries: list[dict[str, Any]] = []
        self._entry_indexes: dict[tuple[str, str], int] = {}
        self._agents: dict[str, dict[str, Any]] = {}

    def started_payload(self) -> dict[str, Any]:
        return {
            "started_at": self._started_at,
            "elapsed_ms": self.elapsed_ms(),
        }

    def start_agent(self, agent: str, summary: str) -> dict[str, Any]:
        with self._lock:
            now = _utc_now()
            state = self._agents.setdefault(agent, {})
            state.update(
                {
                    "status": "running",
                    "started_at": now,
                    "_started_monotonic": time.monotonic(),
                    "duration_ms": 0,
                }
            )
            self._store_entry(agent, f"{agent}-start", summary, "pipeline", append=False)
            return {
                "agent": agent,
                "summary": summary,
                "source": "pipeline",
                "started_at": now,
                "elapsed_ms": self.elapsed_ms(),
            }

    def progress(
        self,
        agent: str,
        summary: str,
        *,
        source: str = "pipeline",
        progress_id: str | None = None,
        append: bool = False,
    ) -> dict[str, Any]:
        clean_summary = _bounded_text(summary)
        stable_id = progress_id or f"{agent}-{len(self._entries) + 1}"
        with self._lock:
            self._store_entry(agent, stable_id, clean_summary, source, append=append)
            return {
                "agent": agent,
                "progress_id": stable_id,
                "text": clean_summary,
                "source": source,
                "append": append,
                "elapsed_ms": self.elapsed_ms(),
            }

    def complete_agent(self, agent: str, summary: str) -> dict[str, Any]:
        with self._lock:
            now = _utc_now()
            state = self._agents.setdefault(agent, {})
            started = state.pop("_started_monotonic", self._started_monotonic)
            duration_ms = max(0, round((time.monotonic() - started) * 1000))
            state.update(
                {
                    "status": "complete",
                    "completed_at": now,
                    "duration_ms": duration_ms,
                }
            )
            self._store_entry(agent, f"{agent}-complete", summary, "pipeline", append=False)
            return {
                "agent": agent,
                "summary": summary,
                "source": "pipeline",
                "completed_at": now,
                "duration_ms": duration_ms,
                "elapsed_ms": self.elapsed_ms(),
            }

    def fail_agent(self, agent: str, summary: str) -> dict[str, Any]:
        with self._lock:
            now = _utc_now()
            state = self._agents.setdefault(agent, {})
            started = state.pop("_started_monotonic", self._started_monotonic)
            duration_ms = max(0, round((time.monotonic() - started) * 1000))
            state.update(
                {
                    "status": "failed",
                    "completed_at": now,
                    "duration_ms": duration_ms,
                }
            )
            self._store_entry(agent, f"{agent}-failed", summary, "pipeline", append=False)
            return {
                "agent": agent,
                "summary": summary,
                "completed_at": now,
                "duration_ms": duration_ms,
                "elapsed_ms": self.elapsed_ms(),
            }

    def finish(self, *, status: str = "completed") -> dict[str, Any]:
        with self._lock:
            self._status = status
            self._completed_at = _utc_now()
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            agents = {
                agent: {
                    key: value
                    for key, value in state.items()
                    if not key.startswith("_")
                }
                for agent, state in self._agents.items()
            }
            return {
                "status": self._status,
                "started_at": self._started_at,
                "completed_at": self._completed_at,
                "duration_ms": self.elapsed_ms(),
                "agents": agents,
                "entries": [dict(entry) for entry in self._entries],
            }

    def elapsed_ms(self) -> int:
        return max(0, round((time.monotonic() - self._started_monotonic) * 1000))

    def _store_entry(
        self,
        agent: str,
        progress_id: str,
        text: str,
        source: str,
        *,
        append: bool,
    ) -> None:
        key = (agent, progress_id)
        if key in self._entry_indexes:
            index = self._entry_indexes[key]
            if append:
                current = str(self._entries[index].get("text") or "")
                text = f"{current}{text}"
            self._entries[index]["text"] = _bounded_text(text)
            self._entries[index]["elapsed_ms"] = self.elapsed_ms()
            return
        entry = {
            "id": progress_id,
            "agent": agent,
            "text": _bounded_text(text),
            "source": source,
            "elapsed_ms": self.elapsed_ms(),
        }
        self._entry_indexes[key] = len(self._entries)
        self._entries.append(entry)


def _bounded_text(value: str, limit: int = 4000) -> str:
    return str(value or "")[-limit:]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
