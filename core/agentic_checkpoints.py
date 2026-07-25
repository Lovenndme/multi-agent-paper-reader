"""Small local checkpoint log for auditable Agentic RAG tool loops."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any


_LOCK = RLock()
_CHECKPOINT_EVENTS = {
    "retrieval_started",
    "tool_complete",
    "coverage_checked",
    "retrieval_complete",
}


def save_agentic_checkpoint(
    event_type: str,
    payload: dict[str, Any],
) -> None:
    """Append one JSON-safe public retrieval state without storing model prompts."""
    if event_type not in _CHECKPOINT_EVENTS or not _checkpoints_enabled():
        return
    run_id = str(payload.get("run_id") or "")[:80]
    agent = str(payload.get("agent") or "")[:80]
    if not run_id or not agent:
        return
    safe_payload = {
        key: value
        for key, value in payload.items()
        if key
        in {
            "run_id",
            "agent",
            "step",
            "tool",
            "summary",
            "evidence_count",
            "steps",
            "fallback_used",
            "sufficient",
            "error",
        }
    }
    now = datetime.now(timezone.utc).isoformat()
    with _LOCK, _connect() as connection:
        connection.execute(
            """
            INSERT INTO agentic_rag_checkpoints (
                run_id, agent, event_type, created_at, state_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                run_id,
                agent,
                event_type,
                now,
                json.dumps(safe_payload, ensure_ascii=False, separators=(",", ":")),
            ),
        )
        if event_type == "retrieval_started":
            cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
            connection.execute(
                "DELETE FROM agentic_rag_checkpoints WHERE created_at < ?",
                (cutoff,),
            )


def load_agentic_checkpoints(run_id: str) -> list[dict[str, Any]]:
    """Load one loop's public checkpoints for diagnostics or future recovery."""
    if not run_id or len(run_id) > 80:
        return []
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT event_type, created_at, state_json
            FROM agentic_rag_checkpoints
            WHERE run_id = ?
            ORDER BY id ASC
            """,
            (run_id,),
        ).fetchall()
    return [
        {
            "event_type": str(row["event_type"]),
            "created_at": str(row["created_at"]),
            "state": json.loads(str(row["state_json"])),
        }
        for row in rows
    ]


def _connect() -> sqlite3.Connection:
    path = _database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 10000")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS agentic_rag_checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            agent TEXT NOT NULL,
            event_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            state_json TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_agentic_rag_run
        ON agentic_rag_checkpoints(run_id, id)
        """
    )
    return connection


def _database_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    data_dir = Path(os.environ.get("PAPER_READER_DATA_DIR") or root / ".paper-reader")
    return data_dir.expanduser().resolve() / "agentic-rag.sqlite3"


def _checkpoints_enabled() -> bool:
    return os.environ.get("AGENTIC_RAG_CHECKPOINTS", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
