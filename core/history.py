"""Persistent local history for completed paper analyses."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from core.evidence import EvidenceSnippet


ROOT = Path(__file__).resolve().parent.parent
_HISTORY_LOCK = RLock()


def save_paper_analysis(
    *,
    pdf_data: bytes,
    result: dict[str, Any],
    snippets: list[EvidenceSnippet],
) -> str:
    """Insert or replace one PDF analysis and return its stable history ID."""
    if not pdf_data:
        raise ValueError("Cannot save an empty PDF.")
    paper = result.get("paper")
    if not isinstance(paper, dict):
        raise ValueError("Analysis result is missing paper metadata.")

    digest = hashlib.sha256(pdf_data).hexdigest()
    now = _utc_now()
    stored_result = dict(result)
    stored_result.pop("analysis_id", None)
    stored_result.pop("history_id", None)
    evidence_json = json.dumps(
        [asdict(snippet) for snippet in snippets],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    result_json = json.dumps(
        stored_result,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    with _HISTORY_LOCK, history_database_connection() as connection:
        existing = connection.execute(
            "SELECT id, created_at, pdf_path FROM paper_history WHERE sha256 = ?",
            (digest,),
        ).fetchone()
        history_id = str(existing["id"]) if existing else uuid.uuid4().hex
        created_at = str(existing["created_at"]) if existing else now
        pdf_path = _paper_directory() / f"{history_id}.pdf"
        _write_pdf_atomic(pdf_path, pdf_data)

        connection.execute(
            """
            INSERT INTO paper_history (
                id, sha256, title, filename, mode, pages, sections_count,
                size_bytes, created_at, updated_at, result_json, evidence_json,
                pdf_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sha256) DO UPDATE SET
                title = excluded.title,
                filename = excluded.filename,
                mode = excluded.mode,
                pages = excluded.pages,
                sections_count = excluded.sections_count,
                size_bytes = excluded.size_bytes,
                updated_at = excluded.updated_at,
                result_json = excluded.result_json,
                evidence_json = excluded.evidence_json,
                pdf_path = excluded.pdf_path
            """,
            (
                history_id,
                digest,
                str(paper.get("title") or paper.get("filename") or "Untitled Paper"),
                str(paper.get("filename") or "paper.pdf"),
                str(result.get("mode") or "live"),
                _safe_int(paper.get("pages")),
                _safe_int(paper.get("sections_count")),
                _safe_int(paper.get("size_bytes")),
                created_at,
                now,
                result_json,
                evidence_json,
                str(pdf_path),
            ),
        )
    return history_id


def list_paper_history(*, limit: int = 100) -> list[dict[str, Any]]:
    """Return recent saved analyses without their large result payloads."""
    bounded_limit = max(1, min(limit, 500))
    with history_database_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, title, filename, mode, pages, sections_count, size_bytes,
                   created_at, updated_at, pdf_path
            FROM paper_history
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (bounded_limit,),
        ).fetchall()
    return [_history_row(row) for row in rows]


def load_paper_analysis(history_id: str) -> dict[str, Any] | None:
    """Load one saved result and its complete evidence snippets."""
    with history_database_connection() as connection:
        row = connection.execute(
            "SELECT * FROM paper_history WHERE id = ?",
            (history_id,),
        ).fetchone()
    if row is None:
        return None

    result = json.loads(str(row["result_json"]))
    raw_snippets = json.loads(str(row["evidence_json"]))
    snippets = [
        EvidenceSnippet(**item)
        for item in raw_snippets
        if isinstance(item, dict)
    ]
    return {
        "history": _history_row(row),
        "result": result,
        "snippets": snippets,
    }


def paper_history_exists(history_id: str) -> bool:
    """Check one saved analysis without decoding its large JSON payloads."""
    with history_database_connection() as connection:
        row = connection.execute(
            "SELECT 1 FROM paper_history WHERE id = ?",
            (history_id,),
        ).fetchone()
    return row is not None


def delete_paper_history(history_id: str) -> bool:
    """Delete one saved analysis and its retained PDF."""
    with _HISTORY_LOCK, history_database_connection() as connection:
        row = connection.execute(
            "SELECT pdf_path FROM paper_history WHERE id = ?",
            (history_id,),
        ).fetchone()
        if row is None:
            return False
        comparison_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'comparison_papers'"
        ).fetchone()
        if comparison_table is not None:
            connection.execute(
                """
                DELETE FROM comparison_workspaces
                WHERE id IN (
                    SELECT comparison_id
                    FROM comparison_papers
                    WHERE paper_history_id = ?
                )
                """,
                (history_id,),
            )
        connection.execute("DELETE FROM paper_history WHERE id = ?", (history_id,))
        pdf_path = Path(str(row["pdf_path"]))
    try:
        pdf_path.unlink(missing_ok=True)
    except OSError:
        pass
    return True


def _connect() -> sqlite3.Connection:
    database_path = _database_path()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 15000")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_history (
            id TEXT PRIMARY KEY,
            sha256 TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            filename TEXT NOT NULL,
            mode TEXT NOT NULL,
            pages INTEGER NOT NULL DEFAULT 0,
            sections_count INTEGER NOT NULL DEFAULT 0,
            size_bytes INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            result_json TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            pdf_path TEXT NOT NULL
        )
        """
    )
    return connection


@contextmanager
def history_database_connection() -> Iterator[sqlite3.Connection]:
    """Yield a transaction-scoped connection and always release its file handle."""
    connection = _connect()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _database_path() -> Path:
    configured = os.environ.get("PAPER_HISTORY_DB")
    if configured:
        return Path(configured).expanduser().resolve()
    return _data_directory() / "history.sqlite3"


def _data_directory() -> Path:
    configured = os.environ.get("PAPER_READER_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return ROOT / ".paper-reader"


def _paper_directory() -> Path:
    directory = _database_path().parent / "papers"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write_pdf_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(".pdf.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _history_row(row: sqlite3.Row) -> dict[str, Any]:
    pdf_path = Path(str(row["pdf_path"]))
    return {
        "id": str(row["id"]),
        "title": str(row["title"]),
        "filename": str(row["filename"]),
        "mode": str(row["mode"]),
        "pages": int(row["pages"]),
        "sections_count": int(row["sections_count"]),
        "size_bytes": int(row["size_bytes"]),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
        "pdf_available": pdf_path.is_file(),
    }


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
