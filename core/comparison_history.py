"""SQLite persistence for comparison workspaces and cross-paper conversations."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from core.comparison import ComparisonCreateRequest, ComparisonSource
from core.conversation_titles import generate_conversation_title, local_conversation_title
from core.history import history_database_connection
from core.semantic_search import semantic_scores


class ComparisonConversationCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=80)


class ComparisonConversationUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=80)


@dataclass(frozen=True)
class ComparisonPromptMemory:
    recent_messages: tuple[dict[str, Any], ...]
    recalled_messages: tuple[dict[str, Any], ...]
    total_messages: int


def save_comparison(
    *,
    result: dict[str, Any],
    sources: list[ComparisonSource],
    request: ComparisonCreateRequest,
) -> str:
    comparison_id = uuid.uuid4().hex
    now = _utc_now()
    title = _clean_title(str(result.get("comparison", {}).get("title") or "多论文对比"))
    with history_database_connection() as connection:
        _ensure_schema(connection)
        connection.execute(
            """
            INSERT INTO comparison_workspaces (
                id, title, focus, custom_focus, result_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                comparison_id,
                title or "多论文对比",
                request.focus,
                request.custom_focus,
                json.dumps(result, ensure_ascii=False, separators=(",", ":")),
                now,
                now,
            ),
        )
        for index, source in enumerate(sources, start=1):
            connection.execute(
                """
                INSERT INTO comparison_papers (
                    comparison_id, paper_history_id, paper_order, label
                ) VALUES (?, ?, ?, ?)
                """,
                (comparison_id, source.history_id, index, source.label),
            )
    return comparison_id


def list_comparisons(*, limit: int = 100) -> list[dict[str, Any]]:
    bounded = max(1, min(limit, 500))
    with history_database_connection() as connection:
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT w.*, COUNT(p.paper_history_id) AS paper_count
            FROM comparison_workspaces w
            LEFT JOIN comparison_papers p ON p.comparison_id = w.id
            GROUP BY w.id
            ORDER BY w.updated_at DESC
            LIMIT ?
            """,
            (bounded,),
        ).fetchall()
        items = [_workspace_row(row) for row in rows]
        for item in items:
            item["papers"] = _workspace_papers(connection, item["id"])
    return items


def load_comparison(comparison_id: str) -> dict[str, Any] | None:
    with history_database_connection() as connection:
        _ensure_schema(connection)
        row = connection.execute(
            """
            SELECT w.*, COUNT(p.paper_history_id) AS paper_count
            FROM comparison_workspaces w
            LEFT JOIN comparison_papers p ON p.comparison_id = w.id
            WHERE w.id = ?
            GROUP BY w.id
            """,
            (comparison_id,),
        ).fetchone()
        if row is None:
            return None
        papers = _workspace_papers(connection, comparison_id)
    return {
        "workspace": _workspace_row(row),
        "papers": papers,
        "result": json.loads(str(row["result_json"])),
    }


def comparison_exists(comparison_id: str) -> bool:
    with history_database_connection() as connection:
        _ensure_schema(connection)
        row = connection.execute(
            "SELECT 1 FROM comparison_workspaces WHERE id = ?",
            (comparison_id,),
        ).fetchone()
    return row is not None


def delete_comparison(comparison_id: str) -> bool:
    with history_database_connection() as connection:
        _ensure_schema(connection)
        cursor = connection.execute(
            "DELETE FROM comparison_workspaces WHERE id = ?",
            (comparison_id,),
        )
    return cursor.rowcount > 0


def create_comparison_conversation(
    comparison_id: str,
    *,
    title: str | None = None,
) -> dict[str, Any]:
    conversation_id = uuid.uuid4().hex
    now = _utc_now()
    with history_database_connection() as connection:
        _ensure_schema(connection)
        exists = connection.execute(
            "SELECT 1 FROM comparison_workspaces WHERE id = ?",
            (comparison_id,),
        ).fetchone()
        if exists is None:
            raise KeyError("Comparison workspace was not found.")
        connection.execute(
            """
            INSERT INTO comparison_chat_conversations (
                id, comparison_id, title, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, comparison_id, _clean_title(title) or "新对话", now, now),
        )
    return get_comparison_conversation_summary(conversation_id)


def list_comparison_conversations(comparison_id: str) -> list[dict[str, Any]]:
    with history_database_connection() as connection:
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT c.*, COUNT(m.id) AS message_count,
                   MAX(m.created_at) AS last_message_at
            FROM comparison_chat_conversations c
            LEFT JOIN comparison_chat_messages m ON m.conversation_id = c.id
            WHERE c.comparison_id = ?
            GROUP BY c.id
            ORDER BY c.updated_at DESC, c.created_at DESC
            """,
            (comparison_id,),
        ).fetchall()
    return [_conversation_row(row) for row in rows]


def get_comparison_conversation_summary(conversation_id: str) -> dict[str, Any]:
    with history_database_connection() as connection:
        _ensure_schema(connection)
        row = connection.execute(
            """
            SELECT c.*, COUNT(m.id) AS message_count,
                   MAX(m.created_at) AS last_message_at
            FROM comparison_chat_conversations c
            LEFT JOIN comparison_chat_messages m ON m.conversation_id = c.id
            WHERE c.id = ?
            GROUP BY c.id
            """,
            (conversation_id,),
        ).fetchone()
    if row is None:
        raise KeyError("Conversation was not found.")
    return _conversation_row(row)


def load_comparison_conversation(conversation_id: str) -> dict[str, Any]:
    conversation = get_comparison_conversation_summary(conversation_id)
    with history_database_connection() as connection:
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT id, role, content, quote, sequence, created_at, model_trace_json
            FROM comparison_chat_messages
            WHERE conversation_id = ?
            ORDER BY sequence ASC
            """,
            (conversation_id,),
        ).fetchall()
    return {
        "conversation": conversation,
        "messages": [_message_row(row) for row in rows],
    }


def rename_comparison_conversation(conversation_id: str, title: str) -> dict[str, Any]:
    clean_title = _clean_title(title)
    if not clean_title:
        raise ValueError("Conversation title cannot be empty.")
    with history_database_connection() as connection:
        _ensure_schema(connection)
        cursor = connection.execute(
            "UPDATE comparison_chat_conversations SET title = ?, updated_at = ? WHERE id = ?",
            (clean_title, _utc_now(), conversation_id),
        )
    if cursor.rowcount == 0:
        raise KeyError("Conversation was not found.")
    return get_comparison_conversation_summary(conversation_id)


def schedule_comparison_conversation_title(
    conversation_id: str,
    question: str,
    *,
    expected_title: str,
) -> bool:
    """Refine an automatic comparison-chat title in a background model call."""
    if not expected_title:
        return False

    def run() -> None:
        generated = generate_conversation_title(question)
        if not generated or generated == expected_title:
            return
        with history_database_connection() as connection:
            _ensure_schema(connection)
            connection.execute(
                """
                UPDATE comparison_chat_conversations
                SET title = ?, updated_at = ?
                WHERE id = ? AND title = ?
                """,
                (generated, _utc_now(), conversation_id, expected_title),
            )

    threading.Thread(
        target=run,
        name=f"comparison-chat-title-{conversation_id[:8]}",
        daemon=True,
    ).start()
    return True


def delete_comparison_conversation(conversation_id: str) -> bool:
    with history_database_connection() as connection:
        _ensure_schema(connection)
        cursor = connection.execute(
            "DELETE FROM comparison_chat_conversations WHERE id = ?",
            (conversation_id,),
        )
    return cursor.rowcount > 0


def add_comparison_message(
    conversation_id: str,
    *,
    role: Literal["user", "assistant"],
    content: str,
    quote: str | None = None,
    model_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    clean_content = content.strip()
    if not clean_content:
        raise ValueError("Message content cannot be empty.")
    message_id = uuid.uuid4().hex
    now = _utc_now()
    with history_database_connection() as connection:
        _ensure_schema(connection)
        conversation = connection.execute(
            "SELECT title FROM comparison_chat_conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if conversation is None:
            raise KeyError("Conversation was not found.")
        row = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence FROM comparison_chat_messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        sequence = int(row["next_sequence"])
        connection.execute(
            """
            INSERT INTO comparison_chat_messages (
                id, conversation_id, role, content, quote, sequence, created_at,
                model_trace_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                conversation_id,
                role,
                clean_content,
                quote or None,
                sequence,
                now,
                json.dumps(model_trace, ensure_ascii=False, separators=(",", ":"))
                if model_trace
                else None,
            ),
        )
        title = str(conversation["title"])
        title_generation_eligible = False
        if role == "user" and sequence == 1 and title == "新对话":
            title = local_conversation_title(clean_content)
            title_generation_eligible = True
        connection.execute(
            "UPDATE comparison_chat_conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, conversation_id),
        )
    return {
        "id": message_id,
        "role": role,
        "content": clean_content,
        "quote": quote or None,
        "sequence": sequence,
        "created_at": now,
        "model_trace": dict(model_trace) if model_trace else None,
        "title_generation_eligible": title_generation_eligible,
        "provisional_title": title if title_generation_eligible else None,
    }


def get_comparison_prompt_memory(
    conversation_id: str,
    query: str,
    *,
    recent_count: int = 16,
    recalled_limit: int = 6,
) -> ComparisonPromptMemory:
    """Return recent turns plus query-relevant older messages without a round cap."""
    with history_database_connection() as connection:
        _ensure_schema(connection)
        exists = connection.execute(
            "SELECT 1 FROM comparison_chat_conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if exists is None:
            raise KeyError("Conversation was not found.")
        recent_rows = connection.execute(
            """
            SELECT id, role, content, quote, sequence, created_at
            FROM comparison_chat_messages
            WHERE conversation_id = ?
            ORDER BY sequence DESC
            LIMIT ?
            """,
            (conversation_id, max(2, recent_count)),
        ).fetchall()
        recent_rows = list(reversed(recent_rows))
        oldest_recent = int(recent_rows[0]["sequence"]) if recent_rows else 2**31
        old_rows = connection.execute(
            """
            SELECT id, role, content, quote, sequence, created_at
            FROM comparison_chat_messages
            WHERE conversation_id = ? AND sequence < ?
            ORDER BY sequence DESC
            LIMIT 2000
            """,
            (conversation_id, oldest_recent),
        ).fetchall()
        count_row = connection.execute(
            "SELECT COUNT(*) AS count FROM comparison_chat_messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
    recalled = _rank_messages(old_rows, query, recalled_limit)
    return ComparisonPromptMemory(
        recent_messages=tuple(_message_row(row) for row in recent_rows),
        recalled_messages=tuple(_message_row(row) for row in recalled),
        total_messages=int(count_row["count"]),
    )


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS comparison_workspaces (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            focus TEXT NOT NULL,
            custom_focus TEXT,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS comparison_papers (
            comparison_id TEXT NOT NULL,
            paper_history_id TEXT NOT NULL,
            paper_order INTEGER NOT NULL,
            label TEXT NOT NULL,
            PRIMARY KEY (comparison_id, paper_history_id),
            UNIQUE (comparison_id, paper_order),
            FOREIGN KEY (comparison_id) REFERENCES comparison_workspaces(id) ON DELETE CASCADE,
            FOREIGN KEY (paper_history_id) REFERENCES paper_history(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS comparison_chat_conversations (
            id TEXT PRIMARY KEY,
            comparison_id TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (comparison_id) REFERENCES comparison_workspaces(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS comparison_chat_messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            quote TEXT,
            sequence INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            model_trace_json TEXT,
            FOREIGN KEY (conversation_id) REFERENCES comparison_chat_conversations(id) ON DELETE CASCADE,
            UNIQUE (conversation_id, sequence)
        );

        CREATE INDEX IF NOT EXISTS idx_comparison_workspaces_updated
        ON comparison_workspaces(updated_at DESC);

        CREATE INDEX IF NOT EXISTS idx_comparison_papers_workspace
        ON comparison_papers(comparison_id, paper_order);

        CREATE INDEX IF NOT EXISTS idx_comparison_chat_workspace
        ON comparison_chat_conversations(comparison_id, updated_at DESC);

        CREATE INDEX IF NOT EXISTS idx_comparison_chat_messages_sequence
        ON comparison_chat_messages(conversation_id, sequence);
        """
    )
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(comparison_chat_messages)").fetchall()
    }
    if "model_trace_json" not in columns:
        connection.execute(
            "ALTER TABLE comparison_chat_messages ADD COLUMN model_trace_json TEXT"
        )


def _workspace_papers(connection: sqlite3.Connection, comparison_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT p.label, p.paper_order, p.paper_history_id,
               h.title, h.filename, h.pages
        FROM comparison_papers p
        LEFT JOIN paper_history h ON h.id = p.paper_history_id
        WHERE p.comparison_id = ?
        ORDER BY p.paper_order
        """,
        (comparison_id,),
    ).fetchall()
    return [
        {
            "label": str(row["label"]),
            "history_id": str(row["paper_history_id"]),
            "title": str(row["title"] or "已删除论文"),
            "filename": str(row["filename"] or ""),
            "pages": int(row["pages"] or 0),
        }
        for row in rows
    ]


def _workspace_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "title": str(row["title"]),
        "focus": str(row["focus"]),
        "custom_focus": str(row["custom_focus"]) if row["custom_focus"] else None,
        "paper_count": int(row["paper_count"]),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
    }


def _conversation_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "comparison_id": str(row["comparison_id"]),
        "title": str(row["title"]),
        "message_count": int(row["message_count"]),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
        "last_message_at": str(row["last_message_at"] or row["updated_at"]),
    }


def _message_row(row: sqlite3.Row) -> dict[str, Any]:
    raw_trace = row["model_trace_json"] if "model_trace_json" in row.keys() else None
    model_trace = None
    if raw_trace:
        try:
            parsed_trace = json.loads(str(raw_trace))
            if isinstance(parsed_trace, dict):
                model_trace = parsed_trace
        except (TypeError, ValueError, json.JSONDecodeError):
            model_trace = None
    return {
        "id": str(row["id"]),
        "role": str(row["role"]),
        "content": str(row["content"]),
        "quote": str(row["quote"]) if row["quote"] else None,
        "sequence": int(row["sequence"]),
        "created_at": str(row["created_at"]),
        "model_trace": model_trace,
    }


def _rank_messages(rows: list[sqlite3.Row], query: str, limit: int) -> list[sqlite3.Row]:
    if not query.strip() or limit <= 0:
        return []
    terms = _memory_terms(query)
    similarities = semantic_scores(
        query,
        [f"{row['content']} {row['quote'] or ''}" for row in rows],
    )
    scored: list[tuple[float, int, sqlite3.Row]] = []
    for index, row in enumerate(rows):
        text = f"{row['content']} {row['quote'] or ''}".lower()
        if similarities is None:
            score = sum(min(text.count(term), 4) * (3.0 if len(term) > 2 else 1.0) for term in terms)
        else:
            score = similarities[index]
        if similarities is not None or score > 0:
            scored.append((score, int(row["sequence"]), row))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [item[2] for item in scored[:limit]]
    selected.sort(key=lambda row: int(row["sequence"]))
    return selected


def _memory_terms(text: str) -> set[str]:
    lowered = text.lower()
    terms = set(re.findall(r"[a-z][a-z0-9_-]{1,}|P[1-4]:[ETF]\d{3}", lowered, re.I))
    for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        terms.add(sequence)
        terms.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return {term.lower() for term in terms if len(term) >= 2}


def _clean_title(value: str | None) -> str:
    return " ".join((value or "").split())[:80]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
