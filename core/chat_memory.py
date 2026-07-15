"""Persistent paper conversations with LangMem-managed long-term memory."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from langmem import create_memory_store_manager
from pydantic import BaseModel, Field

from core.conversation_titles import generate_conversation_title, local_conversation_title
from core.history import history_database_connection
from core.langmem_store import (
    PaperReaderMemory,
    ensure_langmem_schema,
    get_langmem_store,
    list_langmem_memories,
)
from utils.llm import get_chat_llm, get_chat_llm_for_route


RECENT_MESSAGE_COUNT = 12
MAX_RECALLED_TOPICS = 3

LOGGER = logging.getLogger(__name__)

_MEMORY_REFRESH_LOCK = threading.RLock()
_MEMORY_REFRESHING: set[str] = set()
_MEMORY_PENDING: set[str] = set()
_MEMORY_PENDING_ROUTE: dict[str, tuple[str, str, str]] = {}
_MEMORY_THREADS: dict[str, threading.Thread] = {}

_LANGMEM_INSTRUCTIONS = """
Maintain concise long-term memory for a research-paper reading assistant.

Store only information that will materially improve future conversations about this paper:
- user: stable role, expertise, goals, or responsibilities;
- feedback: confirmed preferences and corrections about how answers should be produced;
- project: non-derivable decisions, goals, deadlines, or ongoing reproduction context;
- reference: stable locations where current external information should be checked.

Never store credentials, secrets, transient UI state, temporary navigation, or facts already recoverable from the paper.
An explicit request to remember durable information must create or update a memory. An explicit correction must replace
the contradicted memory instead of adding a duplicate. An explicit request to forget must delete the matching memory.
Keep each memory self-contained, factual, and useful in isolation. Use category, subject, content, and context exactly
as defined by the provided schema. It is correct to make no change when the exchange contains nothing durable.
""".strip()


class ConversationCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=80)


class ConversationUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=80)


@dataclass(frozen=True)
class PromptMemory:
    """Recent turns plus semantically selected LangMem records for one request."""

    recent_messages: tuple[dict[str, Any], ...]
    recalled_messages: tuple[dict[str, Any], ...]
    memory_index: str
    memory_summary: str
    recalled_topics: tuple[dict[str, Any], ...]
    total_messages: int
    memory_message_count: int


def create_conversation(history_id: str, *, title: str | None = None) -> dict[str, Any]:
    """Create an independent persisted conversation for one paper."""
    conversation_id = uuid.uuid4().hex
    now = _utc_now()
    clean_title = _clean_title(title) or "新对话"
    with history_database_connection() as connection:
        _ensure_schema(connection)
        paper = connection.execute("SELECT id FROM paper_history WHERE id = ?", (history_id,)).fetchone()
        if paper is None:
            raise KeyError("Saved paper analysis was not found.")
        connection.execute(
            """
            INSERT INTO chat_conversations (
                id, paper_history_id, title, memory_summary,
                memory_message_count, auto_memory_message_count,
                session_memory_token_count, session_context_token_count,
                created_at, updated_at
            ) VALUES (?, ?, ?, '', 0, 0, 0, 0, ?, ?)
            """,
            (conversation_id, history_id, clean_title, now, now),
        )
    return get_conversation_summary(conversation_id)


def list_conversations(history_id: str) -> list[dict[str, Any]]:
    """List paper conversations by recent activity."""
    with history_database_connection() as connection:
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT c.*, COUNT(m.id) AS message_count, MAX(m.created_at) AS last_message_at
            FROM chat_conversations AS c
            LEFT JOIN chat_messages AS m ON m.conversation_id = c.id
            WHERE c.paper_history_id = ?
            GROUP BY c.id
            ORDER BY c.updated_at DESC, c.created_at DESC
            """,
            (history_id,),
        ).fetchall()
    return [_conversation_row(row) for row in rows]


def get_conversation_summary(conversation_id: str) -> dict[str, Any]:
    """Return conversation metadata without loading message bodies."""
    with history_database_connection() as connection:
        _ensure_schema(connection)
        row = connection.execute(
            """
            SELECT c.*, COUNT(m.id) AS message_count, MAX(m.created_at) AS last_message_at
            FROM chat_conversations AS c
            LEFT JOIN chat_messages AS m ON m.conversation_id = c.id
            WHERE c.id = ?
            GROUP BY c.id
            """,
            (conversation_id,),
        ).fetchone()
    if row is None:
        raise KeyError("Conversation was not found.")
    return _conversation_row(row)


def load_conversation(conversation_id: str) -> dict[str, Any]:
    """Load all immutable original messages for display and restart recovery."""
    conversation = get_conversation_summary(conversation_id)
    with history_database_connection() as connection:
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT id, role, content, quote, sequence, created_at, model_trace_json
            FROM chat_messages
            WHERE conversation_id = ?
            ORDER BY sequence
            """,
            (conversation_id,),
        ).fetchall()
    return {"conversation": conversation, "messages": [_message_row(row) for row in rows]}


def rename_conversation(conversation_id: str, title: str) -> dict[str, Any]:
    clean_title = _clean_title(title)
    if not clean_title:
        raise ValueError("Conversation title cannot be empty.")
    with history_database_connection() as connection:
        _ensure_schema(connection)
        cursor = connection.execute(
            "UPDATE chat_conversations SET title = ?, updated_at = ? WHERE id = ?",
            (clean_title, _utc_now(), conversation_id),
        )
        if cursor.rowcount == 0:
            raise KeyError("Conversation was not found.")
    return get_conversation_summary(conversation_id)


def schedule_conversation_title(
    conversation_id: str,
    question: str,
    *,
    expected_title: str,
) -> bool:
    """Refine an automatic first-question title without blocking answer streaming."""
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
                UPDATE chat_conversations SET title = ?, updated_at = ?
                WHERE id = ? AND title = ?
                """,
                (generated, _utc_now(), conversation_id, expected_title),
            )

    threading.Thread(
        target=run,
        name=f"paper-chat-title-{conversation_id[:8]}",
        daemon=True,
    ).start()
    return True


def delete_conversation(conversation_id: str) -> bool:
    """Delete one conversation and its messages; paper-level LangMem remains shared."""
    with history_database_connection() as connection:
        _ensure_schema(connection)
        cursor = connection.execute("DELETE FROM chat_conversations WHERE id = ?", (conversation_id,))
    return cursor.rowcount > 0


def add_conversation_message(
    conversation_id: str,
    *,
    role: Literal["user", "assistant"],
    content: str,
    quote: str | None = None,
    model_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one immutable original message."""
    clean_content = content.strip()
    if not clean_content:
        raise ValueError("Message content cannot be empty.")
    message_id = uuid.uuid4().hex
    now = _utc_now()
    with history_database_connection() as connection:
        _ensure_schema(connection)
        conversation = connection.execute(
            "SELECT title FROM chat_conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if conversation is None:
            raise KeyError("Conversation was not found.")
        row = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence FROM chat_messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        sequence = int(row["next_sequence"])
        connection.execute(
            """
            INSERT INTO chat_messages (
                id, conversation_id, role, content, quote, sequence, created_at, model_trace_json
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
                json.dumps(model_trace, ensure_ascii=False, separators=(",", ":")) if model_trace else None,
            ),
        )
        title = str(conversation["title"])
        title_generation_eligible = False
        if role == "user" and sequence == 1 and title == "新对话":
            title = local_conversation_title(clean_content)
            title_generation_eligible = True
        connection.execute(
            "UPDATE chat_conversations SET title = ?, updated_at = ? WHERE id = ?",
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


def get_prompt_memory(
    conversation_id: str,
    question: str,
    *,
    recent_count: int = RECENT_MESSAGE_COUNT,
    recalled_message_limit: int = 0,
    recalled_topic_limit: int = MAX_RECALLED_TOPICS,
) -> PromptMemory:
    """Load recent messages and direct local-vector LangMem recall.

    Old raw messages are intentionally not reintroduced as long-term memory. This
    prevents an explicitly forgotten fact from resurfacing through message search.
    """
    del recalled_message_limit
    with history_database_connection() as connection:
        _ensure_schema(connection)
        conversation = connection.execute(
            "SELECT paper_history_id, auto_memory_message_count FROM chat_conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if conversation is None:
            raise KeyError("Conversation was not found.")
        recent_rows = connection.execute(
            """
            SELECT id, role, content, quote, sequence, created_at, model_trace_json
            FROM chat_messages
            WHERE conversation_id = ?
            ORDER BY sequence DESC
            LIMIT ?
            """,
            (conversation_id, max(2, recent_count)),
        ).fetchall()
        recent_rows = list(reversed(recent_rows))
        count_row = connection.execute(
            "SELECT COUNT(*) AS count FROM chat_messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()

    ignore_memory = _should_ignore_memory(question)
    recalled_topics = [] if ignore_memory else list_langmem_memories(
        str(conversation["paper_history_id"]),
        query=question,
        limit=recalled_topic_limit,
    )
    return PromptMemory(
        recent_messages=tuple(_message_row(row) for row in recent_rows),
        recalled_messages=(),
        memory_index="",
        memory_summary="",
        recalled_topics=tuple(recalled_topics),
        total_messages=int(count_row["count"]),
        memory_message_count=int(conversation["auto_memory_message_count"]),
    )


def memory_refresh_needed(conversation_id: str) -> bool:
    return bool(_messages_for_langmem(conversation_id))


def refresh_conversation_memory(
    conversation_id: str,
    *,
    force: bool = False,
    text_provider: str | None = None,
    text_model: str | None = None,
    text_mode: str | None = None,
    **_: Any,
) -> int:
    """Run one LangMem background enrichment over unprocessed complete turns."""
    del force
    batch = _messages_for_langmem(conversation_id)
    if not batch:
        return 0
    with history_database_connection() as connection:
        _ensure_schema(connection)
        conversation = connection.execute(
            "SELECT paper_history_id FROM chat_conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
    if conversation is None:
        return 0
    history_id = str(conversation["paper_history_id"])
    llm = (
        get_chat_llm_for_route(text_provider, text_model, text_mode or "")
        if text_provider and text_model
        else get_chat_llm()
    )
    manager = create_memory_store_manager(
        llm,
        schemas=[PaperReaderMemory],
        instructions=_LANGMEM_INSTRUCTIONS,
        enable_inserts=True,
        enable_deletes=True,
        query_limit=6,
        namespace=("paper-reader", "{paper_history_id}"),
        store=get_langmem_store(),
    )
    config = {"configurable": {"paper_history_id": history_id}}
    messages = [
        {"role": item["role"], "content": item["content"]}
        for item in batch
    ]
    try:
        manager.invoke({"messages": messages}, config=config)
    except Exception as exc:  # keep the cursor unchanged so the next turn retries
        LOGGER.warning("LangMem update failed for conversation %s: %s", conversation_id, exc)
        return 0
    end_sequence = int(batch[-1]["sequence"])
    with history_database_connection() as connection:
        _ensure_schema(connection)
        connection.execute(
            """
            UPDATE chat_conversations
            SET auto_memory_message_count = ?, memory_message_count = ?, updated_at = ?
            WHERE id = ?
            """,
            (end_sequence, end_sequence, _utc_now(), conversation_id),
        )
    return len(batch)


def extract_auto_memory(conversation_id: str, **kwargs: Any) -> int:
    """Compatibility alias for callers migrating from the former memory layer."""
    return refresh_conversation_memory(conversation_id, **kwargs)


def schedule_memory_refresh(
    conversation_id: str,
    *,
    context_token_count: int | None = None,
    text_provider: str | None = None,
    text_model: str | None = None,
    text_mode: str | None = None,
) -> bool:
    """Coalesce LangMem background updates after complete assistant turns."""
    if context_token_count and context_token_count > 0:
        with history_database_connection() as connection:
            _ensure_schema(connection)
            connection.execute(
                "UPDATE chat_conversations SET session_context_token_count = ? WHERE id = ?",
                (int(context_token_count), conversation_id),
            )
    route = (
        (text_provider, text_model, text_mode or "")
        if text_provider and text_model
        else None
    )
    with _MEMORY_REFRESH_LOCK:
        if conversation_id in _MEMORY_REFRESHING:
            _MEMORY_PENDING.add(conversation_id)
            if route:
                _MEMORY_PENDING_ROUTE[conversation_id] = route
            return True
        _MEMORY_REFRESHING.add(conversation_id)

    def run() -> None:
        restart = False
        current_route = route
        try:
            while True:
                refresh_conversation_memory(
                    conversation_id,
                    text_provider=current_route[0] if current_route else None,
                    text_model=current_route[1] if current_route else None,
                    text_mode=current_route[2] if current_route else None,
                )
                with _MEMORY_REFRESH_LOCK:
                    if conversation_id not in _MEMORY_PENDING:
                        break
                    _MEMORY_PENDING.discard(conversation_id)
                    current_route = _MEMORY_PENDING_ROUTE.pop(conversation_id, current_route)
        finally:
            with _MEMORY_REFRESH_LOCK:
                restart = conversation_id in _MEMORY_PENDING
                _MEMORY_PENDING.discard(conversation_id)
                restart_route = _MEMORY_PENDING_ROUTE.pop(conversation_id, None)
                _MEMORY_REFRESHING.discard(conversation_id)
                _MEMORY_THREADS.pop(conversation_id, None)
            if restart:
                schedule_memory_refresh(
                    conversation_id,
                    text_provider=restart_route[0] if restart_route else None,
                    text_model=restart_route[1] if restart_route else None,
                    text_mode=restart_route[2] if restart_route else None,
                )

    worker = threading.Thread(
        target=run,
        name=f"paper-chat-langmem-{conversation_id[:8]}",
        daemon=True,
    )
    with _MEMORY_REFRESH_LOCK:
        _MEMORY_THREADS[conversation_id] = worker
    worker.start()
    return True


def drain_memory_refreshes(timeout: float = 60.0) -> None:
    """Wait softly for in-flight LangMem updates during shutdown or tests."""
    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline:
        with _MEMORY_REFRESH_LOCK:
            workers = [worker for worker in _MEMORY_THREADS.values() if worker.is_alive()]
        if not workers:
            return
        for worker in workers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            worker.join(timeout=min(remaining, 0.25))


def _messages_for_langmem(conversation_id: str) -> list[dict[str, Any]]:
    with history_database_connection() as connection:
        _ensure_schema(connection)
        conversation = connection.execute(
            "SELECT auto_memory_message_count FROM chat_conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if conversation is None:
            return []
        rows = connection.execute(
            """
            SELECT id, role, content, quote, sequence, created_at, model_trace_json
            FROM chat_messages
            WHERE conversation_id = ? AND sequence > ?
            ORDER BY sequence
            LIMIT 2000
            """,
            (conversation_id, int(conversation["auto_memory_message_count"])),
        ).fetchall()
    messages = [_message_row(row) for row in rows]
    if messages and messages[-1]["role"] == "user":
        messages.pop()
    return messages


def _should_ignore_memory(question: str) -> bool:
    lowered = question.casefold()
    english = re.search(r"\b(ignore|do not use|don't use|without using)\b.{0,24}\b(memory|memories)\b", lowered)
    chinese = re.search(r"(?:忽略|不要使用|别用|不使用|清空)(?:.{0,12})(?:记忆|历史记忆|长期记忆)", question)
    return bool(english or chinese)


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS chat_conversations (
            id TEXT PRIMARY KEY,
            paper_history_id TEXT NOT NULL,
            title TEXT NOT NULL,
            memory_summary TEXT NOT NULL DEFAULT '',
            memory_message_count INTEGER NOT NULL DEFAULT 0,
            auto_memory_message_count INTEGER NOT NULL DEFAULT 0,
            session_memory_token_count INTEGER NOT NULL DEFAULT 0,
            session_context_token_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (paper_history_id) REFERENCES paper_history(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS chat_messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            quote TEXT,
            sequence INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            model_trace_json TEXT,
            FOREIGN KEY (conversation_id) REFERENCES chat_conversations(id) ON DELETE CASCADE,
            UNIQUE (conversation_id, sequence)
        );

        CREATE TABLE IF NOT EXISTS chat_memories (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            topic TEXT NOT NULL,
            content TEXT NOT NULL,
            source_start_sequence INTEGER NOT NULL,
            source_end_sequence INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES chat_conversations(id) ON DELETE CASCADE,
            UNIQUE (conversation_id, topic)
        );

        CREATE INDEX IF NOT EXISTS idx_chat_conversations_paper_updated
        ON chat_conversations(paper_history_id, updated_at DESC);

        CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_sequence
        ON chat_messages(conversation_id, sequence);
        """
    )
    columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(chat_messages)")}
    if "model_trace_json" not in columns:
        connection.execute("ALTER TABLE chat_messages ADD COLUMN model_trace_json TEXT")
    conversation_columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(chat_conversations)")
    }
    additions = {
        "auto_memory_message_count": "INTEGER NOT NULL DEFAULT 0",
        "session_memory_token_count": "INTEGER NOT NULL DEFAULT 0",
        "session_context_token_count": "INTEGER NOT NULL DEFAULT 0",
    }
    for name, definition in additions.items():
        if name not in conversation_columns:
            connection.execute(f"ALTER TABLE chat_conversations ADD COLUMN {name} {definition}")
    ensure_langmem_schema(connection)


def _conversation_row(row: sqlite3.Row) -> dict[str, Any]:
    cursor = int(row["auto_memory_message_count"])
    return {
        "id": str(row["id"]),
        "history_id": str(row["paper_history_id"]),
        "title": str(row["title"]),
        "message_count": int(row["message_count"]),
        "memory_message_count": cursor,
        "memory_ready": cursor > 0,
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
        "last_message_at": str(row["last_message_at"] or row["updated_at"]),
    }


def _message_row(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
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


def _clean_title(value: str | None) -> str:
    return " ".join((value or "").split())[:80]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
