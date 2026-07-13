"""Persistent multi-conversation memory for paper follow-up chat."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core.conversation_titles import generate_conversation_title, local_conversation_title
from core.history import history_database_connection
from utils.llm import get_chat_llm, invoke_with_retry, parse_structured_output


RECENT_MESSAGE_COUNT = 12
MEMORY_COMPACTION_BATCH = 12
MAX_RECALLED_MESSAGES = 4
MAX_RECALLED_TOPICS = 3
MAX_RETRIEVAL_CANDIDATES = 2_000

_MEMORY_REFRESH_LOCK = threading.RLock()
_MEMORY_REFRESHING: set[str] = set()


class MemoryTopic(BaseModel):
    """One detailed topic file in the Claude Code-inspired memory layout."""

    topic: str = Field(min_length=1, max_length=80)
    content: str = Field(min_length=1, max_length=8_000)


class ConversationMemoryDigest(BaseModel):
    """Compact memory index plus detailed topic memories."""

    summary: str = Field(min_length=1, max_length=6_000)
    topics: list[MemoryTopic] = Field(default_factory=list, max_length=8)


class ConversationCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=80)


class ConversationUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=80)


@dataclass(frozen=True)
class PromptMemory:
    """Conversation context selected for one model request."""

    recent_messages: tuple[dict[str, Any], ...]
    recalled_messages: tuple[dict[str, Any], ...]
    memory_summary: str
    recalled_topics: tuple[dict[str, Any], ...]
    total_messages: int
    memory_message_count: int


MemorySummarizer = Callable[
    [str, list[dict[str, Any]], list[dict[str, Any]]],
    ConversationMemoryDigest,
]


def create_conversation(history_id: str, *, title: str | None = None) -> dict[str, Any]:
    """Create an independent chat conversation for one saved paper."""
    conversation_id = uuid.uuid4().hex
    now = _utc_now()
    clean_title = _clean_title(title) or "新对话"
    with history_database_connection() as connection:
        _ensure_schema(connection)
        paper = connection.execute(
            "SELECT id FROM paper_history WHERE id = ?",
            (history_id,),
        ).fetchone()
        if paper is None:
            raise KeyError("Saved paper analysis was not found.")
        connection.execute(
            """
            INSERT INTO chat_conversations (
                id, paper_history_id, title, memory_summary,
                memory_message_count, created_at, updated_at
            ) VALUES (?, ?, ?, '', 0, ?, ?)
            """,
            (conversation_id, history_id, clean_title, now, now),
        )
    return get_conversation_summary(conversation_id)


def list_conversations(history_id: str) -> list[dict[str, Any]]:
    """List all conversations for a paper, newest activity first."""
    with history_database_connection() as connection:
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT c.*,
                   COUNT(m.id) AS message_count,
                   MAX(m.created_at) AS last_message_at
            FROM chat_conversations c
            LEFT JOIN chat_messages m ON m.conversation_id = c.id
            WHERE c.paper_history_id = ?
            GROUP BY c.id
            ORDER BY c.updated_at DESC, c.created_at DESC
            """,
            (history_id,),
        ).fetchall()
    return [_conversation_row(row) for row in rows]


def get_conversation_summary(conversation_id: str) -> dict[str, Any]:
    """Return conversation metadata without loading all message bodies."""
    with history_database_connection() as connection:
        _ensure_schema(connection)
        row = connection.execute(
            """
            SELECT c.*,
                   COUNT(m.id) AS message_count,
                   MAX(m.created_at) AS last_message_at
            FROM chat_conversations c
            LEFT JOIN chat_messages m ON m.conversation_id = c.id
            WHERE c.id = ?
            GROUP BY c.id
            """,
            (conversation_id,),
        ).fetchone()
    if row is None:
        raise KeyError("Conversation was not found.")
    return _conversation_row(row)


def load_conversation(conversation_id: str) -> dict[str, Any]:
    """Load complete persisted messages for display and restart recovery."""
    conversation = get_conversation_summary(conversation_id)
    with history_database_connection() as connection:
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT id, role, content, quote, sequence, created_at, model_trace_json
            FROM chat_messages
            WHERE conversation_id = ?
            ORDER BY sequence ASC
            """,
            (conversation_id,),
        ).fetchall()
    return {
        "conversation": conversation,
        "messages": [_message_row(row) for row in rows],
    }


def rename_conversation(conversation_id: str, title: str) -> dict[str, Any]:
    """Rename one conversation."""
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
    """Refine an automatic first-question title without blocking the answer stream."""
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
                UPDATE chat_conversations
                SET title = ?, updated_at = ?
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
    """Delete one conversation, its messages, and topic memories."""
    with history_database_connection() as connection:
        _ensure_schema(connection)
        cursor = connection.execute(
            "DELETE FROM chat_conversations WHERE id = ?",
            (conversation_id,),
        )
    return cursor.rowcount > 0


def add_conversation_message(
    conversation_id: str,
    *,
    role: Literal["user", "assistant"],
    content: str,
    quote: str | None = None,
    model_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one immutable original message to a conversation."""
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
    recalled_message_limit: int = MAX_RECALLED_MESSAGES,
    recalled_topic_limit: int = MAX_RECALLED_TOPICS,
) -> PromptMemory:
    """Load recent turns plus query-relevant topic and original-message memories."""
    with history_database_connection() as connection:
        _ensure_schema(connection)
        conversation = connection.execute(
            "SELECT memory_summary, memory_message_count FROM chat_conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if conversation is None:
            raise KeyError("Conversation was not found.")
        recent_rows = connection.execute(
            """
            SELECT id, role, content, quote, sequence, created_at
            FROM chat_messages
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
            FROM chat_messages
            WHERE conversation_id = ? AND sequence < ?
            ORDER BY sequence DESC
            LIMIT ?
            """,
            (conversation_id, oldest_recent, MAX_RETRIEVAL_CANDIDATES),
        ).fetchall()
        topic_rows = connection.execute(
            """
            SELECT id, topic, content, source_start_sequence, source_end_sequence, updated_at
            FROM chat_memories
            WHERE conversation_id = ?
            ORDER BY updated_at DESC
            LIMIT 200
            """,
            (conversation_id,),
        ).fetchall()
        count_row = connection.execute(
            "SELECT COUNT(*) AS count FROM chat_messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()

    terms = _memory_terms(question)
    recalled_messages = _rank_rows(old_rows, terms, recalled_message_limit, kind="message")
    recalled_topics = _rank_rows(topic_rows, terms, recalled_topic_limit, kind="topic")
    return PromptMemory(
        recent_messages=tuple(_message_row(row) for row in recent_rows),
        recalled_messages=tuple(_message_row(row) for row in recalled_messages),
        memory_summary=str(conversation["memory_summary"] or ""),
        recalled_topics=tuple(_topic_row(row) for row in recalled_topics),
        total_messages=int(count_row["count"]),
        memory_message_count=int(conversation["memory_message_count"]),
    )


def memory_refresh_needed(conversation_id: str) -> bool:
    """Return whether enough old messages are ready for the next compaction batch."""
    return bool(_messages_for_compaction(conversation_id))


def refresh_conversation_memory(
    conversation_id: str,
    *,
    summarizer: MemorySummarizer | None = None,
) -> int:
    """Compact eligible old messages into an index and topic memories."""
    processed = 0
    summarize = summarizer or _summarize_with_model
    while True:
        batch = _messages_for_compaction(conversation_id)
        if not batch:
            break
        with history_database_connection() as connection:
            _ensure_schema(connection)
            conversation = connection.execute(
                "SELECT memory_summary FROM chat_conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            topic_rows = connection.execute(
                "SELECT topic, content FROM chat_memories WHERE conversation_id = ? ORDER BY topic",
                (conversation_id,),
            ).fetchall()
        if conversation is None:
            break
        existing_topics = [dict(row) for row in topic_rows]
        try:
            digest = summarize(str(conversation["memory_summary"] or ""), existing_topics, batch)
        except Exception:
            digest = _fallback_digest(str(conversation["memory_summary"] or ""), batch)
        _store_digest(conversation_id, digest, batch)
        processed += len(batch)
    return processed


def schedule_memory_refresh(conversation_id: str) -> bool:
    """Refresh memory in a daemon thread so answer latency is unaffected."""
    if not memory_refresh_needed(conversation_id):
        return False
    with _MEMORY_REFRESH_LOCK:
        if conversation_id in _MEMORY_REFRESHING:
            return False
        _MEMORY_REFRESHING.add(conversation_id)

    def run() -> None:
        try:
            refresh_conversation_memory(conversation_id)
        finally:
            with _MEMORY_REFRESH_LOCK:
                _MEMORY_REFRESHING.discard(conversation_id)

    threading.Thread(
        target=run,
        name=f"paper-chat-memory-{conversation_id[:8]}",
        daemon=True,
    ).start()
    return True


def _messages_for_compaction(conversation_id: str) -> list[dict[str, Any]]:
    with history_database_connection() as connection:
        _ensure_schema(connection)
        conversation = connection.execute(
            "SELECT memory_message_count FROM chat_conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if conversation is None:
            return []
        max_row = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) AS max_sequence FROM chat_messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        compact_through = int(max_row["max_sequence"]) - RECENT_MESSAGE_COUNT
        start_after = int(conversation["memory_message_count"])
        if compact_through - start_after < MEMORY_COMPACTION_BATCH:
            return []
        rows = connection.execute(
            """
            SELECT id, role, content, quote, sequence, created_at
            FROM chat_messages
            WHERE conversation_id = ? AND sequence > ? AND sequence <= ?
            ORDER BY sequence ASC
            LIMIT ?
            """,
            (conversation_id, start_after, compact_through, MEMORY_COMPACTION_BATCH),
        ).fetchall()
    return [_message_row(row) for row in rows]


def _store_digest(
    conversation_id: str,
    digest: ConversationMemoryDigest,
    batch: list[dict[str, Any]],
) -> None:
    start_sequence = int(batch[0]["sequence"])
    end_sequence = int(batch[-1]["sequence"])
    now = _utc_now()
    with history_database_connection() as connection:
        _ensure_schema(connection)
        connection.execute(
            """
            UPDATE chat_conversations
            SET memory_summary = ?, memory_message_count = ?, updated_at = ?
            WHERE id = ?
            """,
            (digest.summary, end_sequence, now, conversation_id),
        )
        for topic in digest.topics:
            topic_name = _clean_title(topic.topic) or "历史对话"
            connection.execute(
                """
                INSERT INTO chat_memories (
                    id, conversation_id, topic, content, source_start_sequence,
                    source_end_sequence, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(conversation_id, topic) DO UPDATE SET
                    content = excluded.content,
                    source_start_sequence = MIN(chat_memories.source_start_sequence, excluded.source_start_sequence),
                    source_end_sequence = excluded.source_end_sequence,
                    updated_at = excluded.updated_at
                """,
                (
                    uuid.uuid4().hex,
                    conversation_id,
                    topic_name,
                    topic.content,
                    start_sequence,
                    end_sequence,
                    now,
                    now,
                ),
            )


def _summarize_with_model(
    existing_summary: str,
    existing_topics: list[dict[str, Any]],
    batch: list[dict[str, Any]],
) -> ConversationMemoryDigest:
    transcript = "\n".join(
        f"#{item['sequence']} {item['role']}: {item['content']}"
        for item in batch
    )
    topics = "\n\n".join(
        f"## {item['topic']}\n{item['content']}"
        for item in existing_topics
    ) or "无"
    messages = [
            SystemMessage(
                content=(
                    "你负责维护论文追问的长期记忆，工作方式类似精简 MEMORY.md 索引加按需加载的主题文件。"
                    "只保留未来问答真正需要的稳定信息：用户目标、已确认结论、重要数字与证据ID、"
                    "用户纠正、术语约定和尚未解决的问题。不得把模型猜测写成论文事实。"
                    "summary 必须简洁，作为每轮自动加载的记忆索引；topics 保存可按需召回的详细信息。"
                    "只返回一个有效 JSON 对象，不要使用 Markdown。"
                )
            ),
            HumanMessage(
                content=(
                    f"现有记忆索引：\n{existing_summary or '无'}\n\n"
                    f"现有主题记忆：\n{topics}\n\n"
                    f"需要压缩的新对话：\n{transcript}\n\n"
                    "请合并而不是丢弃仍然有效的旧信息；若用户纠正了旧结论，以新结论为准。\n"
                    "严格按以下形状返回："
                    + json.dumps(
                        {
                            "summary": "简洁的长期记忆索引",
                            "topics": [
                                {"topic": "主题名", "content": "该主题的详细长期记忆"}
                            ],
                        },
                        ensure_ascii=False,
                    )
                )
            ),
        ]
    response = invoke_with_retry(get_chat_llm(), messages, retries=1, delay=1.0)
    return parse_structured_output(response, ConversationMemoryDigest)


def _fallback_digest(
    existing_summary: str,
    batch: list[dict[str, Any]],
) -> ConversationMemoryDigest:
    lines = [existing_summary.strip()] if existing_summary.strip() else []
    lines.extend(
        f"- #{item['sequence']} {('用户' if item['role'] == 'user' else '助手')}：{item['content'][:320]}"
        for item in batch
    )
    summary = "\n".join(lines)[-5_800:]
    content = "\n".join(
        f"#{item['sequence']} {item['role']}: {item['content'][:900]}"
        for item in batch
    )[-7_800:]
    return ConversationMemoryDigest(
        summary=summary or "对话已建立长期记忆。",
        topics=[MemoryTopic(topic="历史对话", content=content)],
    )


def _rank_rows(
    rows: list[sqlite3.Row],
    terms: set[str],
    limit: int,
    *,
    kind: Literal["message", "topic"],
) -> list[sqlite3.Row]:
    if not terms or limit <= 0:
        return []
    scored: list[tuple[float, int, sqlite3.Row]] = []
    for row in rows:
        if kind == "topic":
            haystack = f"{row['topic']} {row['content']}".lower()
            sequence = int(row["source_end_sequence"])
        else:
            haystack = f"{row['content']} {row['quote'] or ''}".lower()
            sequence = int(row["sequence"])
        score = sum(min(haystack.count(term), 4) * (3.0 if len(term) > 2 else 1.0) for term in terms)
        evidence_ids = re.findall(r"\b[ETF]\d{3}\b", haystack, re.I)
        if any(evidence_id.lower() in terms for evidence_id in evidence_ids):
            score += 20
        if score > 0:
            scored.append((score, sequence, row))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [item[2] for item in scored[:limit]]
    if kind == "message":
        selected.sort(key=lambda row: int(row["sequence"]))
    return selected


def _memory_terms(text: str) -> set[str]:
    lowered = text.lower()
    terms = {
        token
        for token in re.findall(r"[a-z][a-z0-9_-]{1,}|[ETF]\d{3}", lowered, re.I)
        if token not in {"what", "which", "with", "that", "this", "from", "about", "please"}
    }
    for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        terms.add(sequence)
        terms.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return {term.lower() for term in terms if len(term) >= 2}


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS chat_conversations (
            id TEXT PRIMARY KEY,
            paper_history_id TEXT NOT NULL,
            title TEXT NOT NULL,
            memory_summary TEXT NOT NULL DEFAULT '',
            memory_message_count INTEGER NOT NULL DEFAULT 0,
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

        CREATE INDEX IF NOT EXISTS idx_chat_memories_conversation
        ON chat_memories(conversation_id, updated_at DESC);
        """
    )
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(chat_messages)").fetchall()
    }
    if "model_trace_json" not in columns:
        connection.execute("ALTER TABLE chat_messages ADD COLUMN model_trace_json TEXT")


def _conversation_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "history_id": str(row["paper_history_id"]),
        "title": str(row["title"]),
        "message_count": int(row["message_count"]),
        "memory_message_count": int(row["memory_message_count"]),
        "memory_ready": bool(row["memory_summary"]),
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


def _topic_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "topic": str(row["topic"]),
        "content": str(row["content"]),
        "source_start_sequence": int(row["source_start_sequence"]),
        "source_end_sequence": int(row["source_end_sequence"]),
        "updated_at": str(row["updated_at"]),
    }


def _clean_title(value: str | None) -> str:
    return " ".join((value or "").split())[:80]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
