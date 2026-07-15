"""LangMem schemas and a small SQLite-backed LangGraph memory store."""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from langgraph.store.base import GetOp, Op, PutOp, SearchOp
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from core.history import history_database_connection
from core.semantic_search import LocalFastEmbedEmbeddings, embeddings_enabled


LANGMEM_NAMESPACE_ROOT = "paper-reader"
DEFAULT_RECALL_LIMIT = 3
DEFAULT_RECALL_MIN_SCORE = 0.48


class PaperReaderMemory(BaseModel):
    """One durable paper-reading memory managed by LangMem."""

    category: Literal["user", "feedback", "project", "reference"]
    subject: str = Field(min_length=1, max_length=160)
    content: str = Field(min_length=1, max_length=4_000)
    context: str = Field(default="", max_length=1_000)


class SQLiteBackedMemoryStore(InMemoryStore):
    """Use LangGraph's indexed store while mirroring every mutation to SQLite."""

    def __init__(self) -> None:
        dimensions = _positive_int("EMBEDDING_DIMENSIONS", 384)
        index = (
            {
                "dims": dimensions,
                "embed": LocalFastEmbedEmbeddings(),
                "fields": ["content.subject", "content.content", "content.context"],
            }
            if embeddings_enabled()
            else None
        )
        super().__init__(index=index)
        self._persistence_lock = threading.RLock()
        self._hydrated: set[tuple[str, ...]] = set()
        with history_database_connection() as connection:
            ensure_langmem_schema(connection)

    def batch(self, ops: Iterable[Op]) -> list[Any]:
        operations = list(ops)
        with self._persistence_lock:
            for namespace in _operation_namespaces(operations):
                self._hydrate(namespace)
            results = InMemoryStore.batch(self, operations)
            for operation in operations:
                if isinstance(operation, PutOp):
                    self._persist(operation)
            return results

    def evict_paper(self, history_id: str) -> None:
        namespace = memory_namespace(history_id)
        with self._persistence_lock:
            self._data.pop(namespace, None)
            self._vectors.pop(namespace, None)
            self._hydrated.discard(namespace)

    def clear_runtime_cache(self) -> None:
        with self._persistence_lock:
            self._data.clear()
            self._vectors.clear()
            self._hydrated.clear()

    def _hydrate(self, namespace: tuple[str, ...]) -> None:
        if not _is_paper_namespace(namespace) or namespace in self._hydrated:
            return
        self._hydrated.add(namespace)
        history_id = namespace[1]
        _migrate_legacy_memories(history_id)
        with history_database_connection() as connection:
            ensure_langmem_schema(connection)
            rows = connection.execute(
                """
                SELECT memory_id, value_json
                FROM langmem_memories
                WHERE paper_history_id = ?
                ORDER BY updated_at
                """,
                (history_id,),
            ).fetchall()
        puts: list[PutOp] = []
        for row in rows:
            try:
                value = json.loads(str(row["value_json"]))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                puts.append(PutOp(namespace, str(row["memory_id"]), value))
        if puts:
            InMemoryStore.batch(self, puts)

    def _persist(self, operation: PutOp) -> None:
        namespace = operation.namespace
        if not _is_paper_namespace(namespace):
            return
        history_id = namespace[1]
        with history_database_connection() as connection:
            ensure_langmem_schema(connection)
            if operation.value is None:
                connection.execute(
                    "DELETE FROM langmem_memories WHERE paper_history_id = ? AND memory_id = ?",
                    (history_id, operation.key),
                )
                return
            now = _utc_now()
            connection.execute(
                """
                INSERT INTO langmem_memories (
                    paper_history_id, memory_id, value_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(paper_history_id, memory_id) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (
                    history_id,
                    operation.key,
                    json.dumps(operation.value, ensure_ascii=False, separators=(",", ":")),
                    now,
                    now,
                ),
            )


_STORE_LOCK = threading.RLock()
_STORE: SQLiteBackedMemoryStore | None = None


def get_langmem_store() -> SQLiteBackedMemoryStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = SQLiteBackedMemoryStore()
        return _STORE


def reset_langmem_store() -> None:
    """Drop only the process-local index; persisted SQLite memories remain."""
    global _STORE
    with _STORE_LOCK:
        _STORE = None


def memory_namespace(history_id: str) -> tuple[str, str]:
    clean = re.sub(r"[^a-zA-Z0-9_-]", "", str(history_id))
    if not clean:
        raise ValueError("Paper history identifier is empty or unsafe.")
    return LANGMEM_NAMESPACE_ROOT, clean[:120]


def list_langmem_memories(
    history_id: str,
    *,
    query: str | None = None,
    limit: int = DEFAULT_RECALL_LIMIT,
    min_score: float | None = None,
) -> list[dict[str, Any]]:
    """Return bounded, semantically relevant LangMem records for one paper."""
    if limit <= 0:
        return []
    store = get_langmem_store()
    results = store.search(
        memory_namespace(history_id),
        query=query.strip() if query and query.strip() else None,
        limit=min(max(1, limit), 20),
    )
    threshold = _recall_threshold() if min_score is None else float(min_score)
    output: list[dict[str, Any]] = []
    for item in results:
        if query and item.score is not None and float(item.score) < threshold:
            continue
        parsed = _coerce_memory_value(item.value)
        if parsed is None:
            continue
        output.append(
            {
                "id": item.key,
                "topic": parsed.subject,
                "description": parsed.context,
                "type": parsed.category,
                "content": parsed.content,
                "source_start_sequence": 0,
                "source_end_sequence": 0,
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat(),
                "score": float(item.score) if item.score is not None else None,
            }
        )
    return output


def delete_paper_memories(history_id: str) -> None:
    with history_database_connection() as connection:
        ensure_langmem_schema(connection)
        connection.execute("DELETE FROM langmem_memories WHERE paper_history_id = ?", (history_id,))
        connection.execute("DELETE FROM langmem_migrations WHERE paper_history_id = ?", (history_id,))
    with _STORE_LOCK:
        if _STORE is not None:
            _STORE.evict_paper(history_id)


def ensure_langmem_schema(connection: Any) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS langmem_memories (
            paper_history_id TEXT NOT NULL,
            memory_id TEXT NOT NULL,
            value_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (paper_history_id, memory_id),
            FOREIGN KEY (paper_history_id) REFERENCES paper_history(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS langmem_migrations (
            paper_history_id TEXT PRIMARY KEY,
            migrated_at TEXT NOT NULL,
            FOREIGN KEY (paper_history_id) REFERENCES paper_history(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_langmem_memories_paper_updated
        ON langmem_memories(paper_history_id, updated_at DESC);
        """
    )


def _operation_namespaces(operations: list[Op]) -> set[tuple[str, ...]]:
    namespaces: set[tuple[str, ...]] = set()
    for operation in operations:
        if isinstance(operation, SearchOp):
            namespaces.add(operation.namespace_prefix)
        elif isinstance(operation, (GetOp, PutOp)):
            namespaces.add(operation.namespace)
    return namespaces


def _is_paper_namespace(namespace: tuple[str, ...]) -> bool:
    return len(namespace) == 2 and namespace[0] == LANGMEM_NAMESPACE_ROOT and bool(namespace[1])


def _coerce_memory_value(value: dict[str, Any]) -> PaperReaderMemory | None:
    content = value.get("content") if isinstance(value, dict) else None
    if isinstance(content, PaperReaderMemory):
        return content
    if isinstance(content, dict):
        try:
            return PaperReaderMemory.model_validate(content)
        except Exception:
            return None
    if isinstance(content, str) and content.strip():
        return PaperReaderMemory(
            category="project",
            subject=content.strip()[:120],
            content=content.strip(),
        )
    return None


def _migrate_legacy_memories(history_id: str) -> None:
    """Import the prior SQLite/file memories once, then leave the old files untouched."""
    with history_database_connection() as connection:
        ensure_langmem_schema(connection)
        migrated = connection.execute(
            "SELECT 1 FROM langmem_migrations WHERE paper_history_id = ?",
            (history_id,),
        ).fetchone()
        if migrated:
            return
        existing = connection.execute(
            "SELECT COUNT(*) AS count FROM langmem_memories WHERE paper_history_id = ?",
            (history_id,),
        ).fetchone()
        candidates: list[PaperReaderMemory] = []
        if int(existing["count"]) == 0:
            try:
                rows = connection.execute(
                    """
                    SELECT m.topic, m.content
                    FROM chat_memories AS m
                    JOIN chat_conversations AS c ON c.id = m.conversation_id
                    WHERE c.paper_history_id = ?
                    ORDER BY m.updated_at
                    """,
                    (history_id,),
                ).fetchall()
            except Exception:
                rows = []
            for row in rows:
                topic = str(row["topic"] or "").strip()
                content = str(row["content"] or "").strip()
                if topic and content:
                    candidates.append(
                        PaperReaderMemory(
                            category="project",
                            subject=topic[:160],
                            content=content[:4_000],
                            context="从旧版 SQLite 记忆迁移。",
                        )
                    )
            candidates.extend(_legacy_file_memories(history_id))
            seen: set[tuple[str, str]] = set()
            now = _utc_now()
            for index, memory in enumerate(candidates):
                identity = (memory.subject.casefold(), memory.content.casefold())
                if identity in seen:
                    continue
                seen.add(identity)
                value = {
                    "kind": "PaperReaderMemory",
                    "content": memory.model_dump(mode="json"),
                }
                connection.execute(
                    """
                    INSERT OR IGNORE INTO langmem_memories (
                        paper_history_id, memory_id, value_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        history_id,
                        f"legacy-{index:04d}",
                        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
                        now,
                        now,
                    ),
                )
        connection.execute(
            "INSERT OR REPLACE INTO langmem_migrations (paper_history_id, migrated_at) VALUES (?, ?)",
            (history_id, _utc_now()),
        )


def _legacy_file_memories(history_id: str) -> list[PaperReaderMemory]:
    data_dir = Path(os.environ.get("PAPER_READER_DATA_DIR") or Path(__file__).resolve().parent.parent / ".paper-reader")
    paper_dir = data_dir.expanduser().resolve() / "memory" / "papers" / history_id
    if not paper_dir.exists():
        return []
    output: list[PaperReaderMemory] = []
    for path in sorted(paper_dir.glob("*.md")):
        if path.name == "MEMORY.md":
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            continue
        body = re.sub(r"\A<!-- paper-reader-memory\n.*?\n-->\n", "", raw, count=1, flags=re.DOTALL)
        frontmatter = re.match(r"\A---\n(?P<meta>.*?)\n---\n", body, flags=re.DOTALL)
        metadata: dict[str, str] = {}
        if frontmatter:
            for line in frontmatter.group("meta").splitlines():
                key, separator, value = line.partition(":")
                if separator:
                    metadata[key.strip()] = value.strip().strip("\"").strip("'")
            body = body[frontmatter.end() :]
        content = body.strip()
        subject = (metadata.get("name") or metadata.get("topic") or path.stem).strip()
        if not subject or not content:
            continue
        category = metadata.get("type", "project").strip().lower()
        if category not in {"user", "feedback", "project", "reference"}:
            category = "project"
        output.append(
            PaperReaderMemory(
                category=category,
                subject=subject[:160],
                content=content[:4_000],
                context="从旧版文件记忆迁移。",
            )
        )
    return output


def _recall_threshold() -> float:
    try:
        return max(-1.0, min(1.0, float(os.environ.get("LANGMEM_RECALL_MIN_SCORE", DEFAULT_RECALL_MIN_SCORE))))
    except (TypeError, ValueError):
        return DEFAULT_RECALL_MIN_SCORE


def _positive_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
