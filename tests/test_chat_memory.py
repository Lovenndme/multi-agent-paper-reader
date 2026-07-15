"""Tests for SQLite conversations and the LangMem integration boundary."""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from unittest.mock import patch

from core.chat_memory import (
    add_conversation_message,
    create_conversation,
    get_prompt_memory,
    list_conversations,
    load_conversation,
    memory_refresh_needed,
    refresh_conversation_memory,
    rename_conversation,
)
from core.evidence import EvidenceSnippet
from core.history import delete_paper_history, history_database_connection, save_paper_analysis
from core.langmem_store import (
    PaperReaderMemory,
    get_langmem_store,
    list_langmem_memories,
    memory_namespace,
    reset_langmem_store,
)


class _FakeManager:
    def __init__(self, *, fail: bool = False, memory: PaperReaderMemory | None = None):
        self.fail = fail
        self.memory = memory

    def invoke(self, _payload, *, config):
        if self.fail:
            raise RuntimeError("temporary LangMem failure")
        if self.memory is not None:
            history_id = config["configurable"]["paper_history_id"]
            get_langmem_store().put(
                memory_namespace(history_id),
                "fake-memory",
                {"kind": "PaperReaderMemory", "content": self.memory.model_dump(mode="json")},
            )
        return []


class TestChatMemory(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.previous_data_dir = os.environ.get("PAPER_READER_DATA_DIR")
        self.previous_db = os.environ.pop("PAPER_HISTORY_DB", None)
        os.environ["PAPER_READER_DATA_DIR"] = self.tempdir.name
        reset_langmem_store()
        self.history_id = save_paper_analysis(
            pdf_data=b"%PDF-1.4 test",
            result={
                "mode": "live",
                "paper": {
                    "title": "Memory Test Paper",
                    "filename": "memory.pdf",
                    "pages": 2,
                    "sections_count": 2,
                    "size_bytes": 13,
                },
            },
            snippets=[EvidenceSnippet("E001", "Method", 0, 0, "method evidence")],
        )

    def tearDown(self):
        reset_langmem_store()
        if self.previous_data_dir is None:
            os.environ.pop("PAPER_READER_DATA_DIR", None)
        else:
            os.environ["PAPER_READER_DATA_DIR"] = self.previous_data_dir
        if self.previous_db is not None:
            os.environ["PAPER_HISTORY_DB"] = self.previous_db
        self.tempdir.cleanup()

    def _put_memory(self, key: str, memory: PaperReaderMemory) -> None:
        get_langmem_store().put(
            memory_namespace(self.history_id),
            key,
            {"kind": "PaperReaderMemory", "content": memory.model_dump(mode="json")},
        )

    def test_persists_complete_messages_and_multiple_conversations(self):
        first = create_conversation(self.history_id, title="方法讨论")
        second = create_conversation(self.history_id, title="实验复现")
        user = add_conversation_message(
            first["id"], role="user", content="请解释方法。", quote="方法片段"
        )
        assistant = add_conversation_message(
            first["id"],
            role="assistant",
            content="方法解释。",
            model_trace={
                "provider": "qwen",
                "requested_model": "qwen3.7-max",
                "upstream_model": "qwen3.7-max",
                "verification": "upstream_confirmed",
            },
        )

        restored = load_conversation(first["id"])
        conversations = list_conversations(self.history_id)

        self.assertEqual([item["id"] for item in restored["messages"]], [user["id"], assistant["id"]])
        self.assertEqual(restored["messages"][0]["quote"], "方法片段")
        self.assertEqual(restored["messages"][1]["model_trace"]["upstream_model"], "qwen3.7-max")
        self.assertEqual({item["id"] for item in conversations}, {first["id"], second["id"]})
        self.assertEqual(restored["conversation"]["message_count"], 2)

        renamed = rename_conversation(first["id"], "Transformer 方法细节")
        self.assertEqual(renamed["title"], "Transformer 方法细节")

    def test_recent_context_does_not_resurface_unmanaged_old_messages(self):
        conversation = create_conversation(self.history_id)
        for index in range(10):
            marker = "已要求忘记的 CSSE 背景" if index == 0 else f"普通问题 {index}"
            add_conversation_message(conversation["id"], role="user", content=marker)
            add_conversation_message(conversation["id"], role="assistant", content=f"回答 {index}")

        memory = get_prompt_memory(conversation["id"], "我的专业是什么？")

        self.assertEqual(len(memory.recent_messages), 12)
        self.assertEqual(memory.total_messages, 20)
        self.assertEqual(memory.recalled_messages, ())
        self.assertFalse(any("CSSE" in item["content"] for item in memory.recent_messages))

    def test_langmem_update_advances_cursor_and_failed_update_retries(self):
        conversation = create_conversation(self.history_id)
        add_conversation_message(conversation["id"], role="user", content="请记住标题保留原文。")
        add_conversation_message(conversation["id"], role="assistant", content="明白。")
        self.assertTrue(memory_refresh_needed(conversation["id"]))

        with patch("core.chat_memory.create_memory_store_manager", return_value=_FakeManager(fail=True)):
            self.assertEqual(refresh_conversation_memory(conversation["id"]), 0)
        self.assertEqual(load_conversation(conversation["id"])["conversation"]["memory_message_count"], 0)

        durable = PaperReaderMemory(
            category="feedback",
            subject="章节标题语言",
            content="章节标题优先保留论文原文。",
            context="避免技术术语翻译失真。",
        )
        with patch(
            "core.chat_memory.create_memory_store_manager",
            return_value=_FakeManager(memory=durable),
        ):
            self.assertEqual(refresh_conversation_memory(conversation["id"]), 2)

        restored = load_conversation(conversation["id"])
        self.assertEqual(restored["conversation"]["memory_message_count"], 2)
        self.assertTrue(restored["conversation"]["memory_ready"])
        recalled = get_prompt_memory(conversation["id"], "章节标题应该怎么显示？")
        self.assertEqual(recalled.recalled_topics[0]["type"], "feedback")

    def test_memory_is_shared_across_conversations_and_survives_store_restart(self):
        first = create_conversation(self.history_id)
        second = create_conversation(self.history_id)
        self._put_memory(
            "reproduction",
            PaperReaderMemory(
                category="reference",
                subject="官方复现入口",
                content="官方复现入口是 https://github.com/example/reproduction。",
                context="需要最新说明时检查。",
            ),
        )

        self.assertTrue(get_prompt_memory(first["id"], "官方复现入口在哪里？").recalled_topics)
        reset_langmem_store()
        recalled = get_prompt_memory(second["id"], "去哪里看复现说明？")
        self.assertIn("github.com/example/reproduction", recalled.recalled_topics[0]["content"])

    def test_ignore_memory_proceeds_with_empty_long_term_context(self):
        conversation = create_conversation(self.history_id)
        self._put_memory(
            "rare-result",
            PaperReaderMemory(
                category="project",
                subject="稀有结论",
                content="稀有结论是 X。",
            ),
        )

        memory = get_prompt_memory(conversation["id"], "忽略长期记忆，只根据论文回答。")

        self.assertEqual(memory.memory_index, "")
        self.assertEqual(memory.memory_summary, "")
        self.assertEqual(memory.recalled_topics, ())
        self.assertEqual(memory.recalled_messages, ())

    def test_store_update_and_delete_are_persisted(self):
        namespace = memory_namespace(self.history_id)
        store = get_langmem_store()
        first = PaperReaderMemory(category="user", subject="背景", content="用户是 CSSE 学生。")
        store.put(namespace, "profile", {"kind": "PaperReaderMemory", "content": first.model_dump()})
        corrected = first.model_copy(update={"content": "用户具备 Python 基础。"})
        store.put(namespace, "profile", {"kind": "PaperReaderMemory", "content": corrected.model_dump()})
        reset_langmem_store()
        records = list_langmem_memories(self.history_id, limit=10, min_score=-1)
        self.assertEqual(len(records), 1)
        self.assertNotIn("CSSE", records[0]["content"])

        get_langmem_store().delete(namespace, "profile")
        reset_langmem_store()
        self.assertEqual(list_langmem_memories(self.history_id, limit=10, min_score=-1), [])

    def test_paper_delete_cascades_conversations_and_langmem(self):
        conversation = create_conversation(self.history_id)
        add_conversation_message(conversation["id"], role="user", content="临时问题")
        self._put_memory(
            "delete-probe",
            PaperReaderMemory(category="project", subject="删除探针", content="应随论文删除。"),
        )

        self.assertTrue(delete_paper_history(self.history_id))
        self.assertEqual(list_conversations(self.history_id), [])
        with self.assertRaises(KeyError):
            load_conversation(conversation["id"])
        with history_database_connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM langmem_memories WHERE paper_history_id = ?",
                (self.history_id,),
            ).fetchone()
        self.assertEqual(int(row["count"]), 0)

    def test_one_thousand_messages_remain_fast_and_complete(self):
        conversation = create_conversation(self.history_id)
        self._put_memory(
            "rare-ablation",
            PaperReaderMemory(
                category="project",
                subject="早期稀有消融结论",
                content="早期稀有消融结论是移除路由器后下降 4.2%。",
            ),
        )
        for index in range(1_000):
            add_conversation_message(
                conversation["id"],
                role="user" if index % 2 == 0 else "assistant",
                content=f"常规论文讨论 message {index}",
            )

        started = time.perf_counter()
        memory = get_prompt_memory(conversation["id"], "早期稀有消融结论是什么？")
        elapsed = time.perf_counter() - started

        self.assertIn("4.2%", memory.recalled_topics[0]["content"])
        self.assertEqual(len(load_conversation(conversation["id"])["messages"]), 1_000)
        self.assertLess(elapsed, 1.0)


if __name__ == "__main__":
    unittest.main()
