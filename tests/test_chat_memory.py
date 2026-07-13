"""Tests for persistent multi-conversation paper-chat memory."""

from __future__ import annotations

import os
import tempfile
import time
import unittest

from core.chat_memory import (
    ConversationMemoryDigest,
    MemoryTopic,
    add_conversation_message,
    create_conversation,
    get_prompt_memory,
    list_conversations,
    load_conversation,
    rename_conversation,
    refresh_conversation_memory,
)
from core.evidence import EvidenceSnippet
from core.history import delete_paper_history, save_paper_analysis


class TestChatMemory(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.previous_data_dir = os.environ.get("PAPER_READER_DATA_DIR")
        self.previous_db = os.environ.pop("PAPER_HISTORY_DB", None)
        os.environ["PAPER_READER_DATA_DIR"] = self.tempdir.name
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
        if self.previous_data_dir is None:
            os.environ.pop("PAPER_READER_DATA_DIR", None)
        else:
            os.environ["PAPER_READER_DATA_DIR"] = self.previous_data_dir
        if self.previous_db is not None:
            os.environ["PAPER_HISTORY_DB"] = self.previous_db
        self.tempdir.cleanup()

    def test_persists_complete_messages_and_multiple_conversations(self):
        first = create_conversation(self.history_id, title="方法讨论")
        second = create_conversation(self.history_id, title="实验复现")
        user = add_conversation_message(
            first["id"],
            role="user",
            content="请解释方法。",
            quote="方法片段",
        )
        assistant = add_conversation_message(
            first["id"],
            role="assistant",
            content="方法解释。",
            model_trace={
                "provider": "zhipu",
                "requested_model": "glm-5.2",
                "upstream_model": "glm-5.2",
                "verification": "upstream_confirmed",
            },
        )

        restored = load_conversation(first["id"])
        conversations = list_conversations(self.history_id)

        self.assertEqual([item["id"] for item in restored["messages"]], [user["id"], assistant["id"]])
        self.assertEqual(restored["messages"][0]["quote"], "方法片段")
        self.assertEqual(restored["messages"][1]["model_trace"]["upstream_model"], "glm-5.2")
        self.assertEqual({item["id"] for item in conversations}, {first["id"], second["id"]})
        self.assertEqual(restored["conversation"]["message_count"], 2)

        renamed = rename_conversation(first["id"], "Transformer 方法细节")
        self.assertEqual(renamed["title"], "Transformer 方法细节")
        self.assertEqual(load_conversation(first["id"])["conversation"]["title"], "Transformer 方法细节")

    def test_recent_context_and_query_recall_use_full_history(self):
        conversation = create_conversation(self.history_id)
        for index in range(10):
            user_text = "早期量子损失讨论" if index == 0 else f"普通问题 {index}"
            add_conversation_message(conversation["id"], role="user", content=user_text)
            add_conversation_message(conversation["id"], role="assistant", content=f"回答 {index}")

        memory = get_prompt_memory(conversation["id"], "量子损失是什么？")

        self.assertEqual(len(memory.recent_messages), 12)
        self.assertEqual(memory.total_messages, 20)
        self.assertTrue(any("量子损失" in item["content"] for item in memory.recalled_messages))
        self.assertLess(memory.recalled_messages[0]["sequence"], memory.recent_messages[0]["sequence"])

    def test_compaction_keeps_raw_messages_and_builds_index_and_topics(self):
        conversation = create_conversation(self.history_id)
        for index in range(12):
            add_conversation_message(conversation["id"], role="user", content=f"方法问题 {index} E001")
            add_conversation_message(conversation["id"], role="assistant", content=f"方法回答 {index}")

        def summarize(existing_summary, existing_topics, batch):
            self.assertEqual(existing_summary, "")
            self.assertEqual(existing_topics, [])
            self.assertEqual(len(batch), 12)
            return ConversationMemoryDigest(
                summary="用户正在持续研究方法，并关注 E001。",
                topics=[MemoryTopic(topic="方法", content="已讨论方法机制及 E001 证据。")],
            )

        processed = refresh_conversation_memory(conversation["id"], summarizer=summarize)
        restored = load_conversation(conversation["id"])
        memory = get_prompt_memory(conversation["id"], "方法证据 E001")

        self.assertEqual(processed, 12)
        self.assertEqual(len(restored["messages"]), 24)
        self.assertEqual(restored["conversation"]["memory_message_count"], 12)
        self.assertTrue(restored["conversation"]["memory_ready"])
        self.assertIn("持续研究方法", memory.memory_summary)
        self.assertEqual(memory.recalled_topics[0]["topic"], "方法")

    def test_paper_delete_cascades_conversations(self):
        conversation = create_conversation(self.history_id)
        add_conversation_message(conversation["id"], role="user", content="临时问题")

        self.assertTrue(delete_paper_history(self.history_id))
        self.assertEqual(list_conversations(self.history_id), [])
        with self.assertRaises(KeyError):
            load_conversation(conversation["id"])

    def test_retrieval_performance_with_one_thousand_messages(self):
        conversation = create_conversation(self.history_id)
        for index in range(1_000):
            marker = "稀有消融结论" if index == 111 else "常规讨论"
            add_conversation_message(
                conversation["id"],
                role="user" if index % 2 == 0 else "assistant",
                content=f"{marker} message {index}",
            )

        started = time.perf_counter()
        memory = get_prompt_memory(conversation["id"], "稀有消融结论")
        elapsed = time.perf_counter() - started

        self.assertTrue(any("稀有消融结论" in item["content"] for item in memory.recalled_messages))
        self.assertLess(elapsed, 1.0)


if __name__ == "__main__":
    unittest.main()
