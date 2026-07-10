"""Tests for grounded paper follow-up chat context."""

import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.chat import (
    MAX_EVIDENCE_ITEMS,
    ChatHistoryTurn,
    PaperChatRequest,
    build_chat_messages,
    compact_analysis_context,
    demo_chat_reply,
)


class TestPaperChat(unittest.TestCase):
    def test_builds_messages_with_context_history_and_selected_excerpt(self):
        request = PaperChatRequest(
            question="这个结论为什么成立？",
            selected_text="检索器质量与下游性能高度相关。",
            history=[
                ChatHistoryTurn(role="user", content="先概括方法。"),
                ChatHistoryTurn(role="assistant", content="方法结合了检索与生成。"),
            ],
            context={
                "paper": {"title": "RAG"},
                "summary_output": {"one_sentence_summary": "检索增强生成。"},
                "secret": "must not enter prompt",
            },
        )

        messages = build_chat_messages(request)

        self.assertIsInstance(messages[0], SystemMessage)
        self.assertIsInstance(messages[1], HumanMessage)
        self.assertIsInstance(messages[2], AIMessage)
        self.assertIn("RAG", messages[0].content)
        self.assertNotIn("must not enter prompt", messages[0].content)
        self.assertIn("<selected_excerpt>", messages[-1].content)
        self.assertIn("这个结论为什么成立", messages[-1].content)

    def test_compacts_evidence_and_drops_unknown_context_fields(self):
        context = {
            "paper": {"title": "Test"},
            "evidence_index": [
                {"id": f"E{index:03d}", "preview": "evidence"}
                for index in range(MAX_EVIDENCE_ITEMS + 8)
            ],
            "api_key": "not-allowed",
        }

        compact = json.loads(compact_analysis_context(context))

        self.assertEqual(len(compact["evidence_index"]), MAX_EVIDENCE_ITEMS)
        self.assertNotIn("api_key", compact)

    def test_demo_reply_mentions_selected_text_and_live_model(self):
        reply = demo_chat_reply(
            PaperChatRequest(
                question="解释一下",
                selected_text="一段需要解释的论文结论",
            )
        )

        self.assertIn("一段需要解释的论文结论", reply)
        self.assertIn("GLM-5.2", reply)


if __name__ == "__main__":
    unittest.main()
