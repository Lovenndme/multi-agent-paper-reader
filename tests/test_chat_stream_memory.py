"""Integration tests for streamed chat persistence metadata."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app import _stream_chat_response
from core.chat import PaperChatRequest
from core.chat_memory import list_conversations, load_conversation
from core.evidence import EvidenceSnippet
from core.history import save_paper_analysis


class TestChatStreamMemory(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.previous_data_dir = os.environ.get("PAPER_READER_DATA_DIR")
        self.previous_db = os.environ.pop("PAPER_HISTORY_DB", None)
        os.environ["PAPER_READER_DATA_DIR"] = self.tempdir.name
        self.history_id = save_paper_analysis(
            pdf_data=b"%PDF-1.4 stream-test",
            result={
                "mode": "live",
                "paper": {
                    "title": "Streaming Memory",
                    "filename": "stream.pdf",
                    "pages": 1,
                    "sections_count": 1,
                    "size_bytes": 20,
                },
            },
            snippets=[EvidenceSnippet("E001", "Abstract", 0, 0, "source")],
        )

    def tearDown(self):
        if self.previous_data_dir is None:
            os.environ.pop("PAPER_READER_DATA_DIR", None)
        else:
            os.environ["PAPER_READER_DATA_DIR"] = self.previous_data_dir
        if self.previous_db is not None:
            os.environ["PAPER_HISTORY_DB"] = self.previous_db
        self.tempdir.cleanup()

    def test_stream_creates_conversation_and_persists_both_messages(self):
        fake_prompt = SimpleNamespace(
            messages=(),
            stats=SimpleNamespace(
                token_budget=48_000,
                estimated_input_tokens=100,
                recent_messages=0,
                recalled_messages=0,
                recalled_topics=0,
                total_persisted_messages=0,
            ),
        )
        request = PaperChatRequest(
            history_id=self.history_id,
            question="这个方法为什么有效？",
            selected_text="方法片段",
            context={"paper": {"title": "Streaming Memory"}},
        )

        with (
            patch("app.build_chat_prompt", return_value=fake_prompt),
            patch("app.stream_chat_reply", return_value=iter(["因为", "证据充分。"])),
        ):
            events = [json.loads(line) for line in _stream_chat_response(request, demo=False)]

        complete = events[-1]
        conversation_id = complete["conversation_id"]
        restored = load_conversation(conversation_id)

        self.assertEqual([event["type"] for event in events], ["token", "token", "complete"])
        self.assertEqual(len(list_conversations(self.history_id)), 1)
        self.assertEqual([item["role"] for item in restored["messages"]], ["user", "assistant"])
        self.assertEqual(restored["messages"][0]["quote"], "方法片段")
        self.assertEqual(restored["messages"][1]["content"], "因为证据充分。")
        self.assertEqual(complete["conversation"]["message_count"], 2)


if __name__ == "__main__":
    unittest.main()
