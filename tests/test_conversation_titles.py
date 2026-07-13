"""Tests for concise automatic paper-chat titles."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.conversation_titles import generate_conversation_title, local_conversation_title


class TestConversationTitles(unittest.TestCase):
    def test_local_title_removes_request_filler_and_bounds_length(self):
        title = local_conversation_title(
            "请帮我详细介绍一下 GRPO 算法的奖励计算过程，以及它和 PPO 到底有什么区别？"
        )

        self.assertFalse(title.startswith("请帮我"))
        self.assertLessEqual(len(title), 32)
        self.assertIn("GRPO", title)

    def test_model_title_is_cleaned_and_keeps_compact_summary(self):
        fake_llm = SimpleNamespace(bind=lambda **_: "fake-bound-chat-client")
        with (
            patch("core.conversation_titles.is_llm_configured", return_value=True),
            patch("core.conversation_titles.get_chat_llm", return_value=fake_llm),
            patch(
                "core.conversation_titles.invoke_with_retry",
                return_value=SimpleNamespace(content="会话标题：GRPO 奖励机制解析？"),
            ),
        ):
            title = generate_conversation_title("举例说明 GRPO 的奖励如何计算")

        self.assertEqual(title, "GRPO 奖励机制解析")


if __name__ == "__main__":
    unittest.main()
