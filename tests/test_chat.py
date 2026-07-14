"""Tests for grounded paper follow-up chat context."""

import json
import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.chat import (
    MAX_EVIDENCE_ITEMS,
    ChatHistoryTurn,
    PaperChatRequest,
    build_chat_prompt,
    build_chat_messages,
    clear_analysis_sessions,
    compact_analysis_context,
    demo_chat_reply,
    get_analysis_session,
    hide_evidence_citations,
    estimate_chat_tokens,
    resolve_chat_model_route,
    retrieve_chat_evidence,
    store_analysis_session,
)
from core.evidence import EvidenceSnippet


class TestPaperChat(unittest.TestCase):
    def setUp(self):
        clear_analysis_sessions()

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
        self.assertIn("不要把长公式或推导塞进表格", messages[0].content)
        self.assertIn("最终回答不要显示 E 编号", messages[0].content)
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
        with patch.dict(
            os.environ,
            {
                "TEXT_PROVIDER": "qwen",
                "MODEL_NAME": "qwen3.7-max",
                "DASHSCOPE_API_KEY": "configured-qwen-key",
            },
            clear=True,
        ):
            reply = demo_chat_reply(
                PaperChatRequest(
                    question="解释一下",
                    selected_text="一段需要解释的论文结论",
                )
            )

        self.assertIn("一段需要解释的论文结论", reply)
        self.assertIn("Qwen / Qwen3.7 Max", reply)

    def test_request_scoped_model_route_does_not_follow_global_provider(self):
        request = PaperChatRequest(
            question="解释方法",
            text_provider="qwen",
            text_model="qwen3.7-plus",
            text_mode="fast",
        )
        with patch.dict(
            os.environ,
            {"TEXT_PROVIDER": "zhipu", "MODEL_NAME": "glm-5.2", "MODEL_MODE": "deep"},
            clear=True,
        ):
            route = resolve_chat_model_route(request)

        self.assertEqual(route, ("qwen", "qwen3.7-plus", "fast"))

    def test_hides_internal_evidence_markers_from_visible_answer(self):
        answer = "结论成立 [E007, p.6]，并得到重复实验支持 [E004, p.5]。\n\n[E009, pp.7-8]"

        cleaned = hide_evidence_citations(answer)

        self.assertEqual(cleaned, "结论成立，并得到重复实验支持。")

    def test_prompt_treats_server_call_details_as_model_identity_source(self):
        request = PaperChatRequest(
            question="你是什么模型？",
            history=[
                ChatHistoryTurn(role="assistant", content="我是 GLM 模型。"),
            ],
            context={"paper": {"title": "Test"}},
        )

        with patch.dict(
            os.environ,
            {
                "TEXT_PROVIDER": "qwen",
                "MODEL_NAME": "qwen3.7-max",
                "DASHSCOPE_API_KEY": "configured-qwen-key",
            },
            clear=True,
        ):
            messages = build_chat_messages(request)

        self.assertNotIn("Alibaba Qwen / Qwen3.7 Max", messages[0].content)
        self.assertIn("以本条回答下方的服务端调用详情为准", messages[0].content)

    def test_session_retrieval_uses_complete_relevant_evidence(self):
        snippets = [
            EvidenceSnippet(
                id="E001",
                section="Method",
                page_start=1,
                page_end=1,
                text="The architecture uses a dual encoder and a gated fusion mechanism.",
            ),
            EvidenceSnippet(
                id="E002",
                section="Experiments",
                page_start=4,
                page_end=4,
                text="Experiments report accuracy, F1 score, and latency on Dataset Alpha.",
            ),
        ]
        analysis_id = store_analysis_session(
            snippets,
            {"paper": {"title": "Test Paper"}, "summary_output": {}},
        )
        request = PaperChatRequest(
            analysis_id=analysis_id,
            question="实验使用了哪些指标？",
        )

        messages = build_chat_messages(request)
        retrieved = retrieve_chat_evidence(
            session=get_analysis_session(analysis_id),
            question=request.question,
            selected_text=None,
            history=[],
        )

        self.assertEqual(retrieved[0].id, "E002")
        self.assertIn("accuracy, F1 score, and latency", messages[0].content)
        self.assertIn("[E002 | text | Experiments | p.5]", messages[0].content)

    def test_agent_citations_bridge_translated_summary_back_to_source(self):
        snippets = [
            EvidenceSnippet(
                id="E001",
                section="Introduction",
                page_start=0,
                page_end=0,
                text="This paper studies a general sequence transduction problem.",
            ),
            EvidenceSnippet(
                id="E003",
                section="Technical Details",
                page_start=2,
                page_end=2,
                text="The decoder attends to all encoder positions in each layer.",
            ),
        ]
        analysis_id = store_analysis_session(
            snippets,
            {
                "paper": {"title": "Test Paper"},
                "method_output": {
                    "proposed_method": "解码器在每一层关注所有编码器位置。",
                    "evidence": [
                        {
                            "id": "E003",
                            "section": "Technical Details",
                            "page": "p.3",
                            "quote": "decoder attends to all encoder positions",
                            "note": "支持解码器机制描述",
                        }
                    ],
                },
            },
        )

        retrieved = retrieve_chat_evidence(
            session=get_analysis_session(analysis_id),
            question="这段解码器机制具体是什么意思？",
            selected_text="解码器在每一层关注所有编码器位置。",
            history=[],
        )

        self.assertEqual(retrieved[0].id, "E003")

    def test_broad_question_gets_diverse_source_sections(self):
        snippets = [
            EvidenceSnippet("E001", "Abstract", 0, 0, "Abstract evidence."),
            EvidenceSnippet("E002", "Method", 1, 1, "Method evidence."),
            EvidenceSnippet("E003", "Experiments", 2, 2, "Experiment evidence."),
            EvidenceSnippet("E004", "Conclusion", 3, 3, "Conclusion evidence."),
        ]
        analysis_id = store_analysis_session(
            snippets,
            {"paper": {"title": "Test Paper"}},
        )

        retrieved = retrieve_chat_evidence(
            session=get_analysis_session(analysis_id),
            question="请详细解释一下这篇论文。",
            selected_text=None,
            history=[],
        )

        self.assertEqual(len(retrieved), 4)
        self.assertEqual(len({item.section for item in retrieved}), 4)

    def test_prompt_uses_full_snippet_not_api_preview_length(self):
        trailing_evidence = "FULL_SOURCE_SENTINEL_AFTER_PREVIEW"
        snippet = EvidenceSnippet(
            "E007",
            "Experiments",
            6,
            6,
            "x" * 260 + trailing_evidence,
        )
        analysis_id = store_analysis_session(
            [snippet],
            {
                "paper": {"title": "Long Evidence"},
                "experiment_output": {
                    "evidence": [
                        {
                            "id": "E007",
                            "section": "Experiments",
                            "page": "p.7",
                            "quote": trailing_evidence,
                            "note": "supports the result",
                        }
                    ]
                },
            },
        )

        messages = build_chat_messages(
            PaperChatRequest(
                analysis_id=analysis_id,
                question=f"请解释 {trailing_evidence}",
            )
        )

        self.assertIn(trailing_evidence, messages[0].content)
        self.assertIn("x" * 240, messages[0].content)

    def test_dynamic_budget_bounds_large_recent_history(self):
        request = PaperChatRequest(
            question="请继续解释实验结论。",
            history=[
                ChatHistoryTurn(
                    role="user" if index % 2 == 0 else "assistant",
                    content=("很长的历史内容" * 900)[:8_000],
                    quote=("引用" * 2_000) if index % 2 == 0 else None,
                )
                for index in range(20)
            ],
            context={"paper": {"title": "Budget Test"}, "summary_output": {"notes": "摘要" * 20_000}},
        )

        with patch.dict(os.environ, {"CHAT_INPUT_TOKEN_BUDGET": "8000"}):
            prompt = build_chat_prompt(request)

        measured = sum(estimate_chat_tokens(message.content) for message in prompt.messages)
        self.assertLessEqual(measured, prompt.stats.token_budget)
        self.assertEqual(prompt.stats.estimated_input_tokens, measured)
        self.assertLess(prompt.stats.recent_messages, 20)


if __name__ == "__main__":
    unittest.main()
