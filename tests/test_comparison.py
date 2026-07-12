"""Tests for evidence-grounded multi-paper comparison and persisted chat."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import _stream_comparison_chat_response, _stream_comparison_response
from core.comparison import (
    ComparisonCreateRequest,
    build_comparison_assessment,
    format_comparison_sources,
    load_comparison_sources,
    sanitize_comparison_output,
)
from core.comparison_chat import ComparisonChatRequest, build_comparison_chat_prompt
from core.comparison_history import (
    add_comparison_message,
    create_comparison_conversation,
    delete_comparison,
    get_comparison_prompt_memory,
    list_comparison_conversations,
    list_comparisons,
    load_comparison,
    load_comparison_conversation,
)
from core.evidence import EvidenceSnippet
from core.history import delete_paper_history, save_paper_analysis
from core.schemas import ComparisonOutput


class TestMultiPaperComparison(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.environment = patch.dict(
            os.environ,
            {
                "PAPER_READER_DATA_DIR": self.tempdir.name,
                "PAPER_HISTORY_DB": str(Path(self.tempdir.name) / "history.sqlite3"),
            },
        )
        self.environment.start()
        self.history_ids = [
            _save_paper("Paper Alpha", "alpha method and accuracy results"),
            _save_paper("Paper Beta", "beta method and F1 results"),
        ]

    def tearDown(self):
        self.environment.stop()
        self.tempdir.cleanup()

    def test_source_package_prefixes_overlapping_evidence_ids(self):
        request = ComparisonCreateRequest(history_ids=self.history_ids, focus="comprehensive")
        sources = load_comparison_sources(request.history_ids)

        package = format_comparison_sources(sources, request)

        self.assertIn("[P1:E001", package)
        self.assertIn("[P2:E001", package)
        self.assertIn("Paper Alpha", package)
        self.assertIn("Paper Beta", package)

    def test_sanitizer_drops_invalid_and_cross_paper_evidence(self):
        request = ComparisonCreateRequest(history_ids=self.history_ids, focus="method")
        sources = load_comparison_sources(request.history_ids)
        raw = ComparisonOutput.model_validate(
            {
                "title": "Method comparison",
                "focus": "ignored",
                "executive_summary": "summary",
                "papers": [],
                "common_ground": [],
                "key_differences": [],
                "dimensions": [
                    {
                        "key": "method",
                        "title": "核心方法",
                        "category": "method",
                        "description": "method",
                        "cells": [
                            {
                                "paper_label": "P1",
                                "summary": "alpha",
                                "evidence_ids": ["P1:E001", "P2:E001", "P1:E999"],
                            }
                        ],
                        "synthesis": "different",
                        "comparability": "direct",
                    }
                ],
                "research_gaps": [],
                "recommendations": [],
                "warnings": [],
            }
        )

        sanitized = sanitize_comparison_output(raw, sources, request)
        assessment = build_comparison_assessment(sanitized)

        self.assertEqual(sanitized.focus, "方法与架构")
        self.assertEqual(sanitized.dimensions[0].cells[0].evidence_ids, ["P1:E001"])
        self.assertEqual(sanitized.dimensions[0].cells[1].paper_label, "P2")
        self.assertEqual(assessment.total_claims, 2)
        self.assertEqual(assessment.referenced_claims, 1)
        self.assertEqual(assessment.evidence_coverage, 50)

    def test_demo_stream_persists_and_restores_comparison(self):
        request = ComparisonCreateRequest(history_ids=self.history_ids, focus="experiment")

        events = [json.loads(line) for line in _stream_comparison_response(request, demo=True)]
        complete = events[-1]
        stored = load_comparison(complete["comparison_id"])

        self.assertEqual(events[0]["type"], "comparison_started")
        self.assertEqual(sum(event["type"] == "paper_loaded" for event in events), 2)
        self.assertEqual(complete["type"], "complete")
        self.assertEqual(len(list_comparisons()), 1)
        self.assertEqual(stored["result"]["comparison"]["papers"][0]["label"], "P1")
        self.assertEqual(stored["workspace"]["paper_count"], 2)

        self.assertTrue(delete_comparison(complete["comparison_id"]))
        self.assertIsNone(load_comparison(complete["comparison_id"]))

    def test_deleting_source_paper_removes_dependent_comparison(self):
        comparison_id = _save_demo_comparison(self.history_ids)

        deleted = delete_paper_history(self.history_ids[0])

        self.assertTrue(deleted)
        self.assertIsNone(load_comparison(comparison_id))

    def test_cross_paper_chat_prompt_and_messages_persist(self):
        comparison_id = _save_demo_comparison(self.history_ids)
        request = ComparisonChatRequest(
            comparison_id=comparison_id,
            question="两篇论文的方法有什么差异？",
        )

        prompt = build_comparison_chat_prompt(request)
        prompt_text = "\n".join(str(message.content) for message in prompt.messages)
        events = [json.loads(line) for line in _stream_comparison_chat_response(request, demo=True)]
        conversation_id = events[-1]["conversation_id"]
        restored = load_comparison_conversation(conversation_id)

        self.assertIn("P1:E001", prompt_text)
        self.assertIn("P2:E001", prompt_text)
        self.assertIn("不要把长公式或推导塞进表格", prompt_text)
        self.assertEqual([message["role"] for message in restored["messages"]], ["user", "assistant"])
        self.assertEqual(len(list_comparison_conversations(comparison_id)), 1)
        self.assertEqual(events[-1]["conversation"]["message_count"], 2)

    def test_cross_paper_chat_keeps_unlimited_messages_and_recalls_older_turns(self):
        comparison_id = _save_demo_comparison(self.history_ids)
        conversation = create_comparison_conversation(comparison_id)
        for index in range(40):
            marker = "早期低秩适配结论" if index == 2 else f"普通消息 {index}"
            add_comparison_message(
                conversation["id"],
                role="user" if index % 2 == 0 else "assistant",
                content=marker,
            )

        memory = get_comparison_prompt_memory(
            conversation["id"],
            "之前的低秩适配结论是什么？",
        )

        self.assertEqual(memory.total_messages, 40)
        self.assertEqual(len(memory.recent_messages), 16)
        self.assertTrue(any("早期低秩适配结论" in item["content"] for item in memory.recalled_messages))


def _save_paper(title: str, evidence_text: str) -> str:
    return save_paper_analysis(
        pdf_data=f"%PDF-1.7 {title}".encode(),
        result={
            "mode": "live",
            "paper": {
                "title": title,
                "filename": f"{title}.pdf",
                "pages": 4,
                "sections_count": 2,
                "size_bytes": 1000,
            },
            "method_output": {
                "research_problem": f"{title} problem",
                "proposed_method": f"{title} method",
            },
            "experiment_output": {"main_results": f"{title} result"},
            "critic_output": {"limitations": [f"{title} limitation"]},
            "summary_output": {"one_sentence_summary": f"{title} summary"},
        },
        snippets=[
            EvidenceSnippet(
                id="E001",
                section="Method and Experiments",
                page_start=1,
                page_end=1,
                text=evidence_text,
            )
        ],
    )


def _save_demo_comparison(history_ids: list[str]) -> str:
    request = ComparisonCreateRequest(history_ids=history_ids, focus="comprehensive")
    sources = load_comparison_sources(history_ids)
    complete = [json.loads(line) for line in _stream_comparison_response(request, demo=True)][-1]
    return str(complete["comparison_id"])


if __name__ == "__main__":
    unittest.main()
