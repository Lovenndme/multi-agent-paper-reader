"""Tests for optional external academic lookup routing."""

import json
import unittest
from unittest.mock import patch

from core.external_knowledge import (
    search_external_academic_sources,
    should_search_external,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class TestExternalKnowledge(unittest.TestCase):
    def test_only_routes_questions_that_need_outside_literature(self):
        self.assertFalse(should_search_external("这个方法的损失函数是什么？"))
        self.assertFalse(should_search_external("它与文中基线相比提升了多少？"))
        self.assertTrue(should_search_external("和其他论文相比有什么优势？"))
        self.assertTrue(should_search_external("What is the latest related work?"))

    def test_parses_semantic_scholar_results(self):
        payload = {
            "data": [
                {
                    "title": "A Related Paper",
                    "year": 2025,
                    "authors": [{"name": "Ada Researcher"}],
                    "abstract": "A related method and its evaluation.",
                    "url": "https://www.semanticscholar.org/paper/example",
                    "externalIds": {"DOI": "10.1000/example"},
                }
            ]
        }
        with patch(
            "core.external_knowledge.urlopen",
            return_value=FakeResponse(payload),
        ):
            sources = search_external_academic_sources(
                "和其他论文相比有什么优势？",
                "Current Paper",
            )

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].id, "S1")
        self.assertEqual(sources[0].year, 2025)
        self.assertIn("semanticscholar.org", sources[0].url)

    def test_uses_paper_recommendations_after_matching_title(self):
        search_payload = {
            "data": [
                {
                    "paperId": "paper-123",
                    "title": "Current Paper",
                }
            ]
        }
        recommendation_payload = {
            "recommendedPapers": [
                {
                    "title": "A Recommended Follow-up",
                    "year": 2026,
                    "authors": [{"name": "R. Scholar"}],
                    "abstract": "A directly related follow-up study.",
                    "url": "https://www.semanticscholar.org/paper/follow-up",
                }
            ]
        }
        with patch(
            "core.external_knowledge.urlopen",
            side_effect=[FakeResponse(search_payload), FakeResponse(recommendation_payload)],
        ):
            sources = search_external_academic_sources(
                "和其他论文相比有什么优势？",
                "Current Paper",
            )

        self.assertEqual(sources[0].title, "A Recommended Follow-up")
        self.assertEqual(sources[0].year, 2026)


if __name__ == "__main__":
    unittest.main()
