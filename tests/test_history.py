"""Tests for persistent paper analysis history."""

import os
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import _stream_demo_analysis, history_analysis, paper_history
from core.chat import clear_analysis_sessions, get_analysis_session
from core.evidence import EvidenceSnippet
from core.history import (
    delete_paper_history,
    history_database_connection,
    list_paper_history,
    load_paper_analysis,
    retained_paper_pdf_path,
    save_paper_analysis,
)
from core.pdf_parser import ParsedPaper, Section


class TestPaperHistory(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.environment = patch.dict(
            os.environ,
            {
                "PAPER_READER_DATA_DIR": self.temp_directory.name,
                "PAPER_HISTORY_DB": str(Path(self.temp_directory.name) / "history.sqlite3"),
            },
        )
        self.environment.start()
        clear_analysis_sessions()

    def tearDown(self):
        self.environment.stop()
        self.temp_directory.cleanup()

    def test_saves_and_loads_complete_analysis_and_pdf(self):
        history_id = save_paper_analysis(
            pdf_data=b"%PDF-1.7 test paper",
            result=_result("Persistent Paper"),
            snippets=[_snippet("complete source evidence")],
        )

        items = list_paper_history()
        loaded = load_paper_analysis(history_id)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["title"], "Persistent Paper")
        self.assertTrue(items[0]["pdf_available"])
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["result"]["summary_output"]["one_sentence_summary"], "summary")
        self.assertEqual(loaded["snippets"][0].text, "complete source evidence")

    def test_database_context_closes_connection_after_success(self):
        with history_database_connection() as connection:
            self.assertEqual(connection.execute("SELECT 1").fetchone()[0], 1)

        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")

    def test_database_context_closes_connection_after_error(self):
        with self.assertRaisesRegex(RuntimeError, "forced failure"):
            with history_database_connection() as connection:
                raise RuntimeError("forced failure")

        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")

    def test_same_pdf_updates_one_stable_history_record(self):
        pdf_data = b"%PDF-1.7 duplicate"
        first_id = save_paper_analysis(
            pdf_data=pdf_data,
            result=_result("First Title"),
            snippets=[_snippet("first evidence")],
        )
        second_id = save_paper_analysis(
            pdf_data=pdf_data,
            result=_result("Updated Title"),
            snippets=[_snippet("updated evidence")],
        )

        loaded = load_paper_analysis(second_id)

        self.assertEqual(first_id, second_id)
        self.assertEqual(len(list_paper_history()), 1)
        self.assertEqual(loaded["history"]["title"], "Updated Title")
        self.assertEqual(loaded["snippets"][0].text, "updated evidence")

    def test_deletes_database_record_and_retained_pdf(self):
        history_id = save_paper_analysis(
            pdf_data=b"%PDF-1.7 delete",
            result=_result("Delete Me"),
            snippets=[_snippet("evidence")],
        )
        loaded = load_paper_analysis(history_id)
        pdf_files = list((Path(self.temp_directory.name) / "papers").glob("*.pdf"))

        deleted = delete_paper_history(history_id)

        self.assertTrue(deleted)
        self.assertIsNotNone(loaded)
        self.assertEqual(len(pdf_files), 1)
        self.assertFalse(pdf_files[0].exists())
        self.assertIsNone(load_paper_analysis(history_id))

    def test_retained_pdf_accessor_rejects_database_paths_outside_paper_store(self):
        history_id = save_paper_analysis(
            pdf_data=b"%PDF-1.7 bound path",
            result=_result("Bound Path"),
            snippets=[_snippet("evidence")],
        )
        self.assertIsNotNone(retained_paper_pdf_path(history_id))
        outside = Path(self.temp_directory.name) / "outside.pdf"
        outside.write_bytes(b"%PDF-1.7 outside")
        with history_database_connection() as connection:
            connection.execute(
                "UPDATE paper_history SET pdf_path = ? WHERE id = ?",
                (str(outside), history_id),
            )

        self.assertIsNone(retained_paper_pdf_path(history_id))

    def test_demo_stream_persists_and_history_api_restores_result(self):
        paper = ParsedPaper(
            title="Streamed Paper",
            full_text="Abstract\nPersistent evidence text.",
            sections=[Section("Abstract", "Persistent evidence text.", 0, 0)],
        )

        events = [
            json.loads(event)
            for event in _stream_demo_analysis(
                paper,
                "streamed.pdf",
                24,
                b"%PDF-1.7 streamed paper",
            )
        ]
        complete = next(event for event in events if event["type"] == "complete")
        listed = paper_history(limit=100)
        restored = history_analysis(complete["history_id"])

        self.assertEqual(len(listed["items"]), 1)
        self.assertEqual(restored["paper"]["title"], "Streamed Paper")
        self.assertEqual(restored["history_id"], complete["history_id"])
        self.assertIsNone(restored["analysis_id"])

    def test_live_history_restore_recreates_full_evidence_chat_session(self):
        history_id = save_paper_analysis(
            pdf_data=b"%PDF-1.7 live restore",
            result=_result("Restored Live Paper"),
            snippets=[_snippet("full restored evidence")],
        )

        restored = history_analysis(history_id)
        session = get_analysis_session(restored["analysis_id"])

        self.assertIsNotNone(session)
        self.assertEqual(session.snippets[0].text, "full restored evidence")


def _result(title: str):
    return {
        "mode": "live",
        "analysis_id": "volatile-session-id",
        "paper": {
            "title": title,
            "filename": "paper.pdf",
            "pages": 3,
            "sections_count": 2,
            "size_bytes": 1024,
        },
        "summary_output": {"one_sentence_summary": "summary"},
        "evidence_index": [{"id": "E001", "preview": "preview"}],
    }


def _snippet(text: str):
    return EvidenceSnippet(
        id="E001",
        section="Abstract",
        page_start=0,
        page_end=0,
        text=text,
    )


if __name__ == "__main__":
    unittest.main()
