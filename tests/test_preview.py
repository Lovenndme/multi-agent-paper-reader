"""Tests for lightweight PDF preview parsing."""

import asyncio
from io import BytesIO
import unittest
from unittest.mock import patch

from fastapi import HTTPException, UploadFile

from app import preview_paper
from core.pdf_parser import ParsedPaper, Section


class TestPaperPreview(unittest.TestCase):
    def test_returns_title_pages_sections_and_size_without_analysis(self):
        parsed = ParsedPaper(
            title="Original Paper Title",
            full_text="paper text",
            sections=[
                Section("Abstract", "summary", 0, 0),
                Section("1. Introduction", "intro", 1, 4),
            ],
        )
        upload = UploadFile(filename="renamed-local-file.pdf", file=BytesIO(b"pdf-bytes"))

        with patch("app.parse_pdf", return_value=parsed) as parser:
            payload = asyncio.run(preview_paper(upload))

        parser.assert_called_once()
        self.assertFalse(parser.call_args.kwargs["layout"])
        self.assertEqual(payload["paper"]["title"], "Original Paper Title")
        self.assertEqual(payload["paper"]["pages"], 5)
        self.assertEqual(payload["paper"]["sections_count"], 2)
        self.assertEqual(payload["paper"]["size_bytes"], 9)
        self.assertEqual(payload["paper"]["sections"][1]["title"], "1. Introduction")

    def test_rejects_non_pdf_upload(self):
        upload = UploadFile(filename="paper.txt", file=BytesIO(b"text"))
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(preview_paper(upload))
        self.assertEqual(raised.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
