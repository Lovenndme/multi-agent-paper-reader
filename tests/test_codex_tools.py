"""Security and contract tests for the capability-bound paper MCP tools."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from core.codex_tools import (
    PaperToolError,
    build_codex_paper_manifest,
    calculate,
    create_codex_tool_context,
    create_codex_tool_context_from_history,
    paper_get_figure,
    paper_get_overview,
    paper_get_page,
    paper_get_page_image,
    paper_get_section,
    paper_get_table,
    paper_get_visual_region,
    paper_search_evidence,
)
from core.evidence import EvidenceSnippet
from core.history import save_paper_analysis
from core.pdf_parser import FigureBlock, ParsedPaper, Section, TableBlock


class TestCodexPaperTools(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tempdir.name) / "data"
        self.pdf_path = Path(self.tempdir.name) / "paper.pdf"
        document = fitz.open()
        page = document.new_page(width=400, height=400)
        page.insert_text((40, 35), "Alpha method page")
        page.draw_rect(fitz.Rect(40, 60, 220, 160))
        page.insert_text((50, 90), "Figure content")
        page.draw_rect(fitz.Rect(40, 210, 300, 310))
        page.insert_text((50, 240), "Table content")
        document.save(self.pdf_path)
        document.close()
        self.paper = ParsedPaper(
            title="Tool Test",
            full_text="Alpha method page",
            sections=[Section("Methods", "The alpha coefficient is 0.42.", 0, 0)],
            figures=[
                FigureBlock(
                    page=0,
                    caption="Figure 1. Alpha architecture",
                    bbox=(40, 60, 220, 160),
                )
            ],
            tables=[
                TableBlock(
                    page=0,
                    caption="Table 1. Scores",
                    rows=[["Model", "Score"], ["Alpha", "0.42"]],
                    bbox=(40, 210, 300, 310),
                )
            ],
            metadata={"author": "Ada Example", "parser_backend": "pymupdf4llm"},
        )
        self.snippets = [
            EvidenceSnippet("E001", "Methods", 0, 0, "The alpha coefficient is 0.42."),
            EvidenceSnippet("F001", "Figure 1", 0, 0, "Caption: Alpha architecture", "figure"),
            EvidenceSnippet("T001", "Table 1", 0, 0, "| Model | Score |\n| --- | --- |\n| Alpha | 0.42 |", "table"),
        ]

    def tearDown(self):
        self.tempdir.cleanup()

    def _context(self):
        return create_codex_tool_context(
            snippets=self.snippets,
            paper=self.paper,
            pdf_path=self.pdf_path,
        )

    def test_context_is_private_and_tools_are_bound_to_it(self):
        with patch.dict(os.environ, {"PAPER_READER_DATA_DIR": str(self.data_dir)}, clear=False):
            handle = self._context()
            try:
                self.assertEqual(handle.path.stat().st_mode & 0o777, 0o600)
                with patch.dict(
                    os.environ,
                    {"PAPER_READER_CODEX_CONTEXT_FILE": str(handle.path)},
                    clear=False,
                ):
                    search = paper_search_evidence("alpha")
                    overview = paper_get_overview()
                    section = paper_get_section("Methods")
                    page = paper_get_page(1)
                    page_metadata, page_png = paper_get_page_image(1)
                    clamped_metadata, _ = paper_get_page_image(1, dpi=999)
                    figure = paper_get_figure("F001")
                    table = paper_get_table("T001")
                    metadata, png = paper_get_visual_region("F001")
                self.assertEqual(search["matches"][0]["id"], "E001")
                self.assertEqual(overview["title"], "Tool Test")
                self.assertEqual(overview["metadata"]["author"], "Ada Example")
                self.assertEqual(overview["counts"], {"sections": 1, "figures": 1, "tables": 1})
                self.assertEqual([item["id"] for item in overview["assets"]], ["F001", "T001"])
                self.assertNotIn(str(self.pdf_path), repr(overview))
                self.assertIn("0.42", section["text"])
                self.assertIn("Alpha method", page["text"])
                self.assertEqual(page_metadata["page"], 1)
                self.assertTrue(page_png.startswith(b"\x89PNG"))
                self.assertEqual(clamped_metadata["dpi"], 144)
                self.assertEqual(set(figure), {"id", "page", "caption", "visual_summary", "bbox_verified"})
                self.assertEqual(set(table), {"id", "page", "caption", "rows", "truncated", "bbox_verified"})
                self.assertEqual(metadata["id"], "F001")
                self.assertEqual(metadata["dpi"], 144)
                self.assertEqual(metadata["quality"], "model-preview")
                self.assertTrue(png.startswith(b"\x89PNG"))
            finally:
                handle.close()
            self.assertFalse(handle.path.exists())

    def test_page_image_rejects_out_of_range_pages(self):
        with patch.dict(os.environ, {"PAPER_READER_DATA_DIR": str(self.data_dir)}, clear=False):
            handle = self._context()
            try:
                with patch.dict(
                    os.environ,
                    {"PAPER_READER_CODEX_CONTEXT_FILE": str(handle.path)},
                    clear=False,
                ):
                    with self.assertRaises(PaperToolError):
                        paper_get_page_image(0)
                    with self.assertRaises(PaperToolError):
                        paper_get_page_image(2)
            finally:
                handle.close()

    def test_mcp_tool_import_does_not_load_secret_environment_from_dotenv(self):
        environment = os.environ.copy()
        secret_markers = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")
        for key in list(environment):
            if any(marker in key.upper() for marker in secret_markers):
                environment.pop(key, None)
        script = (
            "import json, os; before=set(os.environ); import core.codex_tools; "
            "print(json.dumps(sorted(key for key in os.environ if key not in before "
            "and any(marker in key.upper() for marker in "
            "('API_KEY','TOKEN','SECRET','PASSWORD')))))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parent.parent,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(json.loads(completed.stdout.strip()), [])

    def test_new_context_prunes_only_abandoned_capability_files(self):
        context_dir = self.data_dir / "codex-tool-contexts"
        context_dir.mkdir(parents=True)
        stale = context_dir / f"{'a' * 48}.json"
        unrelated = context_dir / "keep-me.txt"
        stale.write_text("{}", encoding="utf-8")
        stale.chmod(0o600)
        os.utime(stale, (1, 1))
        unrelated.write_text("not a capability", encoding="utf-8")

        with patch.dict(os.environ, {"PAPER_READER_DATA_DIR": str(self.data_dir)}, clear=False):
            handle = self._context()
            try:
                self.assertFalse(stale.exists())
                self.assertTrue(unrelated.exists())
                self.assertTrue(handle.path.exists())
            finally:
                handle.close()

    def test_calculator_rejects_python_and_unbounded_work(self):
        self.assertEqual(calculate("sqrt(81) + 7 * 8")["result"], 65.0)
        for expression in (
            "__import__('os').system('id')",
            "(1).__class__",
            "2 ** 1000",
            "[x for x in range(3)]",
        ):
            with self.subTest(expression=expression), self.assertRaises(PaperToolError):
                calculate(expression)

    def test_context_bounds_section_content_before_writing_capability_file(self):
        self.paper.sections[0].content = "x" * 20_000
        with patch.dict(os.environ, {"PAPER_READER_DATA_DIR": str(self.data_dir)}, clear=False):
            handle = self._context()
            try:
                with patch.dict(
                    os.environ,
                    {"PAPER_READER_CODEX_CONTEXT_FILE": str(handle.path)},
                    clear=False,
                ):
                    section = paper_get_section("Methods")
                self.assertEqual(len(section["text"]), 12_000)
                self.assertTrue(section["truncated"])
                self.assertLess(handle.path.stat().st_size, 100_000)
            finally:
                handle.close()

    def test_visual_tool_refuses_unverified_bbox(self):
        self.paper.figures[0].bbox = None
        with patch.dict(os.environ, {"PAPER_READER_DATA_DIR": str(self.data_dir)}, clear=False):
            handle = self._context()
            try:
                with patch.dict(
                    os.environ,
                    {"PAPER_READER_CODEX_CONTEXT_FILE": str(handle.path)},
                    clear=False,
                ):
                    # A missing verified bbox fails closed; the explicit page-image tool
                    # must never become an automatic fallback for a visual-region call.
                    with patch("core.codex_tools.parse_pdf") as parser, self.assertRaises(PaperToolError):
                        paper_get_visual_region("F001")
                    parser.assert_not_called()
            finally:
                handle.close()

    def test_persisted_manifest_keeps_history_tool_contract_and_visual_ids(self):
        result = {
            "mode": "live",
            "paper": {
                "title": self.paper.title,
                "filename": "paper.pdf",
                "pages": 1,
                "sections_count": 1,
                "sections": [{"title": "Methods", "page_start": 0, "page_end": 0, "chars": 30}],
                "metadata": self.paper.metadata,
            },
        }
        with patch.dict(os.environ, {"PAPER_READER_DATA_DIR": str(self.data_dir)}, clear=False):
            history_id = save_paper_analysis(
                pdf_data=self.pdf_path.read_bytes(),
                result=result,
                snippets=self.snippets,
                paper_manifest=build_codex_paper_manifest(self.paper),
            )
            handle = create_codex_tool_context_from_history(history_id)
            try:
                with patch.dict(
                    os.environ,
                    {"PAPER_READER_CODEX_CONTEXT_FILE": str(handle.path)},
                    clear=False,
                ):
                    overview = paper_get_overview()
                    figure = paper_get_figure("F001")
                    table = paper_get_table("T001")
                    metadata, png = paper_get_visual_region("F001")
                self.assertFalse(overview["legacy_reparsed"])
                self.assertEqual([item["id"] for item in overview["assets"]], ["F001", "T001"])
                self.assertEqual(figure["page"], 1)
                self.assertEqual(table["rows"][1], ["Alpha", "0.42"])
                self.assertEqual(metadata["id"], "F001")
                self.assertTrue(png.startswith(b"\x89PNG"))
            finally:
                handle.close()

    def test_stdio_mcp_lists_exact_read_only_surface(self):
        async def inspect_server(context_path: Path) -> None:
            environment = os.environ.copy()
            environment["PAPER_READER_CODEX_CONTEXT_FILE"] = str(context_path)
            environment["PAPER_READER_DATA_DIR"] = str(self.data_dir)
            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "core.codex_tools_mcp"],
                cwd=str(Path(__file__).resolve().parent.parent),
                env=environment,
            )
            async with stdio_client(params) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    await session.initialize()
                    listing = await session.list_tools()
                    self.assertEqual(
                        [tool.name for tool in listing.tools],
                        [
                            "paper_get_overview",
                            "paper_search_evidence",
                            "paper_get_section",
                            "paper_get_page",
                            "paper_get_page_image",
                            "paper_get_figure",
                            "paper_get_table",
                            "paper_get_visual_region",
                            "paper_recall_memory",
                            "calculate",
                        ],
                    )
                    self.assertTrue(
                        all(tool.annotations and tool.annotations.readOnlyHint for tool in listing.tools)
                    )
                    self.assertTrue(
                        all(tool.annotations and tool.annotations.idempotentHint for tool in listing.tools)
                    )
                    self.assertTrue(
                        all(tool.annotations and not tool.annotations.openWorldHint for tool in listing.tools)
                    )

        with patch.dict(os.environ, {"PAPER_READER_DATA_DIR": str(self.data_dir)}, clear=False):
            handle = self._context()
            try:
                asyncio.run(inspect_server(handle.path))
            finally:
                handle.close()


if __name__ == "__main__":
    unittest.main()
