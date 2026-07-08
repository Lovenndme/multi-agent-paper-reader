"""Tests for evidence-grounded snippet selection."""

import unittest

from core.evidence import build_evidence_index, evidence_context_for_agent
from core.pdf_parser import FigureBlock, ParsedPaper, Section, TableBlock


class TestEvidenceIndex(unittest.TestCase):
    def _paper(self):
        return ParsedPaper(
            title="Evidence Test",
            full_text="full",
            sections=[
                Section("Abstract", "This paper proposes a framework.", 0, 0),
                Section("Method", "The model architecture uses a new training objective.", 1, 2),
                Section("Experiments", "Datasets include WMT14. Results use BLEU and ablation tables.", 3, 4),
            ],
            tables=[
                TableBlock(
                    page=4,
                    caption="Table 1: Main results on WMT14.",
                    rows=[
                        ["Model", "BLEU"],
                        ["Baseline", "27.3"],
                        ["Ours", "29.8"],
                    ],
                )
            ],
            figures=[
                FigureBlock(
                    page=1,
                    caption="Figure 1: Model architecture overview.",
                    image_index=1,
                    bbox=(10.0, 10.0, 200.0, 150.0),
                )
            ],
        )

    def test_builds_traceable_snippets(self):
        snippets = build_evidence_index(self._paper(), chunk_chars=80, overlap_chars=10)
        self.assertGreaterEqual(len(snippets), 3)
        self.assertEqual(snippets[0].id, "E001")
        self.assertEqual(snippets[0].page_label, "p.1")

    def test_selects_agent_relevant_context(self):
        snippets = build_evidence_index(self._paper(), chunk_chars=120, overlap_chars=10)
        method_context = evidence_context_for_agent(snippets, "method")
        experiment_context = evidence_context_for_agent(snippets, "experiment")

        self.assertIn("model architecture", method_context)
        self.assertIn("E", method_context)
        self.assertIn("BLEU", experiment_context)

    def test_includes_table_and_figure_evidence(self):
        snippets = build_evidence_index(self._paper(), chunk_chars=120, overlap_chars=10)
        ids = {snippet.id for snippet in snippets}
        self.assertIn("T001", ids)
        self.assertIn("F001", ids)

        experiment_context = evidence_context_for_agent(snippets, "experiment")
        method_context = evidence_context_for_agent(snippets, "method")

        self.assertIn("[T001 | table", experiment_context)
        self.assertIn("| Ours | 29.8 |", experiment_context)
        self.assertIn("[F001 | figure", method_context)
        self.assertIn("Model architecture overview", method_context)


if __name__ == "__main__":
    unittest.main()
