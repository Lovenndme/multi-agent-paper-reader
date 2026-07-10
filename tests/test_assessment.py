"""Tests for deterministic novelty and reliability assessment."""

import unittest

from core.assessment import build_analysis_assessment
from core.evidence import build_evidence_index
from core.pdf_parser import ParsedPaper, Section
from core.schemas import (
    CriticOutput,
    EvidenceItem,
    ExperimentOutput,
    MethodOutput,
    NoveltyDimensionScore,
)


class TestAnalysisAssessment(unittest.TestCase):
    def _complete_paper(self) -> ParsedPaper:
        return ParsedPaper(
            title="Assessment Test",
            full_text="x" * 6000,
            sections=[
                Section("Abstract", "The paper introduces a new research problem.", 0, 0),
                Section("Related Work", "Prior work uses an existing method.", 1, 1),
                Section("Method", "The method proposes a new model architecture.", 2, 3),
                Section("Experiments", "Experiments report benchmark results and datasets.", 4, 5),
                Section("Discussion", "Discussion covers limitations and future work.", 6, 6),
            ],
        )

    def _method(self) -> MethodOutput:
        return MethodOutput(
            research_problem="A research problem.",
            proposed_method="A proposed method.",
            key_components=["Component A"],
            innovations=["Innovation A"],
            differences_from_prior="Different from prior work.",
        )

    def _experiment(self) -> ExperimentOutput:
        return ExperimentOutput(
            datasets=["Dataset A"],
            metrics=["Accuracy"],
            main_results="The method improves accuracy.",
            comparison_with_baselines="It outperforms the baseline.",
            notable_findings=["Finding A"],
        )

    def _critic(self, evidence_ids: list[str]) -> CriticOutput:
        dimensions = [
            NoveltyDimensionScore(
                dimension="problem_originality",
                score=3,
                reason="The problem framing is meaningfully different.",
                evidence_ids=[evidence_ids[0]],
            ),
            NoveltyDimensionScore(
                dimension="method_originality",
                score=4,
                reason="The core mechanism is original.",
                evidence_ids=[evidence_ids[2]],
            ),
            NoveltyDimensionScore(
                dimension="prior_work_difference",
                score=4,
                reason="The method differs substantially from prior work.",
                evidence_ids=[evidence_ids[1], evidence_ids[2]],
            ),
            NoveltyDimensionScore(
                dimension="generality",
                score=3,
                reason="The contribution applies to several settings.",
                evidence_ids=[evidence_ids[3]],
            ),
        ]
        evidence = [
            EvidenceItem(
                id=evidence_id,
                section="Test",
                page=f"p.{index + 1}",
                quote="Evidence quote.",
                note="Supports the assessment.",
            )
            for index, evidence_id in enumerate(evidence_ids)
        ]
        return CriticOutput(
            novelty_score=4,
            novelty_justification="The work contains a substantial new mechanism.",
            novelty_dimensions=dimensions,
            strengths=["Strong method."],
            limitations=["Limited scale."],
            potential_improvements=["Evaluate more settings."],
            broader_impact=None,
            evidence=evidence,
        )

    def test_calculates_weighted_novelty_and_high_reliability(self):
        paper = self._complete_paper()
        snippets = build_evidence_index(paper)
        evidence_ids = [snippet.id for snippet in snippets[:5]]

        assessment = build_analysis_assessment(
            paper,
            snippets,
            self._method(),
            self._experiment(),
            self._critic(evidence_ids),
        )

        self.assertEqual(assessment.novelty.score, 3.7)
        self.assertEqual(assessment.novelty.label, "创新性较高")
        self.assertEqual(len(assessment.novelty.dimensions), 4)
        self.assertGreaterEqual(assessment.reliability.score, 80)
        self.assertEqual(
            assessment.reliability.raw_score,
            assessment.reliability.score,
        )
        self.assertEqual(assessment.reliability.score_cap, 100)
        self.assertEqual(assessment.reliability.level, "high")
        self.assertEqual(assessment.reliability.warnings, [])

    def test_caps_reliability_when_evidence_and_parsing_are_weak(self):
        paper = ParsedPaper(
            title="Weak Evidence",
            full_text="short",
            sections=[Section("Abstract", "A short abstract.", 0, 0)],
        )
        snippets = build_evidence_index(paper)
        critic = CriticOutput(
            novelty_score=3,
            novelty_justification="Insufficient evidence.",
            strengths=["Clear writing."],
            limitations=["Evidence is limited."],
            potential_improvements=["Add experiments."],
            broader_impact=None,
            evidence=[],
        )

        assessment = build_analysis_assessment(
            paper,
            snippets,
            self._method(),
            self._experiment(),
            critic,
        )

        self.assertLessEqual(assessment.reliability.score, 59)
        self.assertEqual(assessment.reliability.score_cap, 59)
        self.assertEqual(assessment.reliability.level, "low")
        self.assertTrue(any("相关工作" in warning for warning in assessment.reliability.warnings))
        self.assertTrue(any("有效证据" in warning for warning in assessment.reliability.warnings))
        self.assertTrue(any("解析内容不足" in warning for warning in assessment.reliability.warnings))

    def test_demo_mode_is_always_low_reliability(self):
        paper = self._complete_paper()
        snippets = build_evidence_index(paper)
        critic = self._critic([snippet.id for snippet in snippets[:5]])

        assessment = build_analysis_assessment(
            paper,
            snippets,
            self._method(),
            self._experiment(),
            critic,
            demo=True,
        )

        self.assertLessEqual(assessment.reliability.score, 39)
        self.assertEqual(assessment.reliability.score_cap, 39)
        self.assertEqual(assessment.reliability.level, "low")
        self.assertTrue(any("Demo 模式" in warning for warning in assessment.reliability.warnings))


if __name__ == "__main__":
    unittest.main()
