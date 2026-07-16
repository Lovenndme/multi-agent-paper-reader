"""Deterministic tests for the opt-in visual resolution benchmark scorer."""

from __future__ import annotations

import copy
import unittest

from tools.benchmark_visual_resolution import CASES, field_matches, score_answer, summarize


class TestVisualResolutionBenchmark(unittest.TestCase):
    def test_field_matching_tolerates_formatting_but_not_wrong_values(self):
        self.assertTrue(field_matches("BERTScore-R", "bertscore r"))
        self.assertTrue(field_matches(["TR₁", "TR₂", "TR₃"], ["TR1", "TR2", "TR3"]))
        self.assertTrue(field_matches(27, 27.0))
        self.assertTrue(field_matches(27.04, 27.0))
        self.assertFalse(field_matches(27.06, 27.0))

    def test_scores_every_required_field(self):
        case = CASES[0]
        answer = copy.deepcopy(case.expected)
        answer["panel_b_segment_count"] = 2

        score = score_answer(case, answer)

        self.assertEqual(score["correct"], len(case.expected) - 1)
        self.assertEqual(score["total"], len(case.expected))
        self.assertFalse(score["fields"]["panel_b_segment_count"]["correct"])

    def test_summary_uses_repeat_majority_and_kind_specific_decision(self):
        results = []
        wrong_figure_fields = {
            ("figure_1", "panel_titles"),
            ("figure_2", "phase1_mask_layer"),
        }
        for profile in ("120dpi", "144dpi", "native"):
            for repeat in range(1, 4):
                for case in CASES:
                    answer = copy.deepcopy(case.expected)
                    if profile == "144dpi":
                        for case_id, field_name in wrong_figure_fields:
                            if case.id == case_id:
                                answer[field_name] = "wrong"
                    score = score_answer(case, answer)
                    results.append(
                        {
                            "key": f"r{repeat}:{profile}:{case.id}",
                            "repeat": repeat,
                            "profile": profile,
                            "case": case.id,
                            "kind": case.kind,
                            "status": "passed",
                            "elapsed_seconds": 2.0 if profile == "native" else 1.0,
                            "asset": {"bytes": 200 if profile == "native" else 100, "pixels": 400 if profile == "native" else 100},
                            "answer": answer,
                            "score": score,
                            "web_search_used": False,
                            "tools_used": [],
                        }
                    )

        summary = summarize(results)

        self.assertEqual(summary["profiles"]["native"]["correct"], 38)
        self.assertEqual(summary["profiles"]["native"]["total"], 38)
        self.assertEqual(summary["profiles"]["144dpi"]["correct"], 36)
        self.assertEqual(summary["profiles"]["native"]["repeat_consistency_percent"], 100.0)
        self.assertTrue(summary["decisions"]["figure"]["native_is_materially_better"])
        self.assertFalse(summary["decisions"]["table"]["native_is_materially_better"])
        self.assertEqual(summary["recommended_model_policy"], "hybrid-by-visual-kind")
        self.assertEqual(summary["comparisons"]["native_vs_144dpi"]["mean_latency_ratio"], 2.0)


if __name__ == "__main__":
    unittest.main()
