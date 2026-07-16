"""Tests for the user-facing single-paper analysis boundary."""

from core.public_analysis import (
    public_agent_output,
    public_analysis_payload,
    sanitize_visible_text,
)


def test_sanitize_visible_text_removes_supported_internal_markers():
    text = "结论 [E001]、(T002)、（T003、T009）以及 F011 均只供内部使用。"

    assert sanitize_visible_text(text) == "结论，以及 均只供内部使用。"


def test_sanitize_visible_text_preserves_domain_terms_and_comparison_ids():
    text = "F1-score、T2-weighted、P1:E003 与参考文献 [1] 都应保留。"

    assert sanitize_visible_text(text) == text


def test_public_agent_output_removes_internal_fields_recursively():
    output = {
        "main_results": "结果提升。[T002]",
        "evidence": [{"id": "T002"}],
        "novelty_dimensions": [
            {
                "reason": "机制差异有限（T003、T009）。",
                "evidence_ids": ["T003", "T009"],
            }
        ],
    }

    public = public_agent_output(output)

    assert public["main_results"] == "结果提升。"
    assert "evidence" not in public
    assert public["novelty_dimensions"] == [{"reason": "机制差异有限。"}]


def test_public_analysis_payload_keeps_only_evidence_count():
    payload = {
        "mode": "live",
        "paper": {"title": "Paper"},
        "evidence_index": [{"id": "E001"}, {"id": "T001"}],
        "experiment_output": {
            "main_results": "提升明显。[T001]",
            "evidence": [{"id": "T001"}],
        },
        "assessment": {
            "novelty": {
                "dimensions": [
                    {
                        "reason": "有支持。[E001]",
                        "evidence_ids": ["E001"],
                    }
                ]
            }
        },
    }

    public = public_analysis_payload(payload)

    assert public["evidence_count"] == 2
    assert "evidence_index" not in public
    assert public["experiment_output"] == {"main_results": "提升明显。"}
    assert public["assessment"]["novelty"]["dimensions"] == [{"reason": "有支持。"}]
    assert payload["evidence_index"][0]["id"] == "E001"
