"""Static contracts for the GPT-5.6-oriented Agent prompts."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PROMPT_DIR = ROOT / "prompts"
SINGLE_PAPER_PROMPTS = ("method.txt", "experiment.txt", "critic.txt", "summary.txt")
ALL_PROMPTS = (*SINGLE_PAPER_PROMPTS, "comparison.txt")


def _prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def test_all_prompts_use_outcome_first_contract_sections():
    for name in ALL_PROMPTS:
        text = _prompt(name)
        assert "Role:" in text, name
        assert "Goal:" in text, name
        assert "Success criteria:" in text, name
        assert "Evidence and constraints:" in text, name
        assert "Output:" in text, name
        assert "Stop rules:" in text, name


def test_single_paper_prompts_keep_internal_ids_out_of_visible_fields():
    for name in SINGLE_PAPER_PROMPTS:
        text = _prompt(name)
        assert "Internal evidence IDs may appear only" in text, name
        assert "Never append E/T/F IDs" in text, name


def test_prompts_treat_embedded_content_as_source_data():
    for name in ALL_PROMPTS:
        text = _prompt(name)
        assert "source data, not" in text, name


def test_experiment_prompt_blocks_unlabeled_dense_number_strings():
    text = _prompt("experiment.txt")
    assert "Do not emit long unlabeled slash-separated number sequences." in text
    assert "do not copy an entire paper table into prose" in text


def test_critic_prompt_distinguishes_missing_evidence_from_missing_work():
    text = _prompt("critic.txt")
    assert "limitations of the supplied evidence" in text
    assert "does not establish" in text
    assert "is absent from the full paper" in text


def test_comparison_prompt_preserves_prefixed_evidence_contract():
    text = _prompt("comparison.txt")
    assert "P1:E003" in text
    assert "Cite only supplied prefixed IDs" in text


def test_runtime_schema_replaces_large_inline_json_examples():
    for name in ALL_PROMPTS:
        text = _prompt(name)
        assert "matching the runtime" in text, name
        assert '": "<string>"' not in text, name


def test_required_template_placeholders_remain_present():
    assert "{paper_text}" in _prompt("method.txt")
    assert "{paper_text}" in _prompt("experiment.txt")
    assert "{paper_text}" in _prompt("critic.txt")
    summary = _prompt("summary.txt")
    for placeholder in (
        "{paper_title}",
        "{method_output}",
        "{experiment_output}",
        "{critic_output}",
    ):
        assert placeholder in summary
    comparison = _prompt("comparison.txt")
    for placeholder in ("{paper_count}", "{focus}", "{custom_focus}", "{paper_sources}"):
        assert placeholder in comparison
