import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from core.schemas import ExperimentOutput
from utils.llm import parse_structured_output, stream_structured_with_retry


class FakeStreamingLLM:
    def stream(self, _messages):
        for part in (
            '{"datasets":["WMT14"],',
            '"metrics":["BLEU"],',
            '"main_results":"达到较强结果。",',
            '"comparison_with_baselines":"优于基线。",',
            '"ablation_study":null,',
            '"notable_findings":["训练更高效。"]}',
        ):
            yield AIMessage(content=part)


class FakeRepairChain:
    def invoke(self, _messages):
        return ExperimentOutput(
            datasets=["MATH-500"],
            metrics=["pass@1"],
            main_results="修复调用返回完整对象。",
            comparison_with_baselines="与基线相比表现更好。",
            ablation_study=None,
            notable_findings=["自动修复了错误的顶层数组。"],
        )


class FakeNeedsRepairLLM:
    def stream(self, _messages):
        yield AIMessage(content="not valid json")

    def with_structured_output(self, _schema):
        return FakeRepairChain()


class TestStructuredOutputParsing(unittest.TestCase):
    def test_accepts_fenced_json_from_compatible_provider(self):
        response = AIMessage(
            content="""```json
{
  "datasets": ["WMT 2014 English-German"],
  "metrics": ["BLEU"],
  "main_results": "The model reaches strong translation quality.",
  "comparison_with_baselines": "It outperforms recurrent baselines.",
  "ablation_study": null,
  "notable_findings": ["Attention-only models train efficiently."]
}
```"""
        )

        parsed = parse_structured_output(response, ExperimentOutput)

        self.assertEqual(parsed.datasets, ["WMT 2014 English-German"])
        self.assertEqual(parsed.metrics, ["BLEU"])
        self.assertIsNone(parsed.ablation_study)

    def test_stream_structured_output_emits_tokens_and_parses_json(self):
        tokens = []
        with patch("utils.llm.get_llm", return_value=FakeStreamingLLM()):
            parsed = stream_structured_with_retry(
                ExperimentOutput,
                [HumanMessage(content="Analyze experiments.")],
                on_token=tokens.append,
            )

        self.assertGreater(len(tokens), 1)
        self.assertEqual(parsed.datasets, ["WMT14"])
        self.assertEqual(parsed.notable_findings, ["训练更高效。"])

    def test_accepts_experiment_list_as_provider_shape_fallback(self):
        parsed = parse_structured_output(
            '["MATH-500", "AIME 2024", "GPQA Diamond"]',
            ExperimentOutput,
        )

        self.assertEqual(parsed.datasets, ["MATH-500", "AIME 2024", "GPQA Diamond"])
        self.assertIn("未按完整结构输出", parsed.main_results)

    def test_stream_structured_output_repairs_wrong_top_level_shape(self):
        tokens = []
        with patch("utils.llm.get_llm", return_value=FakeNeedsRepairLLM()):
            parsed = stream_structured_with_retry(
                ExperimentOutput,
                [HumanMessage(content="Analyze experiments.")],
                on_token=tokens.append,
            )

        self.assertIn("自动修正", "".join(tokens))
        self.assertEqual(parsed.datasets, ["MATH-500"])


if __name__ == "__main__":
    unittest.main()
