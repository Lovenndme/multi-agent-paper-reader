import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from core.chat_memory import ConversationMemoryDigest
from core.schemas import ExperimentOutput
from utils.llm import (
    get_api_key,
    get_base_url,
    get_chat_llm,
    get_llm,
    get_vision_llm,
    is_vision_configured,
    parse_structured_output,
    start_text_model_call_trace,
    stream_structured_with_retry,
    update_text_model_call_trace,
)


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
    def tearDown(self):
        get_llm.cache_clear()
        get_chat_llm.cache_clear()
        get_vision_llm.cache_clear()

    def test_chat_model_uses_separate_low_temperature(self):
        get_chat_llm.cache_clear()
        with (
            patch.dict(os.environ, {"CHAT_TEMPERATURE": "0.2"}),
            patch("utils.llm.get_api_key", return_value="test-key"),
        ):
            chat_llm = get_chat_llm()

        self.assertEqual(chat_llm.temperature, 0.2)

    def test_qwen_route_uses_its_own_key_base_url_and_model(self):
        get_llm.cache_clear()
        with patch.dict(
            os.environ,
            {
                "TEXT_PROVIDER": "qwen",
                "DASHSCOPE_API_KEY": "qwen-test-key",
                "QWEN_BASE_URL": "https://dashscope.example/v1",
                "MODEL_NAME": "qwen3.7-plus",
            },
            clear=True,
        ):
            llm = get_llm()
            api_key = get_api_key()
            base_url = get_base_url()

        self.assertEqual(api_key, "qwen-test-key")
        self.assertEqual(base_url, "https://dashscope.example/v1")
        self.assertEqual(llm.model_name, "qwen3.7-plus")

    def test_model_call_trace_combines_client_route_and_upstream_metadata(self):
        with patch.dict(
            os.environ,
            {
                "TEXT_PROVIDER": "qwen",
                "MODEL_NAME": "qwen3.7-max",
                "DASHSCOPE_API_KEY": "qwen-test-key",
            },
            clear=True,
        ):
            trace = start_text_model_call_trace(
                SimpleNamespace(
                    model_name="qwen3.7-max",
                    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
            )
            update_text_model_call_trace(
                trace,
                SimpleNamespace(
                    response_metadata={
                        "model_name": "qwen3.7-max",
                        "headers": {"x-request-id": "qwen-request-123"},
                    }
                ),
            )

        self.assertEqual(trace["provider"], "qwen")
        self.assertEqual(trace["endpoint_host"], "dashscope.aliyuncs.com")
        self.assertEqual(trace["upstream_model"], "qwen3.7-max")
        self.assertEqual(trace["request_id"], "qwen-request-123")
        self.assertEqual(trace["verification"], "upstream_confirmed")

    def test_doubao_route_uses_ark_key_base_url_and_model(self):
        get_llm.cache_clear()
        with patch.dict(
            os.environ,
            {
                "TEXT_PROVIDER": "doubao",
                "ARK_API_KEY": "doubao-test-key",
                "DOUBAO_BASE_URL": "https://ark.example/api/v3",
                "MODEL_NAME": "doubao-seed-2-0-lite-260215",
            },
            clear=True,
        ):
            llm = get_llm()
            api_key = get_api_key()
            base_url = get_base_url()

        self.assertEqual(api_key, "doubao-test-key")
        self.assertEqual(base_url, "https://ark.example/api/v3")
        self.assertEqual(llm.model_name, "doubao-seed-2-0-lite-260215")

    def test_vision_route_ignores_a_different_legacy_provider(self):
        get_vision_llm.cache_clear()
        with patch.dict(
            os.environ,
            {
                "TEXT_PROVIDER": "zhipu",
                "GLM_API_KEY": "zhipu-test-key",
                "GLM_BASE_URL": "https://open.bigmodel.cn/api/paas/v4",
                "MODEL_NAME": "glm-5.2",
                "VISION_PROVIDER": "qwen",
                "DASHSCOPE_API_KEY": "qwen-test-key",
                "VISION_MODEL_NAME": "qwen3-vl-plus",
                "ENABLE_VISION_SUMMARY": "true",
            },
            clear=True,
        ):
            vision_llm = get_vision_llm()
            configured = is_vision_configured()

        self.assertTrue(configured)
        self.assertEqual(vision_llm.model_name, "glm-5v-turbo")

    def test_deepseek_route_does_not_claim_hosted_vision_support(self):
        with patch.dict(
            os.environ,
            {
                "TEXT_PROVIDER": "deepseek",
                "DEEPSEEK_API_KEY": "deepseek-test-key",
                "VISION_PROVIDER": "deepseek",
                "VISION_MODEL_NAME": "deepseek-vl2",
                "ENABLE_VISION_SUMMARY": "true",
            },
            clear=True,
        ):
            self.assertFalse(is_vision_configured())

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

    def test_accepts_memory_topic_list_as_provider_shape_fallback(self):
        parsed = parse_structured_output(
            '[{"topic":"多头注意力","content":"用户重点关注 E001 及其机制。"}]',
            ConversationMemoryDigest,
        )

        self.assertEqual(parsed.topics[0].topic, "多头注意力")
        self.assertIn("E001", parsed.summary)

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
