"""Tests for public settings metadata and local API-key persistence."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.model_providers import PROVIDERS
from core.settings import (
    ApiKeySettingsRequest,
    ApiKeyValidationError,
    ModelRoutingSettingsRequest,
    ModelRoutingValidationError,
    ProviderApiKeySettingsRequest,
    application_settings_payload,
    configure_glm_api_key,
    configure_model_routing,
    configure_provider_api_key,
)


class TestApplicationSettings(unittest.TestCase):
    def test_public_payload_never_contains_secret_material(self):
        with patch.dict(
            os.environ,
            {"TEXT_PROVIDER": "zhipu", "GLM_API_KEY": "secret-test-value"},
            clear=False,
        ):
            payload = application_settings_payload()

        self.assertTrue(payload["api_key_configured"])
        self.assertEqual(payload["version"], "V1.2.1")
        self.assertEqual(payload["routing"]["text"]["model_label"], "GLM-5.2")
        self.assertEqual(
            {provider["id"] for provider in payload["providers"]},
            {"zhipu", "deepseek", "openai", "qwen", "doubao"},
        )
        self.assertNotIn("secret-test-value", repr(payload))

    def test_request_representation_masks_api_key(self):
        request = ApiKeySettingsRequest(api_key="secret-test-value")
        provider_request = ProviderApiKeySettingsRequest(
            api_key="another-secret-value",
            base_url="https://example.com/v1",
        )

        self.assertNotIn("secret-test-value", repr(request))
        self.assertNotIn("another-secret-value", repr(provider_request))

    def test_every_provider_exposes_multiple_text_models(self):
        for provider in PROVIDERS.values():
            with self.subTest(provider=provider.id):
                self.assertGreaterEqual(len(provider.text_models), 2)

    def test_openai_default_model_uses_the_complete_sol_name(self):
        self.assertEqual(PROVIDERS["openai"].text_models[0].label, "GPT-5.6 Sol")
        self.assertEqual(PROVIDERS["openai"].text_models[0].id, "gpt-5.6")

    def test_openai_catalog_only_contains_gpt_5_6_models(self):
        model_ids = [model.id for model in PROVIDERS["openai"].text_models]

        self.assertEqual(model_ids, ["gpt-5.6", "gpt-5.6-terra", "gpt-5.6-luna"])

    def test_zhipu_text_catalog_matches_live_model_list(self):
        model_ids = [model.id for model in PROVIDERS["zhipu"].text_models]

        self.assertEqual(
            model_ids,
            [
                "glm-5.2",
                "glm-5.1",
                "glm-5",
                "glm-5-turbo",
                "glm-4.7",
                "glm-4.6",
                "glm-4.5-air",
                "glm-4.5",
            ],
        )

    def test_doubao_catalog_exposes_three_multimodal_seed_models(self):
        provider = PROVIDERS["doubao"]

        self.assertEqual(provider.api_key_env, "ARK_API_KEY")
        self.assertEqual(provider.default_base_url, "https://ark.cn-beijing.volces.com/api/v3")
        self.assertEqual(len(provider.text_models), 3)
        self.assertEqual(provider.default_vision_model, "doubao-seed-2-0-pro-260215")

    def test_invalid_key_is_not_persisted(self):
        with tempfile.TemporaryDirectory() as tempdir:
            env_path = Path(tempdir) / ".env"

            with self.assertRaises(ApiKeyValidationError):
                configure_glm_api_key("short", env_path=env_path)

            self.assertFalse(env_path.exists())

    def test_validated_key_is_saved_and_activated(self):
        previous_key = os.environ.get("GLM_API_KEY")
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                env_path = Path(tempdir) / ".env"
                with (
                    patch("core.settings._probe_glm_api_key"),
                    patch("core.settings.reset_llm_clients") as reset_clients,
                ):
                    payload = configure_glm_api_key(
                        "test-key-that-is-long-enough",
                        env_path=env_path,
                    )

                self.assertTrue(payload["api_key_configured"])
                self.assertIn("GLM_API_KEY='test-key-that-is-long-enough'", env_path.read_text())
                if os.name != "nt":
                    self.assertEqual(env_path.stat().st_mode & 0o777, 0o600)
                reset_clients.assert_called_once_with()
        finally:
            if previous_key is None:
                os.environ.pop("GLM_API_KEY", None)
            else:
                os.environ["GLM_API_KEY"] = previous_key

    def test_provider_key_is_saved_under_provider_specific_names(self):
        with tempfile.TemporaryDirectory() as tempdir:
            env_path = Path(tempdir) / ".env"
            with (
                patch.dict(os.environ, {"TEXT_PROVIDER": "qwen"}, clear=False),
                patch("core.settings._probe_provider_api_key", return_value=["qwen3.7-max"]),
                patch("core.settings.reset_llm_clients") as reset_clients,
            ):
                payload = configure_provider_api_key(
                    "qwen",
                    "qwen-key-that-is-long-enough",
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/",
                    env_path=env_path,
                )

            saved = env_path.read_text()
            self.assertIn("DASHSCOPE_API_KEY='qwen-key-that-is-long-enough'", saved)
            self.assertIn(
                "QWEN_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'",
                saved,
            )
            self.assertNotIn("OPENAI_API_KEY", saved)
            self.assertEqual(payload["validation"]["available_model_count"], 1)
            reset_clients.assert_called_once_with()

    def test_text_and_vision_routes_are_persisted_for_one_provider(self):
        with tempfile.TemporaryDirectory() as tempdir:
            env_path = Path(tempdir) / ".env"
            request = ModelRoutingSettingsRequest(
                text_provider="zhipu",
                text_model="glm-4.7",
                vision_enabled=True,
                vision_provider="zhipu",
                vision_model="glm-5v-turbo",
            )
            with (
                patch.dict(
                    os.environ,
                    {"TEXT_PROVIDER": "zhipu", "GLM_API_KEY": "configured-zhipu-key"},
                    clear=False,
                ),
                patch("core.settings.reset_llm_clients") as reset_clients,
            ):
                payload = configure_model_routing(request, env_path=env_path)

            saved = env_path.read_text()
            self.assertIn("TEXT_PROVIDER='zhipu'", saved)
            self.assertIn("MODEL_NAME='glm-4.7'", saved)
            self.assertIn("VISION_PROVIDER='zhipu'", saved)
            self.assertIn("VISION_MODEL_NAME='glm-5v-turbo'", saved)
            self.assertEqual(payload["routing"]["text"]["provider"], "zhipu")
            self.assertEqual(payload["routing"]["vision"]["provider"], "zhipu")
            reset_clients.assert_called_once_with()

    def test_doubao_route_uses_the_same_provider_for_text_and_vision(self):
        with tempfile.TemporaryDirectory() as tempdir:
            env_path = Path(tempdir) / ".env"
            request = ModelRoutingSettingsRequest(
                text_provider="doubao",
                text_model="doubao-seed-2-0-lite-260215",
                vision_enabled=True,
                vision_provider="doubao",
                vision_model="doubao-seed-2-0-pro-260215",
            )
            with (
                patch.dict(
                    os.environ,
                    {"TEXT_PROVIDER": "doubao", "ARK_API_KEY": "configured-doubao-key"},
                    clear=True,
                ),
                patch("core.settings.reset_llm_clients"),
            ):
                payload = configure_model_routing(request, env_path=env_path)

            saved = env_path.read_text()
            self.assertIn("TEXT_PROVIDER='doubao'", saved)
            self.assertIn("MODEL_NAME='doubao-seed-2-0-lite-260215'", saved)
            self.assertIn("VISION_PROVIDER='doubao'", saved)
            self.assertIn("VISION_MODEL_NAME='doubao-seed-2-0-pro-260215'", saved)
            self.assertEqual(payload["routing"]["vision"]["provider"], "doubao")

    def test_unconfigured_provider_route_is_rejected_without_persistence(self):
        with tempfile.TemporaryDirectory() as tempdir:
            env_path = Path(tempdir) / ".env"
            request = ModelRoutingSettingsRequest(
                text_provider="qwen",
                text_model="qwen3.7-plus",
                vision_enabled=True,
                vision_provider="qwen",
                vision_model="qwen3.7-plus",
            )
            with patch.dict(
                os.environ,
                {"TEXT_PROVIDER": "qwen"},
                clear=True,
            ):
                with self.assertRaisesRegex(ModelRoutingValidationError, "配置并验证 API Key"):
                    configure_model_routing(request, env_path=env_path)

            self.assertFalse(env_path.exists())

    def test_cross_provider_vision_route_is_rejected(self):
        request = ModelRoutingSettingsRequest(
            text_provider="zhipu",
            text_model="glm-5.2",
            vision_enabled=True,
            vision_provider="qwen",
            vision_model="qwen3.7-plus",
        )
        with patch.dict(
            os.environ,
            {"TEXT_PROVIDER": "zhipu", "GLM_API_KEY": "configured-zhipu-key"},
            clear=True,
        ):
            with self.assertRaisesRegex(ModelRoutingValidationError, "同一家厂商"):
                configure_model_routing(request)

    def test_deepseek_cannot_be_selected_as_hosted_vision_provider(self):
        request = ModelRoutingSettingsRequest(
            text_provider="deepseek",
            text_model="deepseek-v4-pro",
            vision_enabled=True,
            vision_provider="deepseek",
            vision_model="deepseek-vl2",
        )

        with patch.dict(
            os.environ,
            {"TEXT_PROVIDER": "deepseek", "DEEPSEEK_API_KEY": "configured-deepseek-key"},
            clear=True,
        ):
            with self.assertRaisesRegex(ModelRoutingValidationError, "不提供视觉模型"):
                configure_model_routing(request)


if __name__ == "__main__":
    unittest.main()
