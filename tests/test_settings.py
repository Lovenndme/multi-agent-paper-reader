"""Tests for public settings metadata and local API-key persistence."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.codex_sdk import CodexModelInfo
from core.model_providers import PROVIDERS
from core.settings import (
    ApiKeySettingsRequest,
    ApiKeyValidationError,
    ModelRoutingSettingsRequest,
    ModelRoutingValidationError,
    PROJECT_VERSION,
    ProviderApiKeySettingsRequest,
    application_settings_payload,
    configure_glm_api_key,
    configure_model_routing,
    configure_provider_api_key,
)


class TestApplicationSettings(unittest.TestCase):
    def setUp(self):
        self.codex_service = SimpleNamespace(
            status=MagicMock(
                return_value={
                    "installed": True,
                    "runtime_ready": True,
                    "authenticated": False,
                    "auth_mode": None,
                    "plan_type": None,
                    "message": "本机 Codex 尚未登录 ChatGPT。",
                }
            ),
            models=MagicMock(return_value=()),
        )
        self.codex_service_patch = patch(
            "core.codex_sdk.get_codex_sdk_service",
            return_value=self.codex_service,
        )
        self.codex_service_patch.start()
        self.addCleanup(self.codex_service_patch.stop)

    def test_public_payload_never_contains_secret_material(self):
        with patch.dict(
            os.environ,
            {"TEXT_PROVIDER": "zhipu", "GLM_API_KEY": "secret-test-value"},
            clear=False,
        ):
            payload = application_settings_payload()

        self.assertTrue(payload["api_key_configured"])
        self.assertEqual(payload["version"], PROJECT_VERSION)
        self.assertEqual(payload["routing"]["text"]["model_label"], "GLM-5.2")
        self.assertEqual(
            {provider["id"] for provider in payload["providers"]},
            {
                "zhipu",
                "deepseek",
                "openai",
                "qwen",
                "doubao",
                "anthropic",
                "kimi",
                "codex",
                "custom",
            },
        )
        self.assertNotIn("secret-test-value", repr(payload))

    def test_codex_payload_uses_live_models_without_account_identity(self):
        model = CodexModelInfo(
            id="gpt-test",
            label="GPT Test",
            description="Account model",
            recommended=True,
            supports_image=True,
            default_effort="medium",
            efforts=(("medium", "Balanced"), ("high", "Deeper")),
        )
        self.codex_service.status.return_value = {
            "installed": True,
            "runtime_ready": True,
            "authenticated": True,
            "auth_mode": "chatgpt",
            "plan_type": "plus",
            "message": "已连接本机 Codex 订阅。",
        }
        self.codex_service.models.return_value = (model,)

        with patch.dict(
            os.environ,
            {
                "TEXT_PROVIDER": "codex",
                "MODEL_NAME": "gpt-test",
                "MODEL_MODE": "medium",
            },
            clear=True,
        ):
            payload = application_settings_payload()

        provider = next(item for item in payload["providers"] if item["id"] == "codex")
        self.assertTrue(provider["configured"])
        self.assertEqual(provider["credential_type"], "codex_login")
        self.assertTrue(provider["local_only"])
        self.assertEqual([item["id"] for item in provider["text_models"]], ["gpt-test"])
        self.assertEqual(provider["text_models"][0]["default_mode"], "medium")
        self.assertEqual(payload["routing"]["vision"]["model"], "gpt-test")
        self.assertNotIn("email", repr(payload).lower())
        self.assertNotIn("token", repr(payload).lower())

    def test_codex_route_persists_only_route_metadata(self):
        model = CodexModelInfo(
            id="gpt-test",
            label="GPT Test",
            description="Account model",
            recommended=True,
            supports_image=True,
            default_effort="medium",
            efforts=(("medium", "Balanced"), ("high", "Deeper")),
        )
        self.codex_service.status.return_value = {
            "runtime_ready": True,
            "authenticated": True,
            "auth_mode": "chatgpt",
            "plan_type": "plus",
        }
        self.codex_service.models.return_value = (model,)
        request = ModelRoutingSettingsRequest(
            text_provider="codex",
            text_model="gpt-test",
            text_mode="high",
            vision_enabled=True,
            vision_provider="codex",
            vision_model="gpt-test",
        )

        with tempfile.TemporaryDirectory() as tempdir:
            env_path = Path(tempdir) / ".env"
            with patch.dict(os.environ, {}, clear=True):
                payload = configure_model_routing(request, env_path=env_path)
            saved = env_path.read_text()

        self.assertIn("TEXT_PROVIDER='codex'", saved)
        self.assertIn("MODEL_NAME='gpt-test'", saved)
        self.assertIn("MODEL_MODE='high'", saved)
        self.assertNotIn("API_KEY", saved)
        self.assertNotIn("TOKEN", saved)
        self.assertEqual(payload["routing"]["text"]["provider"], "codex")

    def test_codex_never_falls_back_when_live_catalog_is_empty(self):
        self.codex_service.status.return_value = {
            "installed": True,
            "runtime_ready": True,
            "authenticated": True,
            "auth_mode": "chatgpt",
            "plan_type": "plus",
            "model_catalog_ready": True,
            "model_catalog_status": "empty",
            "message": "已连接本机 Codex 订阅。",
        }
        self.codex_service.models.return_value = ()

        with patch.dict(
            os.environ,
            {
                "TEXT_PROVIDER": "codex",
                "MODEL_NAME": "gpt-5.6-sol",
                "MODEL_MODE": "ultra",
            },
            clear=True,
        ):
            payload = application_settings_payload()
            with self.assertRaisesRegex(ModelRoutingValidationError, "不支持文本模型"):
                configure_model_routing(
                    ModelRoutingSettingsRequest(
                        text_provider="codex",
                        text_model="gpt-5.6-sol",
                        text_mode="ultra",
                    ),
                    env_path=Path(tempfile.gettempdir()) / "must-not-be-written.env",
                )

        provider = next(item for item in payload["providers"] if item["id"] == "codex")
        self.assertEqual(provider["text_models"], [])
        self.assertIsNone(provider["default_text_model"])
        self.assertEqual(payload["routing"]["text"]["model"], "")

    def test_codex_provider_rejects_api_key_persistence(self):
        with tempfile.TemporaryDirectory() as tempdir:
            env_path = Path(tempdir) / ".env"
            with self.assertRaisesRegex(ApiKeyValidationError, "不使用 API Key"):
                configure_provider_api_key(
                    "codex",
                    "must-not-be-saved",
                    env_path=env_path,
                )

            self.assertFalse(env_path.exists())

    def test_request_representation_masks_api_key(self):
        request = ApiKeySettingsRequest(api_key="secret-test-value")
        provider_request = ProviderApiKeySettingsRequest(
            api_key="another-secret-value",
            base_url="https://example.com/v1",
        )

        self.assertNotIn("secret-test-value", repr(request))
        self.assertNotIn("another-secret-value", repr(provider_request))

    def test_every_provider_exposes_a_text_route(self):
        for provider in PROVIDERS.values():
            with self.subTest(provider=provider.id):
                self.assertGreaterEqual(len(provider.text_models), 1)

    def test_anthropic_uses_messages_protocol_and_kimi_is_multimodal(self):
        self.assertEqual(PROVIDERS["anthropic"].protocol, "anthropic")
        self.assertEqual(PROVIDERS["anthropic"].default_base_url, "https://api.anthropic.com")
        self.assertEqual(PROVIDERS["kimi"].default_base_url, "https://api.moonshot.cn/v1")
        self.assertEqual(PROVIDERS["kimi"].default_text_model, "kimi-k2.6")
        self.assertEqual(PROVIDERS["kimi"].default_vision_model, "kimi-k2.6")
        self.assertEqual(
            [mode.id for mode in PROVIDERS["kimi"].text_models[0].modes],
            ["enabled", "disabled"],
        )

    def test_anthropic_catalog_contains_current_supported_generation(self):
        self.assertEqual(
            [model.id for model in PROVIDERS["anthropic"].text_models],
            [
                "claude-fable-5",
                "claude-sonnet-5",
                "claude-opus-4-8",
                "claude-opus-4-7",
                "claude-opus-4-6",
                "claude-sonnet-4-6",
                "claude-haiku-4-5-20251001",
            ],
        )

    def test_openai_default_model_uses_the_complete_sol_name(self):
        self.assertEqual(PROVIDERS["openai"].text_models[0].label, "GPT-5.6 Sol")
        self.assertEqual(PROVIDERS["openai"].text_models[0].id, "gpt-5.6-sol")

    def test_openai_catalog_only_contains_gpt_5_6_models(self):
        model_ids = [model.id for model in PROVIDERS["openai"].text_models]

        self.assertEqual(model_ids, ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"])

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

    def test_glm_5_2_exposes_fast_standard_and_deep_request_modes(self):
        model = PROVIDERS["zhipu"].text_models[0]

        self.assertEqual([mode.id for mode in model.modes], ["standard", "deep", "fast"])
        self.assertEqual(model.modes[0].request_body, {"thinking": {"type": "enabled"}})
        self.assertEqual(
            model.modes[1].request_body,
            {"thinking": {"type": "enabled"}, "reasoning_effort": "max"},
        )
        self.assertEqual(model.modes[2].request_body, {"thinking": {"type": "disabled"}})

    def test_qwen_hybrid_models_expose_real_thinking_toggle(self):
        hybrid_models = PROVIDERS["qwen"].text_models[:4]

        for model in hybrid_models:
            with self.subTest(model=model.id):
                self.assertEqual([mode.id for mode in model.modes], ["thinking", "fast"])
                self.assertEqual(model.modes[0].request_body, {"enable_thinking": True})
                self.assertEqual(model.modes[1].request_body, {"enable_thinking": False})

    def test_doubao_catalog_exposes_three_multimodal_seed_models(self):
        provider = PROVIDERS["doubao"]

        self.assertEqual(provider.api_key_env, "ARK_API_KEY")
        self.assertEqual(provider.default_base_url, "https://ark.cn-beijing.volces.com/api/v3")
        self.assertEqual(len(provider.text_models), 3)
        self.assertEqual(
            [model.id for model in provider.text_models],
            ["doubao-seed-2.1-pro", "doubao-seed-2.1-turbo", "doubao-seed-evolving"],
        )
        self.assertEqual(provider.default_vision_model, "doubao-seed-2.1-pro")

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

    def test_custom_relay_persists_protocol_endpoint_and_model_ids(self):
        with tempfile.TemporaryDirectory() as tempdir:
            env_path = Path(tempdir) / ".env"
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("core.settings._probe_provider_api_key", return_value=[]),
                patch("core.settings.reset_llm_clients"),
            ):
                payload = configure_provider_api_key(
                    "custom",
                    "relay-key-that-is-long-enough",
                    base_url="https://relay.example/v1",
                    protocol="anthropic",
                    provider_name="团队网关",
                    text_model="team-claude",
                    vision_model="team-claude-vision",
                    env_path=env_path,
                )

            saved = env_path.read_text()
            self.assertIn("CUSTOM_API_PROTOCOL='anthropic'", saved)
            self.assertIn("CUSTOM_BASE_URL='https://relay.example/v1'", saved)
            self.assertIn("CUSTOM_TEXT_MODEL='team-claude'", saved)
            custom = next(item for item in payload["providers"] if item["id"] == "custom")
            self.assertEqual(custom["label"], "团队网关")
            self.assertTrue(custom["supports_vision"])

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
                text_model="doubao-seed-2.1-turbo",
                vision_enabled=True,
                vision_provider="doubao",
                vision_model="doubao-seed-2.1-pro",
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
            self.assertIn("MODEL_NAME='doubao-seed-2.1-turbo'", saved)
            self.assertIn("VISION_PROVIDER='doubao'", saved)
            self.assertIn("VISION_MODEL_NAME='doubao-seed-2.1-pro'", saved)
            self.assertEqual(payload["routing"]["vision"]["provider"], "doubao")

    def test_kimi_route_persists_a_real_k2_6_thinking_mode(self):
        with tempfile.TemporaryDirectory() as tempdir:
            env_path = Path(tempdir) / ".env"
            request = ModelRoutingSettingsRequest(
                text_provider="kimi",
                text_model="kimi-k2.6",
                text_mode="disabled",
                vision_enabled=True,
                vision_provider="kimi",
                vision_model="kimi-k2.6",
            )
            with (
                patch.dict(
                    os.environ,
                    {"TEXT_PROVIDER": "kimi", "MOONSHOT_API_KEY": "configured-kimi-key"},
                    clear=True,
                ),
                patch("core.settings.reset_llm_clients"),
            ):
                payload = configure_model_routing(request, env_path=env_path)

            saved = env_path.read_text()
            self.assertIn("MODEL_NAME='kimi-k2.6'", saved)
            self.assertIn("MODEL_MODE='disabled'", saved)
            self.assertEqual(payload["routing"]["text"]["mode"], "disabled")

    def test_custom_route_uses_explicit_text_and_vision_models(self):
        with tempfile.TemporaryDirectory() as tempdir:
            env_path = Path(tempdir) / ".env"
            request = ModelRoutingSettingsRequest(
                text_provider="custom",
                text_model="relay-text-v2",
                vision_enabled=True,
                vision_provider="custom",
                vision_model="relay-vision-v2",
            )
            with (
                patch.dict(
                    os.environ,
                    {
                        "CUSTOM_API_KEY": "configured-relay-key",
                        "CUSTOM_BASE_URL": "https://relay.example/v1",
                        "CUSTOM_API_PROTOCOL": "openai",
                    },
                    clear=True,
                ),
                patch("core.settings.reset_llm_clients"),
            ):
                payload = configure_model_routing(request, env_path=env_path)

            saved = env_path.read_text()
            self.assertIn("MODEL_NAME='relay-text-v2'", saved)
            self.assertIn("VISION_MODEL_NAME='relay-vision-v2'", saved)
            self.assertEqual(payload["routing"]["text"]["model"], "relay-text-v2")
            self.assertEqual(payload["routing"]["vision"]["model"], "relay-vision-v2")

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
            with self.assertRaisesRegex(ModelRoutingValidationError, "未配置可用的视觉模型"):
                configure_model_routing(request)


if __name__ == "__main__":
    unittest.main()
