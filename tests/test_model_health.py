"""Tests for credential-safe cached model catalog health checks."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.model_health import (
    invalidate_model_catalog_health_cache,
    model_catalog_health,
)
from core.model_providers import PROVIDERS


class _FakeModels:
    def __init__(self, model_ids: list[str] | None = None, error: Exception | None = None):
        self.model_ids = model_ids or []
        self.error = error

    def list(self):
        if self.error:
            raise self.error
        return SimpleNamespace(data=[SimpleNamespace(id=model_id) for model_id in self.model_ids])


class _FakeClient:
    def __init__(self, model_ids: list[str] | None = None, error: Exception | None = None):
        self.models = _FakeModels(model_ids, error)


class _StatusError(RuntimeError):
    def __init__(self, status_code: int):
        super().__init__("provider error with secret-do-not-return")
        self.status_code = status_code


class TestModelCatalogHealth(unittest.TestCase):
    def setUp(self):
        invalidate_model_catalog_health_cache()

    def tearDown(self):
        invalidate_model_catalog_health_cache()

    def test_unconfigured_providers_skip_remote_calls(self):
        with (
            patch("core.model_health.provider_api_key", return_value=None),
            patch("core.model_health.OpenAI") as openai_client,
        ):
            payload = model_catalog_health(force=True)

        self.assertEqual(payload["summary"]["unconfigured"], len(PROVIDERS))
        self.assertEqual(payload["summary"]["configured"], 0)
        openai_client.assert_not_called()

    def test_matching_qwen_catalog_is_healthy_and_cached(self):
        qwen = PROVIDERS["qwen"]
        model_ids = [model.id for model in (*qwen.text_models, *qwen.vision_models)]

        def configured_key(provider_id: str):
            return "configured-qwen-key" if provider_id == "qwen" else None

        with (
            patch("core.model_health.provider_api_key", side_effect=configured_key),
            patch("core.model_health.OpenAI", return_value=_FakeClient(model_ids)) as openai_client,
        ):
            first = model_catalog_health(force=True)
            second = model_catalog_health()

        qwen_health = next(item for item in first["providers"] if item["id"] == "qwen")
        self.assertEqual(qwen_health["status"], "ok")
        self.assertEqual(qwen_health["vision_catalog_check"], "verified")
        self.assertEqual(qwen_health["missing_text_models"], [])
        self.assertTrue(second["cached"])
        openai_client.assert_called_once()

    def test_missing_text_model_reports_drift(self):
        deepseek = PROVIDERS["deepseek"]
        available = [deepseek.text_models[0].id]

        def configured_key(provider_id: str):
            return "configured-deepseek-key" if provider_id == "deepseek" else None

        with (
            patch("core.model_health.provider_api_key", side_effect=configured_key),
            patch("core.model_health.OpenAI", return_value=_FakeClient(available)),
        ):
            payload = model_catalog_health(force=True)

        health = next(item for item in payload["providers"] if item["id"] == "deepseek")
        self.assertEqual(health["status"], "drift")
        self.assertEqual(health["missing_text_models"], [deepseek.text_models[1].id])
        self.assertEqual(payload["summary"]["drifted"], 1)

    def test_provider_that_omits_vision_models_does_not_false_alarm(self):
        zhipu = PROVIDERS["zhipu"]
        available = [model.id for model in zhipu.text_models]

        def configured_key(provider_id: str):
            return "configured-zhipu-key" if provider_id == "zhipu" else None

        with (
            patch("core.model_health.provider_api_key", side_effect=configured_key),
            patch("core.model_health.OpenAI", return_value=_FakeClient(available)),
        ):
            payload = model_catalog_health(force=True)

        health = next(item for item in payload["providers"] if item["id"] == "zhipu")
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["vision_catalog_check"], "not_listed")
        self.assertEqual(health["missing_vision_models"], [])

    def test_provider_error_is_sanitized(self):
        def configured_key(provider_id: str):
            return "configured-openai-key" if provider_id == "openai" else None

        with (
            patch("core.model_health.provider_api_key", side_effect=configured_key),
            patch(
                "core.model_health.OpenAI",
                return_value=_FakeClient(error=_StatusError(401)),
            ),
        ):
            payload = model_catalog_health(force=True)

        health = next(item for item in payload["providers"] if item["id"] == "openai")
        self.assertEqual(health["status"], "unavailable")
        self.assertEqual(health["http_status"], 401)
        self.assertNotIn("secret-do-not-return", repr(payload))


if __name__ == "__main__":
    unittest.main()
