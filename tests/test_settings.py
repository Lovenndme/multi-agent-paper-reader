"""Tests for public settings metadata and local API-key persistence."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.settings import (
    ApiKeySettingsRequest,
    ApiKeyValidationError,
    application_settings_payload,
    configure_glm_api_key,
)


class TestApplicationSettings(unittest.TestCase):
    def test_public_payload_never_contains_secret_material(self):
        with patch.dict(os.environ, {"GLM_API_KEY": "secret-test-value"}, clear=False):
            payload = application_settings_payload()

        self.assertTrue(payload["api_key_configured"])
        self.assertEqual(payload["version"], "V1.1.2")
        self.assertNotIn("secret-test-value", repr(payload))

    def test_request_representation_masks_api_key(self):
        request = ApiKeySettingsRequest(api_key="secret-test-value")

        self.assertNotIn("secret-test-value", repr(request))

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
                self.assertEqual(env_path.stat().st_mode & 0o777, 0o600)
                reset_clients.assert_called_once_with()
        finally:
            if previous_key is None:
                os.environ.pop("GLM_API_KEY", None)
            else:
                os.environ["GLM_API_KEY"] = previous_key


if __name__ == "__main__":
    unittest.main()
