"""API tests for local Codex subscription status and login flows."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

import app as app_module
from core.codex_sdk import CodexModelInfo


class TestCodexSettingsAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app_module.app)
        self.model = CodexModelInfo(
            id="gpt-5.6-luna",
            label="GPT-5.6 Luna",
            description="Account model",
            recommended=True,
            supports_image=True,
            default_effort="medium",
            efforts=(("medium", "Balanced"), ("high", "Deeper")),
        )

    def test_status_returns_safe_account_and_model_metadata(self):
        service = SimpleNamespace(
            status=MagicMock(
                return_value={
                    "installed": True,
                    "runtime_ready": True,
                    "authenticated": True,
                    "auth_mode": "chatgpt",
                    "plan_type": "plus",
                    "model_catalog_ready": True,
                    "message": "已连接本机 Codex 订阅。",
                }
            ),
            models=MagicMock(return_value=(self.model,)),
        )

        with patch("app.get_codex_sdk_service", return_value=service):
            response = self.client.get("/api/settings/codex/status?force=true")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["status"]["plan_type"], "plus")
        self.assertEqual(payload["models"][0]["id"], "gpt-5.6-luna")
        self.assertEqual([item["id"] for item in payload["models"][0]["efforts"]], [
            "low", "medium", "high", "xhigh", "max", "ultra"
        ])
        self.assertFalse(payload["models"][0]["efforts"][-1]["available"])
        self.assertNotIn("email", repr(payload).lower())
        self.assertNotIn("access_token", repr(payload).lower())

    def test_login_reuses_an_existing_local_chatgpt_session(self):
        service = SimpleNamespace(
            status=MagicMock(
                return_value={
                    "runtime_ready": True,
                    "authenticated": True,
                    "auth_mode": "chatgpt",
                    "plan_type": "plus",
                }
            ),
            start_chatgpt_login=MagicMock(),
        )

        with patch("app.get_codex_sdk_service", return_value=service):
            response = self.client.post("/api/settings/codex/login")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["already_authenticated"])
        service.start_chatgpt_login.assert_not_called()

    def test_login_start_and_poll_return_only_browser_flow_state(self):
        service = SimpleNamespace(
            status=MagicMock(
                side_effect=[
                    {"runtime_ready": True, "authenticated": False},
                    {
                        "runtime_ready": True,
                        "authenticated": True,
                        "auth_mode": "chatgpt",
                        "plan_type": "plus",
                    },
                ]
            ),
            start_chatgpt_login=MagicMock(
                return_value={
                    "login_id": "login-safe",
                    "auth_url": "https://auth.openai.com/safe-flow",
                }
            ),
            login_state=MagicMock(
                return_value={"status": "success", "message": "Codex 登录成功。"}
            ),
        )

        with patch("app.get_codex_sdk_service", return_value=service):
            started = self.client.post("/api/settings/codex/login")
            polled = self.client.get("/api/settings/codex/login/login-safe")

        self.assertEqual(started.status_code, 200)
        self.assertFalse(started.json()["already_authenticated"])
        self.assertEqual(started.json()["login_id"], "login-safe")
        self.assertEqual(polled.status_code, 200)
        self.assertEqual(polled.json()["status"], "success")
        self.assertEqual(polled.json()["account"]["plan_type"], "plus")

    def test_device_login_and_logout_use_official_local_service_methods(self):
        service = SimpleNamespace(
            status=MagicMock(
                side_effect=[
                    {"runtime_ready": True, "authenticated": False},
                    {"runtime_ready": True, "authenticated": False},
                ]
            ),
            start_chatgpt_device_login=MagicMock(
                return_value={
                    "login_id": "device-safe",
                    "verification_url": "https://auth.openai.com/device",
                    "user_code": "SAFE-CODE",
                }
            ),
            logout=MagicMock(),
        )
        with patch("app.get_codex_sdk_service", return_value=service):
            started = self.client.post("/api/settings/codex/login/device")
            logged_out = self.client.delete("/api/settings/codex/session")

        self.assertEqual(started.status_code, 200)
        self.assertEqual(started.json()["user_code"], "SAFE-CODE")
        self.assertEqual(logged_out.status_code, 200)
        service.logout.assert_called_once_with()

    def test_login_guard_rejects_non_local_clients(self):
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/settings/codex/login",
                "headers": [],
                "client": ("203.0.113.10", 4567),
                "server": ("127.0.0.1", 8000),
                "scheme": "http",
            }
        )

        with self.assertRaises(HTTPException) as error:
            app_module._require_local_codex_request(request)

        self.assertEqual(error.exception.status_code, 403)

    def test_login_guard_rejects_cross_origin_requests_to_localhost(self):
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/settings/codex/login",
                "headers": [
                    (b"host", b"127.0.0.1:8000"),
                    (b"origin", b"https://malicious.example"),
                ],
                "client": ("127.0.0.1", 4567),
                "server": ("127.0.0.1", 8000),
                "scheme": "http",
            }
        )

        with self.assertRaises(HTTPException) as error:
            app_module._require_local_codex_request(request)

        self.assertEqual(error.exception.status_code, 403)

    def test_login_guard_rejects_dns_rebinding_host_even_with_matching_origin(self):
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/settings/codex/login",
                "headers": [
                    (b"host", b"malicious.example"),
                    (b"origin", b"http://malicious.example"),
                ],
                "client": ("127.0.0.1", 4567),
                "server": ("127.0.0.1", 8000),
                "scheme": "http",
            }
        )

        with self.assertRaises(HTTPException) as error:
            app_module._require_local_codex_request(request)

        self.assertEqual(error.exception.status_code, 403)

    def test_login_guard_accepts_same_origin_local_requests(self):
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/settings/codex/login",
                "headers": [
                    (b"host", b"127.0.0.1:8000"),
                    (b"origin", b"http://127.0.0.1:8000"),
                ],
                "client": ("127.0.0.1", 4567),
                "server": ("127.0.0.1", 8000),
                "scheme": "http",
            }
        )

        app_module._require_local_codex_request(request)


if __name__ == "__main__":
    unittest.main()
