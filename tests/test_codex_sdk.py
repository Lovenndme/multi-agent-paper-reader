"""Unit tests for the credential-safe local Codex SDK boundary."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.codex_sdk import (
    CodexSDKError,
    CodexModelInfo,
    CodexSDKService,
    CodexTurnResult,
    _append_external_source,
    _consume_turn,
    _safe_error_message,
    _safe_login_identifier,
    _safe_login_url,
    _security_profile,
    _thread_config,
)


class _FakeTurn:
    thread_id = "thread-local"
    id = "turn-local"

    def __init__(self, events):
        self._events = events
        self.interrupted = False

    def stream(self):
        return iter(self._events)

    def interrupt(self):
        self.interrupted = True


class TestCodexSDKService(unittest.TestCase):
    @staticmethod
    def _runtime_metadata():
        return {
            "installed": True,
            "sdk_version": "0.0.0.dev0",
            "sdk_source_revision": "3f74f00295dcb1346340686bb09c5bfd4f0237c4",
            "sdk_archive_sha256": "f444a6ca308073dab245cd61ed123cf2e46fe5dd8067f5f9713e0aff3c19de47",
            "runtime_version": "0.144.4",
            "binary_version": "0.144.4",
            "sdk_compatible": True,
            "compatibility": "ok",
            "compatibility_message": "compatible",
        }

    def test_status_exposes_plan_but_never_account_identity_or_tokens(self):
        service = CodexSDKService()
        account = SimpleNamespace(
            type=SimpleNamespace(value="chatgpt"),
            plan_type=SimpleNamespace(value="plus"),
            email="private@example.com",
            access_token="secret-token",
        )
        client = SimpleNamespace(
            account=lambda: SimpleNamespace(account=account, requires_openai_auth=True)
        )

        with (
            patch("core.codex_sdk._sdk_runtime_metadata", return_value=self._runtime_metadata()),
            patch.object(service, "_get_client", return_value=client),
        ):
            payload = service.status(force=True)

        self.assertTrue(payload["authenticated"])
        self.assertEqual(payload["auth_mode"], "chatgpt")
        self.assertEqual(payload["plan_type"], "plus")
        self.assertNotIn("private@example.com", repr(payload))
        self.assertNotIn("secret-token", repr(payload))

    def test_status_normalizes_runtime_errors_without_echoing_secrets(self):
        service = CodexSDKService()
        client = SimpleNamespace(
            account=MagicMock(side_effect=RuntimeError("unauthorized bearer secret-token"))
        )

        with (
            patch("core.codex_sdk._sdk_runtime_metadata", return_value=self._runtime_metadata()),
            patch.object(service, "_get_client", return_value=client),
        ):
            payload = service.status(force=True)

        self.assertFalse(payload["authenticated"])
        self.assertIn("登录", payload["message"])
        self.assertNotIn("secret-token", repr(payload))

    def test_models_are_mapped_from_the_live_account_catalog(self):
        service = CodexSDKService()
        model = SimpleNamespace(
            id="gpt-5.6-sol",
            display_name="GPT-5.6 Sol",
            description="Test model",
            is_default=True,
            input_modalities=[SimpleNamespace(value="text"), SimpleNamespace(value="image")],
            default_reasoning_effort=SimpleNamespace(value="medium"),
            supported_reasoning_efforts=[
                SimpleNamespace(
                    reasoning_effort=SimpleNamespace(value="medium"),
                    description="Balanced",
                ),
                SimpleNamespace(
                    reasoning_effort=SimpleNamespace(value="high"),
                    description="Deeper",
                ),
            ],
        )
        client = SimpleNamespace(models=lambda: SimpleNamespace(data=[model]))

        with (
            patch.object(
                service,
                "status",
                return_value={"runtime_ready": True, "authenticated": True},
            ),
            patch.object(service, "_get_client", return_value=client),
        ):
            catalog = service.models(force=True)

        self.assertEqual(len(catalog), 1)
        self.assertEqual(catalog[0].id, "gpt-5.6-sol")
        self.assertTrue(catalog[0].supports_image)
        self.assertEqual(catalog[0].default_effort, "medium")
        self.assertEqual(catalog[0].efforts[1][0], "high")
        catalog_state = service._catalog_metadata(authenticated=True)
        self.assertTrue(catalog_state["model_catalog_ready"])
        self.assertEqual(catalog_state["model_catalog_status"], "ready")
        self.assertEqual(catalog_state["model_catalog_count"], 1)

    def test_successful_empty_live_catalog_is_distinct_from_not_loaded(self):
        service = CodexSDKService()
        client = SimpleNamespace(models=lambda: SimpleNamespace(data=[]))

        with (
            patch.object(
                service,
                "status",
                return_value={"runtime_ready": True, "authenticated": True},
            ),
            patch.object(service, "_get_client", return_value=client),
        ):
            catalog = service.models(force=True)

        self.assertEqual(catalog, ())
        catalog_state = service._catalog_metadata(authenticated=True)
        self.assertTrue(catalog_state["model_catalog_ready"])
        self.assertEqual(catalog_state["model_catalog_status"], "empty")
        self.assertEqual(catalog_state["model_catalog_count"], 0)

    def test_model_catalog_failure_is_fail_closed_and_credential_safe(self):
        service = CodexSDKService()
        client = SimpleNamespace(
            models=MagicMock(side_effect=RuntimeError("transport secret-token"))
        )

        with (
            patch.object(
                service,
                "status",
                return_value={"runtime_ready": True, "authenticated": True},
            ),
            patch.object(service, "_get_client", return_value=client),
            self.assertRaises(CodexSDKError) as error,
        ):
            service.models(force=True)

        catalog_state = service._catalog_metadata(authenticated=True)
        self.assertFalse(catalog_state["model_catalog_ready"])
        self.assertEqual(catalog_state["model_catalog_status"], "error")
        self.assertNotIn("secret-token", str(error.exception))

    def test_run_text_enforces_ephemeral_read_only_safe_turns(self):
        service = CodexSDKService()
        fake_turn = object()
        thread = SimpleNamespace(turn=MagicMock(return_value=fake_turn))
        client = SimpleNamespace(thread_start=MagicMock(return_value=thread))
        sdk = {
            "ApprovalMode": SimpleNamespace(deny_all="deny-all"),
            "Sandbox": SimpleNamespace(read_only="read-only"),
            "ReasoningEffort": lambda value: f"effort:{value}",
            "TextInput": lambda value: ("text", value),
            "ImageInput": lambda value: ("image", value),
        }
        result = CodexTurnResult(
            text="ok",
            model="gpt-5.6-sol",
            thread_id="thread-local",
            turn_id="turn-local",
            status="completed",
        )

        with tempfile.TemporaryDirectory() as tempdir:
            with (
                patch.object(
                    service,
                    "status",
                    return_value={"runtime_ready": True, "authenticated": True},
                ),
                patch.object(service, "_get_client", return_value=client),
                patch.object(
                    service,
                    "models",
                    return_value=(
                        CodexModelInfo(
                            id="gpt-5.6-sol",
                            label="GPT-5.6 Sol",
                            description="test",
                            recommended=True,
                            supports_image=True,
                            default_effort="low",
                            efforts=(("high", "deep"),),
                        ),
                    ),
                ),
                patch("core.codex_sdk._sdk_symbols", return_value=sdk),
                patch("core.codex_sdk._runtime_cwd", return_value=Path(tempdir)),
                patch("core.codex_sdk._consume_turn", return_value=result) as consume,
            ):
                actual = service.run_text(
                    "Analyze this paper.",
                    model="gpt-5.6-sol",
                    effort="high",
                    output_schema={"type": "object"},
                    image_bytes=b"png",
                    timeout=30,
                )

        self.assertEqual(actual, result)
        start_kwargs = client.thread_start.call_args.kwargs
        self.assertTrue(start_kwargs["ephemeral"])
        self.assertEqual(start_kwargs["approval_mode"], "deny-all")
        self.assertEqual(start_kwargs["sandbox"], "read-only")
        self.assertIn("Never run shell commands", start_kwargs["base_instructions"])
        self.assertIn("image generation are allowed", start_kwargs["base_instructions"])
        self.assertFalse(start_kwargs["config"]["features"]["shell_tool"])
        self.assertTrue(start_kwargs["config"]["features"]["image_generation"])
        self.assertFalse(start_kwargs["config"]["features"]["tool_suggest"])
        self.assertFalse(
            start_kwargs["config"]["tools"]["experimental_request_user_input"]["enabled"]
        )
        self.assertFalse(start_kwargs["config"]["skills"]["include_instructions"])
        self.assertEqual(start_kwargs["config"]["developer_instructions"], "")
        self.assertEqual(start_kwargs["config"]["agents"]["max_threads"], 1)
        self.assertIn("not an Ultra turn", start_kwargs["base_instructions"])
        self.assertEqual(start_kwargs["config"]["web_search"], "live")
        turn_args, turn_kwargs = thread.turn.call_args
        self.assertEqual(turn_args[0][0], ("text", "Analyze this paper."))
        self.assertTrue(turn_args[0][1][1].startswith("data:image/png;base64,"))
        self.assertEqual(turn_kwargs["effort"], "effort:high")
        self.assertEqual(turn_kwargs["output_schema"], {"type": "object"})
        self.assertEqual(turn_kwargs["sandbox"], "read-only")
        consume.assert_called_once_with(
            fake_turn,
            model="gpt-5.6-sol",
            effort="high",
            on_token=None,
            timeout=30.0,
        )

    def test_turn_stream_collects_deltas_and_safe_metadata(self):
        events = [
            SimpleNamespace(
                method="item/agentMessage/delta",
                payload=SimpleNamespace(delta="Hello "),
            ),
            SimpleNamespace(
                method="item/agentMessage/delta",
                payload=SimpleNamespace(delta="world"),
            ),
            SimpleNamespace(
                method="turn/completed",
                payload=SimpleNamespace(
                    turn=SimpleNamespace(status=SimpleNamespace(value="completed"))
                ),
            ),
        ]
        tokens: list[str] = []

        result = _consume_turn(
            _FakeTurn(events),
            model="gpt-test",
            on_token=tokens.append,
            timeout=30,
        )

        self.assertEqual(result.text, "Hello world")
        self.assertEqual(tokens, ["Hello ", "world"])
        self.assertEqual(result.thread_id, "thread-local")
        self.assertEqual(result.turn_id, "turn-local")

    def test_turn_stream_records_only_safe_tool_search_and_subagent_metadata(self):
        events = [
            SimpleNamespace(
                method="item/completed",
                payload=SimpleNamespace(
                    item=SimpleNamespace(
                        root=SimpleNamespace(type="mcpToolCall", tool="calculate")
                    )
                ),
            ),
            SimpleNamespace(
                method="item/completed",
                payload=SimpleNamespace(
                    item=SimpleNamespace(root=SimpleNamespace(type="imageView"))
                ),
            ),
            SimpleNamespace(
                method="item/completed",
                payload=SimpleNamespace(
                    item=SimpleNamespace(root=SimpleNamespace(type="imageGeneration"))
                ),
            ),
            SimpleNamespace(
                method="item/completed",
                payload=SimpleNamespace(
                    item=SimpleNamespace(
                        root=SimpleNamespace(
                            type="webSearch",
                            action=SimpleNamespace(
                                root=SimpleNamespace(url="https://example.com/source")
                            ),
                        )
                    )
                ),
            ),
            SimpleNamespace(
                method="item/completed",
                payload=SimpleNamespace(
                    item=SimpleNamespace(
                        root=SimpleNamespace(
                            type="collabAgentToolCall",
                            receiver_thread_ids=["subagent-safe"],
                        )
                    )
                ),
            ),
            SimpleNamespace(
                method="item/agentMessage/delta",
                payload=SimpleNamespace(delta="done"),
            ),
            SimpleNamespace(
                method="turn/completed",
                payload=SimpleNamespace(
                    turn=SimpleNamespace(status=SimpleNamespace(value="completed"))
                ),
            ),
        ]

        result = _consume_turn(
            _FakeTurn(events),
            model="gpt-5.6-sol",
            effort="ultra",
            on_token=None,
            timeout=30,
        )

        self.assertEqual(result.tools_used, ("calculate", "view_image", "image_generation"))
        self.assertTrue(result.web_search_used)
        self.assertEqual(result.subagent_count, 1)
        self.assertEqual(result.external_sources[0]["domain"], "example.com")
        self.assertNotIn("subagent-safe", repr(result.external_sources))

    def test_ultra_thread_config_is_bounded_and_standard_disables_subagents(self):
        standard = _thread_config(None, effort="max")
        ultra = _thread_config(None, effort="ultra")

        self.assertFalse(standard["features"]["multi_agent"])
        self.assertTrue(ultra["features"]["multi_agent"])
        self.assertTrue(standard["features"]["image_generation"])
        self.assertTrue(ultra["features"]["image_generation"])
        self.assertEqual(standard["agents"], {"max_threads": 1, "max_depth": 1})
        self.assertEqual(ultra["agents"], {"max_threads": 3, "max_depth": 1})

    def test_security_profile_separates_native_and_paper_tool_boundaries(self):
        profile = _security_profile()

        self.assertEqual(
            profile["native_tools"],
            ["exec", "wait", "plan", "view_image", "image_generation", "tool_search"],
        )
        self.assertTrue(profile["image_generation"])
        self.assertEqual(profile["native_tool_policy"], "runtime_managed_sandboxed")
        self.assertEqual(profile["controlled_artifact_write"], "codex_home_generated_images")
        self.assertEqual(profile["runtime_visible_blocked_tools"], ["apply_patch"])
        self.assertEqual(profile["code_mode"], "model_required_v8")
        self.assertEqual(profile["tool_scope"], "current_paper_only")
        self.assertEqual(profile["standard_max_subagents"], 0)
        self.assertEqual(profile["multi_agent_enforcement"], "thread_capacity")
        self.assertFalse(profile["shell"])
        self.assertFalse(profile["filesystem_write"])

    def test_safe_errors_never_return_raw_runtime_messages(self):
        message = _safe_error_message(RuntimeError("quota token=secret-token"))

        self.assertIn("使用限制", message)
        self.assertNotIn("secret-token", message)
        self.assertNotIsInstance(message, CodexSDKError)

    def test_external_source_urls_reject_credentials_and_non_http_schemes(self):
        output = []
        seen = set()
        _append_external_source(output, seen, "https://user:secret@example.com/private")
        _append_external_source(output, seen, "javascript:alert(1)")
        _append_external_source(output, seen, "https://example.com/paper#section")

        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["url"], "https://example.com/paper")
        self.assertEqual(output[0]["domain"], "example.com")
        self.assertNotIn("secret", repr(output))

    def test_login_redirect_metadata_accepts_only_safe_values(self):
        self.assertEqual(_safe_login_identifier("login-safe_123"), "login-safe_123")
        self.assertEqual(
            _safe_login_url("https://auth.openai.com/authorize?flow=codex"),
            "https://auth.openai.com/authorize?flow=codex",
        )
        self.assertEqual(_safe_login_url("http://127.0.0.1:1455/callback"), "http://127.0.0.1:1455/callback")
        for value in (
            "javascript:alert(1)",
            "http://malicious.example/login",
            "https://user:secret@auth.openai.com/login",
        ):
            with self.subTest(value=value), self.assertRaises(CodexSDKError):
                _safe_login_url(value)


if __name__ == "__main__":
    unittest.main()
