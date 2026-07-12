"""Regression tests for the reproducible Windows packaging workflow."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
LAUNCHER_PATH = ROOT / "packaging" / "launcher.py"


def load_launcher_module():
    spec = importlib.util.spec_from_file_location("paper_reader_windows_launcher", LAUNCHER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the Windows launcher module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load_launcher_module()


class FakeKernel32:
    def __init__(self, wait_results: list[int]) -> None:
        self.wait_results = list(wait_results)
        self.opened_pid: int | None = None
        self.event_signaled = False
        self.terminated_handle: int | None = None

    def OpenProcess(self, _access, _inherit, pid):
        self.opened_pid = pid
        return 101

    def OpenEventW(self, _access, _inherit, _name):
        return 202

    def SetEvent(self, _handle):
        self.event_signaled = True
        return True

    def WaitForSingleObject(self, _handle, _timeout):
        return self.wait_results.pop(0)

    def TerminateProcess(self, handle, _exit_code):
        self.terminated_handle = handle
        return True

    def CloseHandle(self, _handle):
        return True


class TestWindowsLauncherState(unittest.TestCase):
    def test_runtime_state_records_pid_and_executable_identity(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            executable = root / "PaperReader.exe"
            with (
                patch.object(launcher.os, "getpid", return_value=4242),
                patch.object(launcher.sys, "executable", str(executable)),
            ):
                launcher.write_runtime_state(root, 8004)

            payload = json.loads((root / "runtime-state.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["pid"], 4242)
            self.assertEqual(payload["port"], 8004)
            self.assertEqual(payload["version"], "V1.2.0")
            self.assertEqual(payload["executable"], str(executable.resolve()))

    def test_invalid_runtime_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            state = root / "runtime-state.json"
            state.write_text('{"pid":0}', encoding="utf-8")
            self.assertIsNone(launcher.read_runtime_state(root))

            state.write_text("not json", encoding="utf-8")
            self.assertIsNone(launcher.read_runtime_state(root))

    def test_windows_path_comparison_is_case_insensitive(self):
        self.assertTrue(
            launcher.paths_match(
                r"C:\Program Files\Paper Reader\PaperReader.exe",
                r"c:/program files/paper reader/PaperReader.exe",
            )
        )
        self.assertFalse(
            launcher.paths_match(
                r"C:\Program Files\Paper Reader\PaperReader.exe",
                r"C:\Other App\PaperReader.exe",
            )
        )

    def test_uninstall_requests_graceful_exit_for_recorded_pid(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            executable = str((root / "PaperReader.exe").resolve())
            (root / "runtime-state.json").write_text(
                json.dumps({"pid": 4242, "port": 8000, "executable": executable}),
                encoding="utf-8",
            )
            kernel32 = FakeKernel32([launcher.WAIT_OBJECT_0])
            with (
                patch.object(launcher.os, "name", "nt"),
                patch.object(launcher.sys, "executable", executable),
                patch.object(launcher, "windows_kernel32", return_value=kernel32),
                patch.object(launcher, "query_process_executable", return_value=executable),
            ):
                result = launcher.request_shutdown_for_uninstall(root)

            self.assertEqual(result, 0)
            self.assertEqual(kernel32.opened_pid, 4242)
            self.assertTrue(kernel32.event_signaled)
            self.assertIsNone(kernel32.terminated_handle)
            self.assertFalse((root / "runtime-state.json").exists())

    def test_uninstall_fallback_terminates_only_verified_process_handle(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            executable = str((root / "PaperReader.exe").resolve())
            (root / "runtime-state.json").write_text(
                json.dumps({"pid": 5151, "port": 8000, "executable": executable}),
                encoding="utf-8",
            )
            kernel32 = FakeKernel32([launcher.WAIT_TIMEOUT, launcher.WAIT_OBJECT_0])
            with (
                patch.object(launcher.os, "name", "nt"),
                patch.object(launcher.sys, "executable", executable),
                patch.object(launcher, "windows_kernel32", return_value=kernel32),
                patch.object(launcher, "query_process_executable", return_value=executable),
            ):
                result = launcher.request_shutdown_for_uninstall(root)

            self.assertEqual(result, 0)
            self.assertEqual(kernel32.opened_pid, 5151)
            self.assertEqual(kernel32.terminated_handle, 101)


class TestWindowsPackagingMetadata(unittest.TestCase):
    def test_v120_is_consistent_across_runtime_and_installer_metadata(self):
        installer = (ROOT / "packaging" / "installer.iss").read_text(encoding="utf-8")
        version_info = (ROOT / "packaging" / "version_info.txt").read_text(encoding="utf-8")
        env_example = (ROOT / ".env.example").read_text(encoding="utf-8")

        self.assertEqual(launcher.APP_VERSION, "V1.2.0")
        self.assertIn('#define AppVersion "1.2.0"', installer)
        self.assertIn("VersionInfoVersion={#AppVersion}.0", installer)
        self.assertIn("ProductVersion', '1.2.0'", version_info)
        self.assertIn("PAPER_READER_VERSION=V1.2.0", env_example)

    def test_uninstall_uses_pid_targeted_launcher_control(self):
        installer = (ROOT / "packaging" / "installer.iss").read_text(encoding="utf-8")
        self.assertIn("--shutdown-for-uninstall", installer)
        self.assertNotIn("taskkill /IM", installer)

    def test_translation_license_is_retained_and_installed(self):
        installer = (ROOT / "packaging" / "installer.iss").read_text(encoding="utf-8")
        notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        self.assertIn("THIRD_PARTY_NOTICES.md", installer)
        self.assertIn("Copyright (c) 2019 - 2020 kirakira", notices)
        self.assertIn("MIT License", notices)


if __name__ == "__main__":
    unittest.main()
