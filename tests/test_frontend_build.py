"""Regression tests for backend/frontend build version compatibility."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import app as app_module
from core.settings import PROJECT_VERSION


class TestFrontendBuildCompatibility(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app_module.app)
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.frontend_dist = Path(self.temporary_directory.name)
        (self.frontend_dist / "index.html").write_text(
            "<!doctype html><title>Paper Reader</title>",
            encoding="utf-8",
        )
        self.dist_patch = patch.object(
            app_module,
            "FRONTEND_DIST",
            self.frontend_dist,
        )
        self.dist_patch.start()
        self.addCleanup(self.dist_patch.stop)

    def _write_metadata(self, version: str) -> None:
        (self.frontend_dist / app_module.FRONTEND_BUILD_METADATA).write_text(
            json.dumps({"schema_version": 1, "project_version": version}),
            encoding="utf-8",
        )

    def test_matching_build_serves_index_without_long_term_browser_cache(self):
        self._write_metadata(PROJECT_VERSION)

        response = self.client.get("/")
        explicit_index_response = self.client.get("/index.html")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Paper Reader", response.text)
        self.assertEqual(response.headers["cache-control"], "no-cache")
        self.assertEqual(explicit_index_response.status_code, 200)
        self.assertEqual(explicit_index_response.headers["cache-control"], "no-cache")

    def test_missing_frontend_marks_health_unhealthy_and_returns_503(self):
        (self.frontend_dist / "index.html").unlink()
        self.frontend_dist.rmdir()

        root_response = self.client.get("/")
        health_response = self.client.get("/api/health")

        self.assertEqual(root_response.status_code, 503)
        self.assertIn("前端构建不存在或不完整", root_response.json()["detail"])
        self.assertFalse(health_response.json()["ok"])
        self.assertFalse(health_response.json()["frontend_dist"])
        self.assertFalse(health_response.json()["frontend_version_match"])
        self.assertIn("前端构建不存在或不完整", health_response.json()["frontend_error"])

    def test_missing_metadata_blocks_old_ui_but_keeps_api_available(self):
        root_response = self.client.get("/")
        health_response = self.client.get("/api/health")

        self.assertEqual(root_response.status_code, 503)
        self.assertIn("缺少版本元数据", root_response.json()["detail"])
        self.assertIn("npm --prefix frontend-prototype run build", root_response.json()["detail"])
        self.assertEqual(health_response.status_code, 200)
        self.assertFalse(health_response.json()["ok"])
        self.assertIsNone(health_response.json()["frontend_version"])
        self.assertFalse(health_response.json()["frontend_version_match"])
        self.assertIn("缺少版本元数据", health_response.json()["frontend_error"])
        self.assertEqual(root_response.headers["cache-control"], "no-store")

    def test_mismatched_metadata_reports_both_versions_and_blocks_old_ui(self):
        self._write_metadata("V0.0.0")

        root_response = self.client.get("/")
        health_response = self.client.get("/api/health")

        self.assertEqual(root_response.status_code, 503)
        self.assertIn(PROJECT_VERSION, root_response.json()["detail"])
        self.assertIn("V0.0.0", root_response.json()["detail"])
        self.assertFalse(health_response.json()["ok"])
        self.assertEqual(health_response.json()["frontend_version"], "V0.0.0")
        self.assertFalse(health_response.json()["frontend_version_match"])
        self.assertIn("V0.0.0", health_response.json()["frontend_error"])

    def test_health_reports_matching_frontend_version(self):
        self._write_metadata(PROJECT_VERSION)

        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])
        self.assertEqual(response.json()["frontend_version"], PROJECT_VERSION)
        self.assertTrue(response.json()["frontend_version_match"])
        self.assertIsNone(response.json()["frontend_error"])


if __name__ == "__main__":
    unittest.main()
