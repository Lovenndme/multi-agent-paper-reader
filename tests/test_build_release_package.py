"""Tests for the safe, version-gated release archive builder."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.build_release_package import ReleasePackageError, build_release_package


class TestBuildReleasePackage(unittest.TestCase):
    def _git(self, root: Path, *args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _repository(
        self,
        base: Path,
        *,
        tag: str = "V1.2.3",
        source_version: str = "V1.2.3",
        frontend_version: str = "V1.2.3",
        metadata: bool = True,
    ) -> Path:
        root = base / "repo"
        (root / "core").mkdir(parents=True)
        (root / "frontend-prototype" / "dist" / "assets").mkdir(parents=True)
        (root / "node_modules").mkdir()
        (root / "output").mkdir()
        (root / "README.md").write_text("Paper Reader\n", encoding="utf-8")
        (root / "core" / "settings.py").write_text(
            f'PROJECT_VERSION = "{source_version}"\n', encoding="utf-8"
        )
        (root / ".env.example").write_text("API_KEY=replace-me\n", encoding="utf-8")
        (root / ".env").write_text("API_KEY=do-not-ship\n", encoding="utf-8")
        (root / "private.pdf").write_bytes(b"private paper")
        (root / "node_modules" / "cache.js").write_text("cache", encoding="utf-8")
        (root / "output" / "model-response.json").write_text("{}", encoding="utf-8")
        (root / "frontend-prototype" / "dist" / "index.html").write_text(
            "<main>current UI</main>", encoding="utf-8"
        )
        (root / "frontend-prototype" / "dist" / "assets" / "app.js").write_text(
            "console.log('current UI')", encoding="utf-8"
        )
        (root / "frontend-prototype" / "dist" / ".DS_Store").write_bytes(b"local")
        if metadata:
            (root / "frontend-prototype" / "dist" / "build-meta.json").write_text(
                json.dumps({"schema_version": 1, "project_version": frontend_version}),
                encoding="utf-8",
            )

        self._git(root, "init", "-q")
        self._git(root, "config", "user.name", "Release Test")
        self._git(root, "config", "user.email", "release@example.invalid")
        self._git(
            root,
            "add",
            "README.md",
            "core/settings.py",
            ".env.example",
        )
        # Simulate accidentally committed local/sensitive inputs. The packager
        # must still leave them out of the public archive.
        self._git(root, "add", "-f", ".env", "private.pdf", "node_modules", "output")
        self._git(root, "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
        self._git(root, "-c", "tag.gpgSign=false", "tag", tag)
        return root

    def test_builds_reproducible_prefixed_archive_from_tracked_files_and_dist(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self._repository(base)

            first = build_release_package(root, "V1.2.3", base / "artifacts-one")
            second = build_release_package(root, "V1.2.3", base / "artifacts-two")

            self.assertEqual(first.archive.name, "Paper-Reader-V1.2.3.zip")
            self.assertEqual(first.checksum.name, "Paper-Reader-V1.2.3.zip.sha256")
            self.assertEqual(first.sha256, second.sha256)
            self.assertEqual(
                first.sha256,
                hashlib.sha256(first.archive.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                first.checksum.read_text(encoding="utf-8"),
                f"{first.sha256}  {first.archive.name}\n",
            )

            with zipfile.ZipFile(first.archive) as archive:
                names = set(archive.namelist())
            self.assertTrue(names)
            self.assertTrue(all(name.startswith("Paper-Reader-V1.2.3/") for name in names))
            self.assertIn("Paper-Reader-V1.2.3/README.md", names)
            self.assertIn("Paper-Reader-V1.2.3/.env.example", names)
            self.assertIn("Paper-Reader-V1.2.3/frontend-prototype/dist/index.html", names)
            self.assertIn(
                "Paper-Reader-V1.2.3/frontend-prototype/dist/build-meta.json", names
            )
            self.assertNotIn("Paper-Reader-V1.2.3/.env", names)
            self.assertNotIn("Paper-Reader-V1.2.3/private.pdf", names)
            self.assertFalse(any("node_modules" in name for name in names))
            self.assertFalse(any("model-response" in name for name in names))
            self.assertFalse(any(".DS_Store" in name for name in names))

    def test_hard_fails_for_each_version_mismatch(self):
        cases = (
            ("V1.2.3", "V1.2.4", "V1.2.3"),
            ("V1.2.3", "V1.2.3", "V1.2.4"),
            ("V1.2.3", "V1.2.4", "V1.2.4"),
        )
        for index, (tag, source_version, frontend_version) in enumerate(cases):
            with self.subTest(
                tag=tag,
                source_version=source_version,
                frontend_version=frontend_version,
            ), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                root = self._repository(
                    base,
                    tag=tag,
                    source_version=source_version,
                    frontend_version=frontend_version,
                )
                with self.assertRaisesRegex(ReleasePackageError, "Release version mismatch"):
                    build_release_package(root, tag, base / f"artifacts-{index}")

    def test_hard_fails_without_frontend_build_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self._repository(base, metadata=False)
            with self.assertRaisesRegex(ReleasePackageError, "build-meta.json is missing"):
                build_release_package(root, "V1.2.3", base / "artifacts")

    def test_hard_fails_when_tag_does_not_reference_head(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self._repository(base)
            (root / "README.md").write_text("new commit\n", encoding="utf-8")
            self._git(root, "add", "README.md")
            self._git(root, "-c", "commit.gpgsign=false", "commit", "-qm", "later")

            with self.assertRaisesRegex(ReleasePackageError, "HEAD is not"):
                build_release_package(root, "V1.2.3", base / "artifacts")


if __name__ == "__main__":
    unittest.main()
