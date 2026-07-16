"""Build a safe, reproducible source release with the prebuilt web UI.

The archive intentionally contains only committed Git files plus the generated
``frontend-prototype/dist`` tree.  Local configuration, credentials, caches,
analysis outputs, and user documents are excluded even if they were
accidentally committed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence


TAG_PATTERN = re.compile(r"^V(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)$")
ARCHIVE_PREFIX_TEMPLATE = "Paper-Reader-{tag}"
DIST_PATH = PurePosixPath("frontend-prototype/dist")
BUILD_META_PATH = DIST_PATH / "build-meta.json"

_BLOCKED_PARTS = frozenset(
    {
        ".cache",
        ".git",
        ".idea",
        ".mypy_cache",
        ".npm-cache",
        ".paper-reader",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        ".vscode",
        "__pycache__",
        "node_modules",
        "output",
        "outputs",
        "venv",
        "vision_ab_outputs",
    }
)
_BLOCKED_NAMES = frozenset(
    {
        ".env",
        ".npmrc",
        ".pypirc",
        "auth.json",
        "cookies.json",
        "credentials.json",
        "id_dsa",
        "id_ed25519",
        "id_ecdsa",
        "id_rsa",
        "secrets.json",
    }
)
_BLOCKED_SUFFIXES = frozenset(
    {
        ".db",
        ".docx",
        ".jks",
        ".key",
        ".keystore",
        ".log",
        ".p12",
        ".pdf",
        ".pem",
        ".pfx",
        ".sqlite",
        ".sqlite3",
        ".xlsx",
    }
)
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


class ReleasePackageError(RuntimeError):
    """Raised when the checkout is not safe or complete enough to package."""


@dataclass(frozen=True)
class ReleaseArtifact:
    archive: Path
    checksum: Path
    sha256: str
    file_count: int


def _run_git(repo_root: Path, *args: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise ReleasePackageError("Git is required to build a release package.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        raise ReleasePackageError(f"Git command failed: {stderr or 'unknown error'}") from exc
    return completed.stdout


def _validate_tag_checkout(repo_root: Path, tag: str) -> None:
    if not TAG_PATTERN.fullmatch(tag):
        raise ReleasePackageError(
            f"Invalid release tag {tag!r}; expected a stable tag such as V1.6.2."
        )

    head = _run_git(repo_root, "rev-parse", "HEAD").strip()
    tag_commit = _run_git(repo_root, "rev-parse", f"refs/tags/{tag}^{{commit}}").strip()
    if head != tag_commit:
        raise ReleasePackageError(f"HEAD is not the commit referenced by {tag}.")

    dirty = _run_git(repo_root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ReleasePackageError("Tracked files contain uncommitted changes.")


def _project_version(repo_root: Path) -> str:
    settings_path = repo_root / "core" / "settings.py"
    try:
        tree = ast.parse(settings_path.read_text(encoding="utf-8"), filename=str(settings_path))
    except (OSError, SyntaxError) as exc:
        raise ReleasePackageError("Unable to read PROJECT_VERSION from core/settings.py.") from exc

    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "PROJECT_VERSION" for target in targets):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
        break
    raise ReleasePackageError("core/settings.py must define a literal PROJECT_VERSION string.")


def _validate_build_metadata(repo_root: Path, tag: str) -> None:
    metadata_path = repo_root / Path(*BUILD_META_PATH.parts)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReleasePackageError(
            "frontend-prototype/dist/build-meta.json is missing; build the frontend first."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleasePackageError("frontend-prototype/dist/build-meta.json is invalid.") from exc

    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ReleasePackageError("Unsupported frontend build metadata schema.")
    built_version = metadata.get("project_version")
    source_version = _project_version(repo_root)
    if built_version != tag or source_version != tag:
        raise ReleasePackageError(
            "Release version mismatch: "
            f"tag={tag}, frontend={built_version!r}, source={source_version!r}."
        )


def _normalized_relative_path(raw_path: str) -> PurePosixPath:
    path = PurePosixPath(raw_path.replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ReleasePackageError(f"Unsafe repository path: {raw_path!r}")
    return path


def _is_release_safe(path: PurePosixPath) -> bool:
    lowered_parts = tuple(part.lower() for part in path.parts)
    name = lowered_parts[-1]
    if any(part in _BLOCKED_PARTS for part in lowered_parts):
        return False
    if name in _BLOCKED_NAMES:
        return False
    if name.startswith(".env.") and name not in {".env.example", ".env.sample"}:
        return False
    if any(name.endswith(suffix) for suffix in _BLOCKED_SUFFIXES):
        return False
    if name in {".ds_store", "thumbs.db"} or name.endswith((".pyc", ".pyo", "~")):
        return False
    return True


def _tracked_files(repo_root: Path) -> list[PurePosixPath]:
    output = _run_git(repo_root, "ls-files", "-z", "--cached")
    paths: list[PurePosixPath] = []
    for raw in output.decode("utf-8", errors="strict").split("\0"):
        if not raw:
            continue
        path = _normalized_relative_path(raw)
        if path == DIST_PATH or DIST_PATH in path.parents:
            continue
        if _is_release_safe(path):
            paths.append(path)
    return paths


def _tracked_executable_files(repo_root: Path) -> set[PurePosixPath]:
    """Read executable bits from Git rather than platform-dependent stat data."""

    output = _run_git(repo_root, "ls-files", "-z", "--stage")
    executables: set[PurePosixPath] = set()
    for record in output.decode("utf-8", errors="strict").split("\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split("\t", 1)
            mode = metadata.split(" ", 1)[0]
        except ValueError as exc:
            raise ReleasePackageError("Unable to parse the Git index.") from exc
        if mode == "100755":
            path = _normalized_relative_path(raw_path)
            if _is_release_safe(path):
                executables.add(path)
    return executables


def _dist_files(repo_root: Path) -> list[PurePosixPath]:
    dist_root = repo_root / Path(*DIST_PATH.parts)
    if not dist_root.is_dir():
        raise ReleasePackageError("frontend-prototype/dist is missing; build the frontend first.")

    paths: list[PurePosixPath] = []
    for candidate in sorted(dist_root.rglob("*")):
        if candidate.is_symlink():
            raise ReleasePackageError(f"Symlinks are not allowed in frontend dist: {candidate}")
        if not candidate.is_file():
            continue
        relative = DIST_PATH / PurePosixPath(candidate.relative_to(dist_root).as_posix())
        if _is_release_safe(relative):
            paths.append(relative)
    if BUILD_META_PATH not in paths:
        raise ReleasePackageError("The frontend build metadata was excluded or is missing.")
    return paths


def _release_files(repo_root: Path) -> list[PurePosixPath]:
    unique = set(_tracked_files(repo_root))
    unique.update(_dist_files(repo_root))
    paths = sorted(unique, key=str)
    for relative in paths:
        source = repo_root / Path(*relative.parts)
        if source.is_symlink():
            raise ReleasePackageError(f"Symlinks are not allowed in release packages: {relative}")
        if not source.is_file():
            raise ReleasePackageError(f"Release input is missing or not a regular file: {relative}")
    return paths


def _zip_info(name: str, executable: bool) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    permissions = 0o755 if executable else 0o644
    info.external_attr = ((0o100000 | permissions) & 0xFFFF) << 16
    return info


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_release_package(repo_root: Path, tag: str, output_dir: Path) -> ReleaseArtifact:
    """Validate *tag* and create its ZIP plus a SHA-256 sidecar."""

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    _validate_tag_checkout(repo_root, tag)
    _validate_build_metadata(repo_root, tag)
    files = _release_files(repo_root)
    executable_files = _tracked_executable_files(repo_root)
    if not files:
        raise ReleasePackageError("No release files were found.")

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_name = f"Paper-Reader-{tag}.zip"
    archive_path = output_dir / archive_name
    checksum_path = output_dir / f"{archive_name}.sha256"
    archive_prefix = ARCHIVE_PREFIX_TEMPLATE.format(tag=tag)

    fd, temporary_name = tempfile.mkstemp(prefix=f".{archive_name}.", dir=output_dir)
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for relative in files:
                source = repo_root / Path(*relative.parts)
                member = f"{archive_prefix}/{relative.as_posix()}"
                executable = relative in executable_files
                archive.writestr(_zip_info(member, executable), source.read_bytes())
        os.replace(temporary_path, archive_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    checksum = _sha256(archive_path)
    checksum_path.write_text(f"{checksum}  {archive_name}\n", encoding="utf-8", newline="\n")
    return ReleaseArtifact(
        archive=archive_path,
        checksum=checksum_path,
        sha256=checksum,
        file_count=len(files),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Stable V-prefixed release tag, for example V1.6.2")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository checkout to package",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("release-artifacts"),
        help="Directory for the ZIP and checksum",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        artifact = build_release_package(args.repo_root, args.tag, args.output_dir)
    except ReleasePackageError as exc:
        print(f"release package error: {exc}")
        return 2
    print(f"Created {artifact.archive} ({artifact.file_count} files)")
    print(f"SHA-256 {artifact.sha256}")
    print(f"Checksum {artifact.checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
