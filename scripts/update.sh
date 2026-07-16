#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend-prototype"

step() {
  printf '\n==> %s\n' "$1"
}

fail() {
  printf '\nUpdate failed: %s\n' "$1" >&2
  exit 1
}

step "Checking the project checkout"
GIT_CHECKOUT=0
if [[ -e "$ROOT_DIR/.git" ]]; then
  GIT_CHECKOUT=1
  command -v git >/dev/null 2>&1 || fail "Git is required to update this checkout."
  git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || fail "The project directory is not a valid Git worktree."

  # Deliberately ignore untracked files: local .env and .paper-reader data must
  # survive an update. Git itself will still refuse a pull that would overwrite
  # an untracked path.
  git -C "$ROOT_DIR" diff --quiet --ignore-submodules -- \
    || fail "Tracked files have local changes. Commit or restore them before updating."
  git -C "$ROOT_DIR" diff --cached --quiet --ignore-submodules -- \
    || fail "Tracked files have staged changes. Commit or unstage them before updating."

  step "Downloading the latest source with a fast-forward-only pull"
  git -C "$ROOT_DIR" pull --ff-only
else
  printf '%s\n' "No Git metadata was found; source download is skipped."
  printf '%s\n' "If this came from a ZIP, extract the latest release first, then run this script there."
fi

step "Preparing Python 3.10+"
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" && -x "$ROOT_DIR/.venv/Scripts/python.exe" ]]; then
  VENV_PYTHON="$ROOT_DIR/.venv/Scripts/python.exe"
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    BASE_PYTHON="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    BASE_PYTHON="$(command -v python)"
  else
    fail "Python 3.10 or later was not found."
  fi
  "$BASE_PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' \
    || fail "Python 3.10 or later is required."
  "$BASE_PYTHON" -m venv "$ROOT_DIR/.venv"
  VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
  if [[ ! -x "$VENV_PYTHON" && -x "$ROOT_DIR/.venv/Scripts/python.exe" ]]; then
    VENV_PYTHON="$ROOT_DIR/.venv/Scripts/python.exe"
  fi
fi

"$VENV_PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' \
  || fail "The existing .venv does not use Python 3.10 or later."

step "Installing Python dependencies"
"$VENV_PYTHON" -m pip install -r "$ROOT_DIR/requirements.txt"

verify_frontend_build() {
  "$VENV_PYTHON" - "$ROOT_DIR" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
settings_text = (root / "core" / "settings.py").read_text(encoding="utf-8")
match = re.search(r'^PROJECT_VERSION\s*=\s*"([^"]+)"', settings_text, re.MULTILINE)
if not match:
    raise SystemExit("Unable to read PROJECT_VERSION from core/settings.py")

expected = match.group(1)
metadata_path = root / "frontend-prototype" / "dist" / "build-meta.json"
try:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
except FileNotFoundError:
    raise SystemExit(f"Missing frontend build metadata: {metadata_path}")
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"Invalid frontend build metadata: {exc}")

if metadata.get("schema_version") != 1:
    raise SystemExit("Unsupported frontend build metadata schema")
actual = metadata.get("project_version")
if actual != expected:
    raise SystemExit(
        f"Frontend/backend version mismatch: frontend={actual!r}, backend={expected!r}"
    )
print(f"Build metadata verified: {actual}")
PY
}

NEEDS_FRONTEND_BUILD=1
if [[ "$GIT_CHECKOUT" -eq 0 ]] && verify_frontend_build; then
  NEEDS_FRONTEND_BUILD=0
  step "Using the version-matched frontend included in the release package"
fi

if [[ "$NEEDS_FRONTEND_BUILD" -eq 1 ]]; then
  step "Checking Node.js and npm"
  command -v node >/dev/null 2>&1 || fail \
    "Node.js 18 or later is required to rebuild a missing or outdated frontend."
  command -v npm >/dev/null 2>&1 || fail "npm was not found."
  node -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 18 ? 0 : 1)' \
    || fail "Node.js 18 or later is required."

  step "Installing locked frontend dependencies"
  npm --prefix "$FRONTEND_DIR" ci

  step "Building the current frontend"
  npm --prefix "$FRONTEND_DIR" run build
fi

step "Verifying frontend/backend build versions"
verify_frontend_build

printf '\n%s\n' "Update completed successfully. Local .env and .paper-reader data were not modified."
printf '%s\n' "Restart the old service process before using the updated app:"
printf '  %q -m uvicorn app:app --host 127.0.0.1 --port 8000\n' "$VENV_PYTHON"
