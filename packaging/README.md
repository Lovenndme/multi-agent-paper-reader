# Windows installer build

The V1.2.0 release uses a PyInstaller one-folder application wrapped by Inno
Setup. The installed app stores the API key, logs, PDFs, SQLite history, and
runtime state under `%LOCALAPPDATA%\MultiAgentPaperReader`, outside the
application directory.

## Pinned toolchain

- Windows 10/11 x64 or Windows Server 2025
- Python 3.12.10 (the final Python 3.12 release with official Windows installers)
- Node.js 24.14.0 and the committed `frontend-prototype/package-lock.json`
- Inno Setup 6.7.3
- Python packages pinned in `packaging/requirements-windows.txt`

The Windows CI downloads the official Inno Setup 6.7.3 installer and verifies
SHA-256 `9c73c3bae7ed48d44112a0f48e66742c00090bdb5bef71d9d3c056c66e97b732`
before installing it.

## Build from a clean checkout

Install [Inno Setup 6.7.3](https://github.com/jrsoftware/issrc/releases/tag/is-6_7_3),
then run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip==26.1.2
.\.venv\Scripts\python.exe -m pip install -r packaging\requirements-windows.txt
npm.cmd ci --prefix frontend-prototype
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\packaging\build_release.ps1 -SkipSigning
```

`-SkipSigning` is intended for CI and local reproducibility checks. The output
is explicitly unsigned and must not be published as a trusted release.

For an installable preview, omit `-SkipSigning`. The script creates or reuses a
local self-signed certificate named `Multi-Agent Paper Reader Preview`:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\packaging\build_release.ps1
```

The self-signed certificate proves that the generated files were not modified
after this machine signed them, but Windows will not trust that publisher on
other computers.

For a production release, import a CA-issued Authenticode code-signing
certificate into the current user's certificate store and pass its thumbprint:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\packaging\build_release.ps1 `
  -SigningThumbprint "CERTIFICATE_THUMBPRINT"
```

Generated installers are written to `release/`, which is intentionally ignored
by Git. The CI workflow uploads its unsigned installer as a seven-day workflow
artifact for review; it does not commit the binary or publish a GitHub Release.

## Publish V1.2.0

After the source PR is merged and CI is green:

1. Create and push the new annotated `V1.2.0` tag. Do not move an existing tag.
2. Check out that exact tag and build with a CA-issued signing certificate.
3. Verify the Authenticode status and SHA-256 printed by the build script.
4. Upload the installer as the GitHub Release asset:

```powershell
gh release create V1.2.0 `
  release\Multi-Agent-Paper-Reader-1.2.0-Windows-x64-Setup.exe `
  --title "Paper Reader V1.2.0" `
  --generate-notes
```

The Simplified Chinese Inno Setup translation is vendored from
`kira-96/Inno-Setup-Chinese-Simplified-Translation`. Its copyright and complete
MIT license are retained in `THIRD_PARTY_NOTICES.md`, which the installer copies
into the application directory.

During uninstall, the installed launcher reads `runtime-state.json`, verifies
the recorded PID belongs to the installed executable, requests a graceful
shutdown, and uses a PID-targeted fallback only if the process does not exit in
time. No process is terminated by image name.
