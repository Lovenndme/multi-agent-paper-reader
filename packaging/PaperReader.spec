# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


ROOT = Path(SPEC).resolve().parent.parent
PACKAGING = ROOT / "packaging"

hiddenimports = []
for package in (
    "langchain",
    "langchain_core",
    "langchain_openai",
    "langgraph",
    "multipart",
    "openai",
    "pymupdf",
    "uvicorn",
):
    hiddenimports += collect_submodules(package)

datas = [
    (str(ROOT / "frontend-prototype" / "dist"), "frontend-prototype/dist"),
    (str(ROOT / "prompts"), "prompts"),
    (str(ROOT / ".env.example"), "."),
    (str(PACKAGING / "assets" / "paper-reader.ico"), "assets"),
]

a = Analysis(
    [str(PACKAGING / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "streamlit"],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PaperReader",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PACKAGING / "assets" / "paper-reader.ico"),
    version=str(PACKAGING / "version_info.txt"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="PaperReader",
)
