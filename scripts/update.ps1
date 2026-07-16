[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$FrontendDir = Join-Path $RootDir "frontend-prototype"

function Write-Step {
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Description
    )
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

Write-Step "检查项目与本地数据"
$GitMarker = Join-Path $RootDir ".git"
$GitCheckout = Test-Path $GitMarker
if ($GitCheckout) {
    $GitCommand = Get-Command git -ErrorAction SilentlyContinue
    if ($null -eq $GitCommand) {
        throw "当前目录是 Git 工作区，但系统中没有找到 Git。"
    }
    $Git = $GitCommand.Source

    & $Git -C $RootDir rev-parse --is-inside-work-tree *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "项目目录不是有效的 Git 工作区。"
    }

    # 不检查或删除未跟踪文件，确保本机 .env 与 .paper-reader 数据保留。
    # 如果未跟踪文件会被远端同名文件覆盖，Git pull 自身仍会安全拒绝。
    & $Git -C $RootDir diff --quiet --ignore-submodules --
    if ($LASTEXITCODE -ne 0) {
        throw "已跟踪文件存在本地修改。请先提交或还原这些修改，再执行更新。"
    }
    & $Git -C $RootDir diff --cached --quiet --ignore-submodules --
    if ($LASTEXITCODE -ne 0) {
        throw "暂存区存在修改。请先提交或取消暂存，再执行更新。"
    }

    Write-Step "以 fast-forward-only 方式下载最新源码"
    Invoke-Native -Executable $Git -Arguments @("-C", $RootDir, "pull", "--ff-only") -Description "git pull --ff-only"
}
else {
    Write-Host "未检测到 Git 元数据，本次不会自动下载源码。" -ForegroundColor Yellow
    Write-Host "如果项目来自 ZIP，请先解压最新发行版，再在新目录中运行本脚本。" -ForegroundColor Yellow
}

Write-Step "准备 Python 3.10+ 虚拟环境"
$VenvPython = Join-Path $RootDir ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    $BasePython = $null
    $BaseArguments = @()
    $PyCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $PyCommand) {
        $BasePython = $PyCommand.Source
        $BaseArguments = @("-3")
    }
    else {
        $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($null -ne $PythonCommand) {
            $BasePython = $PythonCommand.Source
        }
    }
    if ($null -eq $BasePython) {
        throw "未找到 Python 3.10 或更高版本。"
    }

    $VersionCheck = 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'
    Invoke-Native -Executable $BasePython -Arguments ($BaseArguments + @("-c", $VersionCheck)) -Description "Python 版本检查"
    Invoke-Native -Executable $BasePython -Arguments ($BaseArguments + @("-m", "venv", (Join-Path $RootDir ".venv"))) -Description "创建 .venv"
}

$VersionCheck = 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'
Invoke-Native -Executable $VenvPython -Arguments @("-c", $VersionCheck) -Description ".venv Python 版本检查"

Write-Step "安装 Python 依赖"
Invoke-Native -Executable $VenvPython -Arguments @("-m", "pip", "install", "-r", (Join-Path $RootDir "requirements.txt")) -Description "Python 依赖安装"

$VerifyCode = 'import json,re,sys; from pathlib import Path; root=Path(sys.argv[1]); text=(root/"core"/"settings.py").read_text(encoding="utf-8"); match=re.search(r"^PROJECT_VERSION\s*=\s*\"([^\"]+)\"",text,re.MULTILINE); match or sys.exit("Unable to read PROJECT_VERSION from core/settings.py"); expected=match.group(1); path=root/"frontend-prototype"/"dist"/"build-meta.json"; meta=json.loads(path.read_text(encoding="utf-8")); meta.get("schema_version")==1 or sys.exit("Unsupported frontend build metadata schema"); actual=meta.get("project_version"); actual==expected or sys.exit(f"Frontend/backend version mismatch: frontend={actual!r}, backend={expected!r}"); print(f"Build metadata verified: {actual}")'
$NeedsFrontendBuild = $true
if (-not $GitCheckout) {
    & $VenvPython -c $VerifyCode $RootDir
    if ($LASTEXITCODE -eq 0) {
        $NeedsFrontendBuild = $false
        Write-Step "使用正式发行包中版本已匹配的前端"
    }
}

if ($NeedsFrontendBuild) {
    Write-Step "检查 Node.js 与 npm"
    $NodeCommand = Get-Command node -ErrorAction SilentlyContinue
    if ($null -eq $NodeCommand) {
        throw "需要重建缺失或过期的前端，但未找到 Node.js 18 或更高版本。"
    }
    $NpmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if ($null -eq $NpmCommand) {
        $NpmCommand = Get-Command npm -ErrorAction SilentlyContinue
    }
    if ($null -eq $NpmCommand) {
        throw "未找到 npm。"
    }
    $Node = $NodeCommand.Source
    $Npm = $NpmCommand.Source
    Invoke-Native -Executable $Node -Arguments @("-e", 'process.exit(Number(process.versions.node.split(".")[0]) >= 18 ? 0 : 1)') -Description "Node.js 18+ 版本检查"

    Write-Step "安装锁定的前端依赖"
    Invoke-Native -Executable $Npm -Arguments @("--prefix", $FrontendDir, "ci") -Description "npm ci"

    Write-Step "重新构建当前版本前端"
    Invoke-Native -Executable $Npm -Arguments @("--prefix", $FrontendDir, "run", "build") -Description "npm run build"
}

Write-Step "校验前后端构建版本"
Invoke-Native -Executable $VenvPython -Arguments @("-c", $VerifyCode, $RootDir) -Description "前后端版本一致性校验"

Write-Host ""
Write-Host "更新完成。本机 .env 与 .paper-reader 数据未被修改。" -ForegroundColor Green
Write-Host "请先关闭旧服务窗口，再执行以下命令启动新版本："
Write-Host ".\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000" -ForegroundColor White
