param(
    [switch]$SkipFrontendBuild,
    [switch]$SkipSigning,
    [string]$SigningThumbprint,
    [string]$InnoSetupPath
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$appVersion = '1.2.0'
$installerFileName = "Multi-Agent-Paper-Reader-$appVersion-Windows-x64-Setup.exe"
$release = Join-Path $root 'release'
$buildDirectory = Join-Path $PSScriptRoot 'build'
$distDirectory = Join-Path $PSScriptRoot 'dist'

function Resolve-InnoSetupCompiler([string]$ConfiguredPath) {
    $candidates = [Collections.Generic.List[string]]::new()
    if ($ConfiguredPath) {
        $candidates.Add($ConfiguredPath)
    }
    if ($env:LOCALAPPDATA) {
        $candidates.Add((Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 6\ISCC.exe'))
    }

    $programFilesX86 = [Environment]::GetFolderPath([Environment+SpecialFolder]::ProgramFilesX86)
    if ($programFilesX86) {
        $candidates.Add((Join-Path $programFilesX86 'Inno Setup 6\ISCC.exe'))
    }

    $programFiles = [Environment]::GetFolderPath([Environment+SpecialFolder]::ProgramFiles)
    if ($programFiles) {
        $candidates.Add((Join-Path $programFiles 'Inno Setup 6\ISCC.exe'))
    }

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw 'Inno Setup 6 is missing. Install version 6.7.3 or pass -InnoSetupPath.'
}

function Assert-PackagingChildPath([string]$Path) {
    $resolvedPackaging = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\') + '\'
    $resolvedTarget = [IO.Path]::GetFullPath($Path)
    if (-not $resolvedTarget.StartsWith($resolvedPackaging, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a path outside packaging/: $resolvedTarget"
    }
}

if (-not (Test-Path -LiteralPath $python)) {
    throw 'Project virtual environment is missing. Follow packaging/README.md to create it.'
}
if ($SkipSigning -and $SigningThumbprint) {
    throw 'Do not combine -SkipSigning with -SigningThumbprint.'
}
$iscc = Resolve-InnoSetupCompiler $InnoSetupPath

& $python -c 'import PIL, PyInstaller'
if ($LASTEXITCODE -ne 0) {
    throw 'Pinned Windows build dependencies are missing. Install packaging/requirements-windows.txt.'
}

if (-not $SkipFrontendBuild) {
    if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
        throw 'Node.js/npm is missing. Install Node.js 24.14.0 and run npm.cmd ci in frontend-prototype.'
    }
    Push-Location (Join-Path $root 'frontend-prototype')
    try {
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
    } finally {
        Pop-Location
    }
}

& $python (Join-Path $PSScriptRoot 'generate_icon.py')
if ($LASTEXITCODE -ne 0) { throw 'Icon generation failed.' }

Assert-PackagingChildPath $buildDirectory
Assert-PackagingChildPath $distDirectory
Remove-Item -LiteralPath $buildDirectory -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $distDirectory -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $release -Force | Out-Null

Push-Location $root
try {
    & $python -m PyInstaller --noconfirm --clean `
        --workpath $buildDirectory `
        --distpath $distDirectory `
        (Join-Path $PSScriptRoot 'PaperReader.spec')
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed.' }
} finally {
    Pop-Location
}

if (-not $SkipSigning) {
    if (-not $SigningThumbprint) {
        $certificate = Get-ChildItem Cert:\CurrentUser\My -CodeSigningCert |
            Where-Object { $_.Subject -eq 'CN=Multi-Agent Paper Reader Preview' -and $_.NotAfter -gt (Get-Date) } |
            Sort-Object NotAfter -Descending |
            Select-Object -First 1
        if (-not $certificate) {
            $certificate = New-SelfSignedCertificate `
                -Type CodeSigningCert `
                -Subject 'CN=Multi-Agent Paper Reader Preview' `
                -CertStoreLocation 'Cert:\CurrentUser\My' `
                -KeyAlgorithm RSA `
                -KeyLength 3072 `
                -HashAlgorithm SHA256 `
                -KeyExportPolicy Exportable `
                -NotAfter (Get-Date).AddYears(2)
        }
        $SigningThumbprint = $certificate.Thumbprint
    }

    $signScript = Join-Path $PSScriptRoot 'sign_file.ps1'
    $appExe = Join-Path $PSScriptRoot 'dist\PaperReader\PaperReader.exe'
    & $signScript -Path $appExe -Thumbprint $SigningThumbprint

    $signCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$signScript`" -Thumbprint `"$SigningThumbprint`" -Path `$f"
    & $iscc "/DAppVersion=$appVersion" '/DSigningEnabled=1' "/Spreview=$signCommand" (Join-Path $PSScriptRoot 'installer.iss')
} else {
    $appExe = Join-Path $PSScriptRoot 'dist\PaperReader\PaperReader.exe'
    & $iscc "/DAppVersion=$appVersion" (Join-Path $PSScriptRoot 'installer.iss')
}
if ($LASTEXITCODE -ne 0) { throw 'Installer compilation failed.' }

$installer = Join-Path $release $installerFileName
$appSignature = Get-AuthenticodeSignature -LiteralPath $appExe
$installerSignature = Get-AuthenticodeSignature -LiteralPath $installer
$artifact = Get-Item -LiteralPath $installer
$hash = Get-FileHash -LiteralPath $installer -Algorithm SHA256
$appSigner = if ($appSignature.SignerCertificate) { $appSignature.SignerCertificate.Subject } else { $null }
$installerSigner = if ($installerSignature.SignerCertificate) { $installerSignature.SignerCertificate.Subject } else { $null }

[pscustomobject]@{
    Installer = $artifact.FullName
    SizeBytes = $artifact.Length
    Sha256 = $hash.Hash
    AppSigner = $appSigner
    AppSignatureStatus = $appSignature.Status
    InstallerSigner = $installerSigner
    InstallerSignatureStatus = $installerSignature.Status
    SigningThumbprint = $SigningThumbprint
    SigningMode = if ($SkipSigning) { 'Unsigned CI artifact' } elseif ($appSigner -eq 'CN=Multi-Agent Paper Reader Preview') { 'Self-signed preview' } else { 'Provided certificate' }
} | Format-List
