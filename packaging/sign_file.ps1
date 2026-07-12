param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [Parameter(Mandatory = $true)]
    [string]$Thumbprint
)

$ErrorActionPreference = 'Stop'
$certificate = Get-ChildItem "Cert:\CurrentUser\My\$Thumbprint" -ErrorAction Stop
$signature = Set-AuthenticodeSignature -LiteralPath $Path -Certificate $certificate -HashAlgorithm SHA256
if (-not $signature.SignerCertificate) {
    throw "Authenticode signing did not attach a signer certificate to $Path"
}
Write-Output "signed=$Path"
