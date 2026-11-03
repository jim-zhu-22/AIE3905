param(
    [int]$Port = 8769,
    [switch]$OnlyInbox,
    [switch]$NoBrowser,
    [switch]$InstallOnly,
    [switch]$Repair,
    [switch]$Paused
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv-seg\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    $pythonPath = Join-Path $projectRoot '.venv-demo-windows\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        $basePython = Join-Path $projectRoot '.runtime-windows\python\python.exe'
        if (-not (Test-Path -LiteralPath $basePython)) {
            $archive = Join-Path $projectRoot 'runtime\python-windows-x64.tar.gz'
            $hashFile = $archive + '.sha256'
            $expected = ((Get-Content -LiteralPath $hashFile -Raw).Trim() -split '\s+')[0]
            if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
                throw 'Runtime archive checksum mismatch. Please extract a complete package again.'
            }
            $runtimeDir = Join-Path $projectRoot '.runtime-windows'
            New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
            Write-Host 'Preparing bundled Python...'
            & tar.exe -xzf $archive -C $runtimeDir
            if ($LASTEXITCODE -ne 0) { throw 'Could not extract Python. Windows 10/11 with tar.exe is required.' }
        }
        & $basePython -m venv (Join-Path $projectRoot '.venv-demo-windows')
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated Python environment.' }
    }
}
$demoArgs = @((Join-Path $projectRoot 'start_annotation_demo.py'), '--port', "$Port")
if ($OnlyInbox) { $demoArgs += '--only-inbox' }
if ($NoBrowser) { $demoArgs += '--no-browser' }
if ($InstallOnly) { $demoArgs += '--install-only' }
if ($Repair) { $demoArgs += '--repair' }
if ($Paused) { $demoArgs += '--paused' }
& $pythonPath @demoArgs
exit $LASTEXITCODE
