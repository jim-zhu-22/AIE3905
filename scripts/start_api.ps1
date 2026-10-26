param(
    [ValidateSet('archive','model')][string]$Mode = 'archive',
    [string]$Python,
    [int]$Port = 8000
)
$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectDir
if (-not $Python) {
    $localInterpreter = Join-Path $projectDir '.venv-api\Scripts\python.exe'
    if (Test-Path -LiteralPath $localInterpreter) { $Python = $localInterpreter }
    else { $Python = 'python' }
}
$env:PYTHONUTF8 = '1'
$env:AIE_API_MODE = $Mode
Write-Host "API mode: $Mode | http://127.0.0.1:$Port/"
& $Python -m uvicorn api_service.app:app --host 127.0.0.1 --port $Port --workers 1 --log-config api_service/logging.json
exit $LASTEXITCODE
