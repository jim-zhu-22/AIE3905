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
    elseif (Test-Path -LiteralPath (Join-Path $projectDir '.venv\Scripts\python.exe')) { $Python = Join-Path $projectDir '.venv\Scripts\python.exe' }
    else { $Python = 'python' }
}
$moduleConfig = Join-Path $projectDir 'api_service\modules.local.json'
if (-not $env:AIE_MODULE_CONFIG -and (Test-Path -LiteralPath $moduleConfig)) { $env:AIE_MODULE_CONFIG = $moduleConfig }
$env:PYTHONUTF8 = '1'
$env:AIE_API_MODE = $Mode
Write-Host "API mode: $Mode | http://127.0.0.1:$Port/"
& $Python -m uvicorn api_service.app:app --host 127.0.0.1 --port $Port --workers 1 --log-config api_service/logging.json
exit $LASTEXITCODE
