param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$PipelineArgs
)
$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Project environment missing. Follow perception/README.md to install it.'
}
Push-Location $PSScriptRoot
try {
    & $projectPython -m perception.pipeline.run @PipelineArgs
    $resultCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $resultCode
