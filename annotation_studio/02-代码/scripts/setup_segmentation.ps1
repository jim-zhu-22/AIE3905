param([string]$BasePython = (Join-Path $env:USERPROFILE 'miniconda3\envs\GPU\python.exe'))
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath $BasePython)) { throw 'Specify -BasePython with an existing Python 3.10+ interpreter containing matching torch and torchvision.' }
& $BasePython -c "import torch,torchvision; print('torch',torch.__version__,'CUDA',torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw 'The base environment needs working torch and torchvision.' }
$envPython = Join-Path $projectRoot '..\.venv-seg\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $envPython)) {
    & $BasePython -m venv --system-site-packages (Join-Path $projectRoot '..\.venv-seg')
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create segmentation environment.' }
}
& $envPython -m pip install -r (Join-Path $projectRoot 'requirements-segmentation.txt')
if ($LASTEXITCODE -ne 0) { throw 'Failed to install segmentation dependencies.' }
& $envPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Resolve dependency conflicts before continuing.' }
Write-Host 'Environment ready. Download weights with the first CLI run using --download.'
