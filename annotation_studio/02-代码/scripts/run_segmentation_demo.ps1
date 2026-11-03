param([string]$Image, [string]$BoxesJson, [string]$Prompt = 'a package. a box. a bottle. a bag.', [ValidateSet('auto','cpu','cuda')][string]$Device='auto', [int[]]$SearchRegion, [switch]$Download, [switch]$Material, [switch]$NoOpen)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$envPython=Join-Path $projectRoot '..\.venv-seg\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $envPython)) { throw 'Run scripts/setup_segmentation.ps1 first.' }
if (-not $Image) { $Image=Join-Path $projectRoot '..\04-数据\映界_rgbd数据\2026_08_26_09_19_22_657_color.jpg' }
$output=Join-Path $projectRoot ('..\06-实验结果\测试输出\segmentation_'+(Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
$runArgs=@('-m','rgbd_perception.segmentation','--image',$Image,'--prompt',$Prompt,'--device',$Device,'--output',$output)
if ($BoxesJson) {$runArgs+=@('--boxes-json',$BoxesJson)}
if ($SearchRegion) {if ($SearchRegion.Count -ne 4) {throw 'SearchRegion requires X1,Y1,X2,Y2'}; $runArgs+='--search-region'; $runArgs+=$SearchRegion}
if ($Download) {$runArgs+='--download'}
if ($Material) {$runArgs+=@('--material-checkpoint',(Join-Path $projectRoot '..\06-实验结果\训练记录\week1_resnet18\best.pt'))}
Push-Location $projectRoot
try {
    & $envPython @runArgs
    if ($LASTEXITCODE -ne 0) { throw 'Segmentation failed; see the error above. No CPU fallback was silently used.' }
    if (-not $NoOpen) { Start-Process -FilePath (Join-Path $output 'index.html') }
    Write-Host "Results: $output"
} finally { Pop-Location }
