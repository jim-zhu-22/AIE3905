param([switch]$NoOpen)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$pythonExe=Join-Path $projectRoot '..\.venv-seg\Scripts\python.exe'
$pages=@(
    @{Port=7862; Folder='..\04-数据\processed\yingjie_rgbd_inspection'; Page='annotate.html'},
    @{Port=7863; Folder='..\06-实验结果\测试输出\project_showcase_20261009_v1'; Page='index.html'}
)
foreach ($item in $pages) {
    $folder=Join-Path $projectRoot $item.Folder
    if (-not (Test-Path -LiteralPath (Join-Path $folder $item.Page))) {throw "Missing demo page: $folder"}
    if (-not (Get-NetTCPConnection -LocalPort $item.Port -State Listen -ErrorAction SilentlyContinue)) {
        Start-Process -FilePath $pythonExe -ArgumentList '-m','http.server',"$($item.Port)",'--bind','127.0.0.1' -WorkingDirectory $folder -WindowStyle Hidden
    }
}
if (-not $NoOpen) {Start-Process 'http://127.0.0.1:7863/'}
Write-Host 'Annotation: http://127.0.0.1:7862/annotate.html'
Write-Host 'Project showcase: http://127.0.0.1:7863/'
