param([int]$Port=7862)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$demoRoot=Join-Path $projectRoot '..\04-数据\processed\yingjie_rgbd_inspection'
$pythonExe=Join-Path $projectRoot '..\.venv-seg\Scripts\python.exe'
if (-not (Test-Path -LiteralPath (Join-Path $demoRoot 'annotate.html'))) {throw 'Prepare the annotation demo first.'}
if (-not (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath $pythonExe -ArgumentList '-m','http.server',"$Port",'--bind','127.0.0.1' -WorkingDirectory $demoRoot -WindowStyle Hidden
}
Start-Process "http://127.0.0.1:$Port/annotate.html"
