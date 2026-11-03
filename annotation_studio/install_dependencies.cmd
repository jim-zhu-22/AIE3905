@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_annotation_demo.ps1" -InstallOnly %*
pause
