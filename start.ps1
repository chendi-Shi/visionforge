$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Run setup.ps1 -WithVideo first to create the project environment.'
}
$env:PYTHONUTF8 = '1'
$env:GRADIO_ANALYTICS_ENABLED = 'False'
& $taskPython -c 'import gradio, scenedetect, imageio_ffmpeg, filelock' 2>$null
if ($LASTEXITCODE -ne 0) {
    throw 'Video dependencies missing. Run: .\.venv\Scripts\python.exe -m pip install -r requirements-video.txt'
}
Write-Host 'VisionForge video workbench: http://127.0.0.1:7860'
& $taskPython app.py
exit $LASTEXITCODE

