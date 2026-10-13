param([switch]$WithUI, [switch]$WithVideo)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12+ is required.' }
}
$taskRequirements = if ($WithVideo) {
    if (Test-Path -LiteralPath 'requirements-video-lock.txt') { 'requirements-video-lock.txt' } else { 'requirements-video.txt' }
} elseif ($WithUI) {
    if (Test-Path -LiteralPath 'requirements-ui-lock.txt') { 'requirements-ui-lock.txt' } else { 'requirements-ui.txt' }
} else {
    if (Test-Path -LiteralPath 'requirements-lock.txt') { 'requirements-lock.txt' } else { 'requirements.txt' }
}
& $taskPython -m pip install -r $taskRequirements
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
$taskOllamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
$taskOllama = if ($taskOllamaCommand) { $taskOllamaCommand.Source } else { Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe' }
if (-not (Test-Path -LiteralPath $taskOllama)) { throw 'Install Ollama for Windows and start it before running setup.' }
& $taskOllama pull qwen3-vl:2b
if ($LASTEXITCODE -ne 0) { throw 'Ollama model download failed; start Ollama and retry.' }
Write-Host 'Setup complete. WithVideo: start.ps1. WithUI: python image_app.py. Core: python run.py.'

