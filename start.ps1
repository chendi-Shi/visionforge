$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Create .venv and install the README environment: python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements-robot-lock.txt -r requirements-temporal.txt'
}
$env:PYTHONUTF8 = '1'
$env:GRADIO_ANALYTICS_ENABLED = 'False'
$taskPreviousErrorAction = $ErrorActionPreference
try {
    # Preserve Python's full diagnostic before printing the installation hint.
    $ErrorActionPreference = 'Continue'
    & $taskPython -c 'import app'
    $taskImportExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $taskPreviousErrorAction
}
if ($taskImportExitCode -ne 0) {
    throw 'The app import preflight failed; see the diagnostic above. Install the README environment: .\.venv\Scripts\python.exe -m pip install -r requirements-robot-lock.txt -r requirements-temporal.txt'
}
Write-Host 'VisionForge robot failure localization: http://127.0.0.1:7860'
& $taskPython app.py
exit $LASTEXITCODE

