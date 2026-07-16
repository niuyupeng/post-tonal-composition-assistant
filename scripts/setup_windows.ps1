[CmdletBinding()]
param(
    [string]$VenvPath = ".venv311"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Find-CompatiblePython {
    foreach ($version in @("3.11", "3.10")) {
        & py "-$version" -c "import sys; assert sys.version_info[:2] in {(3, 10), (3, 11)}" 2>$null
        if ($LASTEXITCODE -eq 0) {
            return @("py", "-$version")
        }
    }
    throw @"
Python 3.10 or 3.11 was not found.
Install Python 3.11 x64 from https://www.python.org/downloads/windows/
Ensure the Python Launcher is selected, then rerun:
  .\scripts\setup_windows.ps1
"@
}

$launcher = Find-CompatiblePython
$venvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    & $launcher[0] $launcher[1] -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Virtual-environment creation failed."
    }
}

& $venvPython -c "import sys; assert sys.version_info[:2] in {(3, 10), (3, 11)}, sys.version"
if ($LASTEXITCODE -ne 0) {
    throw "$VenvPath does not use Python 3.10 or 3.11. Remove that environment and rerun this script."
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed."
}

$env:PYTHONPATH = "src"
& $venvPython -c "import sys, torch; print(sys.version); print(torch.__version__); print('CUDA', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"

Write-Host "Environment ready: $venvPython"
Write-Host 'For this PowerShell session: $env:PYTHONPATH = "src"'
Write-Host "If CUDA is unavailable on the RTX machine, install the CUDA build selected at https://pytorch.org/get-started/locally/ inside $VenvPath."
