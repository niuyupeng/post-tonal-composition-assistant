[CmdletBinding()]
param(
    [switch]$Full,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($Full) {
    & "$PSScriptRoot\run_project2_full_local.ps1" -Resume:$Resume
    exit $LASTEXITCODE
}

Write-Host "Running the isolated CPU smoke suite. Formal Project 2 result files are not modified."
& "$PSScriptRoot\smoke_project2.ps1"
exit $LASTEXITCODE
