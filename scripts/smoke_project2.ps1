[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv311\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = Join-Path $Root ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Run .\scripts\setup_windows.ps1 first."
}

$env:PYTHONPATH = Join-Path $Root "src"
$env:POST_TONAL_DEVICE = "cpu"
$env:PYTHONWARNINGS = "ignore"
$SmokeResults = Join-Path $Root "results\smoke_v3"
$SmokeExamples = Join-Path $Root "generated_scores\smoke_v3"
New-Item -ItemType Directory -Force -Path $SmokeResults, $SmokeExamples | Out-Null

Push-Location -LiteralPath $Root
try {
    & $Python -m post_tonal.data.generate_corpus `
        --train-samples 24 `
        --val-samples 4 `
        --test-samples 4 `
        --output data/processed/post_tonal_smoke_v3.pt `
        --vocab-output data/processed/post_tonal_smoke_v3.vocab.json `
        --seed 7 `
        --min-measures 4 `
        --max-measures 6 `
        --min-voices 2 `
        --max-voices 4 `
        --export-musicxml `
        --musicxml-dir data/generated/smoke_v3_musicxml `
        --musicxml-limit 3
    if ($LASTEXITCODE -ne 0) { throw "Smoke corpus generation failed." }

    & $Python -m pytest
    if ($LASTEXITCODE -ne 0) { throw "Tests failed." }

    & $Python -m post_tonal.train --config configs/post_tonal_smoke.yaml
    if ($LASTEXITCODE -ne 0) { throw "Smoke training failed." }

    & $Python -m post_tonal.evaluate `
        --config configs/post_tonal_smoke.yaml `
        --checkpoint runs/smoke_v3/checkpoint.pt `
        --split test `
        --experiment-name cpu_smoke_v3 `
        --output results/smoke_v3/metrics.json `
        --metrics-csv results/smoke_v3/metrics.csv `
        --constraints-csv results/smoke_v3/constraints.csv `
        --examples-output results/smoke_v3/examples.json
    if ($LASTEXITCODE -ne 0) { throw "Smoke evaluation failed." }

    & $Python -m post_tonal.generate `
        --generator transformer `
        --config configs/post_tonal_smoke.yaml `
        --checkpoint runs/smoke_v3/checkpoint.pt `
        --pcset 0,1,4,6 `
        --rhythm_profile pointillistic `
        --gesture fragmented `
        --voices 4 `
        --measures 6 `
        --attempts 2 `
        --num-examples 3 `
        --output-dir generated_scores/smoke_v3
    if ($LASTEXITCODE -ne 0) { throw "Smoke model generation failed." }
}
finally {
    Pop-Location
}

Write-Host "Corrected smoke workflow complete."
Write-Host "Metrics: results/smoke_v3/metrics.json"
Write-Host "Examples: generated_scores/smoke_v3"
