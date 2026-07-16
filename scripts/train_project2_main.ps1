$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Python = ".\.venv311\Scripts\python.exe"
if (-not (Test-Path $Python) -and (Test-Path ".venv\Scripts\python.exe")) {
    $Python = ".\.venv\Scripts\python.exe"
}
if (-not (Test-Path $Python)) { throw "Run .\scripts\setup_windows.ps1 first." }
$env:PYTHONPATH = "src"

function Invoke-Python {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python stage failed with exit code $LASTEXITCODE`: $($Arguments -join ' ')"
    }
}

Invoke-Python @(
    "-m", "post_tonal.train",
    "--config", "configs/post_tonal_main.yaml",
    "--auto-oom-retry",
    "--resume"
)
Invoke-Python @(
    "-m", "post_tonal.evaluate",
    "--config", "configs/post_tonal_main.yaml",
    "--checkpoint", "runs/v3/proposed_constraint_guided_transformer/checkpoint.pt",
    "--split", "test",
    "--experiment-name", "proposed_constraint_guided_transformer",
    "--output", "results/project2_v3_proposed_constraint_guided_transformer_metrics.json",
    "--metrics-csv", "results/project2_v3_metrics.csv",
    "--constraints-csv", "results/project2_v3_constraints.csv",
    "--examples-output", "results/project2_v3_generation_examples.json",
    "--table-output", "paper/tables/project2_v3_main_metrics.tex"
)
