$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Python = ".\.venv311\Scripts\python.exe"
if (-not (Test-Path $Python) -and (Test-Path ".venv\Scripts\python.exe")) {
    $Python = ".\.venv\Scripts\python.exe"
}
if (-not (Test-Path $Python)) { throw "Run .\scripts\setup_windows.ps1 first." }
$env:PYTHONPATH = "src"

& $Python -m post_tonal.make_tables `
    --metrics-csv results/project2_v3_metrics.csv `
    --constraints-csv results/project2_v3_constraints.csv `
    --main-table paper/tables/project2_v3_main_results.tex `
    --ablation-table paper/tables/project2_v3_ablation_results.tex

& $Python -m post_tonal.plot_results `
    --metrics-csv results/project2_v3_metrics.csv `
    --output results/project2_v3_constraint_summary.svg

Write-Host "Available tables written to paper/tables."
