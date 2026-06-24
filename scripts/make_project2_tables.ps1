$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Python = "python"
if (Test-Path ".venv\Scripts\python.exe") {
    $Python = ".\.venv\Scripts\python.exe"
}
$env:PYTHONPATH = "src"

& $Python -m post_tonal.make_tables `
    --metrics-csv results/project2_metrics.csv `
    --constraints-csv results/project2_constraints.csv `
    --main-table paper/tables/project2_main_results.tex `
    --ablation-table paper/tables/project2_ablation_results.tex

& $Python -m post_tonal.plot_results `
    --metrics-csv results/project2_metrics.csv `
    --output results/project2_constraint_summary.svg

Write-Host "Available tables written to paper/tables."
