$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Python = "python"
if (Test-Path ".venv\Scripts\python.exe") {
    $Python = ".\.venv\Scripts\python.exe"
}
$env:PYTHONPATH = "src"

& $Python -m post_tonal.train --config configs/post_tonal_main.yaml
& $Python -m post_tonal.evaluate `
    --config configs/post_tonal_main.yaml `
    --checkpoint runs/proposed_constraint_guided_transformer/checkpoint.pt `
    --split test `
    --experiment-name proposed_constraint_guided_transformer `
    --output results/proposed_constraint_guided_transformer_metrics.json `
    --metrics-csv results/project2_metrics.csv `
    --constraints-csv results/project2_constraints.csv `
    --examples-output results/project2_generation_examples.json `
    --table-output paper/tables/main_metrics.tex
