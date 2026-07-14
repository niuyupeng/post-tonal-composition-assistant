param(
    [switch]$Full
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Python = "python"
if (Test-Path ".venv\Scripts\python.exe") {
    $Python = ".\.venv\Scripts\python.exe"
}
$env:PYTHONPATH = "src"

if ($Full) {
    & ".\scripts\run_project2_full_local.ps1"
    exit $LASTEXITCODE
}

Write-Host "Running CPU-safe Project 2 verification suite. Use -Full for the long RTX local run."
$env:POST_TONAL_DEVICE = "cpu"

$aggregateFiles = @(
    "results/project2_metrics.csv",
    "results/project2_constraints.csv",
    "results/project2_generation_examples.json"
)
foreach ($file in $aggregateFiles) {
    if (Test-Path $file) {
        Remove-Item -LiteralPath $file -Force
    }
}

& $Python -m pytest

& $Python -m post_tonal.data.generate_corpus `
    --train-samples 24 `
    --val-samples 4 `
    --test-samples 4 `
    --output data/processed/post_tonal_smoke.pt `
    --vocab-output data/processed/post_tonal_smoke.vocab.json `
    --seed 7 `
    --min-measures 4 `
    --max-measures 6 `
    --min-voices 2 `
    --max-voices 4 `
    --export-musicxml `
    --musicxml-dir data/generated/smoke_musicxml `
    --musicxml-limit 3

& $Python -m post_tonal.train --config configs/post_tonal_smoke.yaml

& $Python -m post_tonal.evaluate `
    --config configs/post_tonal_smoke.yaml `
    --checkpoint runs/smoke/checkpoint.pt `
    --split test `
    --experiment-name cpu_smoke_transformer `
    --output results/cpu_smoke_transformer_metrics.json `
    --metrics-csv results/project2_metrics.csv `
    --constraints-csv results/project2_constraints.csv `
    --examples-output results/project2_generation_examples.json `
    --table-output paper/tables/cpu_smoke_transformer.tex

& $Python -m post_tonal.evaluate `
    --config configs/post_tonal_smoke.yaml `
    --split test `
    --experiment-name cpu_smoke_rule_reference `
    --output results/cpu_smoke_rule_reference_metrics.json `
    --metrics-csv results/project2_metrics.csv `
    --constraints-csv results/project2_constraints.csv `
    --examples-output results/project2_generation_examples.json `
    --table-output paper/tables/cpu_smoke_rule_reference.tex

& $Python -m post_tonal.make_tables `
    --metrics-csv results/project2_metrics.csv `
    --constraints-csv results/project2_constraints.csv `
    --main-table paper/tables/project2_main_results.tex `
    --ablation-table paper/tables/project2_ablation_results.tex

& $Python -m post_tonal.plot_results `
    --metrics-csv results/project2_metrics.csv `
    --output results/project2_constraint_summary.svg

& $Python -m post_tonal.prepare_expert_eval `
    --output-dir expert_eval/project2 `
    --count 20 `
    --examples-json results/project2_generation_examples.json `
    --experiment proposed_constraint_guided_transformer

Write-Host "CPU-safe Project 2 verification suite complete."
Write-Host "Aggregate metrics: results/project2_metrics.csv"
Write-Host "Expert package: expert_eval/project2"
