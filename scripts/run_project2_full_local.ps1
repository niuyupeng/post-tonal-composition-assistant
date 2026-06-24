$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Python = "python"
if (Test-Path ".venv311\Scripts\python.exe") {
    $Python = ".\.venv311\Scripts\python.exe"
} elseif (Test-Path ".venv\Scripts\python.exe") {
    $Python = ".\.venv\Scripts\python.exe"
}
$env:PYTHONPATH = "src"
$env:POST_TONAL_DEVICE = "auto"

if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}
if (-not (Test-Path "results")) {
    New-Item -ItemType Directory -Path "results" | Out-Null
}

$logPath = "logs/project2_full_run.log"
Start-Transcript -Path $logPath -Force | Out-Null

try {
    & $Python -m post_tonal.full_run env-check

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
    Get-ChildItem -Path "data/processed" -Filter "project2_*.pt" -ErrorAction SilentlyContinue | Remove-Item -Force
    Get-ChildItem -Path "data/processed" -Filter "project2_*.vocab.json" -ErrorAction SilentlyContinue | Remove-Item -Force

    & $Python -m pytest
    & $Python -m post_tonal.full_run write-split-summary `
        --config configs/post_tonal_main.yaml `
        --output results/project2_full_split_summary.json

    $experiments = @(
        @{Name="rule_baseline"; Config="configs/post_tonal_rule_baseline.yaml"; RunDir="runs/rule_baseline"; Train=$false},
        @{Name="vanilla_transformer"; Config="configs/post_tonal_transformer_vanilla.yaml"; RunDir="runs/vanilla_transformer"; Train=$true},
        @{Name="proposed_constraint_guided_transformer"; Config="configs/post_tonal_main.yaml"; RunDir="runs/proposed_constraint_guided_transformer"; Train=$true},
        @{Name="transformer_no_constraints"; Config="configs/post_tonal_transformer_no_constraints.yaml"; RunDir="runs/transformer_no_constraints"; Train=$true},
        @{Name="no_constraints"; Config="configs/post_tonal_no_constraints.yaml"; RunDir="runs/no_constraints"; Train=$true},
        @{Name="serial_only"; Config="configs/post_tonal_serial_only.yaml"; RunDir="runs/serial_only"; Train=$true},
        @{Name="pcset_only"; Config="configs/post_tonal_pcset_only.yaml"; RunDir="runs/pcset_only"; Train=$true},
        @{Name="rhythm_only"; Config="configs/post_tonal_rhythm_only.yaml"; RunDir="runs/rhythm_only"; Train=$true},
        @{Name="gesture_only"; Config="configs/post_tonal_gesture_only.yaml"; RunDir="runs/gesture_only"; Train=$true},
        @{Name="without_pcset_constraints"; Config="configs/post_tonal_without_pcset_constraints.yaml"; RunDir="runs/without_pcset_constraints"; Train=$true},
        @{Name="without_serial_constraints"; Config="configs/post_tonal_without_serial_constraints.yaml"; RunDir="runs/without_serial_constraints"; Train=$true},
        @{Name="without_rhythm_constraints"; Config="configs/post_tonal_without_rhythm_constraints.yaml"; RunDir="runs/without_rhythm_constraints"; Train=$true},
        @{Name="without_gesture_constraints"; Config="configs/post_tonal_without_gesture_constraints.yaml"; RunDir="runs/without_gesture_constraints"; Train=$true}
    )

    foreach ($exp in $experiments) {
        Write-Host "==== Project 2 experiment: $($exp.Name) ===="
        if ($exp.Train) {
            & $Python -m post_tonal.train --config $exp.Config
            $checkpoint = "$($exp.RunDir)/checkpoint.pt"
        } else {
            $checkpoint = $null
        }

        $evalArgs = @(
            "-m", "post_tonal.evaluate",
            "--config", $exp.Config,
            "--split", "test",
            "--experiment-name", $exp.Name,
            "--output", "results/$($exp.Name)_metrics.json",
            "--metrics-csv", "results/project2_metrics.csv",
            "--constraints-csv", "results/project2_constraints.csv",
            "--examples-output", "results/project2_generation_examples.json"
        )
        if ($checkpoint -ne $null) {
            $evalArgs += @("--checkpoint", $checkpoint)
        }
        & $Python @evalArgs
    }

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
        --seed 2026

    & $Python -m post_tonal.full_run write-report `
        --output results/project2_full_run_report.md

    Write-Host "Full Project 2 local pipeline complete."
    Write-Host "Metrics CSV: results/project2_metrics.csv"
    Write-Host "Constraint CSV: results/project2_constraints.csv"
    Write-Host "Examples JSON: results/project2_generation_examples.json"
    Write-Host "Expert package: expert_eval/project2"
} finally {
    Stop-Transcript | Out-Null
}
