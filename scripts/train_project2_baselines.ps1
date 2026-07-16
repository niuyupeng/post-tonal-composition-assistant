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

$configs = @(
    @{Name="transformer_no_constraints"; Path="configs/post_tonal_transformer_no_constraints.yaml"; Run="runs/v3/transformer_no_constraints"},
    @{Name="serial_only"; Path="configs/post_tonal_serial_only.yaml"; Run="runs/v3/serial_only"},
    @{Name="pcset_only"; Path="configs/post_tonal_pcset_only.yaml"; Run="runs/v3/pcset_only"},
    @{Name="rhythm_only"; Path="configs/post_tonal_rhythm_only.yaml"; Run="runs/v3/rhythm_only"},
    @{Name="gesture_only"; Path="configs/post_tonal_gesture_only.yaml"; Run="runs/v3/gesture_only"}
)

foreach ($cfg in $configs) {
    Invoke-Python @(
        "-m", "post_tonal.train",
        "--config", $cfg.Path,
        "--auto-oom-retry",
        "--resume"
    )
    Invoke-Python @(
        "-m", "post_tonal.evaluate",
        "--config", $cfg.Path,
        "--checkpoint", "$($cfg.Run)/checkpoint.pt",
        "--split", "test",
        "--experiment-name", $cfg.Name,
        "--output", "results/project2_v3_$($cfg.Name)_metrics.json",
        "--metrics-csv", "results/project2_v3_metrics.csv",
        "--constraints-csv", "results/project2_v3_constraints.csv",
        "--examples-output", "results/project2_v3_generation_examples.json",
        "--table-output", "paper/tables/project2_v3_$($cfg.Name)_metrics.tex"
    )
}
