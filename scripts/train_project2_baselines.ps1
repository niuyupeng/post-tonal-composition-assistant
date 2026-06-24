$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Python = "python"
if (Test-Path ".venv\Scripts\python.exe") {
    $Python = ".\.venv\Scripts\python.exe"
}
$env:PYTHONPATH = "src"

$configs = @(
    @{Name="transformer_no_constraints"; Path="configs/post_tonal_transformer_no_constraints.yaml"; Run="runs/transformer_no_constraints"},
    @{Name="serial_only"; Path="configs/post_tonal_serial_only.yaml"; Run="runs/serial_only"},
    @{Name="pcset_only"; Path="configs/post_tonal_pcset_only.yaml"; Run="runs/pcset_only"},
    @{Name="rhythm_only"; Path="configs/post_tonal_rhythm_only.yaml"; Run="runs/rhythm_only"},
    @{Name="gesture_only"; Path="configs/post_tonal_gesture_only.yaml"; Run="runs/gesture_only"}
)

foreach ($cfg in $configs) {
    & $Python -m post_tonal.train --config $cfg.Path
    & $Python -m post_tonal.evaluate `
        --config $cfg.Path `
        --checkpoint "$($cfg.Run)/checkpoint.pt" `
        --split test `
        --experiment-name $cfg.Name `
        --output "results/$($cfg.Name)_metrics.json" `
        --metrics-csv results/project2_metrics.csv `
        --constraints-csv results/project2_constraints.csv `
        --examples-output results/project2_generation_examples.json `
        --table-output "paper/tables/$($cfg.Name)_metrics.tex"
}
