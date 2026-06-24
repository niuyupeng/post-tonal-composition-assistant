$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Python = "python"
if (Test-Path ".venv\Scripts\python.exe") {
    $Python = ".\.venv\Scripts\python.exe"
}
$env:PYTHONPATH = "src"
$env:POST_TONAL_DEVICE = "cpu"

& $Python -m post_tonal.data.generate_corpus `
    --num-samples 32 `
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

& $Python -m pytest
& $Python -m post_tonal.train --config configs/post_tonal_smoke.yaml
& $Python -m post_tonal.evaluate `
    --config configs/post_tonal_smoke.yaml `
    --checkpoint runs/smoke/checkpoint.pt `
    --output results/smoke_metrics.json `
    --table-output paper/tables/smoke_metrics.tex

1..3 | ForEach-Object {
    & $Python -m post_tonal.generate `
        --pcset 0,1,4,6 `
        --row random `
        --row_form P0 `
        --rhythm_profile pointillistic `
        --gesture fragmented `
        --voices 4 `
        --measures 8 `
        --seed $_ `
        --output "generated_scores/smoke_example_$_.musicxml" `
        --report "results/smoke_example_$_.json"
}

Write-Host "Smoke workflow complete: results/smoke_metrics.json"
