[CmdletBinding()]
param(
    [int[]]$Seeds = @(42, 43, 44),
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RequiredSeeds = @(42, 43, 44)
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv311\Scripts\python.exe"
$ResultDir = Join-Path $Root "results\multiseed_controlled"
$LogPath = Join-Path $Root "logs\project2_multiseed_controlled.log"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python 3.11 environment not found: $Python"
}
New-Item -ItemType Directory -Force -Path $ResultDir, (Split-Path -Parent $LogPath) | Out-Null
$env:PYTHONPATH = Join-Path $Root "src"
$env:PYTHONWARNINGS = "ignore"

$invalidSeeds = @($Seeds | Where-Object { $_ -notin $RequiredSeeds })
if ($invalidSeeds.Count -gt 0 -or @($Seeds | Select-Object -Unique).Count -ne $Seeds.Count) {
    throw "Seeds must be a unique subset of 42, 43, and 44."
}

function Invoke-LoggedPython {
    param([string[]]$Arguments)
    $previousPreference = $ErrorActionPreference
    $exitCode = 1
    Push-Location -LiteralPath $Root
    try {
        $ErrorActionPreference = "Continue"
        & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
        Pop-Location
    }
    if ($exitCode -ne 0) {
        throw "Python stage failed with exit code ${exitCode}: $($Arguments -join ' ')"
    }
}

function Test-CompletePerSample {
    param(
        [string]$Path,
        [int]$Attempts,
        [int]$TrainingSeed,
        [string]$CheckpointSha256,
        [string]$DataSha256,
        [string]$VocabSha256
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    try {
        $payload = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        if ($payload.num_samples -ne 2000 -or $payload.candidate_attempts -ne $Attempts) {
            return $false
        }
        if ($payload.sampling_protocol -ne "per_sample_generator_batch_v1") {
            return $false
        }
        if (@($payload.samples).Count -ne 2000) {
            return $false
        }
        if ($payload.provenance.checkpoint_sha256 -ne $CheckpointSha256 -or
            $payload.provenance.checkpoint_training_seed -ne $TrainingSeed -or
            $payload.provenance.data_sha256 -ne $DataSha256 -or
            $payload.provenance.vocab_sha256 -ne $VocabSha256 -or
            $payload.provenance.dataset_split -ne "test" -or
            $payload.provenance.dataset_split_size -ne 2000) {
            return $false
        }
        $fingerprints = @($payload.samples | Where-Object {
            $_.first_candidate_sha256 -is [string] -and $_.first_candidate_sha256.Length -eq 64
        })
        return $fingerprints.Count -eq 2000
    }
    catch {
        return $false
    }
}

$cudaCheck = & $Python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__); print(torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) {
    throw "CUDA validation failed; no controlled generation was started."
}
$cudaCheck | Tee-Object -FilePath $LogPath -Append
$DataSha256 = (Get-FileHash -LiteralPath (Join-Path $Root "data\processed\project2_main.pt") -Algorithm SHA256).Hash.ToLowerInvariant()
$VocabSha256 = (Get-FileHash -LiteralPath (Join-Path $Root "data\processed\project2_main.vocab.json") -Algorithm SHA256).Hash.ToLowerInvariant()

foreach ($seed in $Seeds) {
    $checkpoint = Join-Path $Root "runs\multiseed\seed_$seed\checkpoint.pt"
    if (-not (Test-Path -LiteralPath $checkpoint)) {
        throw "Missing checkpoint for seed $seed`: $checkpoint"
    }
    $checkpointSha256 = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
    $singleMetrics = Join-Path $ResultDir "seed_${seed}_single_metrics.json"
    $singleSamples = Join-Path $ResultDir "seed_${seed}_single_per_sample.json"
    $rerankedMetrics = Join-Path $ResultDir "seed_${seed}_reranked_metrics.json"
    $rerankedSamples = Join-Path $ResultDir "seed_${seed}_reranked_per_sample.json"

    if (-not ($Resume -and
        (Test-CompletePerSample $singleSamples 1 $seed $checkpointSha256 $DataSha256 $VocabSha256) -and
        (Test-Path -LiteralPath $singleMetrics))) {
        "START seed $seed K=1 $(Get-Date -Format o)" | Tee-Object -FilePath $LogPath -Append
        Invoke-LoggedPython @(
            "-m", "post_tonal.evaluate",
            "--config", (Join-Path $Root "configs\post_tonal_multiseed_controlled_single_candidate.yaml"),
            "--checkpoint", $checkpoint,
            "--split", "test",
            "--experiment-name", "multiseed_seed${seed}_single_candidate",
            "--output", $singleMetrics,
            "--per-sample-output", $singleSamples
        )
        if (-not (Test-CompletePerSample $singleSamples 1 $seed $checkpointSha256 $DataSha256 $VocabSha256)) {
            throw "Seed $seed K=1 output failed the 2,000-condition provenance gate."
        }
        "END seed $seed K=1 $(Get-Date -Format o)" | Tee-Object -FilePath $LogPath -Append
    }

    if (-not ($Resume -and
        (Test-CompletePerSample $rerankedSamples 4 $seed $checkpointSha256 $DataSha256 $VocabSha256) -and
        (Test-Path -LiteralPath $rerankedMetrics))) {
        "START seed $seed K=4 $(Get-Date -Format o)" | Tee-Object -FilePath $LogPath -Append
        Invoke-LoggedPython @(
            "-m", "post_tonal.evaluate",
            "--config", (Join-Path $Root "configs\post_tonal_multiseed_controlled_constraint_reranked.yaml"),
            "--checkpoint", $checkpoint,
            "--split", "test",
            "--experiment-name", "multiseed_seed${seed}_constraint_reranked",
            "--output", $rerankedMetrics,
            "--per-sample-output", $rerankedSamples
        )
        if (-not (Test-CompletePerSample $rerankedSamples 4 $seed $checkpointSha256 $DataSha256 $VocabSha256)) {
            throw "Seed $seed K=4 output failed the 2,000-condition provenance gate."
        }
        "END seed $seed K=4 $(Get-Date -Format o)" | Tee-Object -FilePath $LogPath -Append
    }

    Invoke-LoggedPython @(
        "-m", "post_tonal.analyze_controlled_results",
        "--single", $singleSamples,
        "--reranked", $rerankedSamples,
        "--output-json", (Join-Path $ResultDir "seed_${seed}_statistics.json"),
        "--output-csv", (Join-Path $ResultDir "seed_${seed}_statistics.csv"),
        "--output-table", (Join-Path $ResultDir "seed_${seed}_statistics.tex"),
        "--bootstrap-seed", (52042 + $seed),
        "--bootstrap-samples", "10000"
    )
}

$allRequiredOutputsComplete = $true
foreach ($seed in $RequiredSeeds) {
    $checkpoint = Join-Path $Root "runs\multiseed\seed_$seed\checkpoint.pt"
    if (-not (Test-Path -LiteralPath $checkpoint)) {
        $allRequiredOutputsComplete = $false
        continue
    }
    $checkpointSha256 = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
    $singleSamples = Join-Path $ResultDir "seed_${seed}_single_per_sample.json"
    $rerankedSamples = Join-Path $ResultDir "seed_${seed}_reranked_per_sample.json"
    if (-not (Test-CompletePerSample $singleSamples 1 $seed $checkpointSha256 $DataSha256 $VocabSha256) -or
        -not (Test-CompletePerSample $rerankedSamples 4 $seed $checkpointSha256 $DataSha256 $VocabSha256)) {
        $allRequiredOutputsComplete = $false
    }
}

if (-not $allRequiredOutputsComplete) {
    Write-Output "Completed selected seeds: $($Seeds -join ', '). Cross-seed aggregation is deferred until seeds 42, 43, and 44 all pass the provenance gate."
    return
}

$aggregateArguments = @("-m", "post_tonal.analyze_multiseed_controlled")
foreach ($seed in $RequiredSeeds) {
    $aggregateArguments += @(
        "--seed", "$seed",
        "--single", (Join-Path $ResultDir "seed_${seed}_single_per_sample.json"),
        "--reranked", (Join-Path $ResultDir "seed_${seed}_reranked_per_sample.json")
    )
}
$aggregateArguments += @(
    "--output-json", (Join-Path $Root "results\project2_multiseed_controlled_statistics.json"),
    "--output-csv", (Join-Path $Root "results\project2_multiseed_controlled_statistics.csv"),
    "--output-table", (Join-Path $Root "paper\tables\project2_multiseed_controlled_results.tex"),
    "--bootstrap-seed", "52042",
    "--bootstrap-samples", "10000",
    "--expected-conditions", "2000"
)
Invoke-LoggedPython $aggregateArguments

Write-Output "Completed cross-seed controlled decoding for seeds: $($RequiredSeeds -join ', ')"
