[CmdletBinding()]
param(
    [switch]$Resume,
    [switch]$Fresh,
    [switch]$Promote,
    [switch]$NoPromote
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv311\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python 3.11 environment not found: $Python. Run .\scripts\setup_windows.ps1 first."
}

$env:PYTHONPATH = Join-Path $Root "src"
$env:POST_TONAL_DEVICE = "auto"
$env:PYTHONWARNINGS = "ignore"
$LogPath = Join-Path $Root "logs\project2_v3_full_run.log"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
$ResolvedRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
if ($Promote -and $NoPromote) {
    throw "Use either -Promote or -NoPromote, not both."
}
$ShouldPromote = -not $NoPromote

function Assert-WorkspacePath {
    param([string]$Path)
    $resolved = [System.IO.Path]::GetFullPath($Path)
    if (-not $resolved.StartsWith($ResolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing filesystem mutation outside workspace: $resolved"
    }
}

function Invoke-LoggedPython {
    param([string[]]$Arguments)
    $previousPreference = $ErrorActionPreference
    $exitCode = 1
    Push-Location -LiteralPath $Root
    try {
        $ErrorActionPreference = "Continue"
        "COMMAND $Python $($Arguments -join ' ')" | Tee-Object -FilePath $LogPath -Append
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

function Test-EvaluationComplete {
    param(
        [string]$Path,
        [string]$ConfigPath,
        [AllowNull()]
        [string]$CheckpointPath,
        [string]$ExportDirectory
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    if (-not (Test-Path -LiteralPath $ExportDirectory)) {
        return $false
    }
    $xmlExports = @(Get-ChildItem -LiteralPath $ExportDirectory -Filter "*_test_*.musicxml" -File)
    $reportExports = @(Get-ChildItem -LiteralPath $ExportDirectory -Filter "*_test_*.json" -File)
    if ($xmlExports.Count -ne 20 -or $reportExports.Count -ne 20) {
        return $false
    }
    try {
        $payload = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        $configHash = (Get-FileHash -LiteralPath (Join-Path $Root $ConfigPath) -Algorithm SHA256).Hash.ToLowerInvariant()
        if ([string]::IsNullOrWhiteSpace($CheckpointPath)) {
            $checkpointHashMatches = $null -eq $payload.provenance.checkpoint_sha256
        }
        elseif (-not (Test-Path -LiteralPath $CheckpointPath)) {
            return $false
        }
        else {
            $checkpointHash = (Get-FileHash -LiteralPath $CheckpointPath -Algorithm SHA256).Hash.ToLowerInvariant()
            $checkpointHashMatches = $payload.provenance.checkpoint_sha256 -eq $checkpointHash
        }
        return (
            $payload.num_samples -eq 2000 -and
            $null -ne $payload.content_span_ratio -and
            $null -ne $payload.musicxml_measure_adherence_rate -and
            $payload.musicxml_export_success_rate -eq 1.0 -and
            $payload.musicxml_measure_adherence_rate -eq 1.0 -and
            $payload.musicxml_voice_adherence_rate -eq 1.0 -and
            $payload.provenance.data_sha256 -eq $script:DataSha256 -and
            $payload.provenance.config_sha256 -eq $configHash -and
            $checkpointHashMatches
        )
    }
    catch {
        return $false
    }
}

function Test-TrainingComplete {
    param(
        [string]$RunDir,
        [string]$ConfigPath
    )
    $checkpoint = Join-Path $RunDir "checkpoint.pt"
    $summaryPath = Join-Path $RunDir "train_summary.json"
    if (-not (Test-Path -LiteralPath $checkpoint) -or -not (Test-Path -LiteralPath $summaryPath)) {
        return $false
    }
    try {
        $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
        $configHash = (Get-FileHash -LiteralPath (Join-Path $Root $ConfigPath) -Algorithm SHA256).Hash.ToLowerInvariant()
        $checkpointHash = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
        return (
            $summary.completed -eq $true -and
            $summary.epochs_ran -gt 0 -and
            @($summary.history).Count -eq $summary.epochs_ran -and
            $summary.config_sha256 -eq $configHash -and
            $summary.data_sha256 -eq $script:DataSha256 -and
            $summary.checkpoint_sha256 -eq $checkpointHash
        )
    }
    catch {
        return $false
    }
}

function Test-TrainingArtifactsPresent {
    param([string]$RunDir)
    foreach ($name in @("checkpoint.pt", "last_checkpoint.pt", "train_summary.json", "metrics.csv")) {
        if (Test-Path -LiteralPath (Join-Path $RunDir $name)) {
            return $true
        }
    }
    return $false
}

function Test-MetricsRowPresent {
    param([string]$ExperimentName)
    $metricsPath = Join-Path $Root "results\project2_v3_metrics.csv"
    if (-not (Test-Path -LiteralPath $metricsPath)) {
        return $false
    }
    $rows = @(
        Import-Csv -LiteralPath $metricsPath |
            Where-Object {
                $_.experiment -eq $ExperimentName -and
                $_.split -eq "test" -and
                [int]$_.num_samples -eq 2000
            }
    )
    return $rows.Count -eq 1
}

function Assert-ResourceGate {
    $os = Get-CimInstance Win32_OperatingSystem
    $freePhysicalGB = [double]$os.FreePhysicalMemory / 1MB
    $commitCounter = Get-Counter "\Memory\Committed Bytes", "\Memory\Commit Limit"
    $committedBytes = [double]$commitCounter.CounterSamples[0].CookedValue
    $commitLimit = [double]$commitCounter.CounterSamples[1].CookedValue
    $freeCommitGB = ($commitLimit - $committedBytes) / 1GB
    if ($freePhysicalGB -lt 6.0) {
        throw ("Resource gate failed: free physical memory is {0:N2} GiB; at least 6 GiB is required." -f $freePhysicalGB)
    }
    if ($freeCommitGB -lt 12.0) {
        throw ("Resource gate failed: free commit capacity is {0:N2} GiB; at least 12 GiB is required." -f $freeCommitGB)
    }
    "RESOURCE_GATE free_physical_gib=$([math]::Round($freePhysicalGB, 3)) free_commit_gib=$([math]::Round($freeCommitGB, 3))" |
        Tee-Object -FilePath $LogPath -Append
}

if ($Fresh) {
    $targets = @(
        (Join-Path $Root "runs\v3"),
        (Join-Path $Root "results\project2_v3_metrics.csv"),
        (Join-Path $Root "results\project2_v3_constraints.csv"),
        (Join-Path $Root "results\project2_v3_generation_examples.json"),
        (Join-Path $Root "results\project2_v3_full_split_summary.json"),
        (Join-Path $Root "results\project2_v3_full_run_report.md"),
        (Join-Path $Root "results\eval_musicxml_v3"),
        (Join-Path $Root "expert_eval\project2_v3"),
        $LogPath
    )
    foreach ($target in $targets) {
        Assert-WorkspacePath $target
        if (Test-Path -LiteralPath $target) {
            Remove-Item -LiteralPath $target -Recurse -Force
        }
    }
    $processedPath = Join-Path $Root "data\processed"
    Assert-WorkspacePath $processedPath
    Get-ChildItem -LiteralPath $processedPath -Filter "project2_v3_*" -ErrorAction SilentlyContinue |
        Remove-Item -Force
}

Invoke-LoggedPython @("-m", "post_tonal.full_run", "env-check")
Invoke-LoggedPython @("-m", "pytest")
Invoke-LoggedPython @(
    "-m", "post_tonal.full_run", "write-split-summary",
    "--config", "configs/post_tonal_main.yaml",
    "--output", "results/project2_v3_full_split_summary.json"
)
$DataSha256 = (Get-FileHash -LiteralPath (Join-Path $Root "data\processed\project2_v3_main.pt") -Algorithm SHA256).Hash.ToLowerInvariant()

$experiments = @(
    @{Name="rule_baseline"; Config="configs/post_tonal_rule_baseline.yaml"; RunDir="runs/v3/rule_baseline"; Train=$false},
    @{Name="proposed_constraint_guided_transformer"; Config="configs/post_tonal_main.yaml"; RunDir="runs/v3/proposed_constraint_guided_transformer"; Train=$true},
    @{Name="vanilla_transformer"; Config="configs/post_tonal_transformer_vanilla.yaml"; RunDir="runs/v3/vanilla_transformer"; Train=$false; ReuseFrom="runs/v3/proposed_constraint_guided_transformer"},
    @{Name="transformer_no_constraints"; Config="configs/post_tonal_transformer_no_constraints.yaml"; RunDir="runs/v3/transformer_no_constraints"; Train=$true},
    @{Name="without_pcset_constraints"; Config="configs/post_tonal_without_pcset_constraints.yaml"; RunDir="runs/v3/without_pcset_constraints"; Train=$true},
    @{Name="without_serial_constraints"; Config="configs/post_tonal_without_serial_constraints.yaml"; RunDir="runs/v3/without_serial_constraints"; Train=$true},
    @{Name="without_rhythm_constraints"; Config="configs/post_tonal_without_rhythm_constraints.yaml"; RunDir="runs/v3/without_rhythm_constraints"; Train=$true},
    @{Name="without_gesture_constraints"; Config="configs/post_tonal_without_gesture_constraints.yaml"; RunDir="runs/v3/without_gesture_constraints"; Train=$true},
    @{Name="serial_only"; Config="configs/post_tonal_serial_only.yaml"; RunDir="runs/v3/serial_only"; Train=$true},
    @{Name="pcset_only"; Config="configs/post_tonal_pcset_only.yaml"; RunDir="runs/v3/pcset_only"; Train=$true},
    @{Name="rhythm_only"; Config="configs/post_tonal_rhythm_only.yaml"; RunDir="runs/v3/rhythm_only"; Train=$true},
    @{Name="gesture_only"; Config="configs/post_tonal_gesture_only.yaml"; RunDir="runs/v3/gesture_only"; Train=$true},
    @{Name="no_constraints"; Config="configs/post_tonal_no_constraints.yaml"; RunDir="runs/v3/no_constraints"; Train=$true}
)

foreach ($exp in $experiments) {
    Write-Host "==== Corrected Project 2 experiment: $($exp.Name) ===="
    $checkpoint = Join-Path $Root "$($exp.RunDir)\checkpoint.pt"
    $runDirectory = Join-Path $Root $exp.RunDir
    $usesCheckpoint = $exp.Train -or $exp.ContainsKey("ReuseFrom")
    if ($exp.ContainsKey("ReuseFrom")) {
        $sourceRun = Join-Path $Root $exp.ReuseFrom
        $sourceCheckpoint = Join-Path $sourceRun "checkpoint.pt"
        $sourceSummary = Join-Path $sourceRun "train_summary.json"
        $sourceConfig = "configs/post_tonal_main.yaml"
        if (-not (Test-TrainingComplete $sourceRun $sourceConfig)) {
            throw "Shared generator source is incomplete: $sourceRun"
        }
        New-Item -ItemType Directory -Force -Path $runDirectory | Out-Null
        Copy-Item -LiteralPath $sourceCheckpoint -Destination $checkpoint -Force
        $summary = Get-Content -LiteralPath $sourceSummary -Raw | ConvertFrom-Json
        $summary | Add-Member -NotePropertyName shared_checkpoint_source -NotePropertyValue $exp.ReuseFrom -Force
        $summary | Add-Member -NotePropertyName shared_checkpoint_rationale -NotePropertyValue "Vanilla K=1 and proposed K=4 decoding share one trained conditional generator." -Force
        $summary | Add-Member -NotePropertyName config_path -NotePropertyValue $exp.Config -Force
        $summary | Add-Member -NotePropertyName config_sha256 -NotePropertyValue ((Get-FileHash -LiteralPath (Join-Path $Root $exp.Config) -Algorithm SHA256).Hash.ToLowerInvariant()) -Force
        $summary | Add-Member -NotePropertyName checkpoint_sha256 -NotePropertyValue ((Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()) -Force
        $summaryPath = Join-Path $runDirectory "train_summary.json"
        Assert-WorkspacePath $summaryPath
        $summaryJson = $summary | ConvertTo-Json -Depth 100
        [System.IO.File]::WriteAllText(
            $summaryPath,
            $summaryJson,
            [System.Text.UTF8Encoding]::new($false)
        )
    }
    elseif ($exp.Train) {
        if (Test-TrainingComplete $runDirectory $exp.Config) {
            Write-Host "Training already complete; preserving checkpoint for $($exp.Name)."
        }
        else {
            if ((Test-TrainingArtifactsPresent $runDirectory) -and -not $Resume) {
                throw "Incomplete training artifacts exist for $($exp.Name). Use -Resume to continue or -Fresh to restart explicitly."
            }
            Assert-ResourceGate
            $trainArguments = @("-m", "post_tonal.train", "--config", $exp.Config, "--auto-oom-retry")
            if ($Resume) {
                $trainArguments += "--resume"
            }
            Invoke-LoggedPython $trainArguments
        }
    }
    if ($usesCheckpoint -and -not (Test-TrainingComplete $runDirectory $exp.Config)) {
        throw "Missing checkpoint after training: $checkpoint"
    }

    $metricsOutput = Join-Path $Root "results\project2_v3_$($exp.Name)_metrics.json"
    $evaluationCheckpoint = if ($usesCheckpoint) { $checkpoint } else { $null }
    $evaluationExport = Join-Path $Root "results\eval_musicxml_v3\$($exp.Name)"
    if (-not ((Test-EvaluationComplete $metricsOutput $exp.Config $evaluationCheckpoint $evaluationExport) -and (Test-MetricsRowPresent $exp.Name))) {
        $evalArguments = @(
            "-m", "post_tonal.evaluate",
            "--config", $exp.Config,
            "--split", "test",
            "--experiment-name", $exp.Name,
            "--output", $metricsOutput,
            "--metrics-csv", "results/project2_v3_metrics.csv",
            "--constraints-csv", "results/project2_v3_constraints.csv",
            "--examples-output", "results/project2_v3_generation_examples.json",
            "--export-dir", "results/eval_musicxml_v3/$($exp.Name)"
        )
        if ($usesCheckpoint) {
            $evalArguments += @("--checkpoint", $checkpoint)
        }
        Invoke-LoggedPython $evalArguments
    }
    Write-Host "DONE $($exp.Name)"
}

Invoke-LoggedPython @(
    "-m", "post_tonal.make_tables",
    "--metrics-csv", "results/project2_v3_metrics.csv",
    "--constraints-csv", "results/project2_v3_constraints.csv",
    "--main-table", "paper/tables/project2_v3_main_results.tex",
    "--ablation-table", "paper/tables/project2_v3_ablation_results.tex"
)
Invoke-LoggedPython @(
    "-m", "post_tonal.plot_results",
    "--metrics-csv", "results/project2_v3_metrics.csv",
    "--output", "results/project2_v3_constraint_summary.svg"
)
Invoke-LoggedPython @(
    "-m", "post_tonal.prepare_expert_eval",
    "--output-dir", "expert_eval/project2_v3",
    "--count", "20",
    "--examples-json", "results/project2_v3_generation_examples.json",
    "--experiment", "proposed_constraint_guided_transformer"
)
Invoke-LoggedPython @(
    "-m", "post_tonal.generate",
    "--generator", "transformer",
    "--config", "configs/post_tonal_main.yaml",
    "--checkpoint", "runs/v3/proposed_constraint_guided_transformer/checkpoint.pt",
    "--pcset", "0,1,4,6",
    "--rhythm_profile", "pointillistic",
    "--gesture", "fragmented",
    "--voices", "4",
    "--measures", "8",
    "--attempts", "4",
    "--num-examples", "20",
    "--output-dir", "results/eval_musicxml_v3/proposed_constraint_guided_transformer"
)
Invoke-LoggedPython @("-m", "pytest")

$expectedExperiments = @(
    "rule_baseline",
    "vanilla_transformer",
    "proposed_constraint_guided_transformer",
    "transformer_no_constraints",
    "without_pcset_constraints",
    "without_serial_constraints",
    "without_rhythm_constraints",
    "without_gesture_constraints",
    "serial_only",
    "pcset_only",
    "rhythm_only",
    "gesture_only",
    "no_constraints"
)
$metricsRows = @(Import-Csv -LiteralPath (Join-Path $Root "results\project2_v3_metrics.csv"))
foreach ($name in $expectedExperiments) {
    $matching = @($metricsRows | Where-Object { $_.experiment -eq $name -and $_.split -eq "test" })
    if ($matching.Count -ne 1 -or [int]$matching[0].num_samples -ne 2000) {
        throw "Final metrics gate failed for experiment: $name"
    }
}
$neuralRuns = @($experiments | Where-Object { $_.Train -or $_.ContainsKey("ReuseFrom") })
foreach ($exp in $neuralRuns) {
    if (-not (Test-TrainingComplete (Join-Path $Root $exp.RunDir) $exp.Config)) {
        throw "Final checkpoint gate failed for: $($exp.Name)"
    }
}
foreach ($table in @(
    (Join-Path $Root "paper\tables\project2_v3_main_results.tex"),
    (Join-Path $Root "paper\tables\project2_v3_ablation_results.tex")
)) {
    if (-not (Test-Path -LiteralPath $table) -or (Get-Item -LiteralPath $table).Length -eq 0) {
        throw "Final table gate failed: $table"
    }
}
$expertXml = @(Get-ChildItem -LiteralPath (Join-Path $Root "expert_eval\project2_v3\musicxml") -Filter "*.musicxml")
$expertReports = @(Get-ChildItem -LiteralPath (Join-Path $Root "expert_eval\project2_v3\analysis_reports") -Filter "*.json")
if ($expertXml.Count -lt 20 -or $expertReports.Count -lt 20) {
    throw "Final expert-package gate failed: XML=$($expertXml.Count), reports=$($expertReports.Count)"
}

Invoke-LoggedPython @(
    "-m", "post_tonal.full_run", "write-report",
    "--output", "results/project2_v3_full_run_report.md",
    "--metrics", "results/project2_v3_metrics.csv",
    "--constraints", "results/project2_v3_constraints.csv",
    "--examples", "results/project2_v3_generation_examples.json",
    "--split-summary", "results/project2_v3_full_split_summary.json",
    "--expert-dir", "expert_eval/project2_v3",
    "--run-root", "runs/v3",
    "--log-path", "logs/project2_v3_full_run.log",
    "--main-table", "paper/tables/project2_v3_main_results.tex",
    "--ablation-table", "paper/tables/project2_v3_ablation_results.tex",
    "--incidents", "results/project2_v3_run_incidents.json"
)

if ($ShouldPromote) {
    Copy-Item -LiteralPath (Join-Path $Root "results\project2_v3_metrics.csv") -Destination (Join-Path $Root "results\project2_metrics.csv") -Force
    Copy-Item -LiteralPath (Join-Path $Root "results\project2_v3_constraints.csv") -Destination (Join-Path $Root "results\project2_constraints.csv") -Force
    Invoke-LoggedPython @(
        "-m", "post_tonal.full_run", "promote-generation-examples",
        "--source", "results/project2_v3_generation_examples.json",
        "--output", "results/project2_generation_examples.json",
        "--source-root", "results/eval_musicxml_v3",
        "--destination-root", "results/eval_musicxml"
    )
    Copy-Item -LiteralPath (Join-Path $Root "results\project2_v3_full_split_summary.json") -Destination (Join-Path $Root "results\project2_full_split_summary.json") -Force
    Copy-Item -LiteralPath (Join-Path $Root "results\project2_v3_full_run_report.md") -Destination (Join-Path $Root "results\project2_full_run_report.md") -Force
    Copy-Item -LiteralPath (Join-Path $Root "results\project2_v3_run_incidents.json") -Destination (Join-Path $Root "results\project2_full_run_incidents.json") -Force
    Copy-Item -LiteralPath (Join-Path $Root "results\project2_v3_constraint_summary.svg") -Destination (Join-Path $Root "results\project2_constraint_summary.svg") -Force
    Copy-Item -LiteralPath (Join-Path $Root "paper\tables\project2_v3_main_results.tex") -Destination (Join-Path $Root "paper\tables\project2_main_results.tex") -Force
    Copy-Item -LiteralPath (Join-Path $Root "paper\tables\project2_v3_ablation_results.tex") -Destination (Join-Path $Root "paper\tables\project2_ablation_results.tex") -Force
    Copy-Item -LiteralPath $LogPath -Destination (Join-Path $Root "logs\project2_full_run.log") -Force
    foreach ($exp in $neuralRuns) {
        $sourceRun = Join-Path $Root $exp.RunDir
        $destinationRun = Join-Path $Root ("runs\" + $exp.Name)
        New-Item -ItemType Directory -Force -Path $destinationRun | Out-Null
        Copy-Item -Path (Join-Path $sourceRun "*") -Destination $destinationRun -Recurse -Force
    }
    $canonicalEvaluation = Join-Path $Root "results\eval_musicxml"
    Assert-WorkspacePath $canonicalEvaluation
    if (Test-Path -LiteralPath $canonicalEvaluation) {
        Remove-Item -LiteralPath $canonicalEvaluation -Recurse -Force
    }
    Copy-Item -LiteralPath (Join-Path $Root "results\eval_musicxml_v3") -Destination $canonicalEvaluation -Recurse -Force
    $canonicalExpert = Join-Path $Root "expert_eval\project2"
    Assert-WorkspacePath $canonicalExpert
    if (Test-Path -LiteralPath $canonicalExpert) {
        Remove-Item -LiteralPath $canonicalExpert -Recurse -Force
    }
    Invoke-LoggedPython @(
        "-m", "post_tonal.prepare_expert_eval",
        "--output-dir", "expert_eval/project2",
        "--count", "20",
        "--examples-json", "results/project2_generation_examples.json",
        "--experiment", "proposed_constraint_guided_transformer"
    )
    Invoke-LoggedPython @(
        "-m", "post_tonal.full_run", "write-report",
        "--output", "results/project2_full_run_report.md",
        "--metrics", "results/project2_metrics.csv",
        "--constraints", "results/project2_constraints.csv",
        "--examples", "results/project2_generation_examples.json",
        "--split-summary", "results/project2_full_split_summary.json",
        "--expert-dir", "expert_eval/project2",
        "--run-root", "runs/v3",
        "--log-path", "logs/project2_v3_full_run.log",
        "--main-table", "paper/tables/project2_main_results.tex",
        "--ablation-table", "paper/tables/project2_ablation_results.tex",
        "--incidents", "results/project2_full_run_incidents.json"
    )
    Copy-Item -LiteralPath $LogPath -Destination (Join-Path $Root "logs\project2_full_run.log") -Force
    Write-Host "Corrected v3 outputs promoted to canonical result paths."
}

Write-Host "Corrected Project 2 full pipeline complete."
Write-Host "Metrics: results/project2_v3_metrics.csv"
Write-Host "Expert examples: expert_eval/project2_v3"
Write-Host "Log: logs/project2_v3_full_run.log"
