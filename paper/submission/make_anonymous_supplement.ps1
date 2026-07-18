[CmdletBinding()]
param(
    [string]$Output = "paper/submission/project2_anonymous_supplement.zip"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$OutputPath = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $Output))
$TempRoot = Join-Path $ProjectRoot "tmp/submission"
$StageRoot = Join-Path $TempRoot "project2_anonymous_supplement"
$AnonymousReport = Join-Path $PSScriptRoot "project2_full_run_report_anonymous.md"

if (-not $OutputPath.StartsWith($ProjectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Output must remain inside the project workspace."
}

if (Test-Path $StageRoot) {
    $ResolvedStage = (Resolve-Path $StageRoot).Path
    if (-not $ResolvedStage.StartsWith($ProjectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe staging path."
    }
    Remove-Item -LiteralPath $ResolvedStage -Recurse -Force
}

New-Item -ItemType Directory -Force $StageRoot | Out-Null

$ReportText = Get-Content (Join-Path $ProjectRoot "results/project2_full_run_report.md") -Raw -Encoding utf8
$ReportText = $ReportText.Replace($ProjectRoot, "<PROJECT_ROOT>")
$ReportText = $ReportText.Replace($ProjectRoot.Replace("\", "/"), "<PROJECT_ROOT>")
Set-Content -LiteralPath $AnonymousReport -Value $ReportText -Encoding utf8

$Copies = @(
    @{ Source = "paper/main_anonymous.pdf"; Destination = "manuscript" },
    @{ Source = "paper/submission/README.md"; Destination = "documentation" },
    @{ Source = "paper/submission/figure_alt_text.md"; Destination = "documentation" },
    @{ Source = "paper/submission/supplementary_materials_inventory.md"; Destination = "documentation" },
    @{ Source = "paper/submission/project2_full_run_report_anonymous.md"; Destination = "results" },
    @{ Source = "results/project2_metrics.csv"; Destination = "results" },
    @{ Source = "results/project2_constraints.csv"; Destination = "results" },
    @{ Source = "results/project2_generation_examples.json"; Destination = "results" },
    @{ Source = "results/project2_full_split_summary.json"; Destination = "results" },
    @{ Source = "paper/figures/project2_full_results_source.csv"; Destination = "figure_sources" },
    @{ Source = "paper/figures/make_project2_full_results.py"; Destination = "figure_sources" },
    @{ Source = "expert_eval/project2"; Destination = "expert_eval" }
)

foreach ($Item in $Copies) {
    $SourcePath = Join-Path $ProjectRoot $Item.Source
    if (-not (Test-Path $SourcePath)) {
        throw "Missing supplement input: $($Item.Source)"
    }
    $DestinationPath = Join-Path $StageRoot $Item.Destination
    New-Item -ItemType Directory -Force $DestinationPath | Out-Null
    Copy-Item -LiteralPath $SourcePath -Destination $DestinationPath -Recurse -Force
}

$TextExtensions = @(".md", ".json", ".csv", ".py", ".xml", ".musicxml")
$IdentityPattern = "niuyupeng|github\.com|C:\\Users\\nyp|/Users/nyp|\\nyp\\"
$IdentityHits = Get-ChildItem $StageRoot -Recurse -File |
    Where-Object { $TextExtensions -contains $_.Extension.ToLowerInvariant() } |
    Select-String -Pattern $IdentityPattern -CaseSensitive:$false
if ($IdentityHits) {
    $IdentityHits | ForEach-Object { Write-Error $_.ToString() }
    throw "Anonymous supplement identity audit failed."
}

New-Item -ItemType Directory -Force (Split-Path $OutputPath -Parent) | Out-Null
if (Test-Path $OutputPath) {
    Remove-Item -LiteralPath $OutputPath -Force
}
Compress-Archive -Path (Join-Path $StageRoot "*") -DestinationPath $OutputPath -CompressionLevel Optimal
if (-not (Test-Path $OutputPath) -or (Get-Item $OutputPath).Length -eq 0) {
    throw "Anonymous supplement archive was not created."
}

Write-Output "Anonymous supplement: $OutputPath"
Write-Output "Files: $((Get-ChildItem $StageRoot -Recurse -File).Count)"
Write-Output "Identity audit: PASS"
