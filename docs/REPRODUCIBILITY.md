# Reproducibility Guide

## Environment

The completed full experiment recorded:

- Windows 11
- Python 3.11.9
- PyTorch 2.5.1+cu121
- CUDA available on NVIDIA GeForce RTX 4060 Ti 16 GB
- no out-of-memory batch-size adjustment

Use Python 3.10 or 3.11. The setup script creates `.venv311`, installs the
project dependencies, and prints the Python/PyTorch/CUDA audit:

```powershell
.\scripts\setup_windows.ps1
.\.venv311\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
```

For the exact archived CUDA build:

```powershell
python -m pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

`requirements-cuda121.txt` records the CUDA-specific PyTorch build and the
project's supported dependency ranges. It is not presented as a complete lock
for every transitive package: the archived full-run report explicitly recorded
Python 3.11.9, PyTorch 2.5.1+cu121, CUDA availability, and the GPU, but did not
record a contemporaneous full `pip freeze`. PyTorch wheel availability depends
on Python, operating system, and CUDA index.

## Verification and Smoke Run

```powershell
$env:PYTHONPATH = "src"
python -m pytest
.\scripts\smoke_project2.ps1
```

Smoke outputs are isolated and must never be copied into canonical paths.

## Full Experiment

```powershell
.\scripts\run_project2_full_local.ps1 -Resume
```

The wrapper generates or verifies the explicit 20,000/2,000/2,000 corpus,
trains all required neural configurations, evaluates the rule and neural rows,
builds result tables/figures, exports 20 proposed-model examples, and runs the
completion gate. Use `-NoPromote` to retain staging outputs for audit.

Formal settings are defined in `configs/post_tonal_main.yaml`: seed 42, 60
epochs, physical batch size 16, fp16, 384 hidden units, six layers, six heads,
maximum sequence length 256, zero Windows data-loader workers, early-stopping
patience 10, and gradient clipping at 1.0. The runner's OOM policy preserves
the full corpus and reduces physical batch size before using accumulation.

## Artifact Verification

The canonical full-run report is the primary completion record. Check:

```powershell
Get-Content results/project2_full_split_summary.json
Import-Csv results/project2_metrics.csv | Select-Object experiment,num_samples
Get-ChildItem expert_eval/project2/musicxml -Filter *.musicxml
Get-ChildItem expert_eval/project2/analysis_reports -Filter *.json
```

Expected gates:

- split counts `20000 / 2000 / 2000`, `smoke=false`, seed 42;
- 13 canonical metric rows, each evaluated on 2,000 conditions;
- 12 required neural checkpoint paths recorded with SHA-256 hashes;
- 20 expert-package MusicXML files and 20 reports;
- MusicXML parse, requested-measure, and requested-part checks pass.

The processed corpus and checkpoints are intentionally ignored by Git:

```text
data/processed/project2_v3_main.pt
data/processed/project2_v3_main.vocab.json
runs/v3/<experiment>/checkpoint.pt
```

Use the hashes in `results/project2_full_split_summary.json` and
`results/project2_full_run_report.md` when comparing regenerated artifacts.

## Tables and Figures

```powershell
$env:PYTHONPATH = "src"
python -m post_tonal.make_tables `
  --metrics-csv results/project2_metrics.csv `
  --constraints-csv results/project2_constraints.csv `
  --main-table paper/tables/project2_main_results.tex `
  --ablation-table paper/tables/project2_ablation_results.tex

python -m post_tonal.plot_results `
  --metrics-csv results/project2_metrics.csv `
  --output results/project2_constraint_summary.svg
```

Do not manually edit generated numerical tables. Numerical manuscript claims
must trace to canonical CSV/JSON files or the split/full-run reports.
