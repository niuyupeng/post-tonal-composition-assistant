# Project 2 Corrected Full Run Report

- Completion gate: INCOMPLETE

## Commands Executed
- `.\.venv311\Scripts\python.exe --version`
- `.\.venv311\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"`
- `.\.venv311\Scripts\python.exe -m pytest -q`
- `.\scripts\smoke_project2.ps1`
- `.\.venv311\Scripts\python.exe -m post_tonal.evaluate --config configs/post_tonal_rule_baseline.yaml --split test --experiment-name rule_baseline --output results/project2_v3_rule_baseline_metrics.json --metrics-csv results/project2_v3_metrics.csv --constraints-csv results/project2_v3_constraints.csv --examples-output results/project2_v3_generation_examples.json`
- `.\.venv311\Scripts\python.exe -u -m post_tonal.train --config configs/post_tonal_main.yaml --auto-oom-retry --resume`

## Environment Information
- Python: 3.11.9
- PyTorch: 2.5.1+cu121
- CUDA available: True
- CUDA device: NVIDIA GeForce RTX 4060 Ti
- Peak process RAM across completed training runs: 4.637 GiB
- Peak allocated CUDA memory across completed training runs: 0.667 GiB

## CUDA Check Output
- cuda_available: True
- cuda_device_count: 1
- cuda_device_name_0: NVIDIA GeForce RTX 4060 Ti

## Corpus Split Counts
- Train: 20000
- Validation: 2000
- Test: 2000
- Smoke: False
- Corpus format: post_tonal_synthetic_v3_windowed
- Sequence strategy: coverage_cycle
- Raw score-body tokens discarded by training: 0

## Experiment Configs Completed
- rule_baseline

## Neural Checkpoints Completed
- proposed_constraint_guided_transformer

## Neural Checkpoint Details
- proposed_constraint_guided_transformer: epochs=60, best_epoch=60, stop_reason=max_epochs, checkpoint=runs/v3/proposed_constraint_guided_transformer/checkpoint.pt, sha256=35ef1047b1c0eda553586d48c3402eb6f35cf38603032292e05bdc9cf40ff61d

## Pending Evaluation Rows
- Missing test metric row: vanilla_transformer
- Missing test metric row: proposed_constraint_guided_transformer
- Missing test metric row: transformer_no_constraints
- Missing test metric row: without_pcset_constraints
- Missing test metric row: without_serial_constraints
- Missing test metric row: without_rhythm_constraints
- Missing test metric row: without_gesture_constraints
- Missing test metric row: serial_only
- Missing test metric row: pcset_only
- Missing test metric row: rhythm_only
- Missing test metric row: gesture_only
- Missing test metric row: no_constraints

## Missing or Incomplete Neural Checkpoints
- Missing or incomplete checkpoint: vanilla_transformer
- Missing or incomplete checkpoint: transformer_no_constraints
- Missing or incomplete checkpoint: without_pcset_constraints
- Missing or incomplete checkpoint: without_serial_constraints
- Missing or incomplete checkpoint: without_rhythm_constraints
- Missing or incomplete checkpoint: without_gesture_constraints
- Missing or incomplete checkpoint: serial_only
- Missing or incomplete checkpoint: pcset_only
- Missing or incomplete checkpoint: rhythm_only
- Missing or incomplete checkpoint: gesture_only
- Missing or incomplete checkpoint: no_constraints

## Failed Experiments
- No experiment failure is inferred from a missing artifact. Consult the command log for explicit tracebacks.

## OOM Adjustments
- None recorded in completed training summaries.

## Final Metrics File Paths
- results/project2_v3_metrics.csv
- results/project2_v3_constraints.csv
- results/project2_v3_generation_examples.json
- results/project2_v3_full_split_summary.json

## Generated MusicXML Examples Path
- expert_eval/project2_v3/musicxml/ (0/0 structurally score-partwise; 0/0 requested-span adherent; 0/0 requested-voice adherent)

## Paper Tables Path
- paper/tables/project2_v3_main_results.tex (missing)
- paper/tables/project2_v3_ablation_results.tex (missing)

## Remaining TODOs
- Complete neural training for: vanilla_transformer, transformer_no_constraints, without_pcset_constraints, without_serial_constraints, without_rhythm_constraints, without_gesture_constraints, serial_only, pcset_only, rhythm_only, gesture_only, no_constraints.
- Produce exactly one 2,000-sample test metric row for: vanilla_transformer, proposed_constraint_guided_transformer, transformer_no_constraints, without_pcset_constraints, without_serial_constraints, without_rhythm_constraints, without_gesture_constraints, serial_only, pcset_only, rhythm_only, gesture_only, no_constraints.
- Generate the corrected main and ablation tables from complete v3 CSV files.
- Generate and validate at least 20 full-run MusicXML examples and paired analysis reports.
- Add blind expert ratings after human evaluation.
- Add independent, legally supplied MusicXML validation examples when available.
- Complete author metadata, declarations, and live journal-portal checks.
