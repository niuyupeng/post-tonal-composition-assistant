# Project 2 Three-Checkpoint Controlled-Decoding Run Report

## Scope

This report records the completed aligned comparison between single-candidate sampling (`K=1`) and four-candidate constraint reranking (`K=4`) for the independently trained seed-42, seed-43, and seed-44 checkpoints. It is separate from the primary-checkpoint controlled table and from the 13-row exploratory configuration archive.

## Environment

- Date: 2026-07-16 (Asia/Shanghai)
- OS: Windows 11
- Python: 3.11.9 (`.venv311`)
- PyTorch: 2.5.1+cu121
- CUDA available: true
- GPU: NVIDIA GeForce RTX 4060 Ti, 16GB
- Processed corpus: `data/processed/project2_main.pt`
- Split: 20,000 train / 2,000 validation / 2,000 test, `smoke=false`

## Command

```powershell
$env:PYTHONPATH = "src"
.\scripts\run_project2_multiseed_controlled.ps1 -Seeds 42,43,44 -Resume
```

The seeds were executed in authorized foreground stages with the same runner:

```powershell
.\scripts\run_project2_multiseed_controlled.ps1 -Seeds 42 -Resume
.\scripts\run_project2_multiseed_controlled.ps1 -Seeds 43 -Resume
.\scripts\run_project2_multiseed_controlled.ps1 -Seeds 44 -Resume
```

The full console trace is stored in `logs/project2_multiseed_controlled.log`.

## Protocol

- Checkpoints: `runs/multiseed/seed_42/checkpoint.pt`, `seed_43/checkpoint.pt`, and `seed_44/checkpoint.pt`
- Test conditions per checkpoint: 2,000
- Serial conditions per checkpoint: 914
- Non-serial conditions per checkpoint: 1,086
- Evaluation seed schedule: `42042 + sample_index`
- Generation batch size: 32
- Sampling protocol: `per_sample_generator_batch_v1`
- Single-candidate condition: one sampled continuation
- Reranked condition: four sampled continuations with deterministic minimum-penalty selection
- Bootstrap: 10,000 crossed percentile resamples, seed 52042
- Multiple-endpoint adjustment: none

Before aggregation, the runner verifies the test-split size, per-item seed schedule, candidate count, batch size, sampling protocol, finite endpoint values, dataset/vocabulary/checkpoint hashes, condition alignment, and first-candidate SHA-256 identity for every within-checkpoint pair. The aggregate covers 14 prespecified endpoints. Missing or non-finite endpoint values cause the analysis to fail rather than being skipped.

## Execution Timeline

| Checkpoint | Condition | Start | End | Status |
|---|---|---|---|---|
| seed 42 | K=1 | 13:33:04 | 13:37:56 | complete |
| seed 42 | K=4 | 13:37:56 | 13:55:35 | complete |
| seed 43 | K=1 | 13:56:28 | 14:00:49 | complete |
| seed 43 | K=4 | 14:00:49 | 14:16:45 | complete |
| seed 44 | K=1 | 14:22:04 | 14:26:06 | complete |
| seed 44 | K=4 | 14:26:06 | 14:41:30 | complete |

No CUDA out-of-memory error occurred and no dataset, candidate budget, or endpoint was reduced. GPU and host-memory safety gates were checked before the final seed. Continuous peak RAM/VRAM telemetry was not recorded for this decoding-only run, so no peak value is claimed.

## Aggregate Findings

Effects are oriented so that positive values favor K=4, except the directionless non-serial aggregate diagnostic.

- Serial row-order accuracy: +0.081514, crossed 95% interval [0.075605, 0.087475], favorable in 3/3 checkpoints.
- Rhythmic-profile distance: +0.050689, [0.043874, 0.058302], favorable in 3/3 checkpoints.
- Density-curve error: +0.227813, [0.202112, 0.254461], favorable in 3/3 checkpoints.
- Gesture consistency: +0.031770, [0.028255, 0.035309], favorable in 3/3 checkpoints.
- Overall interval-vector distance: +0.417667, [0.295000, 0.561175], favorable in 3/3 checkpoints.
- Overall pc-set coverage: -0.000717, [-0.002694, 0.001170], favorable in 1/3 checkpoints; the interval crosses zero.
- Serial pc-set coverage: -0.006625, [-0.010680, -0.003294], favorable in 0/3 checkpoints.
- Serial aggregate completion: -0.006595, [-0.009664, -0.003981], favorable in 0/3 checkpoints.

These are automatic diagnostic effects on one fixed synthetic testbed. Reranking and evaluation use overlapping symbolic diagnostics, the 14 intervals are unadjusted for multiplicity, and only three checkpoints were evaluated. The results do not establish artistic quality, composer preference, or external validity.

## Output Files

- Aggregate JSON: `results/project2_multiseed_controlled_statistics.json`
- Aggregate CSV: `results/project2_multiseed_controlled_statistics.csv`
- Manuscript table: `paper/tables/project2_multiseed_controlled_results.tex`
- Per-checkpoint raw records and statistics: `results/multiseed_controlled/`
- Full trace: `logs/project2_multiseed_controlled.log`

## Final Verification

- Environment recheck: Python 3.11.9, PyTorch 2.5.1+cu121, CUDA available, NVIDIA GeForce RTX 4060 Ti.
- Repository tests: 28 passed; two PyTorch nested-tensor warnings only.
- Aggregate trace: all 14 LaTeX table rows match the aggregate JSON at the reported precision.
- Manuscript trace: the replicated endpoint values quoted in the Abstract and Results match the aggregate JSON.
- PDF QA: identified and anonymous manuscripts each compile to 23 pages with no layout, citation, reference, or rerun warnings; all 46 rendered pages were inspected.
- Expert package: 20/20 MusicXML files parse as `score-partwise`, with 20 paired reports and both blind-rating forms.

## Remaining External Work

- Blind ratings by composers or post-tonal analysts are not yet collected.
- Validation on independently supplied, legally usable MusicXML is not yet available.
- Author/affiliation metadata and live journal-portal formatting remain manual submission tasks.
