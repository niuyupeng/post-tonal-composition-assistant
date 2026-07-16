# Project 2 Independent-Seed Training and Controlled-Decoding Report

Run dates: 2026-07-15 to 2026-07-16 (Asia/Shanghai)

## Environment

- OS: Windows 11
- Python: 3.11.9 (`.venv311`)
- PyTorch: 2.5.1+cu121
- CUDA available: true
- GPU: NVIDIA GeForce RTX 4060 Ti, 16GB
- Corpus: `data/processed/project2_main.pt`
- Split: 20,000 train / 2,000 validation / 2,000 test, `smoke=false`

## Commands

```powershell
$env:PYTHONPATH = "src"
$env:PYTHONWARNINGS = "ignore"

.\.venv311\Scripts\python.exe -m post_tonal.train --config configs\post_tonal_multiseed_seed42.yaml
.\.venv311\Scripts\python.exe -m post_tonal.train --config configs\post_tonal_multiseed_seed43.yaml
.\.venv311\Scripts\python.exe -m post_tonal.train --config configs\post_tonal_multiseed_seed44.yaml

.\.venv311\Scripts\python.exe -m post_tonal.evaluate --config configs\post_tonal_multiseed_seed42.yaml --checkpoint runs\multiseed\seed_42\checkpoint.pt --split test --experiment-name multiseed_seed42_teacher_forced --output results\project2_multiseed_seed42_test_metrics.json
.\.venv311\Scripts\python.exe -m post_tonal.evaluate --config configs\post_tonal_multiseed_seed43.yaml --checkpoint runs\multiseed\seed_43\checkpoint.pt --split test --experiment-name multiseed_seed43_teacher_forced --output results\project2_multiseed_seed43_test_metrics.json
.\.venv311\Scripts\python.exe -m post_tonal.evaluate --config configs\post_tonal_multiseed_seed44.yaml --checkpoint runs\multiseed\seed_44\checkpoint.pt --split test --experiment-name multiseed_seed44_teacher_forced --output results\project2_multiseed_seed44_test_metrics.json

.\.venv311\Scripts\python.exe -m post_tonal.analyze_multiseed_training `
  --run-dir runs\multiseed\seed_42 --evaluation-json results\project2_multiseed_seed42_test_metrics.json `
  --run-dir runs\multiseed\seed_43 --evaluation-json results\project2_multiseed_seed43_test_metrics.json `
  --run-dir runs\multiseed\seed_44 --evaluation-json results\project2_multiseed_seed44_test_metrics.json `
  --metrics-csv results\project2_multiseed_training_metrics.csv `
  --summary-json results\project2_multiseed_training_summary.json `
  --latex-table paper\tables\project2_multiseed_training.tex
```

## Completed Runs

| Seed | Epochs | Best epoch | Best validation loss | Test token accuracy | Test loss | Checkpoint SHA256 |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 26 | 16 | 0.8111003876 | 0.6264778590 | 0.8149228723 | `53bdc9028b1555d2a52f71bae8bcfe4cf0bb49f15bc0320689dfa8fe361b3213` |
| 43 | 25 | 15 | 0.8130948930 | 0.6254548535 | 0.8172585757 | `71c484c2f65806a677b94b48110d5acd1162c652a21024c22b8b3e5cfa80c03a` |
| 44 | 26 | 16 | 0.8113609002 | 0.6259040430 | 0.8156992805 | `c17de1a0acf640ed64fc95fd566397756a6df479dcc32147dfa32b800d4af861` |

Across the three runs, mean best validation loss was 0.8118520602 (sample SD 0.0010841779). Mean test token accuracy was 0.6259455852 (sample SD 0.0005127664), and mean test loss was 0.8159602428 (sample SD 0.0011895182). Every test row used all 2,000 test fragments.

## Resource Audit

| Seed | Lifetime peak process-tree RAM | Peak total VRAM | Increment above baseline | Minimum available RAM | Minimum commit remaining | Threshold action |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 4.691GB | 2,024MB | 926MB | 7.830GB | 33.435GB | none |
| 43 | 4.599GB | 2,032MB | 859MB | 9.127GB | 39.461GB | none |
| 44 | 4.600GB | 1,850MB | 709MB | 9.630GB | 39.761GB | none |

No completed run encountered CUDA OOM. Seed 43 and seed 44 ran while Project 3 used short CPU-only evaluation processes; no Project 3 GPU process overlapped these runs. The Project 2 GPU processes exited after evaluation, and the device returned to the desktop baseline.

## Evidence Boundary

The training aggregate covers validation loss, teacher-forced test loss, and full-sequence token accuracy. Constraint metrics in the generic evaluation JSON are derived from target events when model generation is disabled and are deliberately excluded by `post_tonal.analyze_multiseed_training`. Constraint-guided generation is supported separately by the aligned K=1/K=4 records described below.

The report now separates two controlled evidence levels. `project2_controlled_statistics.*` remains the primary-checkpoint seed-42 comparison used in the existing controlled table. The new `project2_multiseed_controlled_statistics.*` analysis repeats that aligned generation protocol for independently trained seed-42, seed-43, and seed-44 checkpoints. The two tables are not pooled because one reports the primary checkpoint in detail and the other estimates variation across three trained checkpoints.

## Three-Checkpoint Controlled Decoding

The completed replication can be reproduced or resumed with:

```powershell
.\scripts\run_project2_multiseed_controlled.ps1 -Seeds 42,43,44 -Resume
```

The current log records PyTorch 2.5.1+cu121 on an NVIDIA GeForce RTX 4060 Ti. Each checkpoint processed the same 2,000 test conditions, including 914 serial and 1,086 non-serial requests. K=1 and K=4 used evaluation seed 42042, the per-condition schedule 42042 through 44041, generation batch size 32, and `per_sample_generator_batch_v1`. Every K=4 first-candidate SHA256 matches its K=1 counterpart. The aggregate binds the checkpoint, config, corpus, and vocabulary SHA256 values for each run.

Shared provenance hashes are config `51cea62e77f9cc62da7087d89902ced369a90800f7335c3044a462ca94f56946`, corpus `6ae2f4579b80b3461723d3726bb8165ec4f7f649e1d648e662d00d5d48888e42`, and vocabulary `3e3719444cb449da26df6cc79d98cebe1add138425d8adb3c83e9d4b6b26dd3c`.

| Seed | K=1 start-end | K=4 start-end | Checkpoint SHA256 |
|---:|---|---|---|
| 42 | 13:33:04-13:37:56 | 13:37:56-13:55:35 | `53bdc9028b1555d2a52f71bae8bcfe4cf0bb49f15bc0320689dfa8fe361b3213` |
| 43 | 13:56:28-14:00:49 | 14:00:49-14:16:45 | `71c484c2f65806a677b94b48110d5acd1162c652a21024c22b8b3e5cfa80c03a` |
| 44 | 14:22:04-14:26:06 | 14:26:06-14:41:30 | `c17de1a0acf640ed64fc95fd566397756a6df479dcc32147dfa32b800d4af861` |

Each seed-level comparison used 10,000 paired percentile-bootstrap resamples. The final analysis used 10,000 crossed percentile-bootstrap resamples over training seeds and shared aligned conditions, bootstrap seed 52042, and no multiple-endpoint adjustment. Positive effects favor K=4 except for the non-serial aggregate diagnostic, which is the raw reranked-minus-single change.

| Endpoint | Mean effect | Seed SD | Crossed 95% CI | Positive seeds | Conditions per seed |
|---|---:|---:|---:|---:|---:|
| PC coverage, all | -0.0007 | 0.0009 | [-0.0027, +0.0012] | 1/3 | 2,000 |
| PC coverage, non-serial | +0.0043 | 0.0012 | [+0.0023, +0.0064] | 3/3 | 1,086 |
| PC coverage, serial | -0.0066 | 0.0020 | [-0.0107, -0.0033] | 0/3 | 914 |
| Interval-vector distance, all | +0.4177 | 0.0510 | [+0.2950, +0.5612] | 3/3 | 2,000 |
| Interval-vector distance, non-serial | +0.2047 | 0.0494 | [+0.1372, +0.2778] | 3/3 | 1,086 |
| Interval-vector distance, serial | +0.6707 | 0.0592 | [+0.4230, +0.9595] | 3/3 | 914 |
| Row-order accuracy, serial | +0.0815 | 0.0023 | [+0.0756, +0.0875] | 3/3 | 914 |
| Aggregate completion, all | -0.0031 | 0.0006 | [-0.0047, -0.0018] | 0/3 | 2,000 |
| Aggregate completion, serial | -0.0066 | 0.0011 | [-0.0097, -0.0040] | 0/3 | 914 |
| Aggregate completion, non-serial diagnostic | -0.0002 | 0.0005 | [-0.0013, +0.0009] | -- | 1,086 |
| Rhythmic-profile distance, all | +0.0507 | 0.0056 | [+0.0439, +0.0583] | 3/3 | 2,000 |
| Density-curve error, all | +0.2278 | 0.0165 | [+0.2021, +0.2545] | 3/3 | 2,000 |
| Gesture consistency, all | +0.0318 | 0.0009 | [+0.0283, +0.0353] | 3/3 | 2,000 |
| Range-violation rate, all | +0.0003 | 0.0001 | [+0.0002, +0.0006] | 3/3 | 2,000 |

The repeated direction is endpoint specific. K=4 favored interval-vector, row-order, rhythm, density, gesture, and range diagnostics for all three checkpoints. It also improved non-serial pc-set coverage, while serial pc-set coverage and serial aggregate completion moved in the opposite direction for all three checkpoints. Overall pc-set coverage and the non-serial aggregate diagnostic crossed zero. These endpoints are automatic diagnostics that overlap the reranking objective. They do not measure composer preference or artistic usefulness.

## Output Paths

- `runs/multiseed/seed_42/`
- `runs/multiseed/seed_43/`
- `runs/multiseed/seed_44/`
- `results/project2_multiseed_training_metrics.csv`
- `results/project2_multiseed_training_summary.json`
- `results/project2_multiseed_seed42_resources.json`
- `results/project2_multiseed_seed43_resources.json`
- `results/project2_multiseed_seed44_resources.json`
- `paper/tables/project2_multiseed_training.tex`
- `results/project2_multiseed_controlled_statistics.json`
- `results/project2_multiseed_controlled_statistics.csv`
- `results/multiseed_controlled/seed_42_*`
- `results/multiseed_controlled/seed_43_*`
- `results/multiseed_controlled/seed_44_*`
- `paper/tables/project2_multiseed_controlled_results.tex`
- `logs/project2_multiseed_controlled.log`

## Remaining External Work

- Blind expert ratings have not been collected.
- Legally supplied external MusicXML validation has not been run.
- Author metadata, current JNMR portal requirements, and an archival release DOI require user or publisher input.
- Direct HTTPS `git push` was blocked by connection resets, but authenticated GitHub Git Database API transport subsequently synchronized the exact local trees and commit hashes to the project branch without rewriting history.

## Final Verification

- The final repository test suite passed 28 tests after controlled-replication integration.
- All three checkpoint SHA256 values were recomputed from the saved files and match the aggregate CSV.
- The aggregate mean and sample standard deviation values were recomputed from the three per-seed rows and match the JSON summary and rounded manuscript values.
- The fixed full split remains 20,000/2,000/2,000 with `smoke=false`, and each teacher-forced evaluation contains 2,000 test fragments.
- The final identified and anonymous manuscripts each compile to 23 pages without layout, citation, reference, or rerun warnings. Fresh renders of all 46 pages passed visual inspection, including the three-checkpoint results table and the anonymized repository statement.
- The final numerical trace matched the 14-row controlled table and manuscript endpoint values to `results/project2_multiseed_controlled_statistics.json`.
- The expert package contains 20 parseable `score-partwise` MusicXML files, 20 paired JSON reports, and both blind-rating forms.
