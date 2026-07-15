# Project 2 Independent-Seed Run Report

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

The three-seed aggregate covers validation loss, teacher-forced test loss, and full-sequence token accuracy. Constraint metrics in the generic evaluation JSON are derived from target events when model generation is disabled and are deliberately excluded by `post_tonal.analyze_multiseed_training`. The controlled K=1 versus K=4 generation result remains the 2,000-condition seed-42 experiment. Cross-seed constraint-guided decoding is not claimed.

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

## Remaining External Work

- Blind expert ratings have not been collected.
- Controlled K=1/K=4 generation has not been repeated across all three training seeds.
- Author metadata, current JNMR portal requirements, and an archival release DOI require user or publisher input.
- Direct HTTPS `git push` was blocked by connection resets, but authenticated GitHub Git Database API transport subsequently synchronized the exact local trees and commit hashes to the project branch without rewriting history.

## Final Verification

- The full repository test suite passed: 26 tests after the controlled multiseed provenance and crossed-bootstrap preflight update.
- All three checkpoint SHA256 values were recomputed from the saved files and match the aggregate CSV.
- The aggregate mean and sample standard deviation values were recomputed from the three per-seed rows and match the JSON summary and rounded manuscript values.
- The fixed full split remains 20,000/2,000/2,000 with `smoke=false`, and each teacher-forced evaluation contains 2,000 test fragments.
- The identified and anonymous manuscripts each compile to 21 pages without layout or reference warnings; fresh renders of every page passed visual inspection.
- The expert package contains 20 parseable `score-partwise` MusicXML files, 20 paired JSON reports, and both blind-rating forms.
