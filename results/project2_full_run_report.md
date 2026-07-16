# Project 2 Full Run Report

Run label: formal_retrain_2026_06_17_plus_full_2000_eval

## Commands Executed
- `.\.venv311\Scripts\python.exe -m post_tonal.full_run env-check`
- `.\.venv311\Scripts\python.exe -m pytest`
- `.\scripts\run_project2_full_local.ps1` was started. It generated the full corpus split, ran pytest, and evaluated `rule_baseline`; PowerShell then stopped at `vanilla_transformer` because stderr/progress output was surfaced as `NativeCommandError`.
- Existing output directories were archived to `archive/full_run_before_retrain_20260617_133804/`.
- Foreground train/evaluate commands were run for `vanilla_transformer`, `proposed_constraint_guided_transformer`, `transformer_no_constraints`, `without_pcset_constraints`, `without_serial_constraints`, `without_rhythm_constraints`, `without_gesture_constraints`, `serial_only`, `pcset_only`, `rhythm_only`, `gesture_only`, and `no_constraints`.
- After auditing the first final CSV, model constraint evaluation was found to be capped at 100 samples by config. The full experiment configs were updated from `constraint_metric_samples: 100` to `constraint_metric_samples: 2000`.
- The previous 100-sample CSV/JSON summaries were preserved as `results/project2_metrics_before_2000_eval_20260617_211757.csv`, `results/project2_constraints_before_2000_eval_20260617_211757.csv`, and `results/project2_generation_examples_before_2000_eval_20260617_211757.json`.
- All 13 experiments were re-evaluated on the full 2000-sample test split, writing fresh final rows to `results/project2_metrics.csv` and `results/project2_constraints.csv`.
- `.\.venv311\Scripts\python.exe -m post_tonal.make_tables --metrics-csv results/project2_metrics.csv --constraints-csv results/project2_constraints.csv --main-table paper/tables/project2_main_results.tex --ablation-table paper/tables/project2_ablation_results.tex`
- `.\.venv311\Scripts\python.exe -m post_tonal.plot_results --metrics-csv results/project2_metrics.csv --output results/project2_constraint_summary.svg`
- Copied 20 full-evaluation proposed-model MusicXML files and JSON analysis reports into `expert_eval/project2/`.

## Environment Information
- Python: 3.11.9
- PyTorch: 2.5.1+cu121
- CUDA available: True
- CUDA device count: 1
- CUDA device: NVIDIA GeForce RTX 4060 Ti
- Operating mode: local CUDA, fp16 enabled, batch size 16

## Corpus Split Counts
- Train: 20000
- Validation: 2000
- Test: 2000
- Smoke: False
- Random seed: 42
- Generation config path: `configs/post_tonal_main.yaml`
- Split summary: `results/project2_full_split_summary.json`
- Split summary timestamp: `2026-06-17T05:41:29.919882+00:00`

## Completed Configs
- `configs/post_tonal_rule_baseline.yaml`
- `configs/post_tonal_transformer_vanilla.yaml`
- `configs/post_tonal_main.yaml`
- `configs/post_tonal_transformer_no_constraints.yaml`
- `configs/post_tonal_without_pcset_constraints.yaml`
- `configs/post_tonal_without_serial_constraints.yaml`
- `configs/post_tonal_without_rhythm_constraints.yaml`
- `configs/post_tonal_without_gesture_constraints.yaml`
- `configs/post_tonal_serial_only.yaml`
- `configs/post_tonal_pcset_only.yaml`
- `configs/post_tonal_rhythm_only.yaml`
- `configs/post_tonal_gesture_only.yaml`
- `configs/post_tonal_no_constraints.yaml`

## Final Evaluation Rows
- `rule_baseline`: split=test, metric_samples=2000, token_accuracy=NA, MusicXML_export=1.0
- `vanilla_transformer`: split=test, metric_samples=2000, token_accuracy=0.627900227546692, MusicXML_export=1.0
- `proposed_constraint_guided_transformer`: split=test, metric_samples=2000, token_accuracy=0.6253677487373352, MusicXML_export=1.0
- `transformer_no_constraints`: split=test, metric_samples=2000, token_accuracy=0.6241833243370056, MusicXML_export=1.0
- `without_pcset_constraints`: split=test, metric_samples=2000, token_accuracy=0.6176029920578003, MusicXML_export=1.0
- `without_serial_constraints`: split=test, metric_samples=2000, token_accuracy=0.6180790061950684, MusicXML_export=1.0
- `without_rhythm_constraints`: split=test, metric_samples=2000, token_accuracy=0.6057550873756409, MusicXML_export=1.0
- `without_gesture_constraints`: split=test, metric_samples=2000, token_accuracy=0.6556383647918701, MusicXML_export=1.0
- `serial_only`: split=test, metric_samples=2000, token_accuracy=0.6593442106246948, MusicXML_export=1.0
- `pcset_only`: split=test, metric_samples=2000, token_accuracy=0.6351425023078918, MusicXML_export=1.0
- `rhythm_only`: split=test, metric_samples=2000, token_accuracy=0.6389482254981995, MusicXML_export=1.0
- `gesture_only`: split=test, metric_samples=2000, token_accuracy=0.5821579647064209, MusicXML_export=1.0
- `no_constraints`: split=test, metric_samples=2000, token_accuracy=0.6238990297317505, MusicXML_export=1.0

## Failed or Retried Stages
- The initial full wrapper stopped during `vanilla_transformer` because PowerShell surfaced stderr/progress output as `NativeCommandError`; training continued with foreground/cmd-log commands.
- The first `no_constraints` training attempt failed with CUDA illegal memory access after writing an intermediate checkpoint. `runs/no_constraints` was cleared, and the same full-size batch=16 fp16 configuration succeeded on retry.
- The first post-training evaluation summary had model constraint metrics capped at 100 samples. This was corrected by changing full experiment configs to `constraint_metric_samples: 2000` and rerunning all 13 evaluations.

## OOM Adjustments
- No CUDA out-of-memory occurred.
- Dataset size was not reduced.
- Batch size was not reduced.
- Gradient accumulation was not needed.

## Final Output File Paths
- `results/project2_metrics.csv`
- `results/project2_constraints.csv`
- `results/project2_generation_examples.json`
- `results/project2_full_split_summary.json`
- `results/project2_full_run_report.md`
- `results/project2_constraint_summary.svg`
- `results/project2_multiseed_controlled_statistics.csv`
- `results/project2_multiseed_controlled_statistics.json`
- `results/multiseed_controlled/`
- `paper/tables/project2_main_results.tex`
- `paper/tables/project2_ablation_results.tex`
- `paper/tables/project2_multiseed_controlled_results.tex`
- `results/eval_musicxml/proposed_constraint_guided_transformer/`
- `expert_eval/project2/musicxml/`
- `expert_eval/project2/analysis_reports/`
- `expert_eval/project2/manifest.json`
- `logs/project2_full_run_retrain.log`
- `logs/project2_full_eval_2000.log`

## Remaining TODOs
- Manuscript claims must be traced to the matching evidence family: archived configuration rows in `project2_metrics.csv`/`project2_constraints.csv`, the primary-checkpoint paired decoding effects in `project2_controlled_statistics.*`, independent-seed teacher-forced diagnostics in `project2_multiseed_training_*`, and the three-checkpoint controlled replication in `project2_multiseed_controlled_statistics.*`.
- Blind expert ratings, legally supplied external MusicXML validation, author metadata, an archival DOI, and current JNMR portal verification remain pending.

## Primary-Checkpoint Controlled Manuscript Evaluation (2026-07-14)

- Loaded `runs/proposed_constraint_guided_transformer/checkpoint.pt` for both decoding conditions; no retraining or checkpoint change was performed.
- Evaluated the complete 2,000-item test split with evaluation seed 42042 and per-item seeds 42042--44041.
- `controlled_single_candidate` used one sampled continuation per request.
- `controlled_constraint_reranked` used four continuations per request and selected the minimum weighted symbolic penalty.
- All 2,000 sample IDs and condition bundles align exactly between the two per-sample files; 914 requests are serial and 1,086 are non-serial.
- Paired percentile-bootstrap intervals use 10,000 resamples. No multiple-endpoint adjustment is applied.
- The controlled aggregate rows are stored in `results/project2_controlled_metrics.csv` and `results/project2_controlled_constraints.csv`.
- Per-sample evidence is stored in `results/controlled_single_candidate_per_sample.json` and `results/controlled_constraint_reranked_per_sample.json`.
- Paired statistics are stored in `results/project2_controlled_statistics.json` and `results/project2_controlled_statistics.csv`.
- The generated main table is `paper/tables/project2_controlled_results.tex`; the effect figure is `paper/figures/controlled_effects.pdf`.
- The evidence-bound manuscript compiled with XeLaTeX to `paper/main.pdf` without layout or reference warnings, and the post-change test run completed with 16 passing tests.
- This table remains the primary-checkpoint analysis. It is not pooled with or replaced by the later three-checkpoint replication.

## Post-run credibility work (2026-07-15)

- Added three independent-seed configs under `configs/post_tonal_multiseed_seed42.yaml`, `post_tonal_multiseed_seed43.yaml`, and `post_tonal_multiseed_seed44.yaml` without changing the completed formal-run artifacts above.
- Added tested gradient accumulation support. Replication configs use physical batch size 8 and two accumulation steps, preserving an effective batch size of 16.
- A seed-43 batch-16 replication reached 26 complete epochs before CUDA OOM; its partial artifacts are preserved under `runs/multiseed/seed_43_batch16_oom_20260714_2254/` and are not reported as results.
- A seed-42 batch-8/accumulation-2 replication reached 17 complete epochs before it was stopped during concurrent GPU use by unrelated projects. Its partial artifacts are preserved under `runs/multiseed/seed_42_concurrent_gpu_contention_20260715_0007/` and are not reported as results.
- Those two interrupted directories remain diagnostic only and are excluded from every aggregate result.
- Rebuilt `expert_eval/project2/` from 20 structurally valid `controlled_constraint_reranked` outputs. The MusicXML creator is anonymous, encoding dates are removed, target conditions are included in the rating forms, and automatic reports are identified as material to withhold from raters.
- Added `paper/main_anonymous.tex` and the gated submission drafts under `paper/submission/`. Author identity, affiliations, funding, conflicts, originality confirmation, archive DOI, and exact live-portal formatting remain author or journal inputs.
- The post-change test run completed with 17 passing tests.

## Completed independent-seed replication (2026-07-15 to 2026-07-16)

- An explicit GPU slot was used to complete fresh runs for seeds 42, 43, and 44 on the fixed full corpus. No smoke data or interrupted checkpoint entered the analysis.
- All runs used physical batch size 8, two accumulation steps, effective batch size 16, fp16, gradient clipping, and early-stopping patience 10.
- Seed 42 ran 26 epochs and selected epoch 16 with validation loss 0.8111003876.
- Seed 43 ran 25 epochs and selected epoch 15 with validation loss 0.8130948930.
- Seed 44 ran 26 epochs and selected epoch 16 with validation loss 0.8113609002.
- Every saved checkpoint loaded on CPU with 78 finite tensors and 11,091,137 parameter values. SHA256 hashes are recorded in `results/project2_multiseed_training_metrics.csv`.
- Each checkpoint was evaluated teacher-forced on all 2,000 test fragments. The aggregate token accuracy is 0.6259456 with sample SD 0.0005128; test loss is 0.8159602 with sample SD 0.0011895.
- These replication values are sequence-model diagnostics. They are not substituted for either the primary-checkpoint or three-checkpoint K=1/K=4 generation results.
- Resource traces are stored in `results/project2_multiseed_seed42_resources.json` through `project2_multiseed_seed44_resources.json`. No run crossed the 6GB available-memory or 12GB commit-headroom stop line, and no CUDA OOM occurred.
- Per-seed metrics, aggregate statistics, and the generated table are stored in `results/project2_multiseed_training_metrics.csv`, `results/project2_multiseed_training_summary.json`, and `paper/tables/project2_multiseed_training.tex`.

## Completed three-checkpoint controlled decoding replication (2026-07-16)

The aligned K=1/K=4 decoding comparison was repeated in foreground for the independently trained seed-42, seed-43, and seed-44 checkpoints. The reproducibility command is:

```powershell
.\scripts\run_project2_multiseed_controlled.ps1 -Seeds 42,43,44 -Resume
```

- Environment lines in `logs/project2_multiseed_controlled.log` record PyTorch 2.5.1+cu121 and an NVIDIA GeForce RTX 4060 Ti.
- Each checkpoint was evaluated on the same 2,000 test conditions. The condition partition contains 914 serial and 1,086 non-serial requests.
- K=1 and K=4 used evaluation seed 42042 with per-condition seeds 42042 through 44041, generation batch size 32, and protocol `per_sample_generator_batch_v1`.
- For every seed and condition, the K=4 first-candidate SHA256 matches the K=1 candidate. The aggregate provenance also binds the three checkpoint hashes, config hash, corpus hash, vocabulary hash, split name, and split size.
- Seed 42 K=1/K=4 ran from 13:33:04 to 13:55:35, seed 43 from 13:56:28 to 14:16:45, and seed 44 from 14:22:04 to 14:41:30 Asia/Shanghai. All six generation stages and all three paired analyses completed without a logged CUDA OOM.
- Each seed-level analysis used 10,000 paired percentile-bootstrap resamples. The final analysis used 10,000 crossed percentile-bootstrap resamples over training seeds and shared aligned test conditions, with bootstrap seed 52042 and no multiple-endpoint adjustment.

The crossed analysis reports favorably oriented effects, where positive values favor K=4 except for the explicitly diagnostic non-serial aggregate row:

| Endpoint | Mean effect | Seed SD | Crossed 95% CI | Positive seeds |
|---|---:|---:|---:|---:|
| PC coverage, all | -0.0007 | 0.0009 | [-0.0027, +0.0012] | 1/3 |
| PC coverage, non-serial | +0.0043 | 0.0012 | [+0.0023, +0.0064] | 3/3 |
| PC coverage, serial | -0.0066 | 0.0020 | [-0.0107, -0.0033] | 0/3 |
| Interval-vector distance, all | +0.4177 | 0.0510 | [+0.2950, +0.5612] | 3/3 |
| Interval-vector distance, non-serial | +0.2047 | 0.0494 | [+0.1372, +0.2778] | 3/3 |
| Interval-vector distance, serial | +0.6707 | 0.0592 | [+0.4230, +0.9595] | 3/3 |
| Row-order accuracy, serial | +0.0815 | 0.0023 | [+0.0756, +0.0875] | 3/3 |
| Aggregate completion, all | -0.0031 | 0.0006 | [-0.0047, -0.0018] | 0/3 |
| Aggregate completion, serial | -0.0066 | 0.0011 | [-0.0097, -0.0040] | 0/3 |
| Aggregate completion, non-serial diagnostic | -0.0002 | 0.0005 | [-0.0013, +0.0009] | -- |
| Rhythmic-profile distance, all | +0.0507 | 0.0056 | [+0.0439, +0.0583] | 3/3 |
| Density-curve error, all | +0.2278 | 0.0165 | [+0.2021, +0.2545] | 3/3 |
| Gesture consistency, all | +0.0318 | 0.0009 | [+0.0283, +0.0353] | 3/3 |
| Range-violation rate, all | +0.0003 | 0.0001 | [+0.0002, +0.0006] | 3/3 |

The replication supports an endpoint-specific claim. K=4 favored interval-vector, row-order, rhythm, density, gesture, and range diagnostics across all three checkpoints, while serial pc-set coverage and serial aggregate completion moved in the unfavorable direction across all three. Overall pc-set coverage and the non-serial aggregate diagnostic crossed zero. These automatic endpoints overlap the reranking objective and do not provide human evidence of artistic quality.

Primary outputs:

- `results/project2_multiseed_controlled_statistics.json`
- `results/project2_multiseed_controlled_statistics.csv`
- `results/multiseed_controlled/seed_42_*`
- `results/multiseed_controlled/seed_43_*`
- `results/multiseed_controlled/seed_44_*`
- `paper/tables/project2_multiseed_controlled_results.tex`
- `logs/project2_multiseed_controlled.log`

## Final artifact and manuscript QA (2026-07-16)

- The final repository test suite passed 28 tests after the controlled multiseed provenance and endpoint-completeness checks and after manuscript integration.
- `results/project2_metrics.csv` contains all 13 expected full-run experiment rows, each evaluated on 2,000 test fragments.
- All 12 required neural checkpoints and all three independent-seed checkpoints exist; the latter hashes match `results/project2_multiseed_training_metrics.csv`.
- The expert package contains 20 MusicXML files and 20 JSON reports. All 20 XML roots parse as `score-partwise`; both blind-rating forms exist.
- `paper/main.pdf` and `paper/main_anonymous.pdf` each compile to 23 non-empty pages. The final logs contain no overfull or underfull boxes, undefined references, undefined citations, or rerun warnings, and all 46 rendered pages passed visual inspection.
- A final numerical trace matched all 14 primary controlled rows and all 14 three-checkpoint controlled rows to their archived JSON sources; the five replicated endpoints quoted in the Abstract and Results were also checked against the aggregate JSON.
