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
- `paper/tables/project2_main_results.tex`
- `paper/tables/project2_ablation_results.tex`
- `results/eval_musicxml/proposed_constraint_guided_transformer/`
- `expert_eval/project2/musicxml/`
- `expert_eval/project2/analysis_reports/`
- `expert_eval/project2/manifest.json`
- `logs/project2_full_run_retrain.log`
- `logs/project2_full_eval_2000.log`

## Remaining TODOs
- Expert human ratings have not been collected.
- Manuscript claims should be updated only from `results/project2_metrics.csv`, `results/project2_constraints.csv`, and the JSON reports produced by this run.
- Optional LaTeX compilation may require local CJK/TeX setup.

## Controlled Manuscript Evaluation (2026-07-14)

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
