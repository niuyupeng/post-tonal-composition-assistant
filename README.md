# Neural-Symbolic Post-Tonal Composition Assistant

Chinese title: 融合音级集合、序列与节奏轮廓约束的神经符号后调性作曲辅助方法

This repository is a runnable research codebase for score-level post-tonal composition assistance. It generates short contemporary art-music fragments conditioned on pitch-class sets, interval vectors, twelve-tone rows, serial transformations, rhythmic profiles, gesture labels, and instrumentation/voice count. Outputs are symbolic scores, MusicXML files, and explainable post-tonal analysis reports.

## Legal Data Strategy

Post-1945 contemporary scores are often copyrighted. This project therefore does not scrape, bundle, or train on copyrighted 20th/21st-century scores. The default corpus is reproducible and rule-generated from explicit post-tonal constraints. Users may optionally validate with their own MusicXML examples, but those examples are not required and are not included here.

## Setup

Windows PowerShell:

```powershell
.\scripts\setup_windows.ps1
```

Manual setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

Install a CUDA-enabled PyTorch build from the official PyTorch selector if your local environment does not already provide it. The code auto-detects CUDA and uses CPU for smoke tests.

## CPU Smoke Test

```powershell
.\scripts\smoke_project2.ps1
```

The smoke run generates a tiny synthetic corpus, runs pytest, trains for one epoch, evaluates computed metrics, exports MusicXML examples, and writes `results/smoke_metrics.json`.

The newer aggregate verification runner also builds project-level CSVs, LaTeX tables, a plot, and the expert-evaluation package:

```powershell
.\scripts\run_project2_all_experiments.ps1
```

## Local CUDA Training

```powershell
.\scripts\run_project2_full_local.ps1
```

The full local script targets an RTX 4060 Ti 16GB class GPU and runs corpus generation, rule baseline evaluation, vanilla Transformer training, proposed constraint-guided Transformer training/evaluation, four constraint-removal ablations, table generation, plots, and expert-evaluation package creation. It also runs on CPU, but full training will be much slower.

## Generate MusicXML Examples

```powershell
$env:PYTHONPATH = "src"
python -m post_tonal.generate `
  --pcset 0,1,4,6 `
  --row random `
  --row_form P0 `
  --rhythm_profile pointillistic `
  --gesture fragmented `
  --voices 4 `
  --measures 8 `
  --output generated_scores/example.musicxml `
  --report results/example_report.json
```

## Reproduce Metrics and Tables

```powershell
.\scripts\run_project2_all_experiments.ps1
# or, for the long local GPU run:
.\scripts\run_project2_full_local.ps1
```

Metrics are computed from generated or model-produced symbolic events. The manuscript reserves `PENDING_REAL_EXPERIMENT` for evidence that has not been produced, chiefly blind expert ratings and external validation.

The completed independent-seed training replication uses configs `post_tonal_multiseed_seed42.yaml` through `post_tonal_multiseed_seed44.yaml`. Its per-seed test rows, aggregate mean/sample SD, checkpoint hashes, resource audit, and exact commands are recorded in `results/project2_multiseed_run_report.md`. Those values are teacher-forced sequence diagnostics and remain separate from both controlled decoding analyses.

The controlled decoding evidence now has two deliberately separate levels:

- The primary-checkpoint comparison uses the original proposed-model checkpoint and is reported in `results/project2_controlled_statistics.*` and `paper/tables/project2_controlled_results.tex`.
- The three-checkpoint replication repeats the aligned K=1/K=4 comparison for training seeds 42, 43, and 44. Each checkpoint uses the same 2,000 test conditions, evaluation seed schedule, batch size 32, and per-sample generation protocol. The first K=4 candidate is verified by SHA256 against the corresponding K=1 output for every seed-condition pair.

Run or resume the three-checkpoint comparison with:

```powershell
.\scripts\run_project2_multiseed_controlled.ps1 -Seeds 42,43,44 -Resume
```

The aggregate uses a crossed percentile bootstrap over training seeds and shared aligned test conditions with 10,000 resamples. Favorably oriented effects were positive for all three checkpoints for interval-vector distance (+0.4177, 95% CI [+0.2950, +0.5612]), serial row-order accuracy (+0.0815, [+0.0756, +0.0875]), rhythmic-profile distance (+0.0507, [+0.0439, +0.0583]), density-curve error (+0.2278, [+0.2021, +0.2545]), gesture consistency (+0.0318, [+0.0283, +0.0353]), and range-violation rate (+0.0003, [+0.0002, +0.0006]). The same reranking reduced serial pc-set coverage (-0.0066, [-0.0107, -0.0033]) and serial aggregate completion (-0.0066, [-0.0097, -0.0040]). Overall pc-set coverage crossed zero. These 14 endpoint-wise intervals are not adjusted for multiple comparisons.

Main outputs:

- `results/project2_metrics.csv`
- `results/project2_constraints.csv`
- `results/project2_generation_examples.json`
- `results/project2_constraint_summary.svg`
- `results/project2_multiseed_training_metrics.csv`
- `results/project2_multiseed_training_summary.json`
- `results/project2_multiseed_controlled_statistics.csv`
- `results/project2_multiseed_controlled_statistics.json`
- `results/multiseed_controlled/`
- `paper/tables/project2_main_results.tex`
- `paper/tables/project2_ablation_results.tex`
- `paper/tables/project2_multiseed_training.tex`
- `paper/tables/project2_multiseed_controlled_results.tex`
- `expert_eval/project2/`

## Scope

This project is about score-level contemporary art music composition assistance:

- pitch-class sets and interval vectors
- twelve-tone rows and P/R/I/RI transformations
- aggregate completion
- rhythmic-profile conditioning
- gesture labels
- MusicXML score output and post-tonal analysis

It is not an audio-generation model, pop MIDI generator, accompaniment generator, or text-to-audio system.

## Limitations

The default dataset is synthetic, so it tests constraint satisfaction and pipeline reproducibility rather than style imitation from copyrighted repertoire. The Transformer is intentionally modest for local training. Constraint-guided decoding uses reranking over symbolic candidates for non-differentiable metrics; those penalties are not reported as fake training losses. The three-checkpoint replication uses one fixed corpus and only three training seeds, and its 14 endpoint-wise intervals have no multiplicity adjustment. Reranking and evaluation also share symbolic diagnostics, so the experiment measures optimization of the implemented constraints rather than independent artistic quality. Blind expert ratings, external user-supplied MusicXML validation, author metadata, an archival DOI, and live submission-portal verification remain pending.
