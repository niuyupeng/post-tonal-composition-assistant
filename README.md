# Neural-Symbolic Post-Tonal Composition Assistant

Chinese title: 融合音级集合、序列与节奏轮廓约束的神经符号后调性作曲辅助方法

This repository implements a score-level research system for post-tonal composition assistance. It conditions symbolic generation on pitch-class sets, interval vectors, twelve-tone rows and P/R/I/RI forms, rhythmic profiles, gesture labels, instrumentation, voice count, and requested score span. Outputs are editable MusicXML scores and explainable JSON analysis reports. The project does not generate audio, pop accompaniment, or performance MIDI.

## Evidence Status

The corrected v3 code path is the current experiment protocol. CPU smoke verification and the full 20,000/2,000/2,000 synthetic split are available. Corrected neural full-run metrics are not final until `scripts/run_project2_full_local.ps1` completes every checkpoint, evaluation, table, and MusicXML gate.

Older v2 checkpoints, tables, figures, and numerical manuscript text are historical development artifacts. They must not be cited as corrected final results. The full runner writes v3-prefixed artifacts first and promotes them to canonical paths only after all completion gates pass.

## Legal Data Strategy

The repository does not scrape, redistribute, or train on copyrighted post-1945 scores. Its default corpus is generated reproducibly from explicit symbolic rules. Users may separately validate with MusicXML that they are legally entitled to use, but no external score is bundled.

Serial and non-serial pitch targets are separated in the corrected corpus: serial examples use a twelve-tone row and row form, while non-serial examples use a pitch-class set and interval vector. A held-out stochastic density curve is retained only for evaluation and is removed from model-visible condition metadata.

## Windows Setup

```powershell
.\scripts\setup_windows.ps1
.\.venv311\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
```

The setup script requires Python 3.10 or 3.11. For the formal run, verify the CUDA environment:

```powershell
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

The target local device is an NVIDIA GeForce RTX 4060 Ti with 16 GB VRAM. Formal configs use fp16, physical batch size 16, zero Windows data-loader workers, gradient clipping, and automatic OOM retries at batch sizes 8 and 4 with accumulation.

## CPU Smoke Test

```powershell
.\scripts\smoke_project2.ps1
```

The smoke workflow is isolated from formal outputs. It generates a tiny corpus, runs pytest, trains one CPU epoch, evaluates the neural checkpoint, and exports three MusicXML examples under:

- `results/smoke_v3/`
- `generated_scores/smoke_v3/`
- `runs/smoke_v3/`

## Corrected Full CUDA Run

```powershell
.\scripts\run_project2_full_local.ps1 -Resume
```

Use `-NoPromote` to retain only v3-prefixed outputs while auditing:

```powershell
.\scripts\run_project2_full_local.ps1 -Resume -NoPromote
```

The full runner:

1. validates Python, PyTorch, CUDA, tests, memory, and the explicit corpus split;
2. trains or resumes the shared conditional Transformer and all condition-prefix ablations;
3. evaluates the independent rule reference, single-candidate decoder, four-candidate guided decoder, no-constraint controls, and focused-condition models;
4. writes aggregate CSV/JSON outputs, LaTeX tables, a summary plot, and 20 structurally checked MusicXML examples;
5. promotes corrected outputs to canonical paths only after all required rows, checkpoints, tables, reports, and expert-package files pass.

Training writes both the best `checkpoint.pt` and a resumable `last_checkpoint.pt`. Each run summary records the best epoch, hashes, elapsed time, OOM adjustment, and measured peak RAM/VRAM.

## Direct Primary Training

```powershell
$env:PYTHONPATH = "src"
.\.venv311\Scripts\python.exe -m post_tonal.train `
  --config configs/post_tonal_main.yaml `
  --auto-oom-retry `
  --resume
```

Evaluation:

```powershell
.\.venv311\Scripts\python.exe -m post_tonal.evaluate `
  --config configs/post_tonal_main.yaml `
  --checkpoint runs/v3/proposed_constraint_guided_transformer/checkpoint.pt `
  --split test `
  --experiment-name proposed_constraint_guided_transformer `
  --output results/project2_v3_proposed_constraint_guided_transformer_metrics.json `
  --metrics-csv results/project2_v3_metrics.csv `
  --constraints-csv results/project2_v3_constraints.csv `
  --examples-output results/project2_v3_generation_examples.json
```

## Generate a MusicXML Sketch

```powershell
$env:PYTHONPATH = "src"
.\.venv311\Scripts\python.exe -m post_tonal.generate `
  --generator transformer `
  --config configs/post_tonal_main.yaml `
  --checkpoint runs/v3/proposed_constraint_guided_transformer/checkpoint.pt `
  --pcset 0,1,4,6 `
  --rhythm_profile pointillistic `
  --gesture fragmented `
  --voices 4 `
  --measures 8 `
  --attempts 4 `
  --output generated_scores/example.musicxml `
  --report results/example_report.json
```

Use a row and row form instead of a small pc-set when requesting serial material.

## Corrected Output Paths

Before promotion:

- `results/project2_v3_full_split_summary.json`
- `results/project2_v3_metrics.csv`
- `results/project2_v3_constraints.csv`
- `results/project2_v3_generation_examples.json`
- `results/project2_v3_full_run_report.md`
- `paper/tables/project2_v3_main_results.tex`
- `paper/tables/project2_v3_ablation_results.tex`
- `runs/v3/`
- `expert_eval/project2_v3/`

After every completion gate passes, the runner copies these to the required canonical `project2_*`, `runs/<experiment>/`, and `expert_eval/project2/` paths.

## Metrics

Evaluation reports body-token accuracy and cross-entropy, target pc-set coverage and precision, interval-vector distance, cyclic row-order accuracy, strict complete-form accuracy, serial aggregate completion, rhythmic-profile distance, held-out density-curve error, gesture consistency, instrument-range violations, content-span adherence, requested-voice adherence, and MusicXML structural/span/voice success.

The rule reference generates a new deterministic realization for each test condition. It does not reuse stored target events. Constraint-guided decoding uses non-differentiable metrics only for candidate reranking, not as fabricated differentiable training losses.

## Limitations

The synthetic corpus measures controllability and reproducibility rather than stylistic authenticity. The gesture metric is heuristic, the event vocabulary omits microtonality and advanced engraving detail, one instrument label is repeated across the requested parts, and automatic constraint metrics are not substitutes for composer judgment. Blind expert ratings, legally supplied external MusicXML validation, author metadata, and an archival release remain external tasks.
