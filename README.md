# Auditable Neural-Symbolic Post-Tonal Score Generation

[![Tests](https://github.com/niuyupeng/post-tonal-composition-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/niuyupeng/post-tonal-composition-assistant/actions/workflows/tests.yml)
[![Python 3.10--3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-3776AB.svg)](https://www.python.org/)

Research code and reproducibility materials for:

> **An auditable neural-symbolic framework enables verifiable control of post-tonal score generation**

Project title: *Neural-Symbolic Post-Tonal Composition Assistant with Pitch-Class Set, Serial, and Rhythmic-Profile Constraints*

Chinese title: 融合音级集合、序列与节奏轮廓约束的神经符号后调性作曲辅助方法

This repository implements score-level contemporary art-music composition assistance. A conditional Transformer and a deterministic rule reference operate on pitch-class sets, interval vectors, twelve-tone rows and P/R/I/RI forms, rhythmic profiles, gesture labels, instrumentation, voice count, and requested score span. The system exports editable MusicXML scores and machine-readable post-tonal analysis reports. It does not generate audio, pop accompaniment, or performance MIDI.

## Repository Status

The canonical full experiment is complete. Its fixed procedural corpus contains 20,000 training, 2,000 validation, and 2,000 test fragments (`smoke=false`, seed 42). The archived evaluation contains 13 rows over the same 2,000-condition test split, and the full-run gate reports 260/260 successful MusicXML parse, measure-count, and part-count checks.

These are automatic controllability and structural-validity results on a synthetic distribution. No human artistic-quality claim or transfer claim is made. Blind expert ratings and independent legally supplied MusicXML validation remain future work.

Primary evidence:

- [`results/project2_metrics.csv`](results/project2_metrics.csv): canonical aggregate metrics.
- [`results/project2_constraints.csv`](results/project2_constraints.csv): constraint-specific metrics.
- [`results/project2_generation_examples.json`](results/project2_generation_examples.json): per-example evaluation records.
- [`results/project2_full_split_summary.json`](results/project2_full_split_summary.json): split counts, seed, format, and corpus hashes.
- [`results/project2_full_run_report.md`](results/project2_full_run_report.md): environment, commands, checkpoint hashes, incidents, and completion gates.
- [`expert_eval/project2/`](expert_eval/project2/): 20 MusicXML scores, paired reports, a manifest, and blank rating forms.

See [`docs/RESULTS_PROVENANCE.md`](docs/RESULTS_PROVENANCE.md) before citing any numerical result.

## Legal Data Strategy

No copyrighted post-1945 score is scraped, bundled, or required. The default corpus is generated from explicit, seeded symbolic rules. Serial and non-serial targets are separated: serial samples use a twelve-tone row and transformed form, while non-serial samples use a pitch-class set and interval vector. Optional external MusicXML must be supplied by a user who has the right to use it.

The 24,000-sample processed tensor and neural checkpoints are reproducible derived artifacts and are intentionally excluded from Git because of their size. Their paths and SHA-256 hashes are recorded in the canonical split and run reports. The current peer-review data-access statement is documented in [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md); this private repository must not be described as a public archive.

## Installation

Target platform: Windows 11, Python 3.10 or 3.11, PyTorch, and `music21`. The formal run used Python 3.11.9, PyTorch 2.5.1+cu121, CUDA, and an NVIDIA GeForce RTX 4060 Ti 16 GB.

```powershell
git clone https://github.com/niuyupeng/post-tonal-composition-assistant.git
cd post-tonal-composition-assistant
.\scripts\setup_windows.ps1
.\.venv311\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

The broad dependency ranges are in [`requirements.txt`](requirements.txt). [`requirements-cuda121.txt`](requirements-cuda121.txt) documents the PyTorch build used by the archived run; the run report is authoritative for the recorded Python, PyTorch, and hardware versions.

## CPU Verification

Run the test suite:

```powershell
$env:PYTHONPATH = "src"
python -m pytest
```

Run the isolated CPU smoke workflow:

```powershell
.\scripts\smoke_project2.ps1
```

Smoke artifacts are written below `results/smoke_v3/`, `generated_scores/smoke_v3/`, and `runs/smoke_v3/`; they are never promoted to canonical result paths.

## Full CUDA Reproduction

On the RTX 4060 Ti machine:

```powershell
.\scripts\run_project2_full_local.ps1 -Resume
```

The runner validates Python, CUDA, tests, corpus membership, checkpoint completeness, evaluation rows, tables, and MusicXML exports. It trains or resumes the shared conditional Transformer and condition-removal/focused-condition configurations, evaluates the rule reference and decoding variants, and promotes outputs only after every completion gate passes.

Use `-NoPromote` when auditing v3-prefixed staging artifacts:

```powershell
.\scripts\run_project2_full_local.ps1 -Resume -NoPromote
```

Detailed commands, artifact policy, and verification checks are in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

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

For serial material, provide a twelve-tone row and a form such as `P0`, `R7`, `I5`, or `RI0` instead of a small pitch-class set.

## Metrics

Evaluation reports body-token accuracy and cross-entropy, target pc-set coverage and precision, interval-vector distance, cyclic row-order accuracy, strict serial-form accuracy, aggregate completion, rhythmic-profile distance, held-out density-curve error, gesture consistency, range violations, content-span adherence, voice-count adherence, and MusicXML structural/span/voice success.

Constraint metrics are non-differentiable diagnostics used for evaluation and candidate selection. They are not presented as differentiable training losses. The rule reference creates an independent deterministic realization for each condition and does not replay stored targets.

## Project Layout

```text
configs/                full, ablation, baseline, and smoke configurations
data/                   corpus documentation; generated tensors are ignored
docs/                   data availability and reproducibility records
expert_eval/project2/   20 blinded MusicXML examples and paired reports
paper/                  LaTeX companion source, figures, and generated tables
results/                canonical aggregate evidence and provenance reports
scripts/                Windows setup, smoke, training, and full-run wrappers
src/post_tonal/          data, theory, model, training, evaluation, and export code
tests/                   CPU-safe automated tests
```

## Limitations

The procedural corpus measures controllability and reproducibility, not stylistic authenticity. Gesture scoring is heuristic. The event vocabulary omits microtonality and advanced engraving detail. Instrument labels are simplified. Exact serial-form realization remains unresolved for the neural models, and structural measure-count success can include padded silence. Automatic metrics do not substitute for composer judgment.

## Citation and License

Use [`CITATION.cff`](CITATION.cff) for software citation metadata. The repository remains private during peer review, and no public reuse license has yet been granted; see [`LICENSE`](LICENSE). Before a public release, all authors should approve code and generated-data licenses and update the repository metadata consistently. Third-party dependencies retain their own licenses.
