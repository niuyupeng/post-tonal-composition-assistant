# Project 2 Manuscript Context

## Scope

Score-level neural-symbolic assistance for post-tonal art-music sketches. Inputs include pc sets and interval vectors or twelve-tone rows and transformations, plus rhythm, gesture, instrumentation, voices, and measures. Outputs are event tokens, MusicXML, and analysis reports. Audio generation and copyrighted-score imitation are outside scope.

## Formal design

- One legal procedural corpus with an explicit 20,000/2,000/2,000 split and seed 42.
- Serial and non-serial pitch targets are mutually exclusive.
- Complete condition prefixes are repeated across score-body windows.
- Coverage cycles expose every saved body token; validation and testing enumerate all windows.
- Proposed K=4 and vanilla K=1 use one checkpoint; the contrast changes candidate budget and symbolic scoring together.
- Condition ablations are separately trained K=1 models that hide one prefix field while retaining original targets.
- Held-out stochastic density targets are unavailable to the model and selector.
- The rule reference creates independent deterministic outputs rather than replaying targets.
- Export checks distinguish XML parse, requested measures, requested parts, and generated voice content.

## Evidence status

- Environment: Python 3.11.9, PyTorch 2.5.1+cu121, CUDA, RTX 4060 Ti 16GB.
- Corpus: 20,000/2,000/2,000, seed 42, `smoke=false`.
- Formal matrix: 13/13 test rows, 2,000 conditions per row.
- Neural artifacts: 12 checkpoint paths representing 11 distinct optimization runs; every row reached 60 epochs.
- Resources: peak process RAM 4.637 GiB; peak allocated VRAM 0.667 GiB; no OOM adjustment.
- Exports: 260/260 XML parse, measure, and part checks passed; proposed expert package contains 20/20 files and reports.
- Code verification: 60 tests passed after the final manuscript table-generator edit.
- Manuscript QA: identified and anonymous PDFs each contain 21 inspected pages; both compile without warnings, and the anonymous text audit passes.
- Human expert ratings and independent legal MusicXML validation: not conducted.

Legacy v2 metrics and historical multi-seed development runs are archived and excluded from formal claims.

## Paper position

The contribution is the integration of legal synthetic data, explicit post-tonal conditions, locally trainable symbolic generation, guided candidate selection, explainable diagnostics, and MusicXML output. The Transformer architecture itself is standard. Claims are limited to descriptive automatic controllability on the synthetic testbed.
