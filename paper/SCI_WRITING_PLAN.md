# Project 2 SCI Writing and Submission Plan

## Evidence status

The repository is in the corrected v3 experiment phase. The full synthetic split
has been generated as 20,000 training, 2,000 validation, and 2,000 test samples,
with `smoke=false`. This split is an implementation artifact, not evidence that
neural training or final evaluation has completed.

No numerical manuscript claim may be treated as final until it can be traced to
the corrected v3 CSV or JSON outputs produced by the current run. Legacy v2
metrics, tables, figures, PDFs, and controlled-decoding outputs are retained only
for provenance and must not be cited as current evidence.

## Stage 1: Corrected experiment

1. Pass the complete test suite under Python 3.10 or 3.11.
2. Verify CUDA PyTorch and the RTX 4060 Ti device.
3. Train the proposed seed-42 model to early stopping or the 60-epoch limit.
4. Train the remaining required baseline and ablation configurations when GPU
   scheduling permits.
5. Evaluate every completed checkpoint on the fixed 2,000-sample test split.
6. Generate at least 20 structurally valid MusicXML examples from the corrected
   proposed checkpoint.
7. Build v3 tables and figures only from corrected result files.

Evidence gate: checkpoints, training summaries, metric rows, exact split
metadata, and MusicXML structural checks must all agree.

## Stage 2: Manuscript integration

1. Replace every `PENDING_REAL_EXPERIMENT` value only with a value read from a
   corrected v3 CSV or JSON file.
2. Report the model-selection protocol, early stopping, hardware, software
   versions, and any OOM adjustment exactly as recorded by the run.
3. Keep automatic constraint adherence separate from artistic usefulness.
4. Do not claim composer preference, perceptual validity, or practical utility
   before a properly approved expert study is completed.
5. Preserve the distinction between unordered pc-set constraints and ordered
   serial constraints throughout the analysis.

## Stage 3: Submission package

1. Rebuild identified and anonymous PDFs after corrected results are integrated.
2. Audit every numerical sentence against the v3 evidence ledger.
3. Check references, figures, tables, page rendering, and MusicXML examples.
4. Confirm author, funding, conflict-of-interest, AI-assistance, and repository
   release statements.
5. Commit and publish only after the corrected experiment and manuscript gates
   pass.

## Current execution board

| Workstream | Current status | Completion gate |
|---|---|---|
| Legal synthetic corpus | generated | v3 split summary records 20,000/2,000/2,000 and `smoke=false` |
| Rule baseline | corrected; final v3 evaluation pending | independent rule generation, not target replay |
| Proposed neural model, seed 42 | pending formal run | completed training summary and best checkpoint |
| Remaining baselines and ablations | pending GPU scheduling | all required checkpoints and metric rows |
| Automatic result tables | pending | generated only from corrected v3 CSV files |
| MusicXML expert package | smoke package exists; full package pending | at least 20 corrected full-run examples and reports |
| Manuscript numerical results | pending | every value traces to corrected v3 evidence |
| Expert ratings | not started | approved protocol and collected ratings |
| Submission package | draft only | corrected experiment, PDF QA, and author confirmations |

## Immediate order

1. Finish code, configuration, and documentation regression checks.
2. Run the corrected rule baseline and isolated CPU smoke workflow.
3. Start the proposed seed-42 CUDA training run.
4. Record the best epoch, checkpoint hash, RAM/VRAM peaks, and stop reason.
5. Continue the remaining experiment matrix only under an authorized GPU slot.
6. Generate v3 tables, examples, and the full-run report after all required
   experiments pass their gates.
7. Integrate final numbers into the manuscript and perform submission QA.

The paper remains a work in progress until these gates are satisfied. Missing
quantitative evidence must remain `PENDING_REAL_EXPERIMENT`.
