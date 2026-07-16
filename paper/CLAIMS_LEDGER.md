# Project 2 Corrected Claim Ledger

## Supported now

| Claim | Evidence | Allowed wording |
|---|---|---|
| The corrected procedural split is 20,000/2,000/2,000, seed 42, with `smoke=false`. | `results/project2_v3_full_split_summary.json` | Exact corpus and split fact. |
| The active environment is Python 3.11.9, PyTorch 2.5.1+cu121, CUDA available, RTX 4060 Ti. | Fresh environment command; final run report after completion. | Environment fact, not proof of completed training. |
| The corrected pipeline uses condition-preserving score-body windows and discards no saved body tokens over a coverage cycle. | `src/post_tonal/data/post_tonal_dataset.py`, split summary, tests. | Method statement. |
| Serial and non-serial pitch targets are mutually exclusive in corrected corpus generation. | `src/post_tonal/data/generate_corpus.py`. | Method statement. |
| Held-out stochastic density curves are removed from visible condition metadata and excluded from candidate loss. | `src/post_tonal/data/conditions.py`, `src/post_tonal/generate.py`, tests. | Anti-leakage statement. |
| The rule reference generates an independent deterministic realization for each condition. | `src/post_tonal/evaluate.py`, `configs/post_tonal_rule_baseline.yaml`, tests. | Procedural reference statement; do not call it an oracle target replay. |
| Export checks distinguish structural validity, measure adherence, and voice adherence. | `src/post_tonal/evaluate.py`, `src/post_tonal/export_musicxml.py`, tests. | File-level verification statement. |

## Pending corrected evidence

The following claims are `PENDING_REAL_EXPERIMENT` until every v3 checkpoint, 2,000-sample metric row, table, and expert-package gate passes:

- corrected neural token accuracy and cross-entropy;
- proposed K=4 versus shared-generator K=1 constraint differences;
- condition-ablation comparisons;
- final strict serial-form, aggregate, rhythm, density, gesture, range, span, and voice metrics;
- 20 full-trained-model MusicXML examples;
- peak RAM/VRAM and best epoch for every completed neural run.

## Legacy evidence excluded

All v2 numerical tables, old controlled-decoding statistics, old three-seed results, and figures derived from those artifacts are historical development outputs. They do not support corrected v3 manuscript claims because the corrected protocol changes sequence coverage, target compatibility, metadata visibility, decoding grammar, metric definitions, and MusicXML adherence checks.

## External evidence not available

- blind composer or analyst ratings;
- validation on independent, legally supplied MusicXML;
- author, affiliation, funding, and contribution metadata;
- archival release DOI and live journal-portal compliance.

No numerical sentence may be inserted into Abstract, Results, Discussion, or Conclusion unless it maps to the corrected `project2_v3_*` artifacts or to canonical files promoted from them after the full completion gate.
