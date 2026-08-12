# Results Provenance

## Canonical Evidence Family

Only the following files are the final full-run evidence used for the current
automatic-results claims:

| Artifact | Role |
|---|---|
| `results/project2_metrics.csv` | 13 aggregate evaluation rows |
| `results/project2_constraints.csv` | constraint-specific aggregate values |
| `results/project2_generation_examples.json` | per-example generation/export records |
| `results/project2_full_split_summary.json` | split, seed, format, and corpus hashes |
| `results/project2_full_run_report.md` | commands, environment, checkpoints, incidents, and gates |
| `results/project2_full_run_incidents.json` | recovered runtime incident record |
| `expert_eval/project2/manifest.json` | mapping for 20 proposed-model examples |

The canonical split is 20,000 training, 2,000 validation, and 2,000 test
fragments with seed 42 and `smoke=false`. All numerical claims must be derived
from the files above. Empty cells are not zeros. Metrics are averaged only when
their target is applicable.

## Interpretation Boundaries

- `proposed_constraint_guided_transformer` and `vanilla_transformer` share one
  trained checkpoint. Their comparison changes candidate count and symbolic
  selection together; it is not an isolated model-architecture comparison.
- Strict serial-transformation accuracy is zero for every neural row. High
  aggregate completion does not imply correct P/R/I/RI order.
- MusicXML measure adherence includes explicit padding. It does not imply that
  generated musical content fills the requested span.
- The rule row is a deterministic construction reference, not a learned model.
- Automatic metrics do not establish artistic quality, usefulness, stylistic
  authenticity, or transfer to human-authored contemporary repertoire.

## Non-Canonical Artifacts

Smoke runs, v3 staging copies, exploratory controlled-decoding analyses, and
historical multi-seed development outputs are not substituted for canonical
full-run values. High-volume per-sample development JSON files are excluded
from Git; aggregate records may be retained only as clearly labelled audit
material.

## Regeneration Rule

Tables and figures must be regenerated from the canonical files using the
checked-in scripts. If an experiment is rerun, write to staging paths first and
promote only after every corpus, checkpoint, metric-row, table, and MusicXML
gate passes. Never overwrite the canonical family with partial, smoke, or
manually constructed results.
