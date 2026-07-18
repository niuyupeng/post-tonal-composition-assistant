# Figure Contract: Full-Run Automatic Constraint Effects

## Claim

On the fixed 2,000-condition synthetic test split, increasing the candidate
budget from one to four and applying symbolic reranking moves each plotted
automatic endpoint in the preferred direction, while removing a condition field
from a separately trained K=1 decoder degrades its matching endpoint.

## Evidence hierarchy

1. Canonical source: `results/project2_metrics.csv`.
2. Required rows: proposed K=4, shared-generator K=1, and the four matching
   condition-removal models.
3. Every plotted row must have `split=test` and `num_samples=2000`.

## Panel map

- Panel a: K=1 versus K=4 from one shared trained generator.
- Panel b: full-prefix K=1 model versus separately trained K=1
  condition-removal models.
- Metrics: pc-set coverage, interval-vector distance, row-order accuracy,
  rhythmic-profile distance, and heuristic gesture consistency.

## Main and supplementary boundary

The main figure shows only the endpoints directly tied to reranking and
condition removal. Token accuracy, aggregate completion, strict serial-form
accuracy, held-out density error, range checks, span, voice adherence, and
MusicXML checks remain in the result tables and prose.

## Excluded evidence

Legacy v2 figures, smoke metrics, historical multi-seed experiments, and the
rule reference are not plotted. The rule reference implements constraints
procedurally and is reported separately rather than treated as a learned model.

## Review risks

- The endpoints are automatic diagnostics, not perceptual or artistic ratings.
- Most plotted endpoints also contribute to candidate selection.
- The condition-removal comparisons are descriptive single-seed contrasts.
- Strict serial-form accuracy is zero for the neural configurations and must be
  reported explicitly in the text and tables.

## Reproducible outputs

- Build script: `paper/figures/make_project2_full_results.py`
- Saved source data: `paper/figures/project2_full_results_source.csv`
- Vector files: `paper/figures/project2_full_results.pdf` and `.svg`
- Review raster: `paper/figures/project2_full_results.png`
