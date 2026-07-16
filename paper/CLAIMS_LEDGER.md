# Project 2 Claim Ledger

## Measured and citable in the manuscript

| Claim | Evidence source | Permitted wording |
|---|---|---|
| Corpus split is 20,000/2,000/2,000 with smoke=false and seed 42 | `results/project2_full_split_summary.json` | Exact factual statement |
| Primary model trained on RTX 4060 Ti with Python 3.11.9 and PyTorch 2.5.1+cu121 | `results/project2_full_run_report.md`, checkpoint summary | Exact factual statement |
| Primary run stopped after 25 epochs; best validation loss 0.8111 at epoch 15 | `runs/proposed_constraint_guided_transformer/train_summary.json` | Exact factual statement |
| Thirteen archived aggregate rows exist and use 2,000 test samples | `results/project2_metrics.csv` | Exploratory cross-configuration description |
| Twenty expert-package MusicXML files parse structurally | `expert_eval/project2/manifest.json` and structural check | Structural-export statement only |
| Controlled K=1 and K=4 decoder metrics over 2,000 aligned conditions | `results/project2_controlled_metrics.csv` | Report exact aggregate means; checkpoint and conditions are fixed |
| Paired uncertainty/effect estimates for 14 prespecified endpoints | `results/project2_controlled_statistics.json`, `results/project2_controlled_statistics.csv` | Report paired percentile-bootstrap intervals and disclose no multiplicity adjustment |
| Three proposed-model seeds completed on one fixed corpus; each has a 2,000-item teacher-forced test row | `results/project2_multiseed_training_metrics.csv`, `results/project2_multiseed_training_summary.json` | Report validation/test sequence diagnostics as mean and sample SD; do not call them multi-seed decoding effects |
| Controlled K=1/K=4 decoding completed for checkpoints trained with seeds 42, 43, and 44 on the same 2,000-condition test split | `results/project2_multiseed_controlled_statistics.json`, `results/project2_multiseed_controlled_statistics.csv` | Report endpoint-specific mean effects, crossed percentile-bootstrap intervals, and seed-direction counts |
| The three-checkpoint aggregate contains 14 endpoints, uses 10,000 crossed bootstrap resamples, and applies no multiplicity correction | `results/project2_multiseed_controlled_statistics.json` | Exact protocol statement; disclose fixed corpus/test split and diagnostic overlap |

## Three-checkpoint controlled decoding evidence

Positive effects favor K=4 reranking except for the non-serial aggregate-completion diagnostic, which is stored as the raw reranked-minus-single difference. Each row is computed from checkpoints trained with seeds 42, 43, and 44. The full split contains 2,000 aligned conditions per checkpoint, with 1,086 non-serial and 914 serial conditions.

| Endpoint | Mean effect | 95% crossed interval | Positive seed count | Permitted wording |
|---|---:|---:|---:|---|
| Pc-set coverage, all | -0.000717 | [-0.002694, 0.001170] | 1/3 | Interval crosses zero; do not claim an aggregate direction |
| Pc-set coverage, non-serial | 0.004256 | [0.002307, 0.006430] | 3/3 | K=4 is associated with higher non-serial coverage on this test split |
| Pc-set coverage, serial | -0.006625 | [-0.010680, -0.003294] | 0/3 | K=4 is associated with lower serial coverage on this test split |
| Interval-vector distance, all | 0.417667 | [0.295000, 0.561175] | 3/3 | K=4 favors the implemented interval-vector diagnostic |
| Interval-vector distance, non-serial | 0.204727 | [0.137201, 0.277778] | 3/3 | K=4 favors the implemented non-serial interval-vector diagnostic |
| Interval-vector distance, serial | 0.670678 | [0.423049, 0.959528] | 3/3 | K=4 favors the implemented serial interval-vector diagnostic |
| Row-order accuracy, serial | 0.081514 | [0.075605, 0.087475] | 3/3 | K=4 is associated with higher serial row-order accuracy |
| Aggregate completion, all | -0.003139 | [-0.004722, -0.001764] | 0/3 | K=4 is associated with lower aggregate completion overall |
| Aggregate completion, serial | -0.006595 | [-0.009664, -0.003981] | 0/3 | K=4 is associated with lower serial aggregate completion |
| Aggregate completion, non-serial diagnostic | -0.000230 | [-0.001330, 0.000895] | -- | Raw difference; interval crosses zero |
| Rhythmic-profile distance, all | 0.050689 | [0.043874, 0.058302] | 3/3 | K=4 favors the implemented rhythmic-profile diagnostic |
| Density-curve error, all | 0.227813 | [0.202112, 0.254461] | 3/3 | K=4 favors the implemented density-curve diagnostic |
| Gesture consistency, all | 0.031770 | [0.028255, 0.035309] | 3/3 | K=4 is associated with higher implemented gesture consistency |
| Range-violation rate, all | 0.000344 | [0.000181, 0.000557] | 3/3 | K=4 favors the implemented range diagnostic |

These are endpoint-specific estimates from three checkpoints on one fixed synthetic test split. The intervals are crossed percentile-bootstrap intervals over training checkpoints and aligned conditions. They are not adjusted for the 14 reported endpoints, and the reranking objective reuses diagnostics that also appear in evaluation.

## Implementation facts

| Claim | Code source | Permitted wording |
|---|---|---|
| Conditions are prefix tokens | `src/post_tonal/data/score_tokenizer.py` | Condition-prefix Transformer |
| Generator is a six-layer causal Transformer | `src/post_tonal/models/transformer.py` | Standard causal Transformer; not a new architecture |
| Guidance is inference-time candidate reranking | `src/post_tonal/evaluate.py`, `src/post_tonal/generate.py` | Neural-symbolic constraint-guided decoding |
| Serial transformation accuracy aliases row-order accuracy | `src/post_tonal/theory/analysis_report.py` | Report one endpoint, disclose alias |
| Pc-set coverage returns 1.0 for an empty target | `src/post_tonal/theory/pcset.py` | Treat absent-target coverage as not applicable |
| Rhythm/gesture removal uses fixed defaults | `src/post_tonal/data/generate_corpus.py` | Fixed-default ablation, not token removal |
| General serial samples combine row and small pc-set conditions | `src/post_tonal/data/generate_corpus.py`, `src/post_tonal/models/rule_generator.py` | Disclose the interval-vector/aggregate tension and stratify results |
| Requested and realized measure counts can differ | expert-package MusicXML audit | Do not claim exact length adherence |
| Long gaps and trailing empty measures are not preserved exactly | `score_tokenizer.py`, `export_musicxml.py` | Treat length mismatch as a representation/export limitation, not solely a model failure |

## Interpretation allowed with caveats

- Automatic metrics quantify consistency with implemented symbolic targets on a synthetic corpus.
- Constraint reranking improves row-order, interval-vector, rhythm, density, gesture, and range diagnostics in the controlled comparison, while serial aggregate completion and serial pc-set coverage decrease.
- Across seeds 42, 43, and 44, the direction recurs for several endpoints but not all. Full-set pc-set coverage and non-serial aggregate completion have crossed intervals that include zero.
- Archived independent configurations may reveal trade-offs but cannot isolate causal effects because their seeds and generated datasets differ.
- Structural MusicXML validity supports editability at the file-format level, not engraving or compositional quality.
- Structural validity does not imply that the requested number of measures was realized.

## Claims not currently supported

- The system produces music preferred by composers or listeners.
- The system is stylistically authentic to post-1945 repertoire.
- The proposed architecture is novel or superior to state-of-the-art music Transformers.
- Automatic gesture consistency proves perceptual gesture recognizability.
- Three-checkpoint controlled decoding establishes uniform benefit across all endpoints, external corpora, or composer populations.
- Pc-set coverage alone proves exclusion of non-target pitch classes.
- The controlled findings transfer to independent, legally supplied MusicXML material.

Use `PENDING_REAL_EXPERIMENT` only for required results that have not been produced, chiefly blind expert ratings and validation on independent, legally supplied MusicXML examples. Cross-seed controlled decoding statistics are complete.
