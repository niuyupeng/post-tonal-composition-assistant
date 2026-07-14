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
- Archived independent configurations may reveal trade-offs but cannot isolate causal effects because their seeds and generated datasets differ.
- Structural MusicXML validity supports editability at the file-format level, not engraving or compositional quality.
- Structural validity does not imply that the requested number of measures was realized.

## Claims not currently supported

- The system produces music preferred by composers or listeners.
- The system is stylistically authentic to post-1945 repertoire.
- The proposed architecture is novel or superior to state-of-the-art music Transformers.
- Automatic gesture consistency proves perceptual gesture recognizability.
- Single-seed training establishes stable superiority across random initializations.
- Pc-set coverage alone proves exclusion of non-target pitch classes.

Use `PENDING_REAL_EXPERIMENT` only for required results that have not been produced, chiefly expert ratings and any planned multi-seed statistics.
