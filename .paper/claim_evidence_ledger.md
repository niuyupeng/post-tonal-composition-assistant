# Claim-Evidence Ledger

| ID | Claim | Evidence | Status | Allowed use |
|---|---|---|---|---|
| C1 | The full corpus contains 20,000 training, 2,000 validation, and 2,000 test fragments with smoke=false. | `results/project2_full_split_summary.json` | supported | Exact factual statement |
| C2 | The primary checkpoint was trained with CUDA on an RTX 4060 Ti using Python 3.11.9 and PyTorch 2.5.1+cu121. | `results/project2_full_run_report.md`; `runs/proposed_constraint_guided_transformer/train_summary.json` | supported | Exact factual statement |
| C3 | The primary run stopped after 25 epochs and reached its lowest validation loss at epoch 15. | `runs/proposed_constraint_guided_transformer/train_summary.json` | supported | Report exact loss only from JSON |
| C4 | Conditions are serialized as prefix tokens for a six-layer causal Transformer. | `src/post_tonal/data/score_tokenizer.py`; `src/post_tonal/models/transformer.py` | supported | Implementation description |
| C5 | Guidance is non-differentiable K-candidate symbolic reranking at inference. | `src/post_tonal/generate.py`; `src/post_tonal/evaluate.py` | supported | Method description |
| C6 | At the primary checkpoint, K=4 reranking changes row-order, interval-vector, rhythm, density, gesture, range, aggregate-completion, and pc-set diagnostics under 2,000 paired conditions. | `results/project2_controlled_statistics.json`; `results/project2_controlled_statistics.csv` | supported with qualifications | Label as primary-checkpoint evidence; report endpoint-specific means and paired intervals |
| C7 | All evaluated outputs are structurally parseable MusicXML. | Controlled metrics and `expert_eval/project2/manifest.json` | supported | Structural validity only; not requested-length adherence |
| C8 | The system is useful to contemporary composers. | Blind expert ratings | gap | `PENDING_REAL_EXPERIMENT`; do not infer from automatic metrics |
| C9 | Across checkpoints trained with seeds 42, 43, and 44, K=4 versus K=1 effects recur for some symbolic endpoints, reverse for others, and cross zero for two endpoints on the fixed 2,000-condition test split. | `results/project2_multiseed_controlled_statistics.json`; `results/project2_multiseed_controlled_statistics.csv` | supported with qualifications | Report exact endpoint, subset, mean effect, crossed interval, and seed-direction count; do not replace this mixed pattern with a blanket stability claim |
| C10 | The model reproduces authentic post-1945 style. | Copyright-safe external validation | rejected | Outside the present study and not claimed |
| C11 | The expert package contains 20 anonymous controlled-reranking MusicXML examples with paired condition metadata and withheld automatic reports. | `expert_eval/project2/manifest.json`; package structural audit | supported | Package-preparation statement only; no human-evaluation claim |
| C12 | Three memory-safe proposed-model runs completed and produced aligned 2,000-item teacher-forced test diagnostics. | `results/project2_multiseed_training_metrics.csv`; `results/project2_multiseed_training_summary.json` | supported with qualifications | Report loss and token-accuracy mean/SD only; do not generalize to constraint decoding |
| C13 | Constraint-control findings transfer to independent, legally supplied MusicXML material. | External MusicXML validation set | gap | `PENDING_REAL_EXPERIMENT`; synthetic held-out conditions and structural export checks do not establish external validity |

## Three-Seed Controlled Endpoint Ledger

The effect column follows the orientation stored in the aggregate artifact. Positive values favor K=4 reranking except for non-serial aggregate completion, whose value is the raw reranked-minus-single diagnostic with no preferred direction. Intervals are 95% crossed percentile-bootstrap intervals over three training checkpoints and their shared aligned conditions.

| Endpoint | Conditions per checkpoint | Mean effect | 95% crossed interval | Seeds favoring K=4 | Allowed interpretation |
|---|---:|---:|---:|---:|---|
| Pc-set coverage, all | 2,000 | -0.000717 | [-0.002694, 0.001170] | 1/3 | Interval crosses zero; no directional aggregate claim |
| Pc-set coverage, non-serial | 1,086 | 0.004256 | [0.002307, 0.006430] | 3/3 | Endpoint-specific favorable effect |
| Pc-set coverage, serial | 914 | -0.006625 | [-0.010680, -0.003294] | 0/3 | Endpoint-specific unfavorable effect |
| Interval-vector distance, all | 2,000 | 0.417667 | [0.295000, 0.561175] | 3/3 | Endpoint-specific favorable effect |
| Interval-vector distance, non-serial | 1,086 | 0.204727 | [0.137201, 0.277778] | 3/3 | Endpoint-specific favorable effect |
| Interval-vector distance, serial | 914 | 0.670678 | [0.423049, 0.959528] | 3/3 | Endpoint-specific favorable effect |
| Row-order accuracy, serial | 914 | 0.081514 | [0.075605, 0.087475] | 3/3 | Endpoint-specific favorable effect |
| Aggregate completion, all | 2,000 | -0.003139 | [-0.004722, -0.001764] | 0/3 | Endpoint-specific unfavorable effect |
| Aggregate completion, serial | 914 | -0.006595 | [-0.009664, -0.003981] | 0/3 | Endpoint-specific unfavorable effect |
| Aggregate completion, non-serial diagnostic | 1,086 | -0.000230 | [-0.001330, 0.000895] | -- | Raw difference; interval crosses zero |
| Rhythmic-profile distance, all | 2,000 | 0.050689 | [0.043874, 0.058302] | 3/3 | Endpoint-specific favorable effect |
| Density-curve error, all | 2,000 | 0.227813 | [0.202112, 0.254461] | 3/3 | Endpoint-specific favorable effect |
| Gesture consistency, all | 2,000 | 0.031770 | [0.028255, 0.035309] | 3/3 | Endpoint-specific favorable effect |
| Range-violation rate, all | 2,000 | 0.000344 | [0.000181, 0.000557] | 3/3 | Endpoint-specific favorable effect |

Protocol source: `results/project2_multiseed_controlled_statistics.json`. The aggregate uses 10,000 bootstrap resamples with bootstrap seed 52042. First-candidate alignment was verified by SHA-256 for every seed-condition pair.

## Metric Qualifications

- The three-checkpoint controlled comparison uses seeds 42, 43, and 44 on one fixed synthetic corpus and the same 2,000-condition test split. It does not establish cross-corpus or external generalization.
- Crossed percentile-bootstrap intervals account for training-checkpoint and aligned-condition variation, but only three checkpoints are available.
- Fourteen endpoint intervals are reported without multiplicity correction. Interpret them as endpoint-specific estimates rather than a family-wise inferential claim.
- Pitch-class-set coverage is recall-like and does not penalize generated pitch classes outside the target set.
- Serial transformation accuracy aliases cyclic row-order accuracy and is not an independent endpoint.
- Token accuracy includes condition-prefix positions.
- Empty-target metrics are not applicable, even where legacy code stores a numeric default.
- Rhythm and gesture removal configurations use fixed default labels rather than deleting the token categories.
- Serial samples retain a small pc-set target while cycling through a twelve-tone row, so interval-vector distance and aggregate completion can encode competing objectives.
- Aggregate completion is weighted for every reranked candidate, including non-serial samples with small pc-set targets; this can reward off-set pitch classes.
- The gesture feature named `rest_ratio` sums rest durations across voices and divides by fragment span, so it can exceed one in polyphonic material.
- Reranking and evaluation reuse overlapping diagnostics; controlled differences are partly optimization-aligned and do not independently validate artistic quality.
- Rhythmic distance compares against a fixed seed-1234 profile realization, not the exact stochastic corpus rhythm.
- One instrument label is repeated across all parts, and event decoding is not grammar constrained.
- Voice-count adherence is not measured, and events with voice IDs beyond the requested part count can be omitted during MusicXML export.
- Generated measure counts may differ from the requested 4--16-measure condition.
- Long gaps can be compressed because decoded `BAR` tokens do not advance time and `TIME_SHIFT` is capped at 16 beats; the archived music21 exporter also omits trailing empty measures.
