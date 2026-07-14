# Claim-Evidence Ledger

| ID | Claim | Evidence | Status | Allowed use |
|---|---|---|---|---|
| C1 | The full corpus contains 20,000 training, 2,000 validation, and 2,000 test fragments with smoke=false. | `results/project2_full_split_summary.json` | supported | Exact factual statement |
| C2 | The primary checkpoint was trained with CUDA on an RTX 4060 Ti using Python 3.11.9 and PyTorch 2.5.1+cu121. | `results/project2_full_run_report.md`; `runs/proposed_constraint_guided_transformer/train_summary.json` | supported | Exact factual statement |
| C3 | The primary run stopped after 25 epochs and reached its lowest validation loss at epoch 15. | `runs/proposed_constraint_guided_transformer/train_summary.json` | supported | Report exact loss only from JSON |
| C4 | Conditions are serialized as prefix tokens for a six-layer causal Transformer. | `src/post_tonal/data/score_tokenizer.py`; `src/post_tonal/models/transformer.py` | supported | Implementation description |
| C5 | Guidance is non-differentiable K-candidate symbolic reranking at inference. | `src/post_tonal/generate.py`; `src/post_tonal/evaluate.py` | supported | Method description |
| C6 | Under 2,000 paired conditions, K=4 reranking improves row-order, interval-vector, rhythm, density, gesture, and range metrics but reduces serial aggregate completion and serial pc-set coverage. | `results/project2_controlled_statistics.json`; `results/project2_controlled_statistics.csv` | supported with qualifications | Report endpoint-specific means and paired CIs; no blanket superiority claim |
| C7 | All evaluated outputs are structurally parseable MusicXML. | Controlled metrics and `expert_eval/project2/manifest.json` | supported | Structural validity only; not requested-length adherence |
| C8 | The system is useful to contemporary composers. | Blind expert ratings | gap | `PENDING_REAL_EXPERIMENT`; do not infer from automatic metrics |
| C9 | Findings are stable across training seeds. | Multi-seed runs | gap | `PENDING_REAL_EXPERIMENT`; disclose single-seed limitation |
| C10 | The model reproduces authentic post-1945 style. | Copyright-safe external validation | rejected | Outside the present study and not claimed |

## Metric Qualifications

- Pitch-class-set coverage is recall-like and does not penalize generated pitch classes outside the target set.
- Serial transformation accuracy aliases cyclic row-order accuracy and is not an independent endpoint.
- Token accuracy includes condition-prefix positions.
- Empty-target metrics are not applicable, even where legacy code stores a numeric default.
- Rhythm and gesture removal configurations use fixed default labels rather than deleting the token categories.
- Serial samples retain a small pc-set target while cycling through a twelve-tone row, so interval-vector distance and aggregate completion can encode competing objectives.
- Aggregate completion is weighted for every reranked candidate, including non-serial samples with small pc-set targets; this can reward off-set pitch classes.
- The gesture feature named `rest_ratio` sums rest durations across voices and divides by fragment span, so it can exceed one in polyphonic material.
- Reranking and evaluation reuse the same diagnostics; controlled differences do not independently validate artistic quality.
- Rhythmic distance compares against a fixed seed-1234 profile realization, not the exact stochastic corpus rhythm.
- One instrument label is repeated across all parts, and event decoding is not grammar constrained.
- Voice-count adherence is not measured, and events with voice IDs beyond the requested part count can be omitted during MusicXML export.
- Generated measure counts may differ from the requested 4--16-measure condition.
- Long gaps can be compressed because decoded `BAR` tokens do not advance time and `TIME_SHIFT` is capped at 16 beats; the archived music21 exporter also omits trailing empty measures.
