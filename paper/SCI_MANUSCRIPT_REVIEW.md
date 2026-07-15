# Project 2 SCI Manuscript Review

## Editorial verdict

The repository now supports a defensible full manuscript draft centered on reproducible, score-level post-tonal constraint control. The strongest contribution is not a new Transformer architecture. It is the integration of a legal synthetic corpus, condition-prefix generation, explicit post-tonal diagnostics, inference-time candidate reranking, MusicXML export, and per-score analysis in one auditable workflow.

The primary quantitative evidence is the controlled comparison between single-candidate decoding and four-candidate reranking from the same checkpoint on the same 2,000 test requests. The older 13-row configuration archive remains exploratory because its models use different generated corpora and seeds. The paper can be written now as a bounded proof of concept, but claims about artistic value still require human evidence.

## Major review findings

### Findings that bound the central claim

1. Reranking and evaluation use the same symbolic diagnostics. The controlled result demonstrates optimization of the implemented objective, not independent musical validity.
2. The effect is a trade-off, not uniform superiority. Row-order, interval-vector, rhythm, density, gesture, and range metrics improve, while serial aggregate completion and serial pc-set coverage decrease.
3. Pitch-class-set coverage is recall-like and near ceiling. It does not penalize generated pitch classes outside the target set and cannot establish set purity.
4. Serial transformation accuracy aliases cyclic row-order accuracy and must not be presented as an independent endpoint.
5. Token accuracy includes condition-prefix tokens and is a sequence-model diagnostic, not a compositional-quality measure.
6. The rule generator creates the synthetic targets and is therefore a procedural reference, not an independent learned baseline.
7. Endpoint-wise paired bootstrap intervals are available, but no multiple-endpoint adjustment is applied. Three training seeds quantify variation in teacher-forced sequence diagnostics only.
8. The controlled comparison isolates the inference procedure within one checkpoint and test set; the reranking effect has not been repeated across trained checkpoints.

### Data and representation limitations

9. General serial samples combine an aggregate-spanning row with a smaller pc-set and interval-vector target, creating an explicit objective conflict.
10. Aggregate completion is weighted for non-serial samples even though those requests have no aggregate target; the non-serial value is diagnostic only.
11. Rhythmic distance uses a deterministic seed-1234 profile template rather than the exact stochastic rhythm used to create each corpus target.
12. The heuristic gesture rest-ratio feature can exceed one in polyphony because rest durations are summed across voices.
13. The decoder is not grammar constrained. Voice-count adherence is not scored, and events assigned to voice IDs outside the requested part count can be omitted during export.
14. Requested and written measure counts can differ because `BAR` does not advance decoded time, `TIME_SHIFT` is capped, generation can terminate early, and trailing empty measures are not padded.
15. One instrument class is repeated across all parts; mixed-ensemble orchestration is outside the present evidence.

### Evidence still missing for stronger claims

16. Blind ratings from composers or post-tonal analysts have not been collected. The 20-example controlled-reranking package is anonymous and condition-aware, but usefulness, gesture recognizability, engraving quality, and material coherence remain `PENDING_REAL_EXPERIMENT`.
17. Three memory-safe proposed-model runs are complete, with 2,000-item teacher-forced test rows and sample-standard-deviation summaries. Cross-seed stability of the controlled reranking effect remains `PENDING_REAL_EXPERIMENT` because K=1/K=4 generation was not repeated for each checkpoint.
18. Validation on legally supplied external MusicXML is not yet available. The study must not claim stylistic authenticity to post-1945 repertoire.
19. Author, affiliation, contribution, conflict-of-interest, and data/code-release metadata still require completion before submission.
20. The current manuscript is journal-neutral. Final Journal of New Music Research formatting and declarations must be applied against the current author instructions.

## Resolved during the manuscript sprint

- The condition encoder is now described correctly as prefix tokens in one causal Transformer.
- The primary comparison now uses deterministic, aligned K=1 and K=4 outputs from the same checkpoint and 2,000 conditions.
- Per-condition statistics, serial/non-serial subgroups, win/tie/loss rates, and paired bootstrap intervals are saved.
- Archived cross-configuration rows are labeled exploratory rather than causal ablations.
- The method figure, training diagnostics, controlled-effect figure, and representative full-span score are integrated.
- Related work now covers computer-assisted composition, formal constraints, symbolic generation, and controllable generation with verified primary references.
- MusicXML provenance text is neutral for future neural exports, and the paper limits the parse check to structural validity.
- The expert package now draws from the controlled $K=4$ outputs, removes creator/date metadata, exposes target conditions to raters, and withholds automatic reports from rating materials.
- The Chinese title is valid UTF-8; earlier terminal mojibake was a display-decoding issue rather than a source-file defect.

## Recommended paper positioning

Frame the paper as a reproducible neural-symbolic proof of concept for explainable post-tonal score sketching. The main claim is that finite-candidate symbolic reranking changes explicit constraint-adherence diagnostics under a controlled same-checkpoint design, with measurable improvements and measurable conflicts. Do not claim a novel Transformer, state-of-the-art generation, contemporary-style imitation, or composer preference.

The Journal of New Music Research is the best disciplinary target for this bounded contribution. Applied Sciences remains a broader fallback if speed is prioritized over disciplinary fit. Submission without expert ratings is possible only if the abstract, title, and conclusion remain explicitly limited to automatic constraint behavior and reproducibility.

## Completion gate

The current draft has passed the post-replication numerical audit, the 21-test suite, and fresh page-level QA of both 21-page PDFs; both XeLaTeX builds are free of layout and reference warnings. An anonymous wrapper and submission drafts are prepared. Submission still requires author metadata, live-portal format verification, and a decision on whether to submit the bounded automatic study before expert ratings or cross-seed controlled decoding.
