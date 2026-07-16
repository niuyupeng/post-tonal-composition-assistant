# Project 2 SCI Manuscript Review

## Editorial verdict

The repository now supports a defensible full manuscript draft centered on reproducible, score-level post-tonal constraint control. The contribution is not a new Transformer architecture. It is the integration of a legal synthetic corpus, condition-prefix generation, explicit post-tonal diagnostics, inference-time candidate reranking, MusicXML export, and per-score analysis in one auditable workflow.

The quantitative evidence now has two controlled levels. The existing primary-checkpoint table compares single-candidate decoding with four-candidate reranking on 2,000 aligned test requests. A separate replication repeats the same comparison for independently trained seed-42, seed-43, and seed-44 checkpoints, again using 2,000 shared requests per checkpoint. The older 13-row configuration archive remains exploratory because its models use different generated corpora and seeds. The paper can be written as a bounded proof of concept, but claims about artistic value still require human evidence.

## Major review findings

### Findings that bound the central claim

1. Reranking and evaluation use the same symbolic diagnostics. The controlled result measures optimization of the implemented objective, not independent musical validity.
2. The effect is a trade-off, not uniform superiority. Row-order, interval-vector, rhythm, density, gesture, and range metrics improve, while serial aggregate completion and serial pc-set coverage decrease.
3. Pitch-class-set coverage is recall-like and near ceiling. It does not penalize generated pitch classes outside the target set and cannot establish set purity.
4. Serial transformation accuracy aliases cyclic row-order accuracy and must not be presented as an independent endpoint.
5. Token accuracy includes condition-prefix tokens and is a sequence-model diagnostic, not a compositional-quality measure.
6. The rule generator creates the synthetic targets and is therefore a procedural reference, not an independent learned baseline.
7. Endpoint-wise paired intervals are available for the primary checkpoint, and crossed intervals are available for the three-checkpoint replication. Neither analysis applies a multiple-endpoint adjustment.
8. The repeated comparison supports an endpoint-specific inference, not uniform superiority. Across all three checkpoints, favorably oriented effects were positive for interval-vector distance (+0.4177, crossed 95% CI [+0.2950, +0.5612]), serial row-order accuracy (+0.0815, [+0.0756, +0.0875]), rhythmic-profile distance (+0.0507, [+0.0439, +0.0583]), density-curve error (+0.2278, [+0.2021, +0.2545]), gesture consistency (+0.0318, [+0.0283, +0.0353]), and range-violation rate (+0.0003, [+0.0002, +0.0006]).
9. The same three-checkpoint analysis records consistent conflicts. Serial pc-set coverage decreased by 0.0066 ([-0.0107, -0.0033]) and serial aggregate completion decreased by 0.0066 ([-0.0097, -0.0040]) under the favorable-effect convention. Overall pc-set coverage crossed zero, so the manuscript must not describe coverage as a general reranking gain.

### Data and representation limitations

10. General serial samples combine an aggregate-spanning row with a smaller pc-set and interval-vector target, creating an explicit objective conflict.
11. Aggregate completion is weighted for non-serial samples even though those requests have no aggregate target; the non-serial value is diagnostic only.
12. Rhythmic distance uses a deterministic seed-1234 profile template rather than the exact stochastic rhythm used to create each corpus target.
13. The heuristic gesture rest-ratio feature can exceed one in polyphony because rest durations are summed across voices.
14. The decoder is not grammar constrained. Voice-count adherence is not scored, and events assigned to voice IDs outside the requested part count can be omitted during export.
15. Requested and written measure counts can differ because `BAR` does not advance decoded time, `TIME_SHIFT` is capped, generation can terminate early, and trailing empty measures are not padded.
16. One instrument class is repeated across all parts; mixed-ensemble orchestration is outside the present evidence.

### Evidence still missing for stronger claims

17. Blind ratings from composers or post-tonal analysts have not been collected. The 20-example controlled-reranking package is anonymous and condition-aware, but usefulness, gesture recognizability, engraving quality, and material coherence remain `PENDING_REAL_EXPERIMENT`.
18. The controlled replication uses only three training checkpoints, one synthetic corpus, and one fixed test split. Crossed intervals estimate variation over those seeds and aligned conditions, not over repertoires, composers, or model families.
19. Validation on legally supplied external MusicXML is not yet available. The study must not claim stylistic authenticity to post-1945 repertoire.
20. Author, affiliation, contribution, conflict-of-interest, and data/code-release metadata still require completion before submission.
21. The current manuscript is journal-neutral. Final Journal of New Music Research formatting and declarations must be applied against the current author instructions.

## Resolved during the manuscript sprint

- The condition encoder is now described correctly as prefix tokens in one causal Transformer.
- The primary comparison now uses deterministic, aligned K=1 and K=4 outputs from the same checkpoint and 2,000 conditions.
- Per-condition statistics, serial/non-serial subgroups, win/tie/loss rates, and paired bootstrap intervals are saved.
- The aligned K=1/K=4 protocol has been repeated for training seeds 42, 43, and 44. All 6,000 seed-condition pairs pass the first-candidate SHA256 alignment gate, and the crossed analysis reports 14 endpoint effects with three seed means.
- Archived cross-configuration rows are labeled exploratory rather than causal ablations.
- The method figure, training diagnostics, controlled-effect figure, and representative full-span score are integrated.
- Related work now covers computer-assisted composition, formal constraints, symbolic generation, and controllable generation with verified primary references.
- MusicXML provenance text is neutral for future neural exports, and the paper limits the parse check to structural validity.
- The expert package now draws from the controlled $K=4$ outputs, removes creator/date metadata, exposes target conditions to raters, and withholds automatic reports from rating materials.
- The Chinese title is valid UTF-8; earlier terminal mojibake was a display-decoding issue rather than a source-file defect.

## Recommended paper positioning

Frame the paper as a reproducible neural-symbolic proof of concept for explainable post-tonal score sketching. The main claim is that finite-candidate symbolic reranking changes explicit constraint-adherence diagnostics under aligned same-checkpoint comparisons, and that several favorable and unfavorable directions recur across three independently trained checkpoints. Keep the primary-checkpoint table and the three-checkpoint replication separate. Do not claim a novel Transformer, state-of-the-art generation, contemporary-style imitation, composer preference, or unrestricted cross-seed stability.

The Journal of New Music Research is the best disciplinary target for this bounded contribution. Applied Sciences remains a broader fallback if speed is prioritized over disciplinary fit. Submission without expert ratings is possible only if the abstract, title, and conclusion remain explicitly limited to automatic constraint behavior and reproducibility.

## Completion gate

The controlled replication evidence is complete, and the final repository suite passed 28 tests. The identified and anonymous manuscripts each compile to 23 pages; their logs contain no layout, citation, reference, or rerun warnings, and all 46 rendered pages passed fresh inspection after the three-checkpoint table was integrated. The final numerical trace matched the controlled tables and manuscript endpoint values to archived JSON evidence. An anonymous wrapper and submission drafts are prepared. Submission still requires human-rating decisions, author metadata, live-portal format verification, and an archival DOI.
