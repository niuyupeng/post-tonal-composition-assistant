# Project 2 SCI Manuscript Review

## Overall assessment

The repository contains enough verified engineering and experimental material for a research manuscript draft, but the current LaTeX files are still a project scaffold rather than a submission-ready paper. The strongest defensible story is a reproducible neural-symbolic framework for controllable post-tonal score sketching, evaluated on a legal synthetic corpus. The manuscript should not claim stylistic imitation, perceptual quality, or superiority over human composition.

## Major issues requiring correction

1. The quantitative paragraph in `sections/results.tex` does not match the final 2,000-sample CSV. Every numerical claim must be regenerated from `results/project2_metrics.csv`.
2. The architecture is described too loosely. The implemented model uses condition prefix tokens in one causal Transformer, not separate condition-encoder networks. Constraint guidance is candidate sampling followed by non-differentiable symbolic reranking.
3. Several metrics are not applicable when their target condition is absent. Empty pc-set targets currently produce coverage 1.0 by definition; this must be reported as not applicable rather than as perfect control. Serial transformation accuracy is currently identical to cyclic row-order accuracy and must not be presented as an independent measurement.
4. Pc-set coverage is recall-like: it measures whether target pitch classes occur, not whether off-set pitch classes are excluded. It therefore cannot alone establish pitch-class-set purity.
5. Token accuracy is full-sequence next-token accuracy, including condition-prefix tokens. It is a model diagnostic, not a direct measure of compositional quality.
6. The rule generator constructs the synthetic training corpus and therefore acts as a procedural reference or oracle-style control, not an independent learned baseline.
7. The experiment uses one deterministic seed. No variance estimates, confidence intervals, or significance tests are available.
8. Blind expert ratings are prepared but have not been collected. Claims about compositional usefulness, notation quality, or perceptual coherence remain pending.
9. Related work has almost no in-text citations, and several BibTeX entries need verification.
10. The method figure is a placeholder, tables lack labels, the Chinese title is corrupted, and no rendered score example is included.
11. The original proposed and vanilla configurations use different corpus seeds and therefore different training and test samples. Their aggregate difference is descriptive, not a controlled estimate of the decoding effect.
12. The original evaluation entry point does not explicitly seed stochastic decoding. Key decoding comparisons require a deterministic rerun.
13. Rhythm-removal and gesture-removal configurations collapse the condition to a fixed default label (`medium` or `fragmented`) rather than removing the token. They must be described as fixed-default ablations.
14. `transformer_no_constraints` and `no_constraints` are duplicate design variants trained with different seeds; they are replication-style rows, not distinct methods.
15. General-corpus serial samples retain a small pc-set and interval-vector target even though the rule target cycles through all twelve row pitch classes. Coverage remains easy to satisfy, but interval-vector matching can conflict with aggregate completion. Serial and non-serial subsets must be reported separately.
16. Structural MusicXML success does not establish requested-length adherence. In the 20-example package, the number of written measures differs from the requested metadata and is often below four because generation may terminate or reach the token limit before the requested span.
17. The exporter previously labeled every score as `Rule-generated synthetic corpus`, including neural outputs. This provenance label is misleading and has been neutralized for future exports; archived files must not be classified from the embedded creator field.
18. `TIME_SHIFT` is capped at 16 beats and `BAR` is ignored by event decoding, so long gaps can be compressed. The music21 export path also omits trailing empty measures. Requested-versus-written length is therefore a representation/export limitation, not a clean model-only endpoint.
19. Reranking and evaluation use the same symbolic diagnostics. Controlled gains show that the selector optimizes its implemented objective, not independent musical validity; expert evaluation remains necessary.
20. Rhythmic distance uses a fixed seed-1234 profile realization rather than the exact stochastic rhythm behind each corpus target. Penalty-weight sensitivity and multiple-endpoint adjustment are not available.
21. The score condition contains one instrument class repeated across parts, and token decoding is not grammar constrained. Mixed instrumentation and syntactic validity rates are outside the present evidence.
22. Voice-count adherence is not measured. The exporter iterates over requested part indices, so decoded events assigned to higher voice IDs can be omitted while the file still passes structural parsing.

## Evidence that can be used now

- Legal synthetic corpus with explicit train/validation/test counts of 20,000/2,000/2,000 and seed 42.
- Twelve trained Transformer configurations plus one rule reference, all evaluated on 2,000 test samples.
- Python 3.11.9, PyTorch 2.5.1+cu121, CUDA execution on an RTX 4060 Ti.
- Archived independently trained rows differ in row-order, rhythmic-profile, density-curve, gesture, and token metrics. Because their generated corpora and seeds differ, these values are descriptive and cannot be used as the primary decoder-effect estimate.
- All reported MusicXML export checks succeed structurally for the evaluated examples; requested measure-count adherence is not established.
- Twenty MusicXML examples and paired JSON reports are available for expert review.

## Recommended positioning

Frame the contribution as a proof-of-concept and reproducible evaluation framework for explicit post-tonal control at score level. The novelty is the integration of condition-prefix symbolic generation, post-tonal theory utilities, candidate reranking, MusicXML output, and explainable diagnostics in one legally reproducible system. Avoid claiming a new Transformer architecture.

The primary controlled comparison should use one trained checkpoint and one test set. Single-candidate sampling is the decoder baseline; four-candidate symbolic reranking is the proposed decoder. The first candidate must be identical across both conditions through per-sample deterministic seeding. Existing multi-configuration rows remain exploratory ablations.

## Submission blockers

- Human expert evaluation is the main scientific blocker for claims about usefulness to composers.
- Multi-seed experiments are the main statistical blocker for strong claims of model superiority.
- A target journal and its formatting/word-limit requirements are not yet fixed.
