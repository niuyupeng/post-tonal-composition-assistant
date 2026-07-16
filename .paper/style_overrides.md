# Style Overrides

- Use score-level, post-tonal, pitch-class set, serial transformation, rhythmic profile, gesture, MusicXML, and computer-assisted composition terminology.
- Avoid audio-generation, pop-MIDI, accompaniment, and style-imitation language.
- Describe the procedural comparator as a rule reference or corpus generator, not as an independent learned baseline.
- Describe K=4 inference as candidate reranking, not as a differentiable constraint loss.
- Use descriptive language for archived cross-configuration rows. Use paired language for controlled K=1 versus K=4 comparisons, and distinguish the primary-checkpoint analysis from the three-checkpoint aggregate.
- Describe the three-checkpoint result endpoint by endpoint. Do not use blanket wording such as "stable across seeds," "consistently superior," or "generalizes" when some crossed intervals favor K=4, some favor K=1, and some cross zero.
- Every three-checkpoint numerical claim must trace to `results/project2_multiseed_controlled_statistics.json` or its CSV mirror. State that the analysis uses seeds 42, 43, and 44, one fixed 2,000-condition test split per checkpoint, and crossed percentile bootstrap intervals over training seeds and aligned conditions.
- When interpreting the three-checkpoint aggregate, disclose that 14 endpoint intervals were reported without multiplicity correction and that reranking and evaluation reuse overlapping symbolic diagnostics.
- Reserve `PENDING_REAL_EXPERIMENT` for unavailable evidence, especially blind expert ratings and validation on independent, legally supplied MusicXML examples. Cross-seed controlled decoding is complete and must not be listed as pending. Use `--` for metrics that are not applicable.
