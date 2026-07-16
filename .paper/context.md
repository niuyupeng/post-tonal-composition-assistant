# Project 2 Manuscript Context

## Scope

The paper studies score-level neural-symbolic assistance for post-tonal art-music sketches. Inputs include pitch-class sets, interval vectors, optional twelve-tone rows and transformations, rhythmic profiles, gestures, instrumentation, voice count, and measure count. Outputs are event-token score fragments, MusicXML, and symbolic analysis reports. Audio generation, pop accompaniment, and imitation of copyrighted post-1945 repertoire are outside scope.

## Central Question

Across three independently trained checkpoints and one fixed test split, which explicit post-tonal constraint diagnostics change when four-candidate symbolic reranking is compared with the aligned first sampled candidate?

## Evidence Hierarchy

1. Three-checkpoint controlled evaluation: seeds 42, 43, and 44, the same 2,000 test conditions per checkpoint, aligned first-candidate hashes, and matched K=1/K=4 sampling schedules.
2. Primary-checkpoint controlled evaluation: one checkpoint and the same 2,000 test conditions for the original detailed paired analysis.
3. Full-run training and corpus artifacts: 20,000/2,000/2,000 split, smoke=false, RTX 4060 Ti execution.
4. Archived 13-row experiment table: descriptive only because separately generated datasets and seeds prevent a controlled causal comparison.
5. Expert package: 20 controlled $K=4$ outputs are available as anonymized, structurally valid MusicXML with condition-aware forms, but blind ratings have not been collected.

## Paper Positioning

The neural architecture is a standard six-layer causal Transformer with condition-prefix tokens. The methodological contribution is the score-level integration of legal synthetic data, explicit post-tonal diagnostics, inference-time candidate reranking, MusicXML export, and per-fragment explanations. Claims concern constraint consistency on the synthetic testbed, not stylistic authenticity or composer preference.

## Current Status

- Full training and archived evaluations: complete.
- Primary-checkpoint K=1/K=4 evaluation and paired bootstrap analysis: complete for 2,000 aligned conditions.
- Three-checkpoint K=1/K=4 evaluation: complete for seeds 42, 43, and 44 on the fixed 2,000-condition test split. The aggregate reports 14 endpoints and uses 10,000 crossed percentile-bootstrap resamples over training seeds and aligned conditions, without multiplicity correction.
- Endpoint-level evidence is mixed. Crossed intervals exclude zero in the favorable direction for non-serial pc-set coverage, all three interval-vector-distance strata, serial row-order accuracy, rhythmic-profile distance, density-curve error, gesture consistency, and range-violation rate. They exclude zero in the unfavorable direction for serial pc-set coverage and aggregate completion on the full and serial subsets. The intervals cross zero for full-set pc-set coverage and non-serial aggregate completion.
- Expert package: rebuilt from `controlled_constraint_reranked`; 20/20 files parse as `score-partwise`, creator metadata are anonymous, and encoding dates are removed.
- Multi-seed training replication: seeds 42, 43, and 44 completed with batch 8 and two accumulation steps. All three checkpoints were evaluated teacher-forced on the complete 2,000-item test split and aggregated with sample standard deviations. Earlier OOM and contention directories remain diagnostic only.
- The integrated manuscript passed its final numerical audit and warning-free 23-page identified and anonymous compilations. All 46 rendered pages were inspected after the three-checkpoint controlled evidence was incorporated.
- Primary target: Journal of New Music Research; Applied Sciences is the faster SCIE fallback.
- Author and affiliation metadata: pending.
- Expert ratings and validation on independent, legally supplied MusicXML examples: pending real experiments.

## Three-Seed Controlled Evidence

The authoritative aggregate is `results/project2_multiseed_controlled_statistics.json`, with a tabular mirror in `results/project2_multiseed_controlled_statistics.csv`. Each checkpoint uses 2,000 aligned conditions, including 1,086 non-serial and 914 serial conditions. First-candidate alignment is verified by SHA-256 for every seed-condition pair. Effect signs follow the aggregate artifact: positive favors K=4 reranking, except the non-serial aggregate-completion diagnostic, which is stored as the raw reranked-minus-single difference without a preferred direction.

Interpret these estimates as endpoint-specific evidence on one fixed synthetic corpus and test split. The crossed percentile intervals cover both training-seed and shared-condition variation, but only three checkpoints were evaluated. The 14 endpoint intervals are unadjusted for multiplicity, and the symbolic diagnostics used for reranking overlap with those used for evaluation. These results therefore support qualified statements about implemented constraint diagnostics, not artistic quality, external validity, or universal decoder superiority.
