# Project 2 Manuscript Context

## Scope

The paper studies score-level neural-symbolic assistance for post-tonal art-music sketches. Inputs include pitch-class sets, interval vectors, optional twelve-tone rows and transformations, rhythmic profiles, gestures, instrumentation, voice count, and measure count. Outputs are event-token score fragments, MusicXML, and symbolic analysis reports. Audio generation, pop accompaniment, and imitation of copyrighted post-1945 repertoire are outside scope.

## Central Question

With one checkpoint and one test set fixed, does four-candidate symbolic reranking improve explicit post-tonal constraint metrics over the first sampled candidate?

## Evidence Hierarchy

1. Controlled paired evaluation: same checkpoint, same 2,000 test conditions, and the same per-sample random seeds for K=1 and K=4.
2. Full-run training and corpus artifacts: 20,000/2,000/2,000 split, smoke=false, RTX 4060 Ti execution.
3. Archived 13-row experiment table: descriptive only because separately generated datasets and seeds prevent a controlled causal comparison.
4. Expert package: structurally valid MusicXML examples are available, but blind ratings have not been collected.

## Paper Positioning

The neural architecture is a standard six-layer causal Transformer with condition-prefix tokens. The methodological contribution is the score-level integration of legal synthetic data, explicit post-tonal diagnostics, inference-time candidate reranking, MusicXML export, and per-fragment explanations. Claims concern constraint consistency on the synthetic testbed, not stylistic authenticity or composer preference.

## Current Status

- Full training and archived evaluations: complete.
- Controlled K=1 evaluation: complete.
- Controlled K=4 evaluation and paired bootstrap analysis: complete for 2,000 aligned conditions.
- Evidence-bound Abstract, Results, and Discussion: complete; numerical audit and warning-free 20-page compilation passed.
- Primary target: Journal of New Music Research; Applied Sciences is the faster SCIE fallback.
- Author and affiliation metadata: pending.
- Expert ratings and multi-seed training: pending real experiments.
