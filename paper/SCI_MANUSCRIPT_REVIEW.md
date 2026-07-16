# Project 2 SCI Manuscript Review

## Verdict

The repository contains a defensible corrected method and experiment runner, but the manuscript is not result-complete. The v3 neural checkpoint matrix has not yet finished. Legacy v2 numerical claims have been removed from the active manuscript and may not be restored as corrected evidence.

## Blocking scientific items

1. Complete the corrected shared-generator training and all required condition-prefix ablations.
2. Produce exactly one 2,000-sample v3 metric row for each of the thirteen named configurations.
3. Generate the main and ablation tables from the corrected CSV files.
4. Build the 20-example expert package from the completed proposed checkpoint and verify structure, measures, and voices.
5. Create a full-run report whose checkpoint, hash, table, split, and package gates all pass.

## Resolved methodological issues

- Long fragments use explicit condition-preserving windows instead of silent 256-token truncation.
- Serial examples no longer carry incompatible small-set targets.
- The stochastic target density curve is hidden from decoding and retained only for evaluation.
- Condition ablations change visible prefixes while preserving original held-out targets and split membership.
- The rule reference generates fresh outputs rather than replaying stored target events.
- Decoding uses an event grammar, requested voice indices, instrument ranges, and requested bar counts.
- Pc-set precision and Jaccard supplement coverage.
- Strict complete-form accuracy is distinct from cyclic row-order accuracy.
- Aggregate completion is applicable only to serial targets.
- Gesture density and rhythm metrics are normalized by requested voices; rest coverage is bounded.
- MusicXML checks include requested measure and part counts.
- Training writes resumable state, best epoch, provenance hashes, and resource summaries.

## Claim boundary

The eventual automatic results can support statements about implemented symbolic controllability on a legal synthetic testbed. They cannot by themselves support artistic quality, stylistic authenticity, composer preference, or publication-quality engraving. Those claims require the prepared blind evaluation and independent legal validation material.

## Submission status

The paper is a truthful method-and-experiment skeleton with `PENDING_REAL_EXPERIMENT` placeholders. It must not be submitted until the corrected full run completes, the numerical claim audit is refreshed, author metadata are supplied, and the current journal instructions are checked.
