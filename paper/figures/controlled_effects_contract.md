# Controlled-Effects Figure Contract

## Claim

Four-candidate symbolic reranking changes several implemented constraint metrics under a fixed checkpoint and fixed test set, with the direction and magnitude reported separately rather than summarized as blanket superiority.

## Evidence

- Primary source: `results/project2_controlled_statistics.json`.
- Population: the 2,000 paired test conditions; serial endpoints use the 914 row-conditioned samples and non-serial endpoints use the remaining 1,086 samples.
- Uncertainty: paired nonparametric bootstrap intervals over test conditions.
- Transformation: favorable mean differences and interval endpoints are divided by the absolute K=1 mean and expressed as percentages.

## Panel Scope

The main panel contains one endpoint for each principal condition family: pc-set coverage and interval-vector distance on non-serial samples, row-order accuracy and aggregate completion on serial samples, and rhythmic-profile distance and gesture consistency on all samples.

## Exclusions

- Archived cross-configuration rows are excluded because they use different generated corpora and seeds.
- Density-curve error is omitted because it overlaps with rhythmic-profile distance.
- Range violations and structural MusicXML checks remain in tables because they are engineering validity checks rather than central musical-control endpoints.
- The figure does not report expert preference or compositional usefulness.

## Review Risks

- Percentage changes can look large when a baseline is small; the table must retain raw means and differences.
- A confidence interval that excludes zero is not described as proof of generalization across training seeds.
- The pc-set and aggregate terms can conflict, so favorable orientation is endpoint-specific and does not imply a globally optimal musical result.
