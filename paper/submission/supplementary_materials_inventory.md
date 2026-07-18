# Supplementary Materials Inventory

## Proposed review package

| ID | Contents | Repository location |
|---|---|---|
| S1 | Canonical aggregate and constraint metrics | `results/project2_metrics.csv`, `results/project2_constraints.csv` |
| S2 | Per-example generation records | `results/project2_generation_examples.json` |
| S3 | Full split and anonymized run provenance | `results/project2_full_split_summary.json`, `paper/submission/project2_full_run_report_anonymous.md` |
| S4 | Twenty proposed-model MusicXML examples | `expert_eval/project2/musicxml/` |
| S5 | Twenty paired automatic analysis reports and manifest | `expert_eval/project2/analysis_reports/`, `expert_eval/project2/manifest.json` |
| S6 | Blank blind-rating instruments | `expert_eval/project2/blind_rating_form_project2.md`, `.csv` |
| S7 | Figure 2 source data and build script | `paper/figures/project2_full_results_source.csv`, `make_project2_full_results.py` |

The anonymous review bundle is generated as
`paper/submission/project2_anonymous_supplement.zip` by
`paper/submission/make_anonymous_supplement.ps1`. The script replaces the local
workspace path in the run report and fails if an author, repository, or local
username identifier is detected in staged text files.

## Exclusions

Neural checkpoints and the processed tensor are large reproducible artifacts and
should be deposited only if the selected archive and journal permit their size.
Legacy v2, smoke, and historical multi-seed development outputs are not part of
the formal supplementary evidence. The original
`results/project2_full_run_report.md` retains exact local commands for internal
provenance and is excluded from the anonymous bundle because those commands
contain the local workspace path.
