# JNMR Submission Checklist

Status date: 2026-07-16

## Scientific package

| Item | Status | Evidence or action |
|---|---|---|
| Legal procedural corpus | PASS | No copyrighted post-1945 score is downloaded or bundled. |
| Corrected full split | PASS | `results/project2_v3_full_split_summary.json`: 20,000/2,000/2,000, `smoke=false`. |
| Corrected full neural training | PENDING_REAL_EXPERIMENT | Required v3 checkpoints are incomplete. |
| Corrected aggregate metrics | PENDING_REAL_EXPERIMENT | `results/project2_v3_metrics.csv` does not yet contain all required rows. |
| Corrected LaTeX result tables | PENDING_REAL_EXPERIMENT | Generated only after complete v3 metrics. |
| Full-model expert package | PENDING_REAL_EXPERIMENT | Must contain 20 exact-span, exact-voice MusicXML examples. |
| Human artistic-quality evidence | PENDING_REAL_EXPERIMENT | Rating forms exist; ratings have not been collected. |
| Independent legal MusicXML validation | PENDING_REAL_EXPERIMENT | User-supplied material is not yet available. |

## Manuscript files

| Item | Status | Evidence or action |
|---|---|---|
| Active manuscript excludes legacy v2 numeric claims | PASS | Abstract, Results, Discussion, Limitations, and Conclusion use corrected status. |
| Chinese title encoding | PASS | UTF-8 title in `paper/main.tex` and `README.md`. |
| Method figure matches corrected pipeline | PASS | Conditions to encoders to generator to guided decoder to MusicXML/report. |
| Corrected numeric traceability | PENDING_REAL_EXPERIMENT | Refresh after v3 CSV/JSON generation. |
| Final identified and anonymous compilation | PENDING_REAL_EXPERIMENT | Recompile and inspect after tables are generated. |
| Current JNMR template and limits | MANUAL PORTAL GATE | Verify against live author instructions before submission. |

## Author-supplied information

Author names, order, affiliations, ORCID identifiers, CRediT roles, funding, conflicts, originality confirmation, AI-assistance disclosure, and archival release details remain pending.

Do not mark the package submission-ready until every corrected full-run and author/portal gate is resolved.
