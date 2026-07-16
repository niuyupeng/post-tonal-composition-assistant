# JNMR Submission Checklist

Status date: 2026-07-16

This checklist distinguishes verified repository facts from requirements that must be checked in the live Taylor & Francis submission portal. The dedicated instructions page was not reliably accessible from the current environment, so no abstract limit, word limit, template rule, or declaration requirement is inferred here.

## Scientific package

| Item | Status | Evidence or action |
|---|---|---|
| Scope matches systematic and technological music research | PASS | Target rationale and official overview link are recorded in `.paper/journal_format.md`. |
| Legal data strategy | PASS | Corpus is rule-generated; no copyrighted post-1945 score is downloaded. |
| Full corpus and primary CUDA run | PASS | `results/project2_full_split_summary.json` and `results/project2_full_run_report.md`. |
| Primary-checkpoint controlled comparison | PASS | `results/project2_controlled_statistics.csv` and `.json`; this remains separate from the replication table. |
| Three-checkpoint controlled replication | PASS WITH LIMIT | Seeds 42--44 each contain aligned K=1/K=4 outputs for 2,000 shared conditions. `results/project2_multiseed_controlled_statistics.csv` and `.json` report 14 crossed-bootstrap endpoints with no multiple-endpoint adjustment. |
| Numeric claim traceability | PASS | All 14 primary controlled rows and all 14 three-checkpoint rows match their archived JSON sources; replicated endpoint values quoted in the Abstract and Results were checked directly. |
| Human artistic-quality evidence | NOT CLAIMED | Expert package is prepared, but no ratings have been collected. |
| Multi-seed training diagnostics | PASS WITH LIMIT | Seeds 42--44 completed and have 2,000-item teacher-forced test metrics. These sequence diagnostics remain distinct from the controlled generation replication. |
| Repository tests | PASS | The final suite passed 28 tests after manuscript integration. |

## Manuscript files

| Item | Status | Evidence or action |
|---|---|---|
| Identified working manuscript | PASS WITH AUTHOR GATE | `paper/main.tex` compiles to 23 pages and passed full page inspection. Author and affiliation remain pending. |
| Anonymous review manuscript | PASS | `paper/main_anonymous.tex` compiles to 23 pages, passed full page inspection, and withholds author, affiliation, and repository identity. |
| Figures and tables cited in order | PASS | The integrated three-checkpoint table, figures, exploratory tables, and representative score were inspected in both variants. |
| References compile | PASS | No undefined citations or references remain; journal-specific citation style still requires live-instruction verification. |
| Exact abstract and main-text limits | MANUAL PORTAL GATE | Verify in the current JNMR instructions before upload. |
| Exact template, line numbering, and spacing | MANUAL PORTAL GATE | Apply only after downloading the current official files. |
| Separate figure formats and resolution | MANUAL PORTAL GATE | Confirm accepted formats and upload requirements. |
| Double-anonymous review policy | MANUAL PORTAL GATE | Confirm whether title page and repository must be separated or anonymized. |

## Author-supplied information

| Item | Status | Required action |
|---|---|---|
| Author names and order | PENDING_AUTHOR_INPUT | Complete `author_metadata_form.md`. |
| Affiliations and corresponding author | PENDING_AUTHOR_INPUT | Complete `author_metadata_form.md`. |
| ORCID identifiers | PENDING_AUTHOR_INPUT | Complete `author_metadata_form.md`. |
| CRediT contributions | PENDING_AUTHOR_INPUT | Complete `declarations.md`. |
| Funding and grant numbers | PENDING_AUTHOR_INPUT | Complete `declarations.md`. |
| Competing interests | PENDING_AUTHOR_CONFIRMATION | Complete `declarations.md`. |
| Originality and no concurrent review | PENDING_AUTHOR_CONFIRMATION | Confirm before signing `cover_letter.md`. |
| Preprint or prior submission disclosure | PENDING_AUTHOR_CONFIRMATION | Confirm before submission. |
| AI-assistance disclosure | PENDING_POLICY_AND_AUTHOR_CONFIRMATION | Use the journal's current policy and an accurate account of tool use. |
| Code/data archive URL and DOI | PENDING_RELEASE | Create an anonymous review snapshot or archival public release as permitted. |

## Final upload gate

Do not mark this package submission-ready until every `PENDING_*` and `MANUAL PORTAL GATE` item above has been resolved. Recompile both manuscript variants after applying the official template, then re-run the numerical, citation, anonymity, and figure checks.
