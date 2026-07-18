# JNMR Submission Checklist

Status date: 2026-07-18

## Scientific package

| Item | Status | Evidence or action |
|---|---|---|
| Legal procedural corpus | PASS | No copyrighted post-1945 score is downloaded or bundled. |
| Formal split | PASS | `results/project2_full_split_summary.json`: 20,000/2,000/2,000, seed 42, `smoke=false`. |
| Formal neural training | PASS | All required checkpoints exist; all neural rows reached 60 epochs. |
| Aggregate metrics | PASS | 13 real 2,000-condition rows in the canonical metrics and constraints CSV files. |
| LaTeX result tables | PASS | Regenerated from the canonical CSV files after the full run. |
| Expert package | PASS | 20 MusicXML files and 20 paired JSON reports pass structural gates. |
| Human artistic-quality evidence | NOT CONDUCTED | Blank forms are prepared; no perceptual claim is made. |
| Independent legal MusicXML validation | NOT CONDUCTED | No user-supplied score was provided; no transfer claim is made. |

## Manuscript files

| Item | Status | Evidence or action |
|---|---|---|
| Numeric traceability | PASS | Active claims map to canonical CSV/JSON or full-run provenance files. |
| References | PASS | All bibliography keys resolve and every bibliography entry is cited. |
| Figures and tables | PASS | Source data/build scripts retained; rendered output visually inspected. |
| Identified PDF compilation | PASS WITH AUTHOR FIELDS PENDING | `paper/main.pdf`, 21 pages, clean log. |
| Anonymous PDF compilation | PASS | `paper/main_anonymous.pdf`, 21 pages, clean log and anonymity audit. |
| Test suite | PASS | 60 tests passed after the final table-generator change. |
| Current JNMR format and portal fields | MANUAL PORTAL GATE | Verify against the live author instructions immediately before upload. |

## Author and release inputs

- Author names, order, affiliations, ORCID identifiers, corresponding-author details, and CRediT roles: `PENDING_AUTHOR_INPUT`.
- Funding, competing interests, originality, exclusive submission, and AI-assistance disclosure: `PENDING_AUTHOR_CONFIRMATION`.
- Public archive or journal-compliant anonymous snapshot, license, URL, and DOI: `PENDING_RELEASE`.
- Article type, final template, limits, and portal-specific fields: `PENDING_PORTAL_VERIFICATION`.

The computational experiment is complete. Do not replace the remaining author or portal fields with inferred information.
