# Project 2 SCI Manuscript Review

Status date: 2026-07-18

## Verdict

The formal computational study is complete and internally consistent. The active manuscript reports only values traced to the canonical full-run CSV/JSON files. The anonymous manuscript is technically ready for journal-format adaptation: it compiles without warnings, contains no author or repository identifier, and passed a 21-page visual inspection. The identified submission remains blocked only by author-supplied metadata, declarations, archival release details, and the live journal portal requirements.

## Evidence gates

| Gate | Status | Evidence |
|---|---|---|
| Legal procedural corpus | PASS | `results/project2_full_split_summary.json`: 20,000/2,000/2,000, seed 42, `smoke=false` |
| Formal experiment matrix | PASS | 13 test rows, 2,000 conditions per row in `results/project2_metrics.csv` |
| Neural training artifacts | PASS | 12 required checkpoint paths; all neural rows reached 60 epochs |
| Constraint and export metrics | PASS | `results/project2_constraints.csv` and 260 per-example export records |
| Tables and result figure | PASS | Generated directly from canonical result files |
| Expert inspection package | PASS | 20 MusicXML files, 20 JSON reports, manifest, and blank rating forms |
| Tests | PASS | 60 tests passed on Python 3.11.9 |
| Identified PDF | TECHNICAL PASS | 21 pages; author and affiliation fields intentionally remain pending |
| Anonymous PDF | PASS | 21 pages; no compile warnings; anonymity text audit passed |

## Scientific review

The contribution is defensible as a neural-symbolic, score-level composition-assistance workflow on a legal procedural testbed. The strongest evidence is condition-level automatic controllability, transparent constraint diagnostics, reproducible local training, and structurally valid MusicXML output. The standard Transformer architecture is not presented as the primary novelty.

The manuscript appropriately reports the negative results. Strict serial-form accuracy is 0.0000 for every neural row, the proposed mean content-span ratio is 0.5195, and the rule reference remains much stronger on exact serial and rhythmic realization. The K=1 versus K=4 comparison jointly changes candidate count and symbolic selection, so it is described as guided candidate selection rather than an isolated reranking effect.

Claims remain limited to descriptive automatic results on the synthetic distribution. Human artistic quality, composer usefulness, stylistic authenticity, and transfer to independent contemporary scores are not claimed because blind ratings and external legal MusicXML validation were not conducted.

## PDF and traceability audit

- Both manuscripts compile with XeLaTeX/BibTeX and no undefined references, citation warnings, or box-overflow warnings.
- Every page of both 21-page PDFs was rendered and inspected.
- Wide result tables use standard sideways table pages and remain legible at normal PDF zoom.
- The method diagram, automatic-result figure, and representative MusicXML score render correctly.
- The anonymous PDF contains no GitHub username, repository URL, experiment commit hash, local path, or pending author token.
- All active-manuscript numerical claims map to `results/project2_metrics.csv`, `results/project2_constraints.csv`, `results/project2_generation_examples.json`, or the full split/run reports.

## Remaining external gates

1. Supply author names, order, affiliations, ORCID identifiers, corresponding-author details, and CRediT roles.
2. Confirm funding, competing interests, originality, exclusive submission, and AI-assistance disclosure.
3. Create the public archive or journal-compliant anonymous review snapshot and assign its permanent identifier if required.
4. Check the live Journal of New Music Research portal for the current article type, template, word limits, keyword limits, and upload fields.
5. Collect expert ratings or independent legal validation only if the authors choose to extend the current automatic-evaluation claim scope.

There are no remaining `PENDING_REAL_EXPERIMENT` values in the active manuscript.
