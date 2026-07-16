# Project 2 SCI Writing and Submission Plan

## Stage 1: Evidence lock

- Treat `results/project2_metrics.csv`, `results/project2_constraints.csv`, `results/project2_full_split_summary.json`, `results/project2_controlled_statistics.*`, `results/project2_multiseed_training_*`, `results/project2_multiseed_controlled_statistics.*`, run summaries, and the expert-evaluation manifest as the only quantitative evidence sources.
- Maintain a claim ledger that distinguishes measured results, implementation facts, interpretations, limitations, and pending human evidence.
- Mark metrics as not applicable when the corresponding condition is absent; reserve `PENDING_REAL_EXPERIMENT` for genuinely required but unavailable evidence.

Deliverable: internally consistent result tables and prose with no unsupported numbers.

## Stage 2: Manuscript core

1. Rewrite the title page, abstract, and keywords.
2. Rebuild the introduction around the control problem in post-tonal symbolic composition rather than around the repository.
3. Expand related work into four strands: computer-assisted composition, post-tonal/serial theory, symbolic music generation, and constrained or neural-symbolic generation.
4. Specify the corpus, token representation, causal Transformer, candidate reranking objective, MusicXML export, and analysis metrics mathematically and operationally.
5. Report the complete experimental protocol, hardware, early stopping, test size, baselines, and ablations.
6. Interpret results as trade-offs rather than as blanket superiority.
7. Separate limitations from future work and explicitly bound all artistic claims.

Deliverable: a journal-neutral full English manuscript whose quantitative claims all trace to saved outputs.

## Stage 3: Figures and tables

- Replace the placeholder pipeline with a publication-quality TikZ figure showing condition tokens, causal Transformer sampling, symbolic reranking, and MusicXML/analysis outputs.
- Generate a main comparison table, an ablation table, and a metric-applicability note.
- Add training-curve figures from run CSV files.
- Render representative MusicXML fragments and pair them with compact analysis panels.
- Keep raw model names in supplementary material and use readable display names in the main paper.

Deliverable: figures and tables that can be understood without reading the code.

## Stage 4: Credibility upgrades

Priority A:

- Run the prepared blind expert evaluation on the 20 examples.
- Report rater count, expertise, protocol, medians/means, dispersion, and inter-rater agreement.

Priority B:

- Retain the completed three-seed proposed-model training diagnostics.
- Integrate the completed aligned K=1/K=4 replication for seeds 42, 43, and 44 without pooling it with the primary-checkpoint table.
- Add comparable vanilla-model seeds only if compute time permits and the resulting claim is needed.

Priority C:

- Add a pc-set precision/Jaccard or off-set contamination metric because coverage alone is recall-like.
- Separate serial transformation identification from row-order accuracy if both are retained.

Deliverable: integrate the completed replication while keeping uncollected human evidence and optional metric extensions marked `PENDING_REAL_EXPERIMENT` or outside the claim set.

## Stage 5: Submission package

- Prepare the disciplinary version for the Journal of New Music Research; retain Applied Sciences as the faster SCIE fallback if APC and broader scope are acceptable.
- Download the current JNMR author instructions and adapt the manuscript template and references before submission.
- Prepare highlights, cover letter, graphical abstract if required, code/data availability statement, author contributions, conflict-of-interest statement, and supplementary material.
- Run final checks for LaTeX compilation, reference resolution, figure legibility, numerical consistency, terminology, and repository reproducibility.

Deliverable: submission-ready manuscript and supplementary package.

## Immediate execution order

1. Integrate `paper/tables/project2_multiseed_controlled_results.tex` into the manuscript as a replication table separate from the primary-checkpoint controlled table.
2. Update the Abstract, Results, Discussion, Limitations, Conclusion, and reproducibility text using only `results/project2_multiseed_controlled_statistics.*`.
3. Run a numerical trace from every new value to the aggregate JSON or CSV, then rebuild and inspect both manuscript variants.
4. Commit and publish the completed experiment artifacts and documentation.
5. Decide whether to submit the bounded automatic study before blind expert ratings are collected.

## Fast-track execution board

| Workstream | Status | Evidence gate | Output |
|---|---|---|---|
| Scientific and implementation review | complete | source, configs, checkpoints, and archived outputs inspected | `SCI_MANUSCRIPT_REVIEW.md`, `CLAIMS_LEDGER.md` |
| Core method and experiment prose | complete | implementation facts, full-run artifacts, and the controlled aggregate | Methodology, Experimental Setup, Evaluation Metrics, Limitations, Reproducibility |
| Primary literature and positioning | complete pending journal-style conversion | verified publisher or conference records | Related Work and `references.bib` |
| Controlled $K=1$ evaluation | complete | 2,000 aligned per-sample records | single-candidate JSON and aggregate metrics |
| Controlled $K=4$ evaluation | complete | 2,000 aligned per-sample records from the same checkpoint and conditions | reranked JSON and aggregate metrics |
| Paired statistics | complete | 2,000 sample IDs and condition bundles match; serial $n=914$, non-serial $n=1,086$ | paired CSV/JSON, bootstrap intervals, controlled table |
| Abstract, Results, and Discussion | complete | new numeric claims trace to `project2_multiseed_controlled_statistics.*` | evidence-bound prose with separate primary and replication tables |
| Controlled effect figure | complete | selected endpoints have prespecified favorable directions | PDF/SVG/PNG and source CSV |
| Representative score figure | integrated | controlled example 019 is structurally valid, spans the requested 5/5 measures, and has a matching JSON report | rendered score and self-contained caption |
| Expert package preparation | complete; ratings pending | 20 controlled $K=4$ MusicXML files parse structurally and are anonymized | condition-aware forms, manifest, withheld reports |
| Multi-seed training replication | complete with a stated scope limit | seeds 42--44 completed; each checkpoint has a 2,000-item teacher-forced test row | per-seed CSV, aggregate JSON, and LaTeX table |
| Three-checkpoint controlled decoding | complete with endpoint-specific interpretation | seeds 42--44 each have aligned K=1/K=4 records for 2,000 shared conditions; first candidates match by SHA256 | per-seed statistics, crossed-bootstrap JSON/CSV, and a separate replication table |
| Submission package | prepared with explicit gates | no author identity, policy, or declaration inferred | anonymous wrapper, checklist, cover letter, metadata form, declarations |
| Final manuscript QA | complete | 28 tests passed; 14-row numeric traces passed; both 23-page PDFs and all 46 rendered pages were checked after integration | final identified and anonymous PDFs plus QA records |
| GitHub synchronization | ready for final commit | controlled-replication artifacts, manuscript, and QA are complete locally | commit and push the final evidence set |

The three-checkpoint aggregate, separate replication table, manuscript integration, numerical audit, and PDF page-level QA are complete. The remaining repository action is the final commit and push. No manuscript sentence may substitute archived cross-configuration values for either controlled comparison or merge the primary-checkpoint and three-checkpoint estimates.

## Submission decision gate

The fastest defensible submission is a bounded proof of concept centered on automatic constraint adherence, reproducible synthetic data, and structurally valid score export. The three-checkpoint controlled replication supports recurring directions for selected automatic diagnostics and recurring conflicts for serial pc-set coverage and aggregate completion. It does not provide evidence of composer preference, perceptual gesture recognition, or practical usefulness. Those claims remain outside the paper until blind ratings are collected.
