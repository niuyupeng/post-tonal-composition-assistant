# Project 2 SCI Writing and Submission Plan

## Stage 1: Evidence lock

- Treat `results/project2_metrics.csv`, `results/project2_constraints.csv`, `results/project2_full_split_summary.json`, run summaries, and the expert-evaluation manifest as the only quantitative evidence sources.
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

- Add at least three independent seeds for the proposed and vanilla systems if compute time permits.
- Save per-example metric rows so paired confidence intervals and effect sizes can be computed.

Priority C:

- Add a pc-set precision/Jaccard or off-set contamination metric because coverage alone is recall-like.
- Separate serial transformation identification from row-order accuracy if both are retained.

Deliverable: stronger statistical and human-centered evidence. Until available, related claims remain `PENDING_REAL_EXPERIMENT` or are removed.

## Stage 5: Submission package

- Prepare the disciplinary version for the Journal of New Music Research; retain Applied Sciences as the faster SCIE fallback if APC and broader scope are acceptable.
- Download the current JNMR author instructions and adapt the manuscript template and references before submission.
- Prepare highlights, cover letter, graphical abstract if required, code/data availability statement, author contributions, conflict-of-interest statement, and supplementary material.
- Run final checks for LaTeX compilation, reference resolution, figure legibility, numerical consistency, terminology, and repository reproducibility.

Deliverable: submission-ready manuscript and supplementary package.

## Immediate execution order

1. Correct title encoding, result numbers, model description, table semantics, and labels.
2. Rewrite all core sections with the current verified evidence.
3. Verify and expand the bibliography with primary sources.
4. Build figures and compile the PDF.
5. Decide whether to submit a bounded proof-of-concept now or wait for expert ratings and multi-seed evidence.

## Fast-track execution board

| Workstream | Status | Evidence gate | Output |
|---|---|---|---|
| Scientific and implementation review | complete | source, configs, checkpoints, and archived outputs inspected | `SCI_MANUSCRIPT_REVIEW.md`, `CLAIMS_LEDGER.md` |
| Core method and experiment prose | complete pending final cross-check | implementation facts and full-run artifacts | Methodology, Experimental Setup, Evaluation Metrics, Limitations, Reproducibility |
| Primary literature and positioning | complete pending journal-style conversion | verified publisher or conference records | Related Work and `references.bib` |
| Controlled $K=1$ evaluation | complete | 2,000 aligned per-sample records | single-candidate JSON and aggregate metrics |
| Controlled $K=4$ evaluation | complete | 2,000 aligned per-sample records from the same checkpoint and conditions | reranked JSON and aggregate metrics |
| Paired statistics | complete | 2,000 sample IDs and condition bundles match; serial $n=914$, non-serial $n=1,086$ | paired CSV/JSON, bootstrap intervals, controlled table |
| Abstract, Results, and Discussion | complete pending final audit | all numeric claims sourced from paired CSV/JSON | evidence-bound prose |
| Controlled effect figure | complete | selected endpoints have prespecified favorable directions | PDF/SVG/PNG and source CSV |
| Representative score figure | integrated | controlled example 019 is structurally valid, spans the requested 5/5 measures, and has a matching JSON report | rendered score and self-contained caption |
| Final manuscript QA | complete for the current draft | all numbers traced to artifacts; LaTeX compiles without warnings; 16 tests pass | 20-page PDF and claim audit |
| GitHub synchronization | current manuscript-preparation commits synchronized | remote branch matches local commit | final evidence-bound manuscript commit after controlled analysis |

The critical dependency chain is $K=4$ evaluation $\rightarrow$ paired statistics $\rightarrow$ controlled table and figure $\rightarrow$ Abstract/Results/Discussion $\rightarrow$ final numerical audit. Literature, method prose, figure preparation, testing, and repository commits proceed in parallel. No manuscript sentence may bypass this chain by substituting archived cross-configuration values for the controlled comparison.

## Submission decision gate

The fastest defensible submission is a bounded proof-of-concept centered on automatic constraint adherence, reproducible synthetic data, and structurally valid score export. Composer preference, perceptual gesture recognition, and practical usefulness remain outside that claim set until blind ratings are collected. Multi-seed training and expert evaluation would strengthen the paper, but their absence must be stated as a limitation rather than hidden behind the controlled single-checkpoint analysis.
