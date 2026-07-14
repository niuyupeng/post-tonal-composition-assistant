# Figure and Table Inventory

| ID | Artifact | Source | Claim supported | Status | Main risk |
|---|---|---|---|---|---|
| Fig. 1 | `paper/figures/method_pipeline.tex` | tokenizer, Transformer, evaluator, exporter | Actual training and inference paths | ready | Must not imply separate neural condition encoders |
| Fig. 2 | `paper/figures/training_diagnostics.pdf` | `runs/proposed_constraint_guided_transformer/train_summary.json` | Optimization and checkpoint selection | ready | Token accuracy includes condition-prefix positions |
| Fig. 3 | `paper/figures/controlled_effects.pdf` | `results/project2_controlled_statistics.json` | Same-checkpoint K=4 versus K=1 paired changes | pending K=4 | Relative changes require raw means in the adjacent table |
| Fig. 4 | controlled MusicXML score rendering | controlled K=4 MusicXML and JSON pair | Inspectable score-level output | pending selection | Caption must state requested and serialized measure counts |
| Table 1 | `paper/tables/project2_controlled_results.tex` | paired per-sample statistics | Primary controlled result | pending K=4 | Same diagnostics are used for selection and evaluation |
| Table 2 | `paper/tables/project2_main_results.tex` | archived aggregate CSV | Descriptive reference rows | ready | Different corpora/seeds; row-accuracy denominator varies |
| Table 3 | `paper/tables/project2_ablation_results.tex` | archived aggregate CSV | Exploratory configurations | ready | Not paired ablations; fixed-default rows are not token removal |

The score figure will be selected from the newly seeded controlled K=4 exports, not from a smoke run. Human-rating figures are excluded until real blind ratings exist.
