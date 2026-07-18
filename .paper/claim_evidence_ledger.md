# Claim-Evidence Ledger

| Claim | Status | Evidence |
|---|---|---|
| Corpus split is 20,000/2,000/2,000, seed 42, `smoke=false`. | supported | `results/project2_full_split_summary.json` |
| Formal environment is Python 3.11.9, PyTorch 2.5.1+cu121, CUDA, RTX 4060 Ti 16GB. | supported | `results/project2_full_run_report.md` |
| Formal matrix has 13 test rows with 2,000 conditions each. | supported | `results/project2_metrics.csv` |
| K=1 and K=4 share model parameters; K=4 changes candidate budget and symbolic scoring. | supported | configs, checkpoint hashes, full-run report |
| K=4/K=1 values reported in the manuscript match the canonical CSV. | supported | `results/project2_metrics.csv`, `paper/tables/project2_main_results.tex` |
| Condition-removal comparisons use separately trained K=1 models and unchanged targets. | supported | ablation configs, canonical CSV, evaluation code |
| Removing pc-set, serial, rhythm, or gesture fields degrades the matching automatic endpoint relative to full-prefix K=1. | supported, descriptive | `results/project2_metrics.csv`, `paper/figures/project2_full_results_source.csv` |
| Strict serial-form accuracy is zero for every neural row. | supported | `results/project2_metrics.csv` |
| Proposed mean content-span ratio is 0.5195 despite exact exported measure counts. | supported | canonical CSV and export records |
| 260/260 attempted exports pass parse, measure, and part checks. | supported | canonical CSV, generation JSON, full-run report |
| Twenty proposed-model MusicXML examples and reports pass package gates. | supported | `expert_eval/project2/manifest.json`, full-run report |
| Composers find outputs useful, coherent, or publication-ready. | unsupported; not claimed | Human ratings not collected |
| Method transfers to independent contemporary scores. | unsupported; not claimed | External legal validation unavailable |

Every precise value in Abstract, Results, Discussion, or Conclusion must map to the canonical CSV/JSON files or the split/full-run reports. Legacy v2 and smoke values are rejected for formal claims.
