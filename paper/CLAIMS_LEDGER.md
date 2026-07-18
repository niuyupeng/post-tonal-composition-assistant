# Project 2 Formal Claim Ledger

## Supported experimental claims

| Claim | Evidence | Allowed wording |
|---|---|---|
| Formal split is 20,000/2,000/2,000, seed 42, `smoke=false`. | `results/project2_full_split_summary.json` | Exact corpus fact. |
| Formal environment is Python 3.11.9, PyTorch 2.5.1+cu121, CUDA, RTX 4060 Ti 16GB. | `results/project2_full_run_report.md` | Exact environment fact. |
| Thirteen 2,000-condition test rows passed the final gate. | canonical metrics CSV and full-run report | Completion fact. |
| K=1 and K=4 share a checkpoint; K=4 also changes candidate budget and applies symbolic scoring. | configs, identical checkpoint hash, report | Controlled shared-generator description; do not attribute the contrast to reranking alone. |
| K=4 moves all plotted selection endpoints in the preferred direction relative to aligned K=1. | `results/project2_metrics.csv` | Descriptive single-run result, not significance or universal superiority. |
| Condition removals use K=1 decoding and worsen matching endpoints relative to full-prefix K=1. | configs and canonical CSV | Descriptive ablation result. |
| Strict serial-form accuracy is zero for every neural configuration. | canonical CSV | Required negative result. |
| Rule reference reaches strict form 0.9819 and row accuracy 1.0000. | canonical CSV | Procedural construction reference, not learned baseline. |
| Proposed content-span ratio is 0.5195 despite exact XML measures. | canonical CSV and export code | Structural padding limitation. |
| 260/260 attempted exports pass parse, measure, and part checks. | generation JSON and full-run report | Structural notation fact only. |
| Expert package contains 20 proposed-model MusicXML files and 20 reports. | `expert_eval/project2/manifest.json` | Package-completion fact; no human ratings. |

## Claims not supported

- artistic quality, preference, creativity, coherence, or compositional usefulness;
- stylistic authenticity or imitation of contemporary composers;
- transfer to independent human-authored MusicXML;
- significance, confidence intervals, or stability across training seeds;
- publication-quality engraving.

Legacy v2, smoke, and historical multi-seed development outputs are excluded from the formal manuscript.
