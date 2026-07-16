# Corrected Claim-Evidence Ledger

| Claim | Status | Evidence |
|---|---|---|
| The corpus split is 20,000/2,000/2,000 with seed 42 and `smoke=false`. | supported | `results/project2_v3_full_split_summary.json` |
| The active formal environment uses Python 3.11.9, PyTorch 2.5.1+cu121, CUDA, and RTX 4060 Ti. | supported environment fact | Fresh environment audit; final report after completion |
| Training uses condition-preserving coverage-cycle windows. | supported method | Dataset and training code plus tests |
| Held-out density targets are hidden from decoding. | supported method | Conditions, analysis, candidate-loss code plus tests |
| Rule-reference outputs are independently regenerated. | supported method | Evaluation code/config plus tests |
| Corrected neural metrics show any advantage or trade-off. | gap | Full v3 experiment incomplete |
| Twenty full-model MusicXML examples satisfy structure, measure, and voice gates. | gap | Full proposed evaluation incomplete |
| Composers find the outputs useful or coherent. | gap | Human ratings not collected |
| The method transfers to independent contemporary scores. | gap | Legal external validation unavailable |

Every precise result value in Abstract, Results, Discussion, or Conclusion must map to a completed corrected artifact. Legacy v2 values are rejected for corrected claims.
