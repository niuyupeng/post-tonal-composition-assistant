# Project 2 Manuscript Context

## Scope

Score-level neural-symbolic assistance for post-tonal art-music sketches. Inputs include pc sets and interval vectors or twelve-tone rows and transformations, plus rhythm, gesture, instrumentation, voices, and measures. Outputs are event tokens, MusicXML, and analysis reports. Audio generation and copyrighted-score imitation are outside scope.

## Corrected Design

- One legal procedural corpus with explicit 20,000/2,000/2,000 split and seed 42.
- Serial and non-serial pitch targets are mutually exclusive.
- Complete condition prefixes are repeated across score-body windows.
- Coverage cycles expose every saved body token; validation and testing enumerate all windows.
- The proposed K=4 and vanilla K=1 decoders share one trained generator.
- Ablations hide condition-prefix fields while preserving original targets.
- Held-out stochastic density targets are unavailable to the model and reranker.
- Rule-reference evaluation generates new deterministic outputs rather than replaying targets.
- Export checks include XML structure, requested measures, and requested voices.

## Current Evidence Status

- Environment: Python 3.11.9, PyTorch 2.5.1+cu121, CUDA available, RTX 4060 Ti.
- Corrected full corpus: present, 20,000/2,000/2,000, `smoke=false`.
- Code verification: current suite passes.
- Corrected full neural checkpoints and complete metric matrix: pending.
- Expert ratings and independent legal MusicXML validation: pending.

Legacy v2 metrics and three-seed controlled results are archived development evidence and are excluded from corrected claims.

## Paper Positioning

The contribution is the integration of legal synthetic data, explicit post-tonal conditions, locally trainable symbolic generation, inference-time candidate reranking, explainable diagnostics, and MusicXML output. The Transformer architecture itself is standard. Final claims must be limited to corrected automatic controllability on the synthetic testbed.
