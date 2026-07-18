# Style Overrides

- Use score-level, post-tonal, pitch-class set, serial transformation, rhythmic profile, gesture, MusicXML, and computer-assisted composition terminology.
- Avoid audio-generation, pop-MIDI, accompaniment, and style-imitation framing.
- Call the procedural comparator a rule reference, not a learned baseline or oracle.
- Describe K=4 inference as four-candidate guided selection: it jointly changes candidate budget and symbolic scoring relative to K=1.
- Describe K=1 and K=4 as sharing one trained generator.
- Compare condition-prefix ablations with the full-prefix K=1 model because every ablation uses single-candidate decoding.
- Describe condition ablations as prefix-field removal against unchanged hidden targets.
- Report pc-set coverage together with precision or Jaccard when interpreting pitch purity.
- Keep cyclic row-order accuracy and strict complete-form accuracy separate.
- Average aggregate completion only over serial targets.
- Do not cite legacy v2, smoke, or historical multi-seed numbers as formal results.
- Use `PENDING_REAL_EXPERIMENT` only for genuinely unavailable experiments. Human ratings and external legal validation are explicitly not conducted, not silently treated as pending full-run metrics.
- Insert numerical prose only after tracing it to canonical full-run CSV/JSON artifacts.
