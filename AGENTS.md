# Agent Instructions

- Never fabricate results. Use `PENDING_REAL_EXPERIMENT` for unavailable paper values.
- Do not use copyrighted contemporary scores or scrape post-1945 score sources.
- Keep all models local-trainable on an RTX 4060 Ti with 16GB VRAM.
- Use score-level symbolic and MusicXML language. Do not design audio generation models.
- Always run tests after code changes.
- Prefer legal, reproducible, rule-generated corpora plus optional user-provided validation examples.
- Report computed metrics only when the code actually produced them.
