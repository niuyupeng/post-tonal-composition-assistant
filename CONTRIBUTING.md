# Contributing

Contributions should preserve the project's score-level, post-tonal research
scope and its evidence rules.

1. Create a focused branch from `main`.
2. Keep generated corpora, checkpoints, logs, submission files, and local
   environments out of Git.
3. Add or update tests for behavioral changes.
4. Run `python -m pytest` with `PYTHONPATH=src` before opening a pull request.
5. State which outputs were actually regenerated. Never fill missing results
   with estimates or copied smoke values.

Do not contribute copyrighted contemporary scores. Validation examples must be
original, procedurally generated, public-domain, openly licensed, or supplied
by a user who has the right to use them. Do not add audio-generation, pop-MIDI,
or accompaniment functionality under this repository's research identity.

Bug reports should include the operating system, Python and PyTorch versions,
CUDA status, configuration path, random seed, command, and complete traceback.
Never include credentials, private score files, or identifiable expert-rating
data in an issue.
