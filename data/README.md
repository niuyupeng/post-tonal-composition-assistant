# Data

The repository uses legal, reproducible synthetic data by default.

- `generated/`: optional MusicXML exports for inspection
- `processed/`: machine-readable `.pt` datasets and tokenizer vocabularies

Run:

```powershell
$env:PYTHONPATH = "src"
python -m post_tonal.data.generate_corpus --num-samples 32 --output data/processed/post_tonal_smoke.pt --vocab-output data/processed/post_tonal_smoke.vocab.json --export-musicxml --musicxml-dir data/generated/smoke_musicxml
```

No copyrighted contemporary scores are included.
