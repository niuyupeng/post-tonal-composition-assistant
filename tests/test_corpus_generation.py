from pathlib import Path

import torch

from post_tonal.data.generate_corpus import generate_samples
from post_tonal.data.post_tonal_dataset import PostTonalDataset, describe_splits


def test_corpus_generation(tmp_path: Path):
    data_path = tmp_path / "tiny.pt"
    vocab_path = tmp_path / "tiny.vocab.json"
    samples = generate_samples(
        num_samples=4,
        output=data_path,
        vocab_output=vocab_path,
        seed=4,
        min_measures=4,
        max_measures=4,
        min_voices=2,
        max_voices=2,
    )
    assert data_path.exists()
    assert vocab_path.exists()
    assert len(samples) == 4
    saved = torch.load(data_path, map_location="cpu", weights_only=False)
    assert len(saved["samples"]) == 4
    assert all("token_ids" in sample and sample["events"] for sample in saved["samples"])
    assert describe_splits(data_path) == {"train": 3, "val": 0, "test": 1}


def test_explicit_train_val_test_splits(tmp_path: Path):
    data_path = tmp_path / "split.pt"
    vocab_path = tmp_path / "split.vocab.json"
    generate_samples(
        num_samples=0,
        output=data_path,
        vocab_output=vocab_path,
        seed=9,
        min_measures=4,
        max_measures=4,
        min_voices=2,
        max_voices=2,
        train_samples=3,
        val_samples=2,
        test_samples=1,
    )
    assert describe_splits(data_path) == {"train": 3, "val": 2, "test": 1}
    assert len(PostTonalDataset(data_path, split="train")) == 3
    assert len(PostTonalDataset(data_path, split="val")) == 2
    assert len(PostTonalDataset(data_path, split="test")) == 1
