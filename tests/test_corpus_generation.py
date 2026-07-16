from pathlib import Path

import pytest
import torch

from post_tonal.data.generate_corpus import derive_corpus, generate_samples
from post_tonal.data.post_tonal_dataset import PostTonalDataset, describe_splits
from post_tonal.data.score_tokenizer import ScoreTokenizer
from post_tonal.train import maybe_generate_data


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


def test_describe_splits_rejects_inconsistent_declared_counts(tmp_path: Path):
    data_path = tmp_path / "bad_splits.pt"
    torch.save(
        {
            "samples": [
                {"split": "train"},
                {"split": "val"},
                {"split": "test"},
            ],
            "split_counts": {"train": 3, "val": 0, "test": 0},
        },
        data_path,
    )

    with pytest.raises(ValueError, match="do not match sample membership"):
        describe_splits(data_path)


def test_disabled_generation_fails_when_inputs_are_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="generation is disabled"):
        maybe_generate_data(
            {
                "data_path": str(tmp_path / "missing.pt"),
                "vocab_path": str(tmp_path / "missing.vocab.json"),
                "generate_data": False,
            }
        )


def test_windowed_dataset_covers_complete_score_body(tmp_path: Path):
    data_path = tmp_path / "windowed.pt"
    vocab_path = tmp_path / "windowed.vocab.json"
    samples = generate_samples(
        num_samples=0,
        output=data_path,
        vocab_output=vocab_path,
        seed=15,
        min_measures=8,
        max_measures=8,
        min_voices=4,
        max_voices=4,
        train_samples=1,
        val_samples=1,
        test_samples=1,
    )
    tokenizer = ScoreTokenizer.load(vocab_path)
    source = samples[0]["token_ids"]
    sep_index = source.index(tokenizer.token_to_id["SEP"])
    expected_body_tokens = len(source[sep_index + 1 :])
    dataset = PostTonalDataset(
        data_path,
        max_seq_len=64,
        split="train",
        sep_id=tokenizer.token_to_id["SEP"],
        sequence_mode="all",
        target_tokens_per_window=24,
        seed=15,
    )
    covered = sum(int(dataset[index]["target_token_count"]) for index in range(len(dataset)))
    assert len(dataset) > 1
    assert covered == expected_body_tokens
    assert all(int(dataset[index]["input_ids"].shape[0]) <= 64 for index in range(len(dataset)))


def test_rotating_windows_visit_every_segment(tmp_path: Path):
    data_path = tmp_path / "rotating.pt"
    vocab_path = tmp_path / "rotating.vocab.json"
    generate_samples(
        num_samples=0,
        output=data_path,
        vocab_output=vocab_path,
        seed=18,
        min_measures=8,
        max_measures=8,
        min_voices=4,
        max_voices=4,
        train_samples=1,
        val_samples=1,
        test_samples=1,
    )
    tokenizer = ScoreTokenizer.load(vocab_path)
    dataset = PostTonalDataset(
        data_path,
        max_seq_len=64,
        split="train",
        sep_id=tokenizer.token_to_id["SEP"],
        sequence_mode="rotating",
        target_tokens_per_window=24,
        seed=18,
    )
    window_count = int(dataset[0]["window_count"])
    visited = set()
    for epoch in range(window_count):
        dataset.set_epoch(epoch)
        visited.add(int(dataset[0]["window_index"]))
    assert visited == set(range(window_count))


def test_coverage_cycle_visits_every_window_once(tmp_path: Path):
    data_path = tmp_path / "coverage_cycle.pt"
    vocab_path = tmp_path / "coverage_cycle.vocab.json"
    samples = generate_samples(
        num_samples=0,
        output=data_path,
        vocab_output=vocab_path,
        seed=19,
        min_measures=8,
        max_measures=8,
        min_voices=4,
        max_voices=4,
        train_samples=3,
        val_samples=1,
        test_samples=1,
    )
    tokenizer = ScoreTokenizer.load(vocab_path)
    dataset = PostTonalDataset(
        data_path,
        max_seq_len=64,
        split="train",
        sep_id=tokenizer.token_to_id["SEP"],
        sequence_mode="coverage_cycle",
        target_tokens_per_window=24,
        seed=19,
        coverage_cycle_epochs=4,
    )

    expected_windows = set(dataset._all_windows)
    expected_tokens = 0
    for sample in samples[:3]:
        sep_index = sample["token_ids"].index(tokenizer.token_to_id["SEP"])
        expected_tokens += len(sample["token_ids"][sep_index + 1 :])

    visited_windows: list[tuple[int, int]] = []
    covered_tokens = 0
    epoch_sizes = []
    for epoch in range(4):
        dataset.set_epoch(epoch)
        epoch_sizes.append(len(dataset))
        for index in range(len(dataset)):
            item = dataset[index]
            visited_windows.append((int(item["sample_index"]), int(item["window_index"])))
            covered_tokens += int(item["target_token_count"])

    assert set(visited_windows) == expected_windows
    assert len(visited_windows) == len(expected_windows)
    assert covered_tokens == expected_tokens
    assert max(epoch_sizes) - min(epoch_sizes) <= 1


def test_condition_ablation_preserves_targets_and_split_membership(tmp_path: Path):
    base_path = tmp_path / "base.pt"
    base_vocab = tmp_path / "base.vocab.json"
    derived_path = tmp_path / "without_pcset.pt"
    derived_vocab = tmp_path / "without_pcset.vocab.json"
    generate_samples(
        num_samples=0,
        output=base_path,
        vocab_output=base_vocab,
        seed=21,
        min_measures=4,
        max_measures=4,
        min_voices=2,
        max_voices=2,
        train_samples=3,
        val_samples=2,
        test_samples=1,
    )
    derive_corpus(base_path, derived_path, derived_vocab, "no_pcset")
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    derived = torch.load(derived_path, map_location="cpu", weights_only=False)
    assert derived["split_counts"] == base["split_counts"]
    for source_sample, ablated_sample in zip(base["samples"], derived["samples"]):
        assert ablated_sample["id"] == source_sample["id"]
        assert ablated_sample["split"] == source_sample["split"]
        assert ablated_sample["events"] == source_sample["events"]
        assert ablated_sample["metadata"]["pcset"] == []
        assert ablated_sample["metadata"]["interval_vector"] is None

    tokenizer = ScoreTokenizer.load(base_vocab)
    view = PostTonalDataset(
        base_path,
        max_seq_len=256,
        split="train",
        sep_id=tokenizer.token_to_id["SEP"],
        sequence_mode="rotating",
        target_tokens_per_window=128,
        tokenizer=tokenizer,
        condition_ablation="no_pcset",
    )
    assert view.samples[0]["metadata"]["pcset"] == []
    assert view.samples[0]["_target_metadata"]["pcset"] == base["samples"][0]["metadata"]["pcset"]
    assert view.samples[0]["events"] == base["samples"][0]["events"]
    condition_ids, _ = view._condition_and_body(view.samples[0])
    condition_tokens = tokenizer.decode(condition_ids)
    assert "NO_PCSET" in condition_tokens
    assert not any(token.startswith("PC_") for token in condition_tokens)


def test_evaluation_only_density_target_is_hidden_from_conditions(tmp_path: Path):
    data_path = tmp_path / "corpus.pt"
    vocab_path = tmp_path / "vocab.json"
    generate_samples(
        num_samples=4,
        output=data_path,
        vocab_output=vocab_path,
        seed=17,
        min_measures=4,
        max_measures=4,
        min_voices=2,
        max_voices=2,
    )
    tokenizer = ScoreTokenizer.load(vocab_path)
    dataset = PostTonalDataset(
        data_path,
        max_seq_len=256,
        split="train",
        sep_id=tokenizer.token_to_id["SEP"],
        sequence_mode="rotating",
        tokenizer=tokenizer,
    )
    item = dataset[0]
    assert "target_density_curve" not in item["metadata"]
    assert item["target_metadata"]["target_density_curve"]
