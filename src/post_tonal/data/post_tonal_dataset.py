"""Torch dataset for generated post-tonal score fragments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, random_split


class PostTonalDataset(Dataset):
    def __init__(self, data_path: str | Path, max_seq_len: int | None = None, split: str | None = None) -> None:
        data = torch.load(data_path, map_location="cpu", weights_only=False)
        samples: list[dict[str, Any]] = data["samples"]
        if split is not None:
            samples = [sample for sample in samples if sample.get("split") == split]
        self.samples = samples
        self.max_seq_len = max_seq_len
        self.split = split
        self.split_counts = data.get("split_counts", {})
        if split is not None and not self.samples:
            raise ValueError(f"No samples found for split {split!r} in {data_path}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        ids = list(sample["token_ids"])
        if self.max_seq_len is not None:
            ids = ids[: self.max_seq_len]
            if ids:
                ids[-1] = sample.get("eos_id", ids[-1])
        if len(ids) < 2:
            raise ValueError("Token sequence must contain at least two ids.")
        return {
            "input_ids": torch.tensor(ids[:-1], dtype=torch.long),
            "labels": torch.tensor(ids[1:], dtype=torch.long),
            "metadata": sample.get("metadata", {}),
            "events": sample.get("events", []),
        }


def collate_batch(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for row, item in enumerate(batch):
        length = item["input_ids"].shape[0]
        input_ids[row, :length] = item["input_ids"]
        labels[row, :length] = item["labels"]
        attention_mask[row, :length] = True
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "metadata": [item["metadata"] for item in batch],
        "events": [item["events"] for item in batch],
    }


def train_val_split(dataset: Dataset, val_fraction: float, seed: int) -> tuple[Dataset, Dataset]:
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = max(1, len(dataset) - val_size)
    if train_size + val_size > len(dataset):
        val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def describe_splits(data_path: str | Path) -> dict[str, int]:
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    if "split_counts" in data:
        return {str(k): int(v) for k, v in data["split_counts"].items()}
    counts = {"train": 0, "val": 0, "test": 0}
    for sample in data.get("samples", []):
        split = str(sample.get("split", "train"))
        counts[split] = counts.get(split, 0) + 1
    return counts
