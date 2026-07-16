"""Torch dataset for generated post-tonal score fragments."""

from __future__ import annotations

import math
import zlib
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, random_split

from post_tonal.data.conditions import apply_condition_ablation
from post_tonal.data.score_tokenizer import ScoreTokenizer


class PostTonalDataset(Dataset):
    """Dataset with condition-preserving windows over complete score sequences.

    ``coverage_cycle`` partitions all score-body windows across a fixed number
    of epochs, guaranteeing complete token coverage once per cycle. ``rotating``
    exposes one window per sample and is retained for small experiments. ``all``
    exposes every body window for validation and teacher-forced evaluation.
    ``truncate`` is retained only for legacy checkpoints and tests.
    """

    def __init__(
        self,
        data_path: str | Path,
        max_seq_len: int | None = None,
        split: str | None = None,
        sep_id: int | None = None,
        sequence_mode: str = "truncate",
        target_tokens_per_window: int = 128,
        seed: int = 0,
        tokenizer: ScoreTokenizer | None = None,
        condition_ablation: str | None = None,
        coverage_cycle_epochs: int = 10,
    ) -> None:
        data = torch.load(data_path, map_location="cpu", weights_only=False)
        samples: list[dict[str, Any]] = data["samples"]
        if split is not None:
            samples = [sample for sample in samples if sample.get("split") == split]
        if condition_ablation and tokenizer is None:
            raise ValueError("Condition ablations require a tokenizer.")
        sample_views: list[dict[str, Any]] = []
        for sample in samples:
            target_metadata = sample.get("target_metadata", sample.get("metadata", {}))
            metadata = apply_condition_ablation(target_metadata, condition_ablation)
            view = {
                **sample,
                "metadata": metadata,
                "_target_metadata": target_metadata,
            }
            if tokenizer is not None:
                condition_ids = tokenizer.encode(tokenizer.condition_tokens(metadata))
                view["_condition_token_ids"] = condition_ids
            sample_views.append(view)
        samples = sample_views
        self.samples = samples
        self.max_seq_len = max_seq_len
        self.split = split
        self.sep_id = sep_id
        self.sequence_mode = sequence_mode
        self.target_tokens_per_window = max(1, int(target_tokens_per_window))
        self.seed = int(seed)
        self.condition_ablation = condition_ablation
        self.coverage_cycle_epochs = max(1, int(coverage_cycle_epochs))
        self.epoch = 0
        self.split_counts = data.get("split_counts", {})
        self.format = data.get("format")
        if split is not None and not self.samples:
            raise ValueError(f"No samples found for split {split!r} in {data_path}.")
        if sequence_mode not in {"truncate", "rotating", "coverage_cycle", "all"}:
            raise ValueError(f"Unknown sequence mode: {sequence_mode!r}")
        if sequence_mode in {"rotating", "coverage_cycle", "all"} and (max_seq_len is None or sep_id is None):
            raise ValueError("Windowed sequence modes require max_seq_len and sep_id.")
        self._all_windows: list[tuple[int, int]] = []
        self._epoch_windows: list[tuple[int, int]] = []
        if sequence_mode in {"coverage_cycle", "all"}:
            for sample_index, sample in enumerate(self.samples):
                for window_index in range(self._window_count(sample)):
                    self._all_windows.append((sample_index, window_index))
        if sequence_mode == "coverage_cycle":
            self._set_coverage_epoch(0)

    def __len__(self) -> int:
        if self.sequence_mode == "all":
            return len(self._all_windows)
        if self.sequence_mode == "coverage_cycle":
            return len(self._epoch_windows)
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(0, int(epoch))
        if self.sequence_mode == "coverage_cycle":
            self._set_coverage_epoch(self.epoch)

    def _set_coverage_epoch(self, epoch: int) -> None:
        partition = int(epoch) % self.coverage_cycle_epochs
        self._epoch_windows = self._all_windows[partition :: self.coverage_cycle_epochs]

    def _condition_and_body(self, sample: dict[str, Any]) -> tuple[list[int], list[int]]:
        ids = list(sample["token_ids"])
        if self.sep_id is None:
            return [], ids
        try:
            sep_index = ids.index(self.sep_id)
        except ValueError as exc:
            raise ValueError(f"Sample {sample.get('id')} has no SEP token.") from exc
        condition = list(sample.get("_condition_token_ids", ids[: sep_index + 1]))
        return condition, ids[sep_index + 1 :]

    def _effective_target_size(self, condition_length: int) -> int:
        assert self.max_seq_len is not None
        total_capacity = self.max_seq_len + 1
        available = total_capacity - condition_length
        if available < 2:
            raise ValueError(
                f"Condition prefix length {condition_length} leaves no room in "
                f"a {self.max_seq_len}-token model context."
            )
        return min(self.target_tokens_per_window, available)

    def _window_count(self, sample: dict[str, Any]) -> int:
        condition, body = self._condition_and_body(sample)
        if self.sequence_mode == "truncate" or self.max_seq_len is None:
            return 1
        target_size = self._effective_target_size(len(condition))
        return max(1, math.ceil(len(body) / target_size))

    def _rotating_window_index(self, sample: dict[str, Any]) -> int:
        count = self._window_count(sample)
        if count <= 1:
            return 0
        sample_key = str(sample.get("id", ""))
        offset = zlib.crc32(sample_key.encode("utf-8"), self.seed) % count
        return int((offset + self.epoch) % count)

    def _build_window(
        self,
        sample: dict[str, Any],
        window_index: int,
    ) -> tuple[list[int], list[int], int]:
        condition, body = self._condition_and_body(sample)
        assert self.max_seq_len is not None
        total_capacity = self.max_seq_len + 1
        target_size = self._effective_target_size(len(condition))
        target_start = window_index * target_size
        target = body[target_start : target_start + target_size]
        context_capacity = max(0, total_capacity - len(condition) - len(target))
        context_start = max(0, target_start - context_capacity)
        context = body[context_start:target_start]
        ids = condition + context + target
        if len(ids) < 2:
            raise ValueError("Token sequence window must contain at least two ids.")
        labels = ids[1:]
        target_position = len(condition) + len(context)
        first_target_label = max(0, target_position - 1)
        labels = [-100] * first_target_label + labels[first_target_label:]
        return ids[:-1], labels, len(target)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.sequence_mode == "all":
            sample_index, window_index = self._all_windows[idx]
        elif self.sequence_mode == "coverage_cycle":
            sample_index, window_index = self._epoch_windows[idx]
        else:
            sample_index = idx
            window_index = (
                self._rotating_window_index(self.samples[sample_index])
                if self.sequence_mode == "rotating"
                else 0
            )
        sample = self.samples[sample_index]
        if self.sequence_mode in {"rotating", "coverage_cycle", "all"}:
            input_ids, labels, target_count = self._build_window(sample, window_index)
        else:
            ids = list(sample["token_ids"])
            if self.max_seq_len is not None:
                ids = ids[: self.max_seq_len + 1]
            if len(ids) < 2:
                raise ValueError("Token sequence must contain at least two ids.")
            input_ids = ids[:-1]
            labels = ids[1:]
            target_count = len(labels)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "metadata": sample.get("metadata", {}),
            "target_metadata": sample.get(
                "_target_metadata",
                sample.get("target_metadata", sample.get("metadata", {})),
            ),
            "events": sample.get("events", []),
            "sample_id": sample.get("id"),
            "sample_index": sample_index,
            "window_index": window_index,
            "window_count": self._window_count(sample),
            "target_token_count": target_count,
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
        "target_metadata": [item["target_metadata"] for item in batch],
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
    counts = {"train": 0, "val": 0, "test": 0}
    for sample in data.get("samples", []):
        split = str(sample.get("split", "train"))
        counts[split] = counts.get(split, 0) + 1
    declared = {
        str(key): int(value)
        for key, value in data.get("split_counts", {}).items()
    }
    if declared and declared != counts:
        raise ValueError(
            f"Stored split counts do not match sample membership: "
            f"declared={declared}, actual={counts}."
        )
    return counts
