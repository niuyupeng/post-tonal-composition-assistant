"""Shared utilities for configuration, reproducibility, and event handling."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml


INSTRUMENT_RANGES = {
    "piano": (21, 108),
    "flute": (60, 96),
    "clarinet": (50, 91),
    "violin": (55, 103),
    "cello": (36, 76),
    "generic_voice": (48, 84),
}

INSTRUMENTS = tuple(INSTRUMENT_RANGES.keys())


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def save_json(data: Any, path: str | os.PathLike[str]) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with open(path_obj, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def load_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(preferred: str | None = None) -> torch.device:
    requested = (preferred or os.environ.get("POST_TONAL_DEVICE") or "auto").lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda"):
        return torch.device(requested if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_pcset(text: str | Iterable[int] | None) -> list[int]:
    if text is None or text == "":
        return []
    if isinstance(text, str):
        return sorted({int(part.strip()) % 12 for part in text.split(",") if part.strip() != ""})
    return sorted({int(x) % 12 for x in text})


def parse_row(text: str | Iterable[int] | None) -> list[int] | None:
    if text is None or text == "":
        return None
    if isinstance(text, str):
        if text.lower() == "random":
            return None
        values = [int(part.strip()) % 12 for part in text.split(",") if part.strip() != ""]
    else:
        values = [int(x) % 12 for x in text]
    if len(values) != 12:
        raise ValueError("A twelve-tone row must contain exactly 12 pitch classes.")
    return values


def event_pitch_classes(events: Iterable[dict[str, Any]]) -> list[int]:
    pcs: list[int] = []
    for event in events:
        if not event.get("is_rest", False):
            if "pc" in event:
                pcs.append(int(event["pc"]) % 12)
            elif "pitch" in event and event["pitch"] is not None:
                pcs.append(int(event["pitch"]) % 12)
    return pcs


def instrument_range_violation_rate(events: Iterable[dict[str, Any]]) -> float:
    total = 0
    violations = 0
    for event in events:
        if event.get("is_rest", False) or event.get("pitch") is None:
            continue
        total += 1
        lo, hi = INSTRUMENT_RANGES.get(str(event.get("instrument", "generic_voice")), INSTRUMENT_RANGES["generic_voice"])
        pitch = int(event["pitch"])
        if pitch < lo or pitch > hi:
            violations += 1
    return 0.0 if total == 0 else violations / total


def quantize_duration(duration: float, quantum: float = 0.25) -> float:
    return max(quantum, round(float(duration) / quantum) * quantum)


def ticks(value: float, quantum: float = 0.25) -> int:
    return int(round(float(value) / quantum))
