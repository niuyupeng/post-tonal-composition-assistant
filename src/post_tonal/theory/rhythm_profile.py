"""Rhythmic-profile generation and metrics."""

from __future__ import annotations

import random
from typing import Iterable

import numpy as np


RHYTHM_PROFILES = (
    "sparse",
    "medium",
    "dense",
    "additive",
    "burst",
    "sustained",
    "pointillistic",
)


def _rng(rng: random.Random | None = None, seed: int | None = None) -> random.Random:
    return rng if rng is not None else random.Random(seed)


def generate_rhythmic_profile(
    profile: str,
    measures: int = 4,
    quantum: float = 0.25,
    rng: random.Random | None = None,
    seed: int | None = None,
) -> list[dict[str, float | bool]]:
    """Generate onset/duration/rest events for a rhythmic profile."""

    if profile not in RHYTHM_PROFILES:
        raise ValueError(f"Unknown rhythmic profile: {profile}")
    local_rng = _rng(rng, seed)
    total_beats = max(1, int(measures)) * 4.0
    events: list[dict[str, float | bool]] = []
    t = 0.0
    additive_pattern = [0.25, 0.5, 0.75, 1.0]
    additive_idx = 0

    while t < total_beats - 1e-9:
        if profile == "sparse":
            dur = local_rng.choice([1.0, 1.5, 2.0])
            is_rest = local_rng.random() < 0.45
        elif profile == "medium":
            dur = local_rng.choice([0.5, 0.75, 1.0])
            is_rest = local_rng.random() < 0.2
        elif profile == "dense":
            dur = local_rng.choice([0.25, 0.25, 0.5])
            is_rest = local_rng.random() < 0.08
        elif profile == "additive":
            dur = additive_pattern[additive_idx % len(additive_pattern)]
            additive_idx += 1
            is_rest = local_rng.random() < 0.18
        elif profile == "burst":
            in_burst = int(t // 2) % 3 == 0
            dur = local_rng.choice([0.25, 0.5]) if in_burst else local_rng.choice([1.0, 1.5])
            is_rest = (not in_burst) and local_rng.random() < 0.6
        elif profile == "sustained":
            dur = local_rng.choice([1.5, 2.0, 3.0, 4.0])
            is_rest = local_rng.random() < 0.15
        else:  # pointillistic
            dur = local_rng.choice([0.25, 0.25, 0.5])
            is_rest = local_rng.random() < 0.35

        dur = max(quantum, round(dur / quantum) * quantum)
        if t + dur > total_beats:
            dur = max(quantum, round((total_beats - t) / quantum) * quantum)
        events.append({"onset": round(t, 4), "duration": round(dur, 4), "is_rest": bool(is_rest)})
        t = round(t + dur, 4)

    return events


def density_curve(events: Iterable[dict], measures: int | None = None, beats_per_measure: float = 4.0) -> list[float]:
    material = list(events)
    if measures is None:
        max_end = 0.0
        for event in material:
            max_end = max(max_end, float(event.get("onset", 0.0)) + float(event.get("duration", 0.0)))
        measures = max(1, int(np.ceil(max_end / beats_per_measure)))
    curve = [0.0 for _ in range(measures)]
    for event in material:
        if event.get("is_rest", False):
            continue
        measure_idx = min(measures - 1, int(float(event.get("onset", 0.0)) // beats_per_measure))
        curve[measure_idx] += 1.0
    return curve


def rhythmic_profile_distance(
    generated_events: Iterable[dict],
    target_profile: str,
    measures: int,
    seed: int = 1234,
) -> float:
    generated = np.array(density_curve(generated_events, measures), dtype=np.float32)
    target_events = generate_rhythmic_profile(target_profile, measures=measures, seed=seed)
    target = np.array(density_curve(target_events, measures), dtype=np.float32)
    scale = max(1.0, float(target.max(initial=0.0)))
    return float(np.mean(np.abs(generated - target)) / scale)


def density_curve_error(generated_events: Iterable[dict], reference_events: Iterable[dict], measures: int) -> float:
    a = np.array(density_curve(generated_events, measures), dtype=np.float32)
    b = np.array(density_curve(reference_events, measures), dtype=np.float32)
    return float(np.mean(np.abs(a - b)))
