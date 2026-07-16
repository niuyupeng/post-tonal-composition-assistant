"""Gesture labels and lightweight consistency metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np


GESTURE_LABELS = (
    "pointillistic",
    "sustained",
    "fragmented",
    "registral_expansion",
    "cluster_like",
    "silence_gap",
    "rhythmic_burst",
)


def compute_gesture_features(
    events: Iterable[dict],
    total_beats: float | None = None,
    voice_count: int | None = None,
) -> dict[str, float]:
    material = list(events)
    notes = [event for event in material if not event.get("is_rest", False) and event.get("pitch") is not None]
    rests = [event for event in material if event.get("is_rest", False)]
    if total_beats is None:
        total_beats = 0.0
        for event in material:
            total_beats = max(total_beats, float(event.get("onset", 0.0)) + float(event.get("duration", 0.0)))
    total_beats = max(float(total_beats or 0.0), 1e-6)
    if voice_count is None:
        voice_count = max([int(event.get("voice", 0)) for event in material], default=0) + 1
    voice_count = max(1, int(voice_count))

    onsets = np.array([float(event.get("onset", 0.0)) for event in notes], dtype=np.float32)
    pitches = np.array([float(event.get("pitch", 60)) for event in notes], dtype=np.float32)
    durations = np.array([float(event.get("duration", 0.25)) for event in notes], dtype=np.float32)
    rest_coverage = 0.0
    for voice in range(voice_count):
        intervals = sorted(
            (
                max(0.0, float(event.get("onset", 0.0))),
                min(
                    total_beats,
                    max(0.0, float(event.get("onset", 0.0)))
                    + max(0.0, float(event.get("duration", 0.0))),
                ),
            )
            for event in rests
            if int(event.get("voice", 0)) == voice
        )
        merged_end = 0.0
        for start, end in intervals:
            if end <= start:
                continue
            if start >= merged_end:
                rest_coverage += end - start
                merged_end = end
            elif end > merged_end:
                rest_coverage += end - merged_end
                merged_end = end

    return {
        "onset_dispersion": float(np.std(onsets % 4.0)) if len(onsets) else 0.0,
        "register_spread": float(np.max(pitches) - np.min(pitches)) if len(pitches) else 0.0,
        "note_density": float(len(notes) / (total_beats * voice_count)),
        "average_duration": float(np.mean(durations)) if len(durations) else 0.0,
        "rest_ratio": min(1.0, rest_coverage / (total_beats * voice_count)),
    }


def _closeness(value: float, target: float, tolerance: float) -> float:
    return max(0.0, 1.0 - abs(value - target) / max(tolerance, 1e-6))


def gesture_consistency_score(
    events: Iterable[dict],
    gesture_label: str,
    total_beats: float | None = None,
    voice_count: int | None = None,
) -> float:
    if gesture_label not in GESTURE_LABELS:
        raise ValueError(f"Unknown gesture label: {gesture_label}")
    features = compute_gesture_features(events, total_beats=total_beats, voice_count=voice_count)

    if gesture_label == "pointillistic":
        scores = [
            _closeness(features["average_duration"], 0.3, 0.5),
            _closeness(features["note_density"], 0.8, 0.8),
            _closeness(features["rest_ratio"], 0.35, 0.35),
        ]
    elif gesture_label == "sustained":
        scores = [
            _closeness(features["average_duration"], 2.5, 2.0),
            _closeness(features["note_density"], 0.25, 0.4),
            _closeness(features["rest_ratio"], 0.15, 0.25),
        ]
    elif gesture_label == "fragmented":
        scores = [
            _closeness(features["average_duration"], 0.6, 0.6),
            _closeness(features["rest_ratio"], 0.45, 0.35),
            _closeness(features["onset_dispersion"], 1.0, 1.0),
        ]
    elif gesture_label == "registral_expansion":
        scores = [
            _closeness(features["register_spread"], 36.0, 24.0),
            _closeness(features["note_density"], 0.7, 0.8),
        ]
    elif gesture_label == "cluster_like":
        scores = [
            _closeness(features["register_spread"], 12.0, 12.0),
            _closeness(features["note_density"], 1.2, 1.0),
        ]
    elif gesture_label == "silence_gap":
        scores = [
            _closeness(features["rest_ratio"], 0.6, 0.35),
            _closeness(features["note_density"], 0.35, 0.5),
        ]
    else:  # rhythmic_burst
        scores = [
            _closeness(features["note_density"], 1.6, 1.2),
            _closeness(features["average_duration"], 0.35, 0.45),
        ]
    return float(sum(scores) / len(scores))
