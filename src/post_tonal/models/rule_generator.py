"""Rule-based neural-symbolic baseline for constrained post-tonal sketches."""

from __future__ import annotations

import random
from typing import Any

from post_tonal.theory.pcset import normalize_pcset
from post_tonal.theory.rhythm_profile import generate_rhythmic_profile
from post_tonal.theory.serial import generate_twelve_tone_row, row_form
from post_tonal.utils import INSTRUMENT_RANGES, INSTRUMENTS, quantize_duration


DEFAULT_PCSETS = [
    [0, 1, 4],
    [0, 1, 4, 6],
    [0, 2, 3, 5],
    [0, 1, 2, 5, 6],
    [0, 2, 4, 7, 9],
    [0, 1, 5, 6, 8],
    [0, 3, 4, 7, 8, 11],
]


class RuleGenerator:
    """Generate score-level events directly from symbolic constraints."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def generate(self, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        measures = min(16, max(4, int(metadata.get("measures", 4))))
        voices = min(8, max(1, int(metadata.get("voices", 2))))
        profile = metadata.get("rhythm_profile", "medium")
        gesture = metadata.get("gesture", "fragmented")
        instrument = metadata.get("instrument", self.rng.choice(INSTRUMENTS))
        pcset = normalize_pcset(metadata.get("pcset") or self.rng.choice(DEFAULT_PCSETS))
        row = metadata.get("row")
        row_form_label = metadata.get("row_form")
        serial_sequence: list[int] | None = None
        if row_form_label:
            if not row:
                row = generate_twelve_tone_row(rng=self.rng)
                metadata["row"] = row
            serial_sequence = row_form(row, row_form_label)

        events: list[dict[str, Any]] = []
        for voice in range(voices):
            voice_events = generate_rhythmic_profile(profile, measures=measures, rng=self.rng)
            offset = self._voice_offset(gesture, voice)
            for event in voice_events:
                if self.rng.random() < self._drop_probability(gesture, profile):
                    continue
                onset = float(event["onset"]) + offset
                if onset >= measures * 4.0:
                    continue
                duration = min(float(event["duration"]), measures * 4.0 - onset)
                is_rest = bool(event["is_rest"]) or self.rng.random() < self._extra_rest_probability(gesture)
                duration = self._shape_duration(duration, gesture)
                events.append(
                    {
                        "onset": round(onset, 4),
                        "duration": round(quantize_duration(duration), 4),
                        "voice": voice,
                        "instrument": instrument,
                        "is_rest": is_rest,
                        "gesture": gesture,
                        "rhythm_profile": profile,
                    }
                )

        events.sort(key=lambda e: (float(e["onset"]), int(e["voice"])))
        note_index = 0
        note_total = max(1, sum(1 for event in events if not event.get("is_rest", False)))
        for event in events:
            if event.get("is_rest", False):
                continue
            if serial_sequence:
                pc = serial_sequence[note_index % len(serial_sequence)]
            else:
                pc = self.rng.choice(pcset)
            pitch = self._choose_pitch(pc, instrument, gesture, note_index / note_total, int(event["voice"]), voices)
            event["pc"] = int(pc) % 12
            event["pitch"] = pitch
            note_index += 1
        return events

    def _voice_offset(self, gesture: str, voice: int) -> float:
        if gesture in {"pointillistic", "fragmented", "rhythmic_burst"}:
            return (voice % 4) * 0.25
        if gesture == "cluster_like":
            return 0.0
        return (voice % 2) * 0.5

    def _drop_probability(self, gesture: str, profile: str) -> float:
        base = 0.0
        if profile in {"dense", "burst", "pointillistic"}:
            base += 0.25
        if gesture == "sustained":
            base += 0.35
        if gesture == "silence_gap":
            base += 0.45
        return min(0.75, base)

    def _extra_rest_probability(self, gesture: str) -> float:
        return {
            "silence_gap": 0.35,
            "fragmented": 0.18,
            "pointillistic": 0.15,
            "rhythmic_burst": 0.08,
            "sustained": 0.05,
            "registral_expansion": 0.05,
            "cluster_like": 0.04,
        }.get(gesture, 0.1)

    def _shape_duration(self, duration: float, gesture: str) -> float:
        if gesture == "sustained":
            return max(duration, self.rng.choice([1.5, 2.0, 3.0]))
        if gesture in {"pointillistic", "rhythmic_burst"}:
            return min(duration, self.rng.choice([0.25, 0.5]))
        if gesture == "fragmented":
            return min(duration, 1.0)
        return duration

    def _choose_pitch(self, pc: int, instrument: str, gesture: str, progress: float, voice: int, voices: int) -> int:
        lo, hi = INSTRUMENT_RANGES.get(instrument, INSTRUMENT_RANGES["generic_voice"])
        candidates = [midi for midi in range(lo, hi + 1) if midi % 12 == pc % 12]
        if not candidates:
            candidates = [60 + pc % 12]
        if gesture == "registral_expansion":
            target = lo + (hi - lo) * progress
            return min(candidates, key=lambda midi: abs(midi - target))
        if gesture == "cluster_like":
            center = 60 + (voice - voices / 2) * 2
            return min(candidates, key=lambda midi: abs(midi - center))
        if gesture == "sustained":
            center = (lo + hi) / 2 - 6
            return min(candidates, key=lambda midi: abs(midi - center))
        if gesture == "pointillistic":
            return self.rng.choice(candidates)
        center = (lo + hi) / 2 + (voice - voices / 2) * 4
        local = sorted(candidates, key=lambda midi: abs(midi - center))[: max(1, min(4, len(candidates)))]
        return self.rng.choice(local)
