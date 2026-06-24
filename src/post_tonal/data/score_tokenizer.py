"""Simple event tokenizer for score-level post-tonal fragments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from post_tonal.theory.gesture import GESTURE_LABELS
from post_tonal.theory.rhythm_profile import RHYTHM_PROFILES
from post_tonal.utils import INSTRUMENTS, load_json, save_json, ticks


ROW_FORM_TOKENS = [f"{form}{n}" for form in ("P", "R", "I", "RI") for n in range(12)]


class ScoreTokenizer:
    """A compact, event-based symbolic tokenizer.

    Conditions are represented as prefix tokens. Musical events are encoded with
    time shifts, voice ids, pitch class/octave or rest, and duration ticks.
    """

    def __init__(self, vocab: dict[str, int] | None = None) -> None:
        self.token_to_id = vocab or self.default_vocab()
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}

    @staticmethod
    def default_vocab() -> dict[str, int]:
        tokens: list[str] = [
            "PAD",
            "BOS",
            "EOS",
            "SEP",
            "BAR",
            "REST",
            "NO_ROW",
            "NO_PCSET",
        ]
        tokens += [f"TIME_SHIFT_{i}" for i in range(65)]
        tokens += [f"VOICE_{i}" for i in range(8)]
        tokens += [f"PITCH_{i}" for i in range(12)]
        tokens += [f"OCTAVE_{i}" for i in range(10)]
        tokens += [f"DUR_{i}" for i in range(1, 33)]
        tokens += [f"PC_{i}" for i in range(12)]
        tokens += [f"ROWPC_{i}" for i in range(12)]
        tokens += [f"IV_{slot}_{value}" for slot in range(6) for value in range(33)]
        tokens += [f"PROFILE_{label}" for label in RHYTHM_PROFILES]
        tokens += [f"GESTURE_{label}" for label in GESTURE_LABELS]
        tokens += [f"ROWFORM_{label}" for label in ROW_FORM_TOKENS]
        tokens += [f"VOICES_{i}" for i in range(1, 9)]
        tokens += [f"MEASURES_{i}" for i in range(1, 17)]
        tokens += [f"INSTRUMENT_{name}" for name in INSTRUMENTS]
        return {token: idx for idx, token in enumerate(tokens)}

    @property
    def pad_id(self) -> int:
        return self.token_to_id["PAD"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["BOS"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["EOS"]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def condition_tokens(self, metadata: dict[str, Any]) -> list[str]:
        tokens: list[str] = ["BOS"]
        pcset = metadata.get("pcset") or []
        if pcset:
            tokens.extend(f"PC_{int(pc) % 12}" for pc in pcset)
        else:
            tokens.append("NO_PCSET")
        iv = metadata.get("interval_vector")
        if iv:
            for slot, value in enumerate(iv[:6]):
                tokens.append(f"IV_{slot}_{min(32, int(value))}")
        row = metadata.get("row")
        if row:
            tokens.extend(f"ROWPC_{int(pc) % 12}" for pc in row)
        else:
            tokens.append("NO_ROW")
        row_form = metadata.get("row_form")
        if row_form:
            tokens.append(f"ROWFORM_{row_form}")
        rhythm_profile = metadata.get("rhythm_profile", "medium")
        gesture = metadata.get("gesture", "fragmented")
        instrument = metadata.get("instrument", "generic_voice")
        tokens.extend(
            [
                f"PROFILE_{rhythm_profile}",
                f"GESTURE_{gesture}",
                f"VOICES_{min(8, max(1, int(metadata.get('voices', 1))))}",
                f"MEASURES_{min(16, max(1, int(metadata.get('measures', 4))))}",
                f"INSTRUMENT_{instrument}",
                "SEP",
            ]
        )
        return tokens

    def events_to_tokens(self, events: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> list[str]:
        metadata = metadata or {}
        tokens = self.condition_tokens(metadata)
        events_sorted = sorted(events, key=lambda e: (float(e.get("onset", 0.0)), int(e.get("voice", 0)), float(e.get("duration", 0.0))))
        previous_onset = 0.0
        current_measure = -1
        for event in events_sorted:
            onset = float(event.get("onset", 0.0))
            measure = int(onset // 4.0)
            while current_measure < measure:
                tokens.append("BAR")
                current_measure += 1
            shift = min(64, max(0, ticks(onset - previous_onset)))
            tokens.append(f"TIME_SHIFT_{shift}")
            tokens.append(f"VOICE_{min(7, max(0, int(event.get('voice', 0))))}")
            dur_tick = min(32, max(1, ticks(float(event.get("duration", 0.25)))))
            if event.get("is_rest", False):
                tokens.append("REST")
                tokens.append(f"DUR_{dur_tick}")
            else:
                pc = int(event.get("pc", int(event.get("pitch", 60)) % 12)) % 12
                pitch = int(event.get("pitch", 60))
                octave = min(9, max(0, int(pitch // 12) - 1))
                tokens.append(f"PITCH_{pc}")
                tokens.append(f"OCTAVE_{octave}")
                tokens.append(f"DUR_{dur_tick}")
            previous_onset = onset
        tokens.append("EOS")
        return tokens

    def tokens_to_events(self, tokens: list[str]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        in_body = False
        onset = 0.0
        voice = 0
        pending_pc: int | None = None
        pending_octave: int | None = None
        pending_rest = False
        for token in tokens:
            if token == "SEP":
                in_body = True
                continue
            if not in_body:
                continue
            if token == "EOS":
                break
            if token.startswith("TIME_SHIFT_"):
                onset += int(token.rsplit("_", 1)[1]) * 0.25
            elif token.startswith("VOICE_"):
                voice = int(token.rsplit("_", 1)[1])
            elif token == "REST":
                pending_rest = True
                pending_pc = None
                pending_octave = None
            elif token.startswith("PITCH_"):
                pending_pc = int(token.rsplit("_", 1)[1])
                pending_rest = False
            elif token.startswith("OCTAVE_"):
                pending_octave = int(token.rsplit("_", 1)[1])
            elif token.startswith("DUR_"):
                duration = int(token.rsplit("_", 1)[1]) * 0.25
                if pending_rest:
                    events.append({"onset": onset, "duration": duration, "voice": voice, "is_rest": True})
                elif pending_pc is not None and pending_octave is not None:
                    pitch = (pending_octave + 1) * 12 + pending_pc
                    events.append(
                        {
                            "onset": onset,
                            "duration": duration,
                            "voice": voice,
                            "is_rest": False,
                            "pitch": pitch,
                            "pc": pending_pc,
                        }
                    )
                pending_pc = None
                pending_octave = None
                pending_rest = False
        return events

    def encode(self, tokens: list[str]) -> list[int]:
        missing = [token for token in tokens if token not in self.token_to_id]
        if missing:
            raise KeyError(f"Unknown tokens: {missing[:8]}")
        return [self.token_to_id[token] for token in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token[int(idx)] for idx in ids if int(idx) in self.id_to_token]

    def save(self, path: str | Path) -> None:
        save_json(self.token_to_id, path)

    @classmethod
    def load(cls, path: str | Path) -> "ScoreTokenizer":
        return cls({str(k): int(v) for k, v in load_json(path).items()})
