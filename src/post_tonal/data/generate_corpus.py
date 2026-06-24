"""Synthetic legal post-tonal corpus generator."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch

from post_tonal.data.score_tokenizer import ScoreTokenizer
from post_tonal.export_musicxml import export_musicxml
from post_tonal.models.rule_generator import DEFAULT_PCSETS, RuleGenerator
from post_tonal.theory.gesture import GESTURE_LABELS
from post_tonal.theory.pcset import interval_vector
from post_tonal.theory.rhythm_profile import RHYTHM_PROFILES
from post_tonal.theory.serial import generate_twelve_tone_row
from post_tonal.utils import INSTRUMENTS, ensure_dir


def make_metadata(
    rng: random.Random,
    min_measures: int,
    max_measures: int,
    min_voices: int,
    max_voices: int,
    focus: str | None = None,
    ablation: str | None = None,
) -> dict[str, Any]:
    pcset = rng.choice(DEFAULT_PCSETS)
    use_row = focus == "serial" or ablation == "serial_only" or rng.random() < 0.45
    row = generate_twelve_tone_row(rng=rng) if use_row else None
    form = rng.choice(["P", "R", "I", "RI"])
    row_form = f"{form}{rng.randrange(12)}" if use_row else None
    rhythm_profile = rng.choice(RHYTHM_PROFILES)
    gesture = rng.choice(GESTURE_LABELS)
    if focus == "gesture":
        gesture = rng.choice(GESTURE_LABELS)
    strip_pcset = ablation in {"no_constraints", "serial_only", "rhythm_only", "gesture_only", "no_pcset"}
    strip_serial = ablation in {"no_constraints", "pcset_only", "rhythm_only", "gesture_only", "no_serial"}
    strip_rhythm = ablation in {"no_constraints", "pcset_only", "serial_only", "gesture_only", "no_rhythm"}
    strip_gesture = ablation in {"no_constraints", "pcset_only", "serial_only", "rhythm_only", "no_gesture"}
    metadata: dict[str, Any] = {
        "pcset": [] if strip_pcset else pcset,
        "interval_vector": None if strip_pcset else interval_vector(pcset),
        "row": None if strip_serial else row,
        "row_form": None if strip_serial else row_form,
        "rhythm_profile": "medium" if strip_rhythm else rhythm_profile,
        "gesture": "fragmented" if strip_gesture else gesture,
        "voices": rng.randint(min_voices, max_voices),
        "measures": rng.randint(min_measures, max_measures),
        "instrument": rng.choice(INSTRUMENTS),
    }
    return metadata


def generate_samples(
    num_samples: int,
    output: str | Path,
    vocab_output: str | Path,
    seed: int = 0,
    min_measures: int = 4,
    max_measures: int = 16,
    min_voices: int = 2,
    max_voices: int = 8,
    export_musicxml_flag: bool = False,
    musicxml_dir: str | Path = "data/generated",
    musicxml_limit: int = 12,
    focus: str | None = None,
    ablation: str | None = None,
    train_samples: int | None = None,
    val_samples: int | None = None,
    test_samples: int | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    tokenizer = ScoreTokenizer()
    generator = RuleGenerator(seed=seed)
    samples: list[dict[str, Any]] = []
    ensure_dir(Path(output).parent)
    ensure_dir(Path(vocab_output).parent)
    if export_musicxml_flag:
        ensure_dir(musicxml_dir)

    if train_samples is not None or val_samples is not None or test_samples is not None:
        split_plan = (
            ["train"] * int(train_samples or 0)
            + ["val"] * int(val_samples or 0)
            + ["test"] * int(test_samples or 0)
        )
        num_samples = len(split_plan)
    else:
        split_plan = []
        train_cut = int(num_samples * 0.8)
        val_cut = int(num_samples * 0.9)
        for idx in range(num_samples):
            split_plan.append("train" if idx < train_cut else "val" if idx < val_cut else "test")

    for idx in range(num_samples):
        metadata = make_metadata(rng, min_measures, max_measures, min_voices, max_voices, focus=focus, ablation=ablation)
        events = generator.generate(metadata)
        tokens = tokenizer.events_to_tokens(events, metadata)
        token_ids = tokenizer.encode(tokens)
        sample = {
            "id": f"synthetic_{idx:06d}",
            "token_ids": token_ids,
            "eos_id": tokenizer.eos_id,
            "metadata": metadata,
            "events": events,
            "split": split_plan[idx],
        }
        samples.append(sample)
        if export_musicxml_flag and idx < musicxml_limit:
            export_musicxml(events, Path(musicxml_dir) / f"{sample['id']}.musicxml", metadata)

    split_counts = {split: sum(1 for sample in samples if sample["split"] == split) for split in ("train", "val", "test")}
    torch.save(
        {
            "format": "post_tonal_synthetic_v2",
            "samples": samples,
            "vocab_size": tokenizer.vocab_size,
            "split_counts": split_counts,
            "seed": seed,
        },
        output,
    )
    tokenizer.save(vocab_output)
    return samples


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--output", type=str, default="data/processed/post_tonal.pt")
    parser.add_argument("--vocab-output", type=str, default="data/processed/post_tonal.vocab.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-measures", type=int, default=4)
    parser.add_argument("--max-measures", type=int, default=16)
    parser.add_argument("--min-voices", type=int, default=2)
    parser.add_argument("--max-voices", type=int, default=8)
    parser.add_argument("--export-musicxml", action="store_true")
    parser.add_argument("--musicxml-dir", type=str, default="data/generated")
    parser.add_argument("--musicxml-limit", type=int, default=12)
    parser.add_argument("--focus", choices=["serial", "gesture"], default=None)
    parser.add_argument(
        "--ablation",
        choices=[
            "no_constraints",
            "serial_only",
            "pcset_only",
            "rhythm_only",
            "gesture_only",
            "no_pcset",
            "no_serial",
            "no_rhythm",
            "no_gesture",
        ],
        default=None,
    )
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    generate_samples(
        num_samples=args.num_samples,
        output=args.output,
        vocab_output=args.vocab_output,
        seed=args.seed,
        min_measures=args.min_measures,
        max_measures=args.max_measures,
        min_voices=args.min_voices,
        max_voices=args.max_voices,
        export_musicxml_flag=args.export_musicxml,
        musicxml_dir=args.musicxml_dir,
        musicxml_limit=args.musicxml_limit,
        focus=args.focus,
        ablation=args.ablation,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
    )


if __name__ == "__main__":
    main()
