"""Prepare a blind expert-evaluation package for Project 2."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from post_tonal.export_musicxml import export_musicxml
from post_tonal.models.rule_generator import DEFAULT_PCSETS, RuleGenerator
from post_tonal.theory.analysis_report import analyze_events
from post_tonal.theory.gesture import GESTURE_LABELS
from post_tonal.theory.pcset import interval_vector
from post_tonal.theory.rhythm_profile import RHYTHM_PROFILES
from post_tonal.theory.serial import generate_twelve_tone_row
from post_tonal.utils import INSTRUMENTS, ensure_dir, save_json


RATING_DIMENSIONS = [
    "post-tonal material coherence",
    "pc-set / interval consistency",
    "serial logic clarity",
    "rhythmic-profile control",
    "gesture recognizability",
    "notational usefulness",
    "usefulness for contemporary composition sketching",
]


def _metadata(idx: int) -> dict:
    pcset = DEFAULT_PCSETS[idx % len(DEFAULT_PCSETS)]
    row = generate_twelve_tone_row(seed=1000 + idx) if idx % 2 == 0 else None
    forms = ["P0", "R0", "I0", "RI0", "P5", "I7"]
    return {
        "pcset": pcset,
        "interval_vector": interval_vector(pcset),
        "row": row,
        "row_form": forms[idx % len(forms)] if row else None,
        "rhythm_profile": RHYTHM_PROFILES[idx % len(RHYTHM_PROFILES)],
        "gesture": GESTURE_LABELS[idx % len(GESTURE_LABELS)],
        "voices": 2 + (idx % 5),
        "measures": 4 + (idx % 5),
        "instrument": INSTRUMENTS[idx % len(INSTRUMENTS)],
    }


def write_forms(root: Path, count: int) -> None:
    md_lines = [
        "# Blind Rating Form: Project 2",
        "",
        "Use a 1-7 scale for each dimension, where 1 = very weak and 7 = very strong.",
        "Do not rate author identity or style imitation; this package evaluates symbolic score sketches only.",
        "",
    ]
    for idx in range(1, count + 1):
        md_lines.append(f"## Example {idx:02d}")
        for dimension in RATING_DIMENSIONS:
            md_lines.append(f"- {dimension}: ___ / 7")
        md_lines.append("- comments: ")
        md_lines.append("")
    (root / "blind_rating_form_project2.md").write_text("\n".join(md_lines), encoding="utf-8")

    csv_path = root / "blind_rating_form_project2.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["example_id"] + [dimension.replace(" ", "_").replace("/", "slash") for dimension in RATING_DIMENSIONS] + ["comments"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(1, count + 1):
            row = {"example_id": f"project2_{idx:02d}"}
            for field in fieldnames[1:]:
                row[field] = ""
            writer.writerow(row)


def prepare_expert_eval(output_dir: str | Path = "expert_eval/project2", count: int = 20, seed: int = 2026) -> dict[str, int]:
    root = ensure_dir(output_dir)
    examples_dir = ensure_dir(root / "musicxml")
    reports_dir = ensure_dir(root / "analysis_reports")
    manifest = []
    for idx in range(count):
        metadata = _metadata(idx)
        events = RuleGenerator(seed=seed + idx).generate(metadata)
        report = analyze_events(events, metadata)
        example_id = f"project2_{idx + 1:02d}"
        musicxml_path = examples_dir / f"{example_id}.musicxml"
        report_path = reports_dir / f"{example_id}.json"
        export_musicxml(events, musicxml_path, metadata)
        save_json({"metadata": metadata, "analysis": report}, report_path)
        manifest.append({"example_id": example_id, "musicxml": str(musicxml_path), "analysis_report": str(report_path)})
    save_json(manifest, root / "manifest.json")
    write_forms(root, count)
    return {"examples": count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="expert_eval/project2")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(prepare_expert_eval(args.output_dir, args.count, args.seed))


if __name__ == "__main__":
    main()
