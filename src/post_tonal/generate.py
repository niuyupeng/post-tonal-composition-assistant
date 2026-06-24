"""Generate constrained post-tonal MusicXML fragments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from post_tonal.export_musicxml import export_musicxml
from post_tonal.models.rule_generator import RuleGenerator
from post_tonal.theory.analysis_report import analysis_text, analyze_events
from post_tonal.theory.pcset import interval_vector
from post_tonal.theory.serial import generate_twelve_tone_row, is_valid_row
from post_tonal.utils import INSTRUMENTS, ensure_dir, parse_pcset, parse_row, save_json


def candidate_loss(report: dict[str, Any], weights: dict[str, float]) -> float:
    loss = 0.0
    loss += weights.get("pcset", 1.0) * (1.0 - float(report.get("pcset_coverage") or 0.0))
    iv_distance = report.get("interval_vector_distance")
    if iv_distance is not None:
        loss += weights.get("interval_vector", 0.05) * float(iv_distance)
    row_acc = report.get("row_order_accuracy")
    if row_acc is not None:
        loss += weights.get("row_order", 1.0) * (1.0 - float(row_acc))
    loss += weights.get("aggregate", 0.5) * (1.0 - float(report.get("aggregate_completion_rate") or 0.0))
    loss += weights.get("rhythm", 0.5) * float(report.get("rhythmic_profile_distance") or 0.0)
    loss += weights.get("gesture", 0.5) * (1.0 - float(report.get("gesture_consistency_score") or 0.0))
    loss += weights.get("range", 2.0) * float(report.get("instrument_range_violation_rate") or 0.0)
    return loss


def generate_fragment(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    pcset = parse_pcset(args.pcset)
    row = parse_row(args.row)
    if args.row == "random" or (args.row_form and row is None):
        row = generate_twelve_tone_row(seed=args.seed)
    if row is not None and not is_valid_row(row):
        raise ValueError("Invalid twelve-tone row.")
    metadata: dict[str, Any] = {
        "pcset": pcset,
        "interval_vector": interval_vector(pcset) if pcset else None,
        "row": row,
        "row_form": args.row_form,
        "rhythm_profile": args.rhythm_profile,
        "gesture": args.gesture,
        "voices": args.voices,
        "measures": args.measures,
        "instrument": args.instrument,
    }
    weights = {
        "pcset": args.pcset_penalty,
        "interval_vector": args.interval_vector_penalty,
        "row_order": args.row_order_penalty,
        "aggregate": args.aggregate_penalty,
        "rhythm": args.rhythm_penalty,
        "gesture": args.gesture_penalty,
        "range": args.range_penalty,
    }
    best_events: list[dict[str, Any]] | None = None
    best_report: dict[str, Any] | None = None
    best_loss = float("inf")
    for attempt in range(max(1, args.attempts)):
        generator = RuleGenerator(seed=args.seed + attempt)
        events = generator.generate(dict(metadata))
        report = analyze_events(events, metadata)
        loss = candidate_loss(report, weights)
        if loss < best_loss:
            best_events, best_report, best_loss = events, report, loss
    assert best_events is not None and best_report is not None
    best_report["decoding_penalty_loss"] = best_loss
    return best_events, metadata, best_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcset", default="0,1,4,6")
    parser.add_argument("--row", default=None, help="'random' or comma-separated 12 pcs")
    parser.add_argument("--row_form", default=None, help="P0/R0/I0/RI0 etc.")
    parser.add_argument("--rhythm_profile", default="medium")
    parser.add_argument("--gesture", default="fragmented")
    parser.add_argument("--voices", type=int, default=4)
    parser.add_argument("--measures", type=int, default=8)
    parser.add_argument("--instrument", choices=INSTRUMENTS, default="generic_voice")
    parser.add_argument("--output", default="generated_scores/generated.musicxml")
    parser.add_argument("--report", default="results/generated_report.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--pcset-penalty", type=float, default=1.0)
    parser.add_argument("--interval-vector-penalty", type=float, default=0.05)
    parser.add_argument("--row-order-penalty", type=float, default=1.0)
    parser.add_argument("--aggregate-penalty", type=float, default=0.5)
    parser.add_argument("--rhythm-penalty", type=float, default=0.5)
    parser.add_argument("--gesture-penalty", type=float, default=0.5)
    parser.add_argument("--range-penalty", type=float, default=2.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    events, metadata, report = generate_fragment(args)
    export_musicxml(events, args.output, metadata)
    ensure_dir(Path(args.report).parent)
    save_json({"metadata": metadata, "analysis": report, "text": analysis_text(report)}, args.report)
    print(analysis_text(report))
    print(f"MusicXML: {args.output}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
