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
from post_tonal.utils import (
    INSTRUMENTS,
    ensure_dir,
    get_device,
    load_yaml,
    parse_pcset,
    parse_row,
    save_json,
)


def candidate_loss(report: dict[str, Any], weights: dict[str, float]) -> float:
    loss = 0.0
    if report.get("pcset"):
        loss += weights.get("pcset", 1.0) * (1.0 - float(report.get("pcset_coverage") or 0.0))
        precision = report.get("pcset_precision")
        if precision is not None:
            loss += weights.get("pcset_precision", 0.5) * (1.0 - float(precision))
    iv_distance = report.get("interval_vector_distance")
    if iv_distance is not None:
        loss += weights.get("interval_vector", 0.05) * float(iv_distance)
    row_acc = report.get("row_order_accuracy")
    if row_acc is not None:
        loss += weights.get("row_order", 1.0) * (1.0 - float(row_acc))
    transformation_acc = report.get("serial_transformation_accuracy")
    if transformation_acc is not None:
        loss += weights.get("serial_transformation", 0.5) * (1.0 - float(transformation_acc))
    if report.get("aggregate_target_applicable", False):
        loss += weights.get("aggregate", 0.5) * (1.0 - float(report.get("aggregate_completion_rate") or 0.0))
    rhythm_distance = report.get("rhythmic_profile_distance")
    if rhythm_distance is not None:
        loss += weights.get("rhythm", 0.5) * float(rhythm_distance)
    gesture_score = report.get("gesture_consistency_score")
    if gesture_score is not None:
        loss += weights.get("gesture", 0.5) * (1.0 - float(gesture_score))
    content_span = report.get("content_span_ratio")
    if content_span is not None:
        loss += weights.get("content_span", 1.0) * (1.0 - float(content_span))
    voice_adherence = report.get("voice_count_adherence")
    if voice_adherence is not None:
        loss += weights.get("voice_count", 0.5) * (1.0 - float(voice_adherence))
    loss += weights.get("range", 2.0) * float(report.get("instrument_range_violation_rate") or 0.0)
    return loss


def _metadata_from_args(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    pcset = parse_pcset(args.pcset)
    row = parse_row(args.row)
    if args.row == "random" or (args.row_form and row is None):
        row = generate_twelve_tone_row(seed=seed)
    if row is not None and not is_valid_row(row):
        raise ValueError("Invalid twelve-tone row.")
    return {
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


def _penalty_weights(args: argparse.Namespace) -> dict[str, float]:
    return {
        "pcset": args.pcset_penalty,
        "pcset_precision": args.pcset_precision_penalty,
        "interval_vector": args.interval_vector_penalty,
        "row_order": args.row_order_penalty,
        "serial_transformation": args.serial_transformation_penalty,
        "aggregate": args.aggregate_penalty,
        "rhythm": args.rhythm_penalty,
        "gesture": args.gesture_penalty,
        "range": args.range_penalty,
        "content_span": args.content_span_penalty,
        "voice_count": args.voice_count_penalty,
    }


def generate_fragment(
    args: argparse.Namespace,
    seed: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    generation_seed = int(args.seed if seed is None else seed)
    metadata = _metadata_from_args(args, generation_seed)
    weights = _penalty_weights(args)
    generator_kind = args.generator
    if generator_kind == "auto":
        generator_kind = "transformer" if args.checkpoint else "rule"

    if generator_kind == "transformer":
        if not args.config or not args.checkpoint:
            raise ValueError("Transformer generation requires both --config and --checkpoint.")
        from post_tonal.data.score_tokenizer import ScoreTokenizer
        from post_tonal.evaluate import (
            generated_events_from_model,
            generation_token_budget,
            load_model,
        )

        config = load_yaml(args.config)
        tokenizer = ScoreTokenizer.load(config["vocab_path"])
        model, checkpoint_config = load_model(args.checkpoint, tokenizer)
        device = get_device(args.device or config.get("device"))
        model.to(device)
        eval_cfg = config.get("evaluation", {})
        base_tokens = int(
            args.max_new_tokens
            if args.max_new_tokens is not None
            else eval_cfg.get("max_new_tokens", config.get("model", {}).get("max_seq_len", 256))
        )
        token_budget = generation_token_budget(
            metadata,
            base_tokens,
            int(eval_cfg.get("max_new_tokens_per_measure", 32)),
            int(eval_cfg.get("max_new_tokens_cap", 768)),
        )
        events = generated_events_from_model(
            model,
            tokenizer,
            metadata,
            attempts=max(1, args.attempts),
            max_new_tokens=token_budget,
            grammar_constrained=args.grammar_constrained,
            weights=weights,
        )
        report = analyze_events(events, metadata)
        provenance = {
            "generator": "transformer",
            "config": str(Path(args.config)),
            "checkpoint": str(Path(args.checkpoint)),
            "checkpoint_training_seed": checkpoint_config.get("seed"),
            "device": str(device),
            "generation_seed": generation_seed,
            "candidate_attempts": max(1, args.attempts),
            "grammar_constrained": args.grammar_constrained,
            "max_new_tokens": token_budget,
        }
    else:
        best_events: list[dict[str, Any]] | None = None
        best_report: dict[str, Any] | None = None
        best_loss = float("inf")
        for attempt in range(max(1, args.attempts)):
            generator = RuleGenerator(seed=generation_seed + attempt)
            events = generator.generate(dict(metadata))
            report = analyze_events(events, metadata)
            loss = candidate_loss(report, weights)
            if loss < best_loss:
                best_events, best_report, best_loss = events, report, loss
        assert best_events is not None and best_report is not None
        events = best_events
        report = best_report
        provenance = {
            "generator": "rule",
            "generation_seed": generation_seed,
            "candidate_attempts": max(1, args.attempts),
        }

    report["decoding_penalty_loss"] = candidate_loss(report, weights)
    return events, metadata, report, provenance


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", choices=["auto", "rule", "transformer"], default="auto")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
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
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-examples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--grammar-constrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pcset-penalty", type=float, default=1.0)
    parser.add_argument("--pcset-precision-penalty", type=float, default=0.5)
    parser.add_argument("--interval-vector-penalty", type=float, default=0.05)
    parser.add_argument("--row-order-penalty", type=float, default=1.0)
    parser.add_argument("--serial-transformation-penalty", type=float, default=0.5)
    parser.add_argument("--aggregate-penalty", type=float, default=0.5)
    parser.add_argument("--rhythm-penalty", type=float, default=0.5)
    parser.add_argument("--gesture-penalty", type=float, default=0.5)
    parser.add_argument("--range-penalty", type=float, default=2.0)
    parser.add_argument("--content-span-penalty", type=float, default=1.0)
    parser.add_argument("--voice-count-penalty", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    count = max(1, int(args.num_examples))
    output_dir = Path(args.output_dir) if args.output_dir else None
    if count > 1 and output_dir is None:
        output_dir = Path("generated_scores/model_examples")
    if output_dir is not None:
        ensure_dir(output_dir)

    for index in range(count):
        events, metadata, report, provenance = generate_fragment(args, seed=args.seed + index)
        if output_dir is None:
            output_path = Path(args.output)
            report_path = Path(args.report)
        else:
            stem = f"example_{index + 1:03d}"
            output_path = output_dir / f"{stem}.musicxml"
            report_path = output_dir / f"{stem}.json"
        export_musicxml(events, output_path, metadata)
        from post_tonal.evaluate import musicxml_matches_request, musicxml_structurally_valid

        structural_ok = musicxml_structurally_valid(output_path)
        measure_ok, voice_ok = (
            musicxml_matches_request(output_path, metadata)
            if structural_ok
            else (False, False)
        )
        save_json(
            {
                "metadata": metadata,
                "analysis": report,
                "provenance": provenance,
                "export_validation": {
                    "structurally_valid": structural_ok,
                    "measure_count_adherent": measure_ok,
                    "voice_count_adherent": voice_ok,
                },
                "text": analysis_text(report),
            },
            report_path,
        )
        print(
            {
                "musicxml": str(output_path),
                "report": str(report_path),
                "structurally_valid": structural_ok,
                "measure_count_adherent": measure_ok,
                "voice_count_adherent": voice_ok,
            }
        )


if __name__ == "__main__":
    main()
