"""Computed analysis reports for generated post-tonal fragments."""

from __future__ import annotations

import math
from typing import Any

from post_tonal.theory.gesture import compute_gesture_features, gesture_consistency_score
from post_tonal.theory.pcset import interval_vector, interval_vector_distance_from_vector, pcset_coverage
from post_tonal.theory.rhythm_profile import (
    density_curve,
    density_curve_distance,
    density_curve_error,
    generate_rhythmic_profile,
    rhythmic_profile_distance,
)
from post_tonal.theory.serial import (
    aggregate_completion_rate,
    cyclic_row_order_accuracy,
    row_form,
    serial_transformation_accuracy,
)
from post_tonal.utils import event_pitch_classes, instrument_range_violation_rate


def analyze_events(events: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    pcs = event_pitch_classes(events)
    measures = int(metadata.get("measures", 4))
    total_beats = measures * 4.0
    pcset = metadata.get("pcset", [])
    target_iv = metadata.get("interval_vector")
    row = metadata.get("row")
    row_form_label = metadata.get("row_form")
    rhythm_profile = metadata.get("rhythm_profile")
    gesture_label = metadata.get("gesture")
    requested_voices = max(1, int(metadata.get("voices", 1)))
    generated_pcset = set(int(pc) % 12 for pc in pcs)
    target_pcset = set(int(pc) % 12 for pc in pcset)
    pcset_precision = (
        None
        if not target_pcset
        else (0.0 if not generated_pcset else len(generated_pcset & target_pcset) / len(generated_pcset))
    )
    pcset_jaccard = (
        None
        if not target_pcset
        else len(generated_pcset & target_pcset) / max(1, len(generated_pcset | target_pcset))
    )
    max_end = max(
        (
            float(event.get("onset", 0.0)) + float(event.get("duration", 0.0))
            for event in events
        ),
        default=0.0,
    )
    realized_content_measures = 0 if max_end <= 0.0 else int(math.ceil(max_end / 4.0))
    realized_voices = {
        int(event.get("voice", 0))
        for event in events
        if 0 <= int(event.get("voice", 0)) < requested_voices
    }

    target_density_curve = metadata.get("target_density_curve")
    if rhythm_profile:
        rhythm_distance = rhythmic_profile_distance(
            events,
            rhythm_profile,
            measures,
            seed=int(metadata.get("rhythm_seed", 1234)),
            voice_count=requested_voices,
        )
        rhythm_reference = "deterministic_profile_template"
        if target_density_curve is not None:
            curve_error = density_curve_distance(
                events,
                target_density_curve,
                measures,
                normalize=False,
            )
            density_curve_reference = "held_out_target_density_curve"
        else:
            target_rhythm_events = generate_rhythmic_profile(
                rhythm_profile,
                measures=measures,
                seed=int(metadata.get("rhythm_seed", 1234)),
            )
            curve_error = density_curve_error(events, target_rhythm_events, measures)
            density_curve_reference = "deterministic_profile_template"
    else:
        rhythm_distance = None
        curve_error = None
        rhythm_reference = None
        density_curve_reference = None
    if gesture_label:
        gesture_features = compute_gesture_features(
            events,
            total_beats=total_beats,
            voice_count=requested_voices,
        )
        gesture_score = gesture_consistency_score(
            events,
            gesture_label,
            total_beats=total_beats,
            voice_count=requested_voices,
        )
    else:
        gesture_features = compute_gesture_features(
            events,
            total_beats=total_beats,
            voice_count=requested_voices,
        )
        gesture_score = None
    report: dict[str, Any] = {
        "note_count": len(pcs),
        "pcset": pcset,
        "pcset_coverage": pcset_coverage(pcs, pcset) if target_pcset else None,
        "pcset_precision": pcset_precision,
        "pcset_jaccard": pcset_jaccard,
        "off_target_pc_rate": None if pcset_precision is None else 1.0 - pcset_precision,
        "generated_interval_vector": interval_vector(pcs),
        "aggregate_completion_rate": aggregate_completion_rate(pcs),
        "aggregate_target_applicable": bool(row and row_form_label),
        "density_curve": density_curve(events, measures),
        "rhythmic_profile_distance": rhythm_distance,
        "density_curve_error": curve_error,
        "rhythm_reference": rhythm_reference,
        "density_curve_reference": density_curve_reference,
        "gesture_features": gesture_features,
        "gesture_consistency_score": gesture_score,
        "instrument_range_violation_rate": instrument_range_violation_rate(events),
        "range_violation_rate": instrument_range_violation_rate(events),
        "requested_measures": measures,
        "realized_content_measures": realized_content_measures,
        "content_span_ratio": min(1.0, max_end / max(total_beats, 1e-6)),
        "requested_voice_count": requested_voices,
        "realized_voice_count": len(realized_voices),
        "voice_count_adherence": len(realized_voices) / requested_voices,
    }
    if target_iv is not None:
        report["interval_vector_distance"] = interval_vector_distance_from_vector(pcs, target_iv)
    else:
        report["interval_vector_distance"] = None
    if row and row_form_label:
        expected = row_form(row, row_form_label)
        report["row_order_accuracy"] = cyclic_row_order_accuracy(pcs, expected)
        report["serial_transformation_accuracy"] = serial_transformation_accuracy(
            pcs,
            row,
            row_form_label,
        )
        report["expected_row_form"] = expected
    else:
        report["row_order_accuracy"] = None
        report["serial_transformation_accuracy"] = None
        report["expected_row_form"] = None
    return report


def analysis_text(report: dict[str, Any]) -> str:
    def display(value: Any, digits: int = 3) -> str:
        return "N/A" if value is None else f"{float(value):.{digits}f}"

    lines = [
        "Post-tonal analysis report",
        f"Note count: {report.get('note_count')}",
        f"PC-set coverage: {display(report.get('pcset_coverage'))}",
        f"PC-set precision: {display(report.get('pcset_precision'))}",
        f"Generated interval vector: {report.get('generated_interval_vector')}",
        f"Interval-vector distance: {report.get('interval_vector_distance')}",
        f"Row-order accuracy: {display(report.get('row_order_accuracy'))}",
        f"Serial transformation accuracy: {display(report.get('serial_transformation_accuracy'))}",
        f"Aggregate completion: {display(report.get('aggregate_completion_rate'))}",
        f"Density curve: {report.get('density_curve')}",
        f"Rhythmic-profile distance: {display(report.get('rhythmic_profile_distance'))}",
        f"Gesture consistency: {display(report.get('gesture_consistency_score'))}",
        f"Register distribution: {report.get('gesture_features', {}).get('register_spread')}",
        f"Instrument range violation rate: {display(report.get('instrument_range_violation_rate'))}",
        f"Content span ratio: {display(report.get('content_span_ratio'))}",
        f"Voice-count adherence: {display(report.get('voice_count_adherence'))}",
    ]
    return "\n".join(lines)
