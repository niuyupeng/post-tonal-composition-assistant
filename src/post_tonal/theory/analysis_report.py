"""Computed analysis reports for generated post-tonal fragments."""

from __future__ import annotations

from typing import Any

from post_tonal.theory.gesture import compute_gesture_features, gesture_consistency_score
from post_tonal.theory.pcset import interval_vector, interval_vector_distance_from_vector, pcset_coverage
from post_tonal.theory.rhythm_profile import density_curve, density_curve_error, generate_rhythmic_profile, rhythmic_profile_distance
from post_tonal.theory.serial import aggregate_completion_rate, cyclic_row_order_accuracy, row_form
from post_tonal.utils import event_pitch_classes, instrument_range_violation_rate


def analyze_events(events: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    pcs = event_pitch_classes(events)
    measures = int(metadata.get("measures", 4))
    total_beats = measures * 4.0
    pcset = metadata.get("pcset", [])
    target_iv = metadata.get("interval_vector")
    row = metadata.get("row")
    row_form_label = metadata.get("row_form")
    rhythm_profile = metadata.get("rhythm_profile", "medium")
    gesture_label = metadata.get("gesture", "fragmented")

    target_rhythm_events = generate_rhythmic_profile(rhythm_profile, measures=measures, seed=int(metadata.get("rhythm_seed", 1234)))
    report: dict[str, Any] = {
        "note_count": len(pcs),
        "pcset": pcset,
        "pcset_coverage": pcset_coverage(pcs, pcset),
        "generated_interval_vector": interval_vector(pcs),
        "aggregate_completion_rate": aggregate_completion_rate(pcs),
        "density_curve": density_curve(events, measures),
        "rhythmic_profile_distance": rhythmic_profile_distance(events, rhythm_profile, measures),
        "density_curve_error": density_curve_error(events, target_rhythm_events, measures),
        "gesture_features": compute_gesture_features(events, total_beats=total_beats),
        "gesture_consistency_score": gesture_consistency_score(events, gesture_label, total_beats=total_beats),
        "instrument_range_violation_rate": instrument_range_violation_rate(events),
        "range_violation_rate": instrument_range_violation_rate(events),
    }
    if target_iv is not None:
        report["interval_vector_distance"] = interval_vector_distance_from_vector(pcs, target_iv)
    else:
        report["interval_vector_distance"] = None
    if row and row_form_label:
        expected = row_form(row, row_form_label)
        report["row_order_accuracy"] = cyclic_row_order_accuracy(pcs, expected)
        report["serial_transformation_accuracy"] = report["row_order_accuracy"]
        report["expected_row_form"] = expected
    else:
        report["row_order_accuracy"] = None
        report["serial_transformation_accuracy"] = None
        report["expected_row_form"] = None
    return report


def analysis_text(report: dict[str, Any]) -> str:
    lines = [
        "Post-tonal analysis report",
        f"Note count: {report.get('note_count')}",
        f"PC-set coverage: {report.get('pcset_coverage'):.3f}",
        f"Generated interval vector: {report.get('generated_interval_vector')}",
        f"Interval-vector distance: {report.get('interval_vector_distance')}",
        f"Row-order accuracy: {report.get('row_order_accuracy')}",
        f"Aggregate completion: {report.get('aggregate_completion_rate'):.3f}",
        f"Density curve: {report.get('density_curve')}",
        f"Rhythmic-profile distance: {report.get('rhythmic_profile_distance'):.3f}",
        f"Gesture consistency: {report.get('gesture_consistency_score'):.3f}",
        f"Register distribution: {report.get('gesture_features', {}).get('register_spread')}",
        f"Instrument range violation rate: {report.get('instrument_range_violation_rate'):.3f}",
    ]
    return "\n".join(lines)
