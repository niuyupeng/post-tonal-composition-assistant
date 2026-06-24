from post_tonal.models.rule_generator import RuleGenerator
from post_tonal.theory.analysis_report import analyze_events
from post_tonal.theory.gesture import compute_gesture_features, gesture_consistency_score
from post_tonal.theory.pcset import interval_vector


def test_gesture_metrics_and_analysis_report():
    metadata = {
        "pcset": [0, 1, 4, 6],
        "interval_vector": interval_vector([0, 1, 4, 6]),
        "row": list(range(12)),
        "row_form": "P0",
        "rhythm_profile": "pointillistic",
        "gesture": "pointillistic",
        "voices": 2,
        "measures": 4,
        "instrument": "generic_voice",
    }
    events = RuleGenerator(seed=6).generate(metadata)
    features = compute_gesture_features(events, total_beats=16.0)
    assert "note_density" in features
    assert 0.0 <= gesture_consistency_score(events, "pointillistic", total_beats=16.0) <= 1.0
    report = analyze_events(events, metadata)
    assert 0.0 <= report["pcset_coverage"] <= 1.0
    assert 0.0 <= report["aggregate_completion_rate"] <= 1.0
    assert report["instrument_range_violation_rate"] == 0.0
