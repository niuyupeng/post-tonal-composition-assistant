import json

import numpy as np
import pytest
import torch

from post_tonal.analyze_controlled_results import analyze_controlled_results, paired_bootstrap_ci
from post_tonal.make_tables import _format
from post_tonal.models.rule_generator import RuleGenerator
from post_tonal.models.transformer import PostTonalTransformer
from post_tonal.theory.analysis_report import analyze_events
from post_tonal.theory.gesture import compute_gesture_features, gesture_consistency_score
from post_tonal.theory.pcset import interval_vector
from post_tonal.utils import set_seed


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


def test_seeded_transformer_sampling_is_reproducible():
    set_seed(17)
    model = PostTonalTransformer(vocab_size=32, hidden_size=24, layers=1, heads=3, max_seq_len=16, dropout=0.0)
    prefix = [1, 4, 7]

    set_seed(91)
    first = model.sample(prefix, eos_id=2, max_new_tokens=8, top_k=8)
    set_seed(91)
    second = model.sample(prefix, eos_id=2, max_new_tokens=8, top_k=8)

    assert first == second


def test_batched_sampling_is_reproducible_and_preserves_first_candidate():
    set_seed(23)
    model = PostTonalTransformer(vocab_size=32, hidden_size=24, layers=1, heads=3, max_seq_len=16, dropout=0.0)
    prefixes = [[1, 4, 7], [1, 5], [1, 9, 3, 6]]
    seeds = [101, 102, 103]

    single_generators = [torch.Generator().manual_seed(seed) for seed in seeds]
    single = model.sample_batch(
        prefixes,
        eos_id=2,
        max_new_tokens=8,
        top_k=8,
        generators=single_generators,
    )
    reranked_generators = [torch.Generator().manual_seed(seed) for seed in seeds]
    reranked_first = model.sample_batch(
        prefixes,
        eos_id=2,
        max_new_tokens=8,
        top_k=8,
        generators=reranked_generators,
    )
    model.sample_batch(
        prefixes,
        eos_id=2,
        max_new_tokens=8,
        top_k=8,
        generators=reranked_generators,
    )

    assert single == reranked_first

    set_seed(seeds[0])
    legacy = model.sample(prefixes[0], eos_id=2, max_new_tokens=8, top_k=8)
    compatible = model.sample_batch(
        [prefixes[0]],
        eos_id=2,
        max_new_tokens=8,
        top_k=8,
        generators=[torch.Generator().manual_seed(seeds[0])],
    )[0]
    assert compatible == legacy


def test_paired_bootstrap_ci_tracks_constant_improvement():
    low, high = paired_bootstrap_ci(np.full(12, 0.25), seed=5, samples=200)
    assert low == 0.25
    assert high == 0.25


def test_controlled_analysis_reports_serial_and_nonserial_subsets(tmp_path):
    common_analysis = {
        "pcset_coverage": 0.5,
        "interval_vector_distance": 2.0,
        "row_order_accuracy": None,
        "aggregate_completion_rate": 0.5,
        "rhythmic_profile_distance": 1.0,
        "density_curve_error": 1.0,
        "gesture_consistency_score": 0.5,
        "range_violation_rate": 0.0,
    }
    single_samples = [
        {
            "sample_id": "nonserial",
            "evaluation_seed": 10,
            "first_candidate_sha256": "a" * 64,
            "metadata": {"row": None, "row_form": None},
            "analysis": dict(common_analysis),
        },
        {
            "sample_id": "serial",
            "evaluation_seed": 11,
            "first_candidate_sha256": "b" * 64,
            "metadata": {"row": list(range(12)), "row_form": "P0"},
            "analysis": {**common_analysis, "row_order_accuracy": 0.25},
        },
    ]
    reranked_samples = json.loads(json.dumps(single_samples))
    reranked_samples[0]["analysis"]["pcset_coverage"] = 0.75
    reranked_samples[1]["analysis"]["row_order_accuracy"] = 0.5
    single_path = tmp_path / "single.json"
    reranked_path = tmp_path / "reranked.json"
    single_path.write_text(json.dumps({"samples": single_samples}), encoding="utf-8")
    reranked_path.write_text(json.dumps({"samples": reranked_samples}), encoding="utf-8")

    result = analyze_controlled_results(
        single_path,
        reranked_path,
        tmp_path / "stats.json",
        tmp_path / "stats.csv",
        tmp_path / "stats.tex",
        bootstrap_samples=50,
    )
    endpoints = {row["endpoint"]: row for row in result["metrics"]}
    assert result["bootstrap_method"] == "paired percentile bootstrap over test conditions"
    assert result["multiple_endpoint_adjustment"] == "none"
    assert result["first_candidate_alignment"] == "verified_by_sha256"
    assert result["first_candidate_fingerprints_verified"] == 2
    assert endpoints["pcset_coverage:all"]["n"] == 2
    assert endpoints["pcset_coverage:non-serial"]["n"] == 1
    assert endpoints["pcset_coverage:serial"]["n"] == 1
    assert endpoints["interval_vector_distance:serial"]["n"] == 1
    assert endpoints["row_order_accuracy:serial"]["n"] == 1
    assert endpoints["aggregate_completion_rate:non-serial"]["n"] == 1
    assert endpoints["aggregate_completion_rate:non-serial"]["higher_is_better"] is None
    assert endpoints["aggregate_completion_rate:non-serial"]["favorable_improvement"] is None

    reranked_samples[1]["first_candidate_sha256"] = "c" * 64
    reranked_path.write_text(json.dumps({"samples": reranked_samples}), encoding="utf-8")
    with pytest.raises(ValueError, match="First candidates differ"):
        analyze_controlled_results(
            single_path,
            reranked_path,
            tmp_path / "bad-stats.json",
            tmp_path / "bad-stats.csv",
            tmp_path / "bad-stats.tex",
            bootstrap_samples=10,
        )

    reranked_samples[1]["first_candidate_sha256"] = "b" * 64
    reranked_samples[0]["analysis"]["pcset_coverage"] = None
    reranked_path.write_text(json.dumps({"samples": reranked_samples}), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing endpoint pcset_coverage:all"):
        analyze_controlled_results(
            single_path,
            reranked_path,
            tmp_path / "missing-stats.json",
            tmp_path / "missing-stats.csv",
            tmp_path / "missing-stats.tex",
            bootstrap_samples=10,
        )


def test_missing_table_metric_is_not_applicable():
    assert _format(None) == "--"
