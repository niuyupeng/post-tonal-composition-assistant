from pathlib import Path

import pytest
import torch
import yaml

from post_tonal.data.score_tokenizer import ScoreTokenizer
from post_tonal.evaluate import evaluate, generated_events_from_rule
from post_tonal.generate import candidate_loss
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


def test_gesture_rest_ratio_uses_bounded_union_coverage():
    events = [
        {"onset": 0.0, "duration": 4.0, "voice": 0, "is_rest": True},
        {"onset": 1.0, "duration": 4.0, "voice": 0, "is_rest": True},
        {"onset": 0.0, "duration": 8.0, "voice": 1, "is_rest": True},
    ]
    features = compute_gesture_features(events, total_beats=4.0, voice_count=2)
    assert features["rest_ratio"] == 1.0


def test_gesture_note_density_is_voice_count_normalized():
    one_voice = [
        {"onset": 0.0, "duration": 0.25, "voice": 0, "pitch": 60, "is_rest": False}
    ]
    two_voices = one_voice + [
        {"onset": 0.0, "duration": 0.25, "voice": 1, "pitch": 64, "is_rest": False}
    ]
    assert compute_gesture_features(one_voice, total_beats=4.0, voice_count=1)["note_density"] == (
        compute_gesture_features(two_voices, total_beats=4.0, voice_count=2)["note_density"]
    )


def test_seeded_transformer_sampling_is_reproducible():
    set_seed(17)
    model = PostTonalTransformer(vocab_size=32, hidden_size=24, layers=1, heads=3, max_seq_len=16, dropout=0.0)
    prefix = [1, 4, 7]

    set_seed(91)
    first = model.sample(prefix, eos_id=2, max_new_tokens=8, top_k=8)
    set_seed(91)
    second = model.sample(prefix, eos_id=2, max_new_tokens=8, top_k=8)

    assert first == second


def test_rule_baseline_generation_is_reproducible_and_independent():
    metadata = {
        "pcset": [0, 1, 4, 6],
        "interval_vector": interval_vector([0, 1, 4, 6]),
        "row": None,
        "row_form": None,
        "rhythm_profile": "medium",
        "gesture": "fragmented",
        "voices": 2,
        "measures": 4,
        "instrument": "piano",
    }
    first = generated_events_from_rule(metadata, seed=91)
    second = generated_events_from_rule(metadata, seed=91)
    third = generated_events_from_rule(metadata, seed=92)
    assert first == second
    assert first != third


def test_transformer_rejects_implicit_sequence_truncation():
    model = PostTonalTransformer(
        vocab_size=32,
        hidden_size=24,
        layers=1,
        heads=3,
        max_seq_len=8,
        dropout=0.0,
    )
    with pytest.raises(ValueError, match="explicit condition-preserving window"):
        model(torch.zeros((1, 9), dtype=torch.long))


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


def test_missing_table_metric_is_not_applicable():
    assert _format(None) == "--"


def test_nonserial_candidate_loss_does_not_reward_aggregate_completion():
    base_report = {
        "pcset": [],
        "interval_vector_distance": None,
        "row_order_accuracy": None,
        "serial_transformation_accuracy": None,
        "aggregate_target_applicable": False,
        "rhythmic_profile_distance": None,
        "gesture_consistency_score": None,
        "content_span_ratio": 1.0,
        "voice_count_adherence": 1.0,
        "instrument_range_violation_rate": 0.0,
    }
    low_aggregate = {**base_report, "aggregate_completion_rate": 0.1}
    high_aggregate = {**base_report, "aggregate_completion_rate": 1.0}
    assert candidate_loss(low_aggregate, {}) == candidate_loss(high_aggregate, {})


def test_saved_target_density_curve_is_held_out_from_rhythm_reranking():
    metadata = {
        "pcset": [0, 4],
        "interval_vector": interval_vector([0, 4]),
        "row": None,
        "row_form": None,
        "rhythm_profile": "sparse",
        "gesture": "fragmented",
        "voices": 1,
        "measures": 4,
        "instrument": "generic_voice",
        "target_density_curve": [1.0, 0.0, 1.0, 0.0],
    }
    events = [
        {"onset": 0.0, "duration": 0.5, "voice": 0, "pitch": 60, "pc": 0, "is_rest": False},
        {"onset": 8.0, "duration": 0.5, "voice": 0, "pitch": 64, "pc": 4, "is_rest": False},
    ]
    report = analyze_events(events, metadata)
    assert report["rhythm_reference"] == "deterministic_profile_template"
    assert report["density_curve_reference"] == "held_out_target_density_curve"
    assert report["density_curve_error"] == 0.0
    assert report["rhythmic_profile_distance"] >= 0.0


def test_serial_transformation_metric_is_independent_from_row_order():
    metadata = {
        "pcset": [],
        "interval_vector": None,
        "row": list(range(12)),
        "row_form": "P0",
        "rhythm_profile": "medium",
        "gesture": "fragmented",
        "voices": 1,
        "measures": 4,
        "instrument": "generic_voice",
    }
    events = [
        {
            "onset": index * 0.25,
            "duration": 0.25,
            "voice": 0,
            "pitch": 60 + pc,
            "pc": pc,
            "is_rest": False,
        }
        for index, pc in enumerate([0, 1, 2, 3, 4, 5, 6, 7])
    ]
    report = analyze_events(events, metadata)
    assert report["row_order_accuracy"] == 1.0
    assert report["serial_transformation_accuracy"] == 0.0


def test_ablation_evaluation_retains_hidden_target_metadata(tmp_path: Path):
    tokenizer = ScoreTokenizer()
    metadata = {
        "pcset": [0, 1],
        "interval_vector": interval_vector([0, 1]),
        "row": None,
        "row_form": None,
        "rhythm_profile": "medium",
        "gesture": "fragmented",
        "voices": 1,
        "measures": 4,
        "instrument": "generic_voice",
        "target_density_curve": [2.0, 0.0, 0.0, 0.0],
    }
    events = [
        {"onset": 0.0, "duration": 0.25, "voice": 0, "pitch": 60, "pc": 0, "is_rest": False},
        {"onset": 0.5, "duration": 0.25, "voice": 0, "pitch": 61, "pc": 1, "is_rest": False},
    ]
    data_path = tmp_path / "ablation.pt"
    vocab_path = tmp_path / "ablation.vocab.json"
    torch.save(
        {
            "format": "post_tonal_synthetic_v3_windowed",
            "samples": [
                {
                    "id": "test-0",
                    "token_ids": tokenizer.encode(tokenizer.events_to_tokens(events, metadata)),
                    "metadata": metadata,
                    "events": events,
                    "split": "test",
                }
            ],
            "split_counts": {"train": 0, "val": 0, "test": 1},
        },
        data_path,
    )
    tokenizer.save(vocab_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "without_pcset_constraints",
                "seed": 42,
                "device": "cpu",
                "condition_ablation": "no_pcset",
                "data_path": str(data_path),
                "vocab_path": str(vocab_path),
                "generate_data": False,
                "model": {"max_seq_len": 256},
                "training": {
                    "batch_size": 1,
                    "sequence_mode": "coverage_cycle",
                    "target_tokens_per_window": 64,
                    "coverage_cycle_epochs": 2,
                },
                "evaluation": {
                    "constraint_metric_samples": 1,
                    "model_generation": False,
                    "export_examples": 0,
                    "generation_examples": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    metrics = evaluate(config_path, None, tmp_path / "metrics.json")
    assert metrics["target_pcset_coverage"] == 1.0
    assert metrics["pcset_precision"] == 1.0
