import json

import pytest

from post_tonal.analyze_multiseed_controlled import analyze_multiseed_controlled


def _analysis(serial: bool, reranked: bool) -> dict:
    delta = 0.1 if reranked else 0.0
    return {
        "pcset_coverage": 0.5 + delta,
        "interval_vector_distance": 2.0 - delta,
        "row_order_accuracy": 0.25 + delta if serial else None,
        "aggregate_completion_rate": 0.5 + delta,
        "serial_transformation_accuracy": 0.25 + delta if serial else None,
        "rhythmic_profile_distance": 1.0 - delta,
        "density_curve_error": 1.5 - delta,
        "gesture_consistency_score": 0.4 + delta,
        "range_violation_rate": 0.2 - delta,
    }


def _write_pair(tmp_path, seed: int):
    row = list(range(12))
    records = [
        ("nonserial", {"row": None, "row_form": None}, False),
        ("serial", {"row": row, "row_form": "P0"}, True),
    ]
    single_samples = []
    reranked_samples = []
    for index, (sample_id, metadata, serial) in enumerate(records):
        fingerprint = f"{seed + index:064x}"[-64:]
        common = {
            "experiment": "controlled",
            "split": "test",
            "sample_index": index,
            "sample_id": sample_id,
            "evaluation_seed": 42042 + index,
            "generation_batch_size": 32,
            "sampling_protocol": "per_sample_generator_batch_v1",
            "first_candidate_sha256": fingerprint,
            "metadata": metadata,
        }
        single_samples.append({**common, "candidate_attempts": 1, "analysis": _analysis(serial, False)})
        reranked_samples.append({**common, "candidate_attempts": 4, "analysis": _analysis(serial, True)})
    common_payload = {
        "evaluation_seed": 42042,
        "sampling_protocol": "per_sample_generator_batch_v1",
        "generation_batch_size": 32,
        "num_samples": 2,
        "provenance": {
            "checkpoint_path": f"runs/multiseed/seed_{seed}/checkpoint.pt",
            "checkpoint_sha256": f"{seed:064x}",
            "checkpoint_training_seed": seed,
            "data_path": "data/processed/project2_main.pt",
            "data_sha256": "d" * 64,
            "vocab_path": "data/processed/project2_main.vocab.json",
            "vocab_sha256": "e" * 64,
            "dataset_split": "test",
            "dataset_split_size": 2,
        },
    }
    single_path = tmp_path / f"seed_{seed}_single.json"
    reranked_path = tmp_path / f"seed_{seed}_reranked.json"
    single_path.write_text(
        json.dumps({**common_payload, "candidate_attempts": 1, "samples": single_samples}),
        encoding="utf-8",
    )
    reranked_path.write_text(
        json.dumps({**common_payload, "candidate_attempts": 4, "samples": reranked_samples}),
        encoding="utf-8",
    )
    return single_path, reranked_path


def test_multiseed_controlled_analysis_validates_alignment_and_aggregates(tmp_path):
    pairs = [_write_pair(tmp_path, seed) for seed in (42, 43, 44)]
    result = analyze_multiseed_controlled(
        [42, 43, 44],
        [pair[0] for pair in pairs],
        [pair[1] for pair in pairs],
        tmp_path / "summary.json",
        tmp_path / "summary.csv",
        tmp_path / "summary.tex",
        bootstrap_samples=100,
        expected_conditions=2,
    )

    assert result["seeds"] == [42, 43, 44]
    assert result["paired_conditions_per_seed"] == 2
    assert result["first_candidate_alignment"] == "verified_by_sha256_for_every_seed_condition"
    rows = {row["endpoint"]: row for row in result["metrics"]}
    pcset = rows["pcset_coverage:all"]
    assert pcset["mean_effect"] == pytest.approx(0.1)
    assert pcset["sample_sd_across_seed_means"] == pytest.approx(0.0)
    assert pcset["crossed_bootstrap_ci95_low"] == pytest.approx(0.1)
    assert pcset["crossed_bootstrap_ci95_high"] == pytest.approx(0.1)
    assert pcset["positive_seed_count"] == 3
    assert rows["aggregate_completion_rate:non-serial"]["positive_seed_count"] is None
    assert (tmp_path / "summary.csv").is_file()
    assert "Crossed 95\\% CI" in (tmp_path / "summary.tex").read_text(encoding="utf-8")


def test_multiseed_controlled_analysis_rejects_unaligned_first_candidate(tmp_path):
    pairs = [_write_pair(tmp_path, seed) for seed in (42, 43)]
    payload = json.loads(pairs[1][1].read_text(encoding="utf-8"))
    payload["samples"][0]["first_candidate_sha256"] = "f" * 64
    pairs[1][1].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="first candidates differ"):
        analyze_multiseed_controlled(
            [42, 43],
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            tmp_path / "summary.json",
            tmp_path / "summary.csv",
            tmp_path / "summary.tex",
            bootstrap_samples=10,
            expected_conditions=2,
        )


def test_multiseed_controlled_analysis_rejects_duplicate_checkpoint(tmp_path):
    pairs = [_write_pair(tmp_path, seed) for seed in (42, 43)]
    second_single = json.loads(pairs[1][0].read_text(encoding="utf-8"))
    second_reranked = json.loads(pairs[1][1].read_text(encoding="utf-8"))
    duplicate = f"{42:064x}"
    second_single["provenance"]["checkpoint_sha256"] = duplicate
    second_reranked["provenance"]["checkpoint_sha256"] = duplicate
    pairs[1][0].write_text(json.dumps(second_single), encoding="utf-8")
    pairs[1][1].write_text(json.dumps(second_reranked), encoding="utf-8")

    with pytest.raises(ValueError, match="distinct checkpoint"):
        analyze_multiseed_controlled(
            [42, 43],
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            tmp_path / "summary.json",
            tmp_path / "summary.csv",
            tmp_path / "summary.tex",
            bootstrap_samples=10,
            expected_conditions=2,
        )


def test_multiseed_controlled_analysis_rejects_incomplete_endpoint(tmp_path):
    pairs = [_write_pair(tmp_path, seed) for seed in (42, 43)]
    payload = json.loads(pairs[1][1].read_text(encoding="utf-8"))
    payload["samples"][0]["analysis"]["pcset_coverage"] = None
    pairs[1][1].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing endpoint pcset_coverage:all"):
        analyze_multiseed_controlled(
            [42, 43],
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            tmp_path / "summary.json",
            tmp_path / "summary.csv",
            tmp_path / "summary.tex",
            bootstrap_samples=10,
            expected_conditions=2,
        )


def test_multiseed_controlled_analysis_rejects_wrong_rng_schedule(tmp_path):
    pairs = [_write_pair(tmp_path, seed) for seed in (42, 43)]
    for path in pairs[1]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["samples"][1]["evaluation_seed"] = 999
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must use evaluation seed 42043"):
        analyze_multiseed_controlled(
            [42, 43],
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            tmp_path / "summary.json",
            tmp_path / "summary.csv",
            tmp_path / "summary.tex",
            bootstrap_samples=10,
            expected_conditions=2,
        )
