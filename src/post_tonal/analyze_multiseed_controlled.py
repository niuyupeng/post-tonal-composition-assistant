"""Aggregate aligned K=1 versus K=4 controlled decoding across training seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from post_tonal.analyze_controlled_results import ENDPOINTS
from post_tonal.utils import ensure_dir, save_json


def _load_payload(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Missing non-empty controlled result: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError(f"Controlled result has no sample list: {source}")
    return payload


def _is_serial(sample: dict[str, Any]) -> bool:
    metadata = sample.get("metadata", {})
    return bool(metadata.get("row") and metadata.get("row_form"))


def _in_subset(sample: dict[str, Any], subset: str) -> bool:
    if subset == "all":
        return True
    if subset == "serial":
        return _is_serial(sample)
    if subset == "non-serial":
        return not _is_serial(sample)
    raise ValueError(f"Unknown controlled subset: {subset}")


def _validate_pair(
    seed: int,
    single_payload: dict[str, Any],
    reranked_payload: dict[str, Any],
    expected_conditions: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    single = single_payload["samples"]
    reranked = reranked_payload["samples"]
    if len(single) != len(reranked):
        raise ValueError(f"Seed {seed} has different K=1 and K=4 sample counts.")
    if single_payload.get("num_samples") != len(single) or reranked_payload.get("num_samples") != len(reranked):
        raise ValueError(f"Seed {seed} payload sample counts do not match their records.")
    if expected_conditions is not None and len(single) != expected_conditions:
        raise ValueError(
            f"Seed {seed} contains {len(single)} conditions; expected exactly {expected_conditions}."
        )
    if single_payload.get("candidate_attempts") != 1:
        raise ValueError(f"Seed {seed} K=1 payload must report one candidate attempt.")
    if reranked_payload.get("candidate_attempts") != 4:
        raise ValueError(f"Seed {seed} K=4 payload must report four candidate attempts.")
    if single_payload.get("sampling_protocol") != "per_sample_generator_batch_v1":
        raise ValueError(f"Seed {seed} does not use the required batched per-sample RNG protocol.")
    if single_payload.get("sampling_protocol") != reranked_payload.get("sampling_protocol"):
        raise ValueError(f"Seed {seed} uses different sampling protocols across conditions.")
    if single_payload.get("generation_batch_size") != reranked_payload.get("generation_batch_size"):
        raise ValueError(f"Seed {seed} uses different generation batch sizes across conditions.")
    if single_payload.get("evaluation_seed") != reranked_payload.get("evaluation_seed"):
        raise ValueError(f"Seed {seed} uses different top-level evaluation seeds across conditions.")

    single_provenance = single_payload.get("provenance")
    reranked_provenance = reranked_payload.get("provenance")
    if not isinstance(single_provenance, dict) or not isinstance(reranked_provenance, dict):
        raise ValueError(f"Seed {seed} lacks evaluation provenance.")
    identity_fields = (
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_training_seed",
        "data_path",
        "data_sha256",
        "vocab_path",
        "vocab_sha256",
        "dataset_split",
        "dataset_split_size",
    )
    for field in identity_fields:
        if single_provenance.get(field) != reranked_provenance.get(field):
            raise ValueError(f"Seed {seed} provenance differs across K=1 and K=4 for {field}.")
    for field in ("checkpoint_sha256", "data_sha256", "vocab_sha256"):
        value = single_provenance.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Seed {seed} lacks a valid {field} provenance digest.")
    if single_provenance.get("checkpoint_training_seed") != seed:
        raise ValueError(f"Seed {seed} does not match the seed embedded in its checkpoint.")
    if single_provenance.get("dataset_split") != "test":
        raise ValueError(f"Seed {seed} was not evaluated on the test split.")
    if single_provenance.get("dataset_split_size") != len(single):
        raise ValueError(f"Seed {seed} provenance reports the wrong test-split size.")

    sample_ids: list[Any] = []
    for left, right in zip(single, reranked):
        sample_id = left.get("sample_id")
        sample_ids.append(sample_id)
        if sample_id != right.get("sample_id"):
            raise ValueError(f"Seed {seed} sample IDs are not aligned.")
        if left.get("metadata") != right.get("metadata"):
            raise ValueError(f"Seed {seed} metadata differs for sample {sample_id}.")
        if left.get("evaluation_seed") != right.get("evaluation_seed"):
            raise ValueError(f"Seed {seed} evaluation seeds differ for sample {sample_id}.")
        left_hash = left.get("first_candidate_sha256")
        right_hash = right.get("first_candidate_sha256")
        if not isinstance(left_hash, str) or len(left_hash) != 64:
            raise ValueError(f"Seed {seed} lacks a K=1 first-candidate fingerprint for {sample_id}.")
        if left_hash != right_hash:
            raise ValueError(f"Seed {seed} first candidates differ for sample {sample_id}.")
    if any(sample_id is None for sample_id in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"Seed {seed} sample IDs must be non-null and unique.")
    return single, single_provenance


def _effect_array(
    single: Sequence[dict[str, Any]],
    reranked: Sequence[dict[str, Any]],
    metric: str,
    subset: str,
    higher_is_better: bool | None,
) -> np.ndarray:
    values: list[float] = []
    for left, right in zip(single, reranked):
        if not _in_subset(left, subset):
            continue
        left_value = left.get("analysis", {}).get(metric)
        right_value = right.get("analysis", {}).get(metric)
        if left_value is None or right_value is None:
            continue
        raw = float(right_value) - float(left_value)
        values.append(raw if higher_is_better is not False else -raw)
    if not values:
        raise ValueError(f"No observations for endpoint {metric}:{subset}")
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"Non-finite effect for endpoint {metric}:{subset}")
    return result


def crossed_bootstrap_ci(
    effects_by_seed: Sequence[np.ndarray],
    seed: int,
    samples: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Resample seed rows and shared condition columns as crossed factors."""
    if len(effects_by_seed) < 2:
        raise ValueError("Crossed bootstrap requires at least two training seeds.")
    sizes = {int(values.size) for values in effects_by_seed}
    if len(sizes) != 1 or 0 in sizes:
        raise ValueError("Aligned seed effects must have one common non-zero condition count.")
    matrix = np.stack(effects_by_seed)
    seed_count, condition_count = matrix.shape
    rng = np.random.default_rng(seed)
    bootstrap_means: list[np.ndarray] = []
    remaining = int(samples)
    while remaining > 0:
        chunk = min(100, remaining)
        selected_seeds = rng.integers(0, seed_count, size=(chunk, seed_count))
        selected_conditions = rng.integers(0, condition_count, size=(chunk, condition_count))
        sampled = matrix[selected_seeds[:, :, None], selected_conditions[:, None, :]]
        bootstrap_means.append(sampled.mean(axis=(1, 2)))
        remaining -= chunk
    values = np.concatenate(bootstrap_means)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], seeds: Sequence[int]) -> None:
    ensure_dir(path.parent)
    seed_fields = [f"seed_{seed}_mean" for seed in seeds]
    fields = [
        "endpoint",
        "metric",
        "label",
        "subset",
        "effect_orientation",
        "n_seeds",
        "n_conditions_per_seed",
        *seed_fields,
        "mean_effect",
        "sample_sd_across_seed_means",
        "crossed_bootstrap_ci95_low",
        "crossed_bootstrap_ci95_high",
        "positive_seed_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_table(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Metric & Subset & Mean effect & Seed SD & Crossed 95\\% CI & Positive seeds \\\\",
        "\\midrule",
    ]
    for row in rows:
        positive = "--" if row["positive_seed_count"] is None else f"{row['positive_seed_count']}/{row['n_seeds']}"
        lines.append(
            f"{row['label']} & {row['subset']} & {row['mean_effect']:+.4f} & "
            f"{row['sample_sd_across_seed_means']:.4f} & "
            f"[{row['crossed_bootstrap_ci95_low']:+.4f}, {row['crossed_bootstrap_ci95_high']:+.4f}] & "
            f"{positive} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\caption{Cross-seed controlled decoding effects. Except for the non-serial aggregate diagnostic, effects are oriented so that positive values favor four-candidate constraint reranking. Seed SD is computed across the three seed-level means; intervals use a crossed bootstrap over training seeds and shared aligned test conditions and are not adjusted for multiple endpoints.}",
            "\\label{tab:multiseed-controlled}",
            "\\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_multiseed_controlled(
    seeds: Sequence[int],
    single_paths: Sequence[str | Path],
    reranked_paths: Sequence[str | Path],
    output_json: str | Path,
    output_csv: str | Path,
    output_table: str | Path,
    bootstrap_seed: int = 52042,
    bootstrap_samples: int = 10_000,
    expected_conditions: int | None = None,
) -> dict[str, Any]:
    if len(seeds) < 2 or len(seeds) != len(single_paths) or len(seeds) != len(reranked_paths):
        raise ValueError("Provide aligned --seed, --single, and --reranked values for at least two seeds.")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate training seeds are not allowed: {list(seeds)}")

    runs: list[dict[str, Any]] = []
    reference_ids: list[Any] | None = None
    reference_metadata: list[Any] | None = None
    reference_evaluation_seeds: list[Any] | None = None
    for seed, single_path, reranked_path in zip(seeds, single_paths, reranked_paths):
        single_payload = _load_payload(single_path)
        reranked_payload = _load_payload(reranked_path)
        single, provenance = _validate_pair(
            seed,
            single_payload,
            reranked_payload,
            expected_conditions,
        )
        reranked = reranked_payload["samples"]
        ids = [sample.get("sample_id") for sample in single]
        metadata = [sample.get("metadata") for sample in single]
        evaluation_seeds = [sample.get("evaluation_seed") for sample in single]
        if reference_ids is None:
            reference_ids = ids
            reference_metadata = metadata
            reference_evaluation_seeds = evaluation_seeds
        elif ids != reference_ids or metadata != reference_metadata or evaluation_seeds != reference_evaluation_seeds:
            raise ValueError(f"Seed {seed} does not use the same aligned test conditions and evaluation seeds.")
        runs.append(
            {
                "seed": int(seed),
                "single_path": Path(single_path).as_posix(),
                "reranked_path": Path(reranked_path).as_posix(),
                "single": single,
                "reranked": reranked,
                "sampling_protocol": single_payload.get("sampling_protocol"),
                "generation_batch_size": single_payload.get("generation_batch_size"),
                "provenance": provenance,
            }
        )

    checkpoint_hashes = [run["provenance"]["checkpoint_sha256"] for run in runs]
    if len(set(checkpoint_hashes)) != len(checkpoint_hashes):
        raise ValueError("Training seeds must use distinct checkpoint SHA256 digests.")
    for field in ("data_sha256", "vocab_sha256"):
        if len({run["provenance"][field] for run in runs}) != 1:
            raise ValueError(f"Training seeds do not share the same {field}.")

    rows: list[dict[str, Any]] = []
    for metric_index, spec in enumerate(ENDPOINTS):
        metric = str(spec["metric"])
        subset = str(spec["subset"])
        higher_is_better = spec["higher_is_better"]
        effects = [
            _effect_array(run["single"], run["reranked"], metric, subset, higher_is_better)
            for run in runs
        ]
        counts = {int(values.size) for values in effects}
        if len(counts) != 1:
            raise ValueError(f"Endpoint {metric}:{subset} is not aligned across seeds: {sorted(counts)}")
        seed_means = [float(values.mean()) for values in effects]
        mean_effect = statistics.fmean(seed_means)
        seed_sd = statistics.stdev(seed_means)
        ci_low, ci_high = crossed_bootstrap_ci(
            effects,
            seed=bootstrap_seed + metric_index,
            samples=bootstrap_samples,
        )
        row: dict[str, Any] = {
            "endpoint": f"{metric}:{subset}",
            "metric": metric,
            "label": str(spec["label"]),
            "subset": subset,
            "effect_orientation": (
                "raw reranked minus single; no favorable direction specified"
                if higher_is_better is None
                else "positive favors constraint reranking"
            ),
            "n_seeds": len(runs),
            "n_conditions_per_seed": next(iter(counts)),
            "mean_effect": mean_effect,
            "sample_sd_across_seed_means": seed_sd,
            "crossed_bootstrap_ci95_low": ci_low,
            "crossed_bootstrap_ci95_high": ci_high,
            "positive_seed_count": None if higher_is_better is None else sum(value > 0.0 for value in seed_means),
        }
        for run, value in zip(runs, seed_means):
            row[f"seed_{run['seed']}_mean"] = value
        if not all(
            math.isfinite(float(row[field]))
            for field in (
                "mean_effect",
                "sample_sd_across_seed_means",
                "crossed_bootstrap_ci95_low",
                "crossed_bootstrap_ci95_high",
            )
        ):
            raise ValueError(f"Non-finite aggregate for endpoint {row['endpoint']}")
        rows.append(row)

    public_runs = [{key: value for key, value in run.items() if key not in {"single", "reranked"}} for run in runs]
    result = {
        "comparison": "cross-seed K=4 constraint reranking versus K=1 sampling",
        "seeds": [int(seed) for seed in seeds],
        "n_seeds": len(seeds),
        "paired_conditions_per_seed": len(reference_ids or []),
        "first_candidate_alignment": "verified_by_sha256_for_every_seed_condition",
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_method": "crossed percentile bootstrap over training seeds and shared aligned test conditions",
        "multiple_endpoint_adjustment": "none",
        "runs": public_runs,
        "metrics": rows,
    }
    save_json(result, output_json)
    _write_csv(Path(output_csv), rows, seeds)
    _write_table(Path(output_table), rows)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="append", type=int, required=True)
    parser.add_argument("--single", action="append", required=True)
    parser.add_argument("--reranked", action="append", required=True)
    parser.add_argument("--output-json", default="results/project2_multiseed_controlled_statistics.json")
    parser.add_argument("--output-csv", default="results/project2_multiseed_controlled_statistics.csv")
    parser.add_argument("--output-table", default="paper/tables/project2_multiseed_controlled_results.tex")
    parser.add_argument("--bootstrap-seed", type=int, default=52042)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--expected-conditions", type=int, default=2000)
    args = parser.parse_args(argv)
    result = analyze_multiseed_controlled(
        args.seed,
        args.single,
        args.reranked,
        args.output_json,
        args.output_csv,
        args.output_table,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
        expected_conditions=args.expected_conditions,
    )
    print({"seeds": result["seeds"], "paired_conditions_per_seed": result["paired_conditions_per_seed"], "metrics": len(result["metrics"])})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
