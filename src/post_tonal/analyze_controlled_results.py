"""Paired statistics for controlled single-candidate versus reranked decoding."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from post_tonal.utils import ensure_dir, save_json


ENDPOINTS: list[dict[str, Any]] = [
    {"metric": "pcset_coverage", "label": "PC coverage", "subset": "all", "higher_is_better": True},
    {
        "metric": "pcset_coverage",
        "label": "PC coverage",
        "subset": "non-serial",
        "higher_is_better": True,
    },
    {
        "metric": "pcset_coverage",
        "label": "PC coverage",
        "subset": "serial",
        "higher_is_better": True,
    },
    {
        "metric": "interval_vector_distance",
        "label": "IV distance",
        "subset": "all",
        "higher_is_better": False,
    },
    {
        "metric": "interval_vector_distance",
        "label": "IV distance",
        "subset": "non-serial",
        "higher_is_better": False,
    },
    {
        "metric": "interval_vector_distance",
        "label": "IV distance",
        "subset": "serial",
        "higher_is_better": False,
    },
    {
        "metric": "row_order_accuracy",
        "label": "Row accuracy",
        "subset": "serial",
        "higher_is_better": True,
    },
    {
        "metric": "aggregate_completion_rate",
        "label": "Aggregate",
        "subset": "all",
        "higher_is_better": True,
    },
    {
        "metric": "aggregate_completion_rate",
        "label": "Aggregate",
        "subset": "serial",
        "higher_is_better": True,
    },
    {
        "metric": "aggregate_completion_rate",
        "label": "Aggregate (diagnostic)",
        "subset": "non-serial",
        "higher_is_better": None,
    },
    {
        "metric": "rhythmic_profile_distance",
        "label": "Rhythm distance",
        "subset": "all",
        "higher_is_better": False,
    },
    {
        "metric": "density_curve_error",
        "label": "Density error",
        "subset": "all",
        "higher_is_better": False,
    },
    {
        "metric": "gesture_consistency_score",
        "label": "Gesture score",
        "subset": "all",
        "higher_is_better": True,
    },
    {
        "metric": "range_violation_rate",
        "label": "Range violation",
        "subset": "all",
        "higher_is_better": False,
    },
]


def paired_bootstrap_ci(
    improvements: np.ndarray,
    seed: int = 42042,
    samples: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(improvements, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Expected a nonempty one-dimensional array of paired improvements.")
    rng = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    remaining = int(samples)
    while remaining > 0:
        chunk = min(500, remaining)
        indices = rng.integers(0, values.size, size=(chunk, values.size))
        means.append(values[indices].mean(axis=1))
        remaining -= chunk
    bootstrap_means = np.concatenate(means)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _load_samples(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"No per-sample records found in {path}.")
    return samples


def _sample_in_subset(sample: dict[str, Any], subset: str) -> bool:
    if subset == "all":
        return True
    metadata = sample.get("metadata", {})
    is_serial = bool(metadata.get("row") and metadata.get("row_form"))
    if subset == "serial":
        return is_serial
    if subset == "non-serial":
        return not is_serial
    raise ValueError(f"Unknown controlled-analysis subset: {subset}")


def analyze_controlled_results(
    single_path: str | Path,
    reranked_path: str | Path,
    output_json: str | Path,
    output_csv: str | Path,
    output_table: str | Path,
    bootstrap_seed: int = 42042,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    single = _load_samples(single_path)
    reranked = _load_samples(reranked_path)
    if len(single) != len(reranked):
        raise ValueError("Controlled result files have different sample counts.")

    aligned_first_candidates = 0
    for left, right in zip(single, reranked):
        if left.get("sample_id") != right.get("sample_id"):
            raise ValueError("Controlled result sample IDs are not aligned.")
        if left.get("metadata") != right.get("metadata"):
            raise ValueError(f"Condition metadata differs for sample {left.get('sample_id')}.")
        if left.get("evaluation_seed") != right.get("evaluation_seed"):
            raise ValueError(f"Evaluation seeds differ for sample {left.get('sample_id')}.")
        left_hash = left.get("first_candidate_sha256")
        right_hash = right.get("first_candidate_sha256")
        if (left_hash is None) != (right_hash is None):
            raise ValueError(f"First-candidate fingerprints are incomplete for sample {left.get('sample_id')}.")
        if left_hash is not None:
            if left_hash != right_hash:
                raise ValueError(f"First candidates differ for sample {left.get('sample_id')}.")
            aligned_first_candidates += 1

    rows: list[dict[str, Any]] = []
    for metric_index, spec in enumerate(ENDPOINTS):
        metric = str(spec["metric"])
        subset = str(spec["subset"])
        paired: list[tuple[float, float]] = []
        for left, right in zip(single, reranked):
            if not _sample_in_subset(left, subset):
                continue
            left_value = left.get("analysis", {}).get(metric)
            right_value = right.get("analysis", {}).get(metric)
            if left_value is None or right_value is None:
                continue
            paired.append((float(left_value), float(right_value)))
        if not paired:
            continue
        single_values = np.asarray([pair[0] for pair in paired], dtype=np.float64)
        reranked_values = np.asarray([pair[1] for pair in paired], dtype=np.float64)
        raw_difference = reranked_values - single_values
        higher_is_better = spec["higher_is_better"]
        if higher_is_better is None:
            effects = raw_difference
            effect_orientation = "raw reranked minus single; no favorable direction specified"
        else:
            effects = raw_difference if higher_is_better else -raw_difference
            effect_orientation = "positive favors constraint reranking"
        ci_low, ci_high = paired_bootstrap_ci(
            effects,
            seed=bootstrap_seed + metric_index,
            samples=bootstrap_samples,
        )
        tolerance = 1e-12
        rows.append(
            {
                "endpoint": f"{metric}:{subset}",
                "metric": metric,
                "label": spec["label"],
                "subset": subset,
                "higher_is_better": higher_is_better,
                "effect_orientation": effect_orientation,
                "n": int(len(paired)),
                "single_mean": float(single_values.mean()),
                "reranked_mean": float(reranked_values.mean()),
                "raw_difference_reranked_minus_single": float(raw_difference.mean()),
                "favorable_improvement": None if higher_is_better is None else float(effects.mean()),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "win_rate": None if higher_is_better is None else float(np.mean(effects > tolerance)),
                "tie_rate": None if higher_is_better is None else float(np.mean(np.abs(effects) <= tolerance)),
                "loss_rate": None if higher_is_better is None else float(np.mean(effects < -tolerance)),
            }
        )

    result = {
        "comparison": "same-checkpoint K=4 constraint reranking versus K=1 sampling",
        "single_path": str(single_path),
        "reranked_path": str(reranked_path),
        "paired_conditions": len(single),
        "first_candidate_alignment": (
            "verified_by_sha256"
            if aligned_first_candidates == len(single)
            else "not_recorded"
        ),
        "first_candidate_fingerprints_verified": aligned_first_candidates,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_method": "paired percentile bootstrap over test conditions",
        "multiple_endpoint_adjustment": "none",
        "improvement_definition": "positive favorable_improvement values favor constraint reranking; endpoints without a prespecified favorable direction report only raw change",
        "metrics": rows,
    }
    save_json(result, output_json)

    csv_path = Path(output_csv)
    ensure_dir(csv_path.parent)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["metric"])
        writer.writeheader()
        writer.writerows(rows)

    table_path = Path(output_table)
    ensure_dir(table_path.parent)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Metric & Subset & $K=1$ & Reranked $K=4$ & Favorable $\\Delta$ & 95\\% CI & $n$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        if row["favorable_improvement"] is None:
            effect_value = row["raw_difference_reranked_minus_single"]
            effect_cell = f"{effect_value:+.4f}$^{{\\dagger}}$"
        else:
            effect_cell = f"{row['favorable_improvement']:+.4f}"
        lines.append(
            f"{row['label']} & {row['subset']} & {row['single_mean']:.4f} & {row['reranked_mean']:.4f} & "
            f"{effect_cell} & "
            f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] & {row['n']} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\caption{Controlled same-checkpoint decoding comparison. Favorable differences are oriented so that positive values favor four-candidate constraint reranking. Confidence intervals are paired percentile bootstrap intervals over test conditions and are not adjusted for multiple endpoints. $^{\\dagger}$The non-serial aggregate row is a raw diagnostic change because no aggregate target is present for that subset.}",
            "\\label{tab:controlled-decoding}",
            "\\end{table*}",
            "",
        ]
    )
    table_path.write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", required=True)
    parser.add_argument("--reranked", required=True)
    parser.add_argument("--output-json", default="results/project2_controlled_statistics.json")
    parser.add_argument("--output-csv", default="results/project2_controlled_statistics.csv")
    parser.add_argument("--output-table", default="paper/tables/project2_controlled_results.tex")
    parser.add_argument("--bootstrap-seed", type=int, default=42042)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    result = analyze_controlled_results(
        args.single,
        args.reranked,
        args.output_json,
        args.output_csv,
        args.output_table,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    print({"paired_conditions": result["paired_conditions"], "metrics": len(result["metrics"])})


if __name__ == "__main__":
    main()
