"""Plot favorable relative changes for the controlled decoder comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


SELECTED_ENDPOINTS = [
    ("pcset_coverage:non-serial", "PC-set coverage\n(non-serial)"),
    ("interval_vector_distance:non-serial", "Interval-vector distance\n(non-serial)"),
    ("row_order_accuracy:serial", "Row-order accuracy\n(serial)"),
    ("aggregate_completion_rate:serial", "Aggregate completion\n(serial)"),
    ("rhythmic_profile_distance:all", "Rhythmic-profile distance\n(all)"),
    ("gesture_consistency_score:all", "Gesture consistency\n(all)"),
]


def make_figure(stats_path: Path, output_stem: Path) -> None:
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    by_endpoint = {row["endpoint"]: row for row in payload["metrics"]}

    source_rows: list[dict[str, float | int | str]] = []
    for endpoint, display_label in SELECTED_ENDPOINTS:
        row = by_endpoint[endpoint]
        baseline = abs(float(row["single_mean"]))
        if baseline == 0.0:
            raise ValueError(f"Cannot express relative change for zero baseline: {endpoint}")
        scale = 100.0 / baseline
        source_rows.append(
            {
                "endpoint": endpoint,
                "display_label": display_label.replace("\n", " "),
                "n": int(row["n"]),
                "single_mean": float(row["single_mean"]),
                "reranked_mean": float(row["reranked_mean"]),
                "favorable_relative_change_percent": float(row["favorable_improvement"]) * scale,
                "ci95_low_percent": float(row["ci95_low"]) * scale,
                "ci95_high_percent": float(row["ci95_high"]) * scale,
            }
        )

    source_path = output_stem.with_name(output_stem.name + "_source.csv")
    source_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(6.9, 4.4))
    y_positions = list(range(len(source_rows)))[::-1]
    values = [float(row["favorable_relative_change_percent"]) for row in source_rows]
    lows = [float(row["ci95_low_percent"]) for row in source_rows]
    highs = [float(row["ci95_high_percent"]) for row in source_rows]
    lower_errors = [value - low for value, low in zip(values, lows)]
    upper_errors = [high - value for value, high in zip(values, highs)]

    ax.axvline(0.0, color="#777777", linewidth=0.9, linestyle="--", zorder=1)
    ax.errorbar(
        values,
        y_positions,
        xerr=[lower_errors, upper_errors],
        fmt="o",
        color="#1f5a78",
        ecolor="#1f5a78",
        markerfacecolor="white",
        markeredgewidth=1.2,
        markersize=5.5,
        capsize=3,
        linewidth=1.2,
        zorder=2,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [f"{label}  (n={row['n']})" for (_, label), row in zip(SELECTED_ENDPOINTS, source_rows)]
    )
    ax.set_xlabel("Favorable change relative to single-candidate decoding (%)")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()

    for suffix in ("pdf", "svg"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, default=Path("results/project2_controlled_statistics.json"))
    parser.add_argument("--output-stem", type=Path, default=Path("paper/figures/controlled_effects"))
    args = parser.parse_args()
    make_figure(args.stats, args.output_stem)


if __name__ == "__main__":
    main()
