"""Build the manuscript result figure from the canonical full-run CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT / "results" / "project2_metrics.csv"
SOURCE_PATH = ROOT / "paper" / "figures" / "project2_full_results_source.csv"
PDF_PATH = ROOT / "paper" / "figures" / "project2_full_results.pdf"
SVG_PATH = ROOT / "paper" / "figures" / "project2_full_results.svg"
PNG_PATH = ROOT / "paper" / "figures" / "project2_full_results.png"

FULL = "proposed_constraint_guided_transformer"
VANILLA = "vanilla_transformer"

METRICS = [
    ("target_pcset_coverage", "PC-set\ncoverage", "higher"),
    ("interval_vector_distance", "Interval-vector\ndistance", "lower"),
    ("row_order_accuracy", "Row-order\naccuracy", "higher"),
    ("rhythmic_profile_distance", "Rhythmic-profile\ndistance", "lower"),
    ("gesture_consistency_score", "Gesture\nconsistency", "higher"),
]

ABLATIONS = {
    "target_pcset_coverage": "without_pcset_constraints",
    "interval_vector_distance": "without_pcset_constraints",
    "row_order_accuracy": "without_serial_constraints",
    "rhythmic_profile_distance": "without_rhythm_constraints",
    "gesture_consistency_score": "without_gesture_constraints",
}

COLORS = {"reference": "#767676", "comparison": "#0072B2"}


def read_rows() -> dict[str, dict[str, str]]:
    with METRICS_PATH.open(newline="", encoding="utf-8") as handle:
        rows = {row["experiment"]: row for row in csv.DictReader(handle)}
    required = {FULL, VANILLA, *ABLATIONS.values()}
    missing = sorted(required.difference(rows))
    if missing:
        raise RuntimeError(f"Missing full-run rows: {missing}")
    for name in required:
        row = rows[name]
        if row.get("split") != "test" or int(row.get("num_samples", 0)) != 2000:
            raise RuntimeError(f"{name} is not a 2,000-sample test row")
    return rows


def write_source(rows: dict[str, dict[str, str]]) -> None:
    fields = [
        "panel",
        "metric",
        "direction",
        "left_experiment",
        "left_value",
        "right_experiment",
        "right_value",
        "split",
        "num_samples",
    ]
    records: list[dict[str, object]] = []
    for metric, _, direction in METRICS:
        records.append(
            {
                "panel": "candidate_reranking",
                "metric": metric,
                "direction": direction,
                "left_experiment": VANILLA,
                "left_value": rows[VANILLA][metric],
                "right_experiment": FULL,
                "right_value": rows[FULL][metric],
                "split": "test",
                "num_samples": 2000,
            }
        )
        ablation = ABLATIONS[metric]
        records.append(
            {
                "panel": "condition_removal",
                "metric": metric,
                "direction": direction,
                "left_experiment": VANILLA,
                "left_value": rows[VANILLA][metric],
                "right_experiment": ablation,
                "right_value": rows[ablation][metric],
                "split": "test",
                "num_samples": 2000,
            }
        )
    with SOURCE_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def limits(values: list[float]) -> tuple[float, float]:
    low, high = min(values), max(values)
    span = high - low
    if span == 0:
        span = max(abs(high), 1.0) * 0.2
    pad = span * 0.36
    return max(0.0, low - pad), high + pad


def add_pair(
    ax: plt.Axes,
    values: list[float],
    labels: list[str],
    title: str,
    direction: str,
) -> None:
    x = [0, 1]
    ax.plot(x, values, color="#444444", linewidth=1.1, zorder=1)
    ax.scatter(
        x,
        values,
        s=54,
        c=[COLORS["reference"], COLORS["comparison"]],
        edgecolors="white",
        linewidths=0.8,
        zorder=2,
    )
    ax.set_xlim(-0.42, 1.42)
    ax.set_ylim(*limits(values))
    ax.set_xticks(x, labels)
    direction_label = "\u2191 higher" if direction == "higher" else "\u2193 lower"
    ax.set_title(f"{title}\n{direction_label}", fontsize=7.4, pad=4, color="#333333")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.tick_params(axis="both", labelsize=7, length=2.5, width=0.6)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.6)
    value_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for xi, value in zip(x, values):
        if 0.99 <= value < 1:
            label = f"{value:.4f}"
        elif value < 1:
            label = f"{value:.3f}"
        else:
            label = f"{value:.2f}"
        ax.text(
            xi,
            value + 0.045 * value_span,
            label,
            ha="center",
            va="bottom",
            fontsize=6.8,
            color="#202020",
        )


def build_figure(rows: dict[str, dict[str, str]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 5, figsize=(7.1, 4.05), constrained_layout=False)
    fig.subplots_adjust(left=0.072, right=0.995, top=0.80, bottom=0.13, wspace=0.62, hspace=1.05)

    for col, (metric, title, direction) in enumerate(METRICS):
        add_pair(
            axes[0, col],
            [float(rows[VANILLA][metric]), float(rows[FULL][metric])],
            ["K=1", "K=4"],
            title,
            direction,
        )
        ablation = ABLATIONS[metric]
        short = {
            "without_pcset_constraints": "No PC",
            "without_serial_constraints": "No row",
            "without_rhythm_constraints": "No rhythm",
            "without_gesture_constraints": "No gesture",
        }[ablation]
        add_pair(
            axes[1, col],
            [float(rows[VANILLA][metric]), float(rows[ablation][metric])],
            ["Full K=1", short],
            title,
            direction,
        )

    fig.text(0.018, 0.965, "a", fontsize=11, fontweight="bold", va="top")
    fig.text(0.047, 0.963, "Guided candidate selection: shared generator", fontsize=8.3, fontweight="bold", va="top")
    fig.text(0.018, 0.505, "b", fontsize=11, fontweight="bold", va="top")
    fig.text(0.047, 0.503, "Condition removal: separately trained K=1 decoders", fontsize=8.3, fontweight="bold", va="top")
    fig.text(
        0.5,
        0.035,
        "Automatic test-set metrics (2,000 conditions per experiment)",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color="#444444",
    )

    fig.savefig(PDF_PATH, bbox_inches="tight")
    fig.savefig(SVG_PATH, bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = read_rows()
    write_source(rows)
    build_figure(rows)
    print(f"Wrote {PDF_PATH.relative_to(ROOT)}")
    print(f"Wrote {SVG_PATH.relative_to(ROOT)}")
    print(f"Wrote {PNG_PATH.relative_to(ROOT)}")
    print(f"Wrote {SOURCE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
