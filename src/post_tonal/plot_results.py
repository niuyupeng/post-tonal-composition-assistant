"""Create lightweight SVG plots from Project 2 CSV metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from post_tonal.utils import ensure_dir


def _float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def plot_results(metrics_csv: str | Path = "results/project2_metrics.csv", output: str | Path = "results/project2_constraint_summary.svg") -> Path:
    metrics_path = Path(metrics_csv)
    rows: list[dict[str, Any]] = []
    if metrics_path.exists():
        with open(metrics_path, newline="", encoding="utf-8") as f:
            rows = [dict(row) for row in csv.DictReader(f)]

    output_path = Path(output)
    ensure_dir(output_path.parent)
    width = 980
    row_height = 34
    height = max(160, 80 + row_height * max(1, len(rows)))
    metrics = [
        ("target_pcset_coverage", "#2f6f9f"),
        ("row_order_accuracy", "#6b8e23"),
        ("gesture_consistency_score", "#9f5f2f"),
        ("musicxml_export_success_rate", "#7b4fa3"),
    ]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="32" font-family="Arial" font-size="18" font-weight="700">Project 2 Constraint Summary</text>',
    ]
    if not rows:
        lines.append('<text x="20" y="72" font-family="Arial" font-size="14">PENDING_REAL_EXPERIMENT</text>')
    for row_idx, row in enumerate(rows):
        y = 70 + row_idx * row_height
        exp = str(row.get("experiment", "experiment")).replace("&", "&amp;")
        lines.append(f'<text x="20" y="{y + 14}" font-family="Arial" font-size="12">{exp}</text>')
        x = 260
        for metric, color in metrics:
            value = max(0.0, min(1.0, _float(row, metric)))
            bar_width = int(value * 130)
            lines.append(f'<rect x="{x}" y="{y}" width="130" height="14" fill="#eeeeee"/>')
            lines.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="14" fill="{color}"/>')
            lines.append(f'<text x="{x}" y="{y + 28}" font-family="Arial" font-size="10">{metric}: {value:.2f}</text>')
            x += 170
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", default="results/project2_metrics.csv")
    parser.add_argument("--output", default="results/project2_constraint_summary.svg")
    args = parser.parse_args()
    print(plot_results(args.metrics_csv, args.output))


if __name__ == "__main__":
    main()
