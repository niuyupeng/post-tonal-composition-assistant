"""Build CSV and LaTeX summary tables for Project 2 experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from post_tonal.evaluate import CONSTRAINT_FIELDS, METRIC_FIELDS
from post_tonal.utils import ensure_dir


MAIN_EXPERIMENTS = {
    "rule_baseline",
    "vanilla_transformer",
    "proposed_constraint_guided_transformer",
}


def _read_rows(metrics_csv: Path) -> list[dict[str, Any]]:
    if metrics_csv.exists():
        with open(metrics_csv, newline="", encoding="utf-8") as f:
            return [dict(row) for row in csv.DictReader(f)]
    rows: list[dict[str, Any]] = []
    for path in sorted(Path("results").glob("*_metrics.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "experiment" not in data:
            data["experiment"] = path.stem.replace("_metrics", "")
        rows.append(data)
    return rows


def _format(value: Any) -> str:
    if value in (None, "", "None"):
        return "PENDING_REAL_EXPERIMENT"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value).replace("_", "\\_")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_table(path: Path, rows: list[dict[str, Any]], caption: str) -> None:
    ensure_dir(path.parent)
    fields = [
        "experiment",
        "token_accuracy",
        "target_pcset_coverage",
        "row_order_accuracy",
        "rhythmic_profile_distance",
        "gesture_consistency_score",
        "musicxml_export_success_rate",
    ]
    header = "Experiment & Token acc. & PC cov. & Row acc. & Rhythm dist. & Gesture & XML \\\\"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        header,
        "\\midrule",
    ]
    for row in rows:
        cells = [_format(row.get(field)) for field in fields]
        lines.append(" & ".join(cells) + " \\\\")
    if not rows:
        lines.append("PENDING\\_REAL\\_EXPERIMENT & PENDING\\_REAL\\_EXPERIMENT & PENDING\\_REAL\\_EXPERIMENT & PENDING\\_REAL\\_EXPERIMENT & PENDING\\_REAL\\_EXPERIMENT & PENDING\\_REAL\\_EXPERIMENT & PENDING\\_REAL\\_EXPERIMENT \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def make_tables(
    metrics_csv: str | Path = "results/project2_metrics.csv",
    constraints_csv: str | Path = "results/project2_constraints.csv",
    main_table: str | Path = "paper/tables/project2_main_results.tex",
    ablation_table: str | Path = "paper/tables/project2_ablation_results.tex",
) -> dict[str, int]:
    metrics_path = Path(metrics_csv)
    rows = _read_rows(metrics_path)
    if rows:
        _write_csv(metrics_path, rows, METRIC_FIELDS)
        _write_csv(Path(constraints_csv), rows, CONSTRAINT_FIELDS)
    main_rows = [row for row in rows if row.get("experiment") in MAIN_EXPERIMENTS]
    ablation_rows = [row for row in rows if row.get("experiment") not in MAIN_EXPERIMENTS]
    _write_table(Path(main_table), main_rows, "Main Project 2 results. Values are computed only for completed local runs.")
    _write_table(Path(ablation_table), ablation_rows, "Project 2 ablation results. Missing values remain marked as pending.")
    return {"rows": len(rows), "main_rows": len(main_rows), "ablation_rows": len(ablation_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", default="results/project2_metrics.csv")
    parser.add_argument("--constraints-csv", default="results/project2_constraints.csv")
    parser.add_argument("--main-table", default="paper/tables/project2_main_results.tex")
    parser.add_argument("--ablation-table", default="paper/tables/project2_ablation_results.tex")
    args = parser.parse_args()
    print(make_tables(args.metrics_csv, args.constraints_csv, args.main_table, args.ablation_table))


if __name__ == "__main__":
    main()
