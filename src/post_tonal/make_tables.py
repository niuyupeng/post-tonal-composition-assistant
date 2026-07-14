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

DISPLAY_NAMES = {
    "rule_baseline": "Rule reference",
    "vanilla_transformer": "Independent vanilla",
    "proposed_constraint_guided_transformer": "Original guided",
    "transformer_no_constraints": "No constraints (seed 44)",
    "without_pcset_constraints": "No pc-set token",
    "without_serial_constraints": "No row token",
    "without_rhythm_constraints": "Fixed rhythm",
    "without_gesture_constraints": "Fixed gesture",
    "serial_only": "Serial only",
    "pcset_only": "PC-set only",
    "rhythm_only": "Rhythm only",
    "gesture_only": "Gesture only",
    "no_constraints": "No constraints (seed 53)",
}

NO_PCSET_TARGET = {
    "transformer_no_constraints",
    "without_pcset_constraints",
    "serial_only",
    "rhythm_only",
    "gesture_only",
    "no_constraints",
}
NO_SERIAL_TARGET = {
    "transformer_no_constraints",
    "without_serial_constraints",
    "pcset_only",
    "rhythm_only",
    "gesture_only",
    "no_constraints",
}
NO_VARIABLE_RHYTHM_TARGET = {
    "transformer_no_constraints",
    "without_rhythm_constraints",
    "serial_only",
    "pcset_only",
    "gesture_only",
    "no_constraints",
}
NO_VARIABLE_GESTURE_TARGET = {
    "transformer_no_constraints",
    "without_gesture_constraints",
    "serial_only",
    "pcset_only",
    "rhythm_only",
    "no_constraints",
}


def _read_rows(metrics_csv: Path) -> list[dict[str, Any]]:
    if metrics_csv.exists():
        with metrics_csv.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
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
        return "--"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value).replace("_", "\\_")


def _applicable(experiment: str, field: str) -> bool:
    if field == "target_pcset_coverage" and experiment in NO_PCSET_TARGET:
        return False
    if field == "row_order_accuracy" and experiment in NO_SERIAL_TARGET:
        return False
    if field == "rhythmic_profile_distance" and experiment in NO_VARIABLE_RHYTHM_TARGET:
        return False
    if field == "gesture_consistency_score" and experiment in NO_VARIABLE_GESTURE_TARGET:
        return False
    return True


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_table(path: Path, rows: list[dict[str, Any]], caption: str, label: str) -> None:
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
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Experiment & Token acc. & PC cov. & Row acc. & Rhythm dist. & Gesture & XML \\\\",
        "\\midrule",
    ]
    for row in rows:
        experiment = str(row.get("experiment", ""))
        cells = [DISPLAY_NAMES.get(experiment, experiment).replace("_", "\\_")]
        for field in fields[1:]:
            cells.append(_format(row.get(field)) if _applicable(experiment, field) else "--")
        lines.append(" & ".join(cells) + " \\\\")
    if not rows:
        lines.append("PENDING\\_REAL\\_EXPERIMENT & -- & -- & -- & -- & -- & -- \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
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
    _write_table(
        Path(main_table),
        main_rows,
        "Archived aggregate results. Independent neural rows use different generated corpora and are descriptive rather than paired comparisons. Row accuracy is averaged only over row-conditioned samples, so its denominator differs from the 2,000-item test-set total and can vary by configuration.",
        "tab:project2-main-results",
    )
    _write_table(
        Path(ablation_table),
        ablation_rows,
        "Exploratory condition and fixed-default configurations. Dashes denote metrics that are not applicable because the corresponding target is absent or fixed. Row accuracy is averaged only over row-conditioned samples.",
        "tab:project2-ablation-results",
    )
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
