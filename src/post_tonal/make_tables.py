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
    "vanilla_transformer": "Shared generator, K=1",
    "proposed_constraint_guided_transformer": "Shared generator, K=4 guided",
    "transformer_no_constraints": "No constraints (seed 42)",
    "without_pcset_constraints": "No pc-set token",
    "without_serial_constraints": "No row token",
    "without_rhythm_constraints": "No rhythm token",
    "without_gesture_constraints": "No gesture token",
    "serial_only": "Serial only",
    "pcset_only": "PC-set only",
    "rhythm_only": "Rhythm only",
    "gesture_only": "Gesture only",
    "no_constraints": "No constraints (seed 53)",
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
        "pcset_precision",
        "interval_vector_distance",
        "row_order_accuracy",
        "aggregate_completion_rate",
        "serial_transformation_accuracy",
        "rhythmic_profile_distance",
        "density_curve_error",
        "gesture_consistency_score",
        "range_violation_rate",
        "content_span_ratio",
        "voice_count_adherence",
        "musicxml_export_success_rate",
        "musicxml_measure_adherence_rate",
    ]
    lines = [
        "\\begin{sidewaystable}[p]",
        "\\centering",
        "\\small",
        "\\resizebox{\\textheight}{!}{%",
        "\\begin{tabular}{lrrrrrrrrrrrrrrr}",
        "\\toprule",
        "Experiment & Token acc. & PC cov. & PC prec. & IV dist. & Row acc. & Aggregate & Form acc. & Rhythm dist. & Density err. & Gesture & Range viol. & Span & Content voices & XML parse & XML measures \\\\",
        "\\midrule",
    ]
    for row in rows:
        experiment = str(row.get("experiment", ""))
        cells = [DISPLAY_NAMES.get(experiment, experiment).replace("_", "\\_")]
        for field in fields[1:]:
            cells.append(_format(row.get(field)))
        lines.append(" & ".join(cells) + " \\\\")
    if not rows:
        lines.append(
            "PENDING\\_REAL\\_EXPERIMENT & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{sidewaystable}",
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
        "Automatic results on the fixed 2,000-condition procedural test split. The guided and single-candidate rows share one checkpoint; the rule reference implements constraints directly. Row and serial-form accuracy are averaged over row-conditioned samples. XML parse and measure adherence are measured over 20 attempted exports per configuration; XML part-count adherence is 1.0000 for all three rows.",
        "tab:project2-main-results",
    )
    _write_table(
        Path(ablation_table),
        ablation_rows,
        "Single-candidate condition-prefix ablations and focused-condition models on the fixed procedural test split. Each model sees its configured prefix while evaluation retains the original held-out targets. Row and serial-form accuracy are averaged over row-conditioned samples. XML parse and measure adherence are measured over 20 attempted exports per configuration; XML part-count adherence is 1.0000 for all rows.",
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
