"""Aggregate aligned training and test artifacts across random seeds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Sequence


PER_SEED_FIELDS = [
    "seed",
    "epochs_ran",
    "best_epoch",
    "best_val_loss",
    "test_token_accuracy",
    "test_model_loss",
    "test_num_samples",
    "batch_size",
    "gradient_accumulation_steps",
    "effective_batch_size",
    "checkpoint_path",
    "checkpoint_sha256",
    "evaluation_path",
]

SCIENTIFIC_METRICS = (
    "best_val_loss",
    "test_token_accuracy",
    "test_model_loss",
)

_RUN_SEED_RE = re.compile(r"^seed_(\d+)$", re.IGNORECASE)
_EMBEDDED_SEED_RE = re.compile(r"(?:^|[_-])seed[_-]?(\d+)(?:$|[_-])", re.IGNORECASE)


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    if not path.is_file():
        raise ValueError(f"Expected {description} to be a file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Expected non-empty {description}: {path}")


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    _require_file(path, description)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {description} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {description}: {path}")
    return payload


def _positive_int(value: Any, field: str, source: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Expected positive integer {field!r} in {source}; got {value!r}")
    return value


def _finite_float(value: Any, field: str, source: Path) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Expected finite numeric {field!r} in {source}; got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected finite numeric {field!r} in {source}; got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Expected finite numeric {field!r} in {source}; got {value!r}")
    return result


def _seed_from_run_dir(run_dir: Path) -> int:
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run directory: {run_dir}")
    if not run_dir.is_dir():
        raise ValueError(f"Expected --run-dir to be a directory: {run_dir}")
    match = _RUN_SEED_RE.fullmatch(run_dir.name)
    if match is None:
        raise ValueError(
            f"Cannot derive seed from run directory {run_dir}; expected a basename such as seed_42"
        )
    return int(match.group(1))


def _embedded_seed(value: str) -> int | None:
    match = _EMBEDDED_SEED_RE.search(value)
    return None if match is None else int(match.group(1))


def _evaluation_seed(payload: dict[str, Any], path: Path) -> int | None:
    candidates: set[int] = set()
    for field in ("seed", "training_seed"):
        if field not in payload:
            continue
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Invalid {field!r} in evaluation JSON {path}: {value!r}")
        candidates.add(value)

    experiment = payload.get("experiment")
    if experiment is not None:
        if not isinstance(experiment, str):
            raise ValueError(f"Invalid 'experiment' in evaluation JSON {path}: {experiment!r}")
        experiment_seed = _embedded_seed(experiment)
        if experiment_seed is not None:
            candidates.add(experiment_seed)

    for part in path.parts:
        path_seed = _embedded_seed(part)
        if path_seed is not None:
            candidates.add(path_seed)

    if len(candidates) > 1:
        raise ValueError(f"Conflicting seed identifiers in evaluation JSON {path}: {sorted(candidates)}")
    return next(iter(candidates), None)


def _best_validation_epoch(metrics_path: Path) -> tuple[int, float, int]:
    _require_file(metrics_path, "training metrics CSV")
    try:
        handle = metrics_path.open("r", newline="", encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Cannot open training metrics CSV {metrics_path}: {exc}") from exc

    with handle:
        reader = csv.DictReader(handle)
        required = {"epoch", "val_loss"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Training metrics CSV {metrics_path} is missing columns: {', '.join(sorted(missing))}"
            )
        history: list[tuple[int, float]] = []
        for line_number, row in enumerate(reader, start=2):
            raw_epoch = row.get("epoch")
            try:
                epoch = int(raw_epoch) if raw_epoch is not None else 0
            except ValueError as exc:
                raise ValueError(
                    f"Invalid epoch in {metrics_path} at line {line_number}: {raw_epoch!r}"
                ) from exc
            if epoch < 1 or str(epoch) != str(raw_epoch).strip():
                raise ValueError(
                    f"Invalid epoch in {metrics_path} at line {line_number}: {raw_epoch!r}"
                )
            val_loss = _finite_float(row.get("val_loss"), "val_loss", metrics_path)
            history.append((epoch, val_loss))

    if not history:
        raise ValueError(f"Training metrics CSV has no epoch rows: {metrics_path}")
    epochs = [epoch for epoch, _ in history]
    expected_epochs = list(range(1, len(history) + 1))
    if epochs != expected_epochs:
        raise ValueError(
            f"Training metrics CSV {metrics_path} has non-sequential epochs: {epochs!r}"
        )
    best_epoch, best_val_loss = min(history, key=lambda item: (item[1], item[0]))
    return best_epoch, best_val_loss, len(history)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_seed_row(run_dir: Path, evaluation_path: Path) -> dict[str, Any]:
    seed = _seed_from_run_dir(run_dir)
    summary_path = run_dir / "train_summary.json"
    metrics_path = run_dir / "metrics.csv"
    checkpoint_path = run_dir / "checkpoint.pt"

    summary = _load_json_object(summary_path, "training summary")
    best_epoch, best_val_loss, metrics_epochs = _best_validation_epoch(metrics_path)
    _require_file(checkpoint_path, "checkpoint")
    evaluation = _load_json_object(evaluation_path, "evaluation JSON")

    evaluation_seed = _evaluation_seed(evaluation, evaluation_path)
    if evaluation_seed is not None and evaluation_seed != seed:
        raise ValueError(
            f"Misaligned inputs: run {run_dir} is seed {seed}, but {evaluation_path} identifies seed "
            f"{evaluation_seed}"
        )
    split = evaluation.get("split")
    if split is not None and (not isinstance(split, str) or split.lower() != "test"):
        raise ValueError(f"Expected test-split evaluation JSON {evaluation_path}; got split {split!r}")

    epochs_ran = _positive_int(summary.get("epochs_ran"), "epochs_ran", summary_path)
    if epochs_ran != metrics_epochs:
        raise ValueError(
            f"Epoch count mismatch for seed {seed}: train_summary.json reports {epochs_ran}, "
            f"but metrics.csv has {metrics_epochs} rows"
        )
    summary_best_val_loss = _finite_float(
        summary.get("best_val_loss"), "best_val_loss", summary_path
    )
    if not math.isclose(summary_best_val_loss, best_val_loss, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            f"Best validation loss mismatch for seed {seed}: train_summary.json reports "
            f"{summary_best_val_loss}, but metrics.csv minimum is {best_val_loss}"
        )

    batch_size = _positive_int(summary.get("batch_size"), "batch_size", summary_path)
    accumulation_steps = _positive_int(
        summary.get("gradient_accumulation_steps"),
        "gradient_accumulation_steps",
        summary_path,
    )
    effective_batch_size = _positive_int(
        summary.get("effective_batch_size"), "effective_batch_size", summary_path
    )
    expected_effective_batch_size = batch_size * accumulation_steps
    if effective_batch_size != expected_effective_batch_size:
        raise ValueError(
            f"Effective batch mismatch for seed {seed}: expected {expected_effective_batch_size} "
            f"from batch_size * gradient_accumulation_steps, got {effective_batch_size}"
        )

    return {
        "seed": seed,
        "epochs_ran": epochs_ran,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_token_accuracy": _finite_float(
            evaluation.get("token_accuracy"), "token_accuracy", evaluation_path
        ),
        "test_model_loss": _finite_float(
            evaluation.get("model_loss"), "model_loss", evaluation_path
        ),
        "test_num_samples": _positive_int(
            evaluation.get("num_samples"), "num_samples", evaluation_path
        ),
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "evaluation_path": evaluation_path.as_posix(),
    }


def _aggregate_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    aggregate: dict[str, dict[str, float | None]] = {}
    for metric in SCIENTIFIC_METRICS:
        values = [float(row[metric]) for row in rows]
        aggregate[metric] = {
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values) if len(values) >= 2 else None,
        }
    return aggregate


def _write_metrics_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_SEED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _mean_sd_cell(summary: dict[str, Any], metric: str) -> str:
    metric_summary = summary["aggregate"][metric]
    mean = metric_summary["mean"]
    sample_std = metric_summary["sample_std"]
    std_text = "--" if sample_std is None else f"{sample_std:.4f}"
    return f"{mean:.4f} $\\pm$ {std_text}"


def _write_latex_table(path: Path, rows: Sequence[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Seed & Epochs & Best epoch & Val. loss & Test token acc. & Test model loss & Eff. batch \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['seed']} & {row['epochs_ran']} & {row['best_epoch']} & "
            f"{row['best_val_loss']:.4f} & {row['test_token_accuracy']:.4f} & "
            f"{row['test_model_loss']:.4f} & {row['effective_batch_size']} \\\\"
        )
    lines.extend(
        [
            "\\midrule",
            "Mean $\\pm$ SD & -- & -- & "
            f"{_mean_sd_cell(summary, 'best_val_loss')} & "
            f"{_mean_sd_cell(summary, 'test_token_accuracy')} & "
            f"{_mean_sd_cell(summary, 'test_model_loss')} & -- \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Independent-seed training replication on a fixed synthetic corpus. "
            "Test metrics are teacher-forced; constraint-guided decoding is evaluated "
            "separately in the controlled seed-42 experiment.}",
            "\\label{tab:multiseed-training}",
            "\\end{table}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_multiseed_training(
    run_dirs: Sequence[str | Path],
    evaluation_jsons: Sequence[str | Path],
    metrics_csv: str | Path,
    summary_json: str | Path,
    latex_table: str | Path,
) -> dict[str, Any]:
    """Validate aligned artifacts and write deterministic per-seed and aggregate outputs."""
    if not run_dirs:
        raise ValueError("At least one --run-dir is required")
    if len(run_dirs) != len(evaluation_jsons):
        raise ValueError(
            "--run-dir and --evaluation-json must be repeated the same number of times "
            f"(got {len(run_dirs)} and {len(evaluation_jsons)})"
        )

    output_paths = [Path(metrics_csv), Path(summary_json), Path(latex_table)]
    resolved_outputs = [path.resolve() for path in output_paths]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("--metrics-csv, --summary-json, and --latex-table must be distinct paths")

    pairs = [(Path(run_dir), Path(evaluation)) for run_dir, evaluation in zip(run_dirs, evaluation_jsons)]
    resolved_evaluations = [evaluation.resolve() for _, evaluation in pairs]
    if len(set(resolved_evaluations)) != len(resolved_evaluations):
        raise ValueError("Each --evaluation-json must identify a distinct file")

    rows = [_build_seed_row(run_dir, evaluation) for run_dir, evaluation in pairs]
    rows.sort(key=lambda row: row["seed"])
    seeds = [int(row["seed"]) for row in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate run seeds are not allowed: {seeds}")
    test_sample_counts = sorted({int(row["test_num_samples"]) for row in rows})
    if len(test_sample_counts) != 1:
        raise ValueError(
            "All evaluations must use the same test-set size; got "
            f"{test_sample_counts}"
        )

    summary = {
        "n": len(rows),
        "seeds": seeds,
        "test_num_samples_per_seed": test_sample_counts[0],
        "sample_standard_deviation_ddof": 1,
        "aggregate": _aggregate_metrics(rows),
        "per_seed": rows,
    }
    _write_metrics_csv(output_paths[0], rows)
    _write_summary_json(output_paths[1], summary)
    _write_latex_table(output_paths[2], rows, summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Training run directory named seed_<integer>; repeat once per seed.",
    )
    parser.add_argument(
        "--evaluation-json",
        action="append",
        required=True,
        help="Aligned test evaluation JSON; repeat in the same order as --run-dir.",
    )
    parser.add_argument("--metrics-csv", required=True, help="Output path for per-seed CSV rows.")
    parser.add_argument("--summary-json", required=True, help="Output path for aggregate JSON.")
    parser.add_argument("--latex-table", required=True, help="Output path for the compact LaTeX table.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        summary = aggregate_multiseed_training(
            run_dirs=args.run_dir,
            evaluation_jsons=args.evaluation_json,
            metrics_csv=args.metrics_csv,
            summary_json=args.summary_json,
            latex_table=args.latex_table,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Aggregated {summary['n']} seeds: {summary['seeds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
