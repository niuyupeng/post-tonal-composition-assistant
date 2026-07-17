"""Full-run helper commands for Project 2 experiment execution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

import torch

from post_tonal.data.post_tonal_dataset import describe_splits
from post_tonal.train import maybe_generate_data
from post_tonal.utils import load_yaml, save_json


EXPECTED_EXPERIMENTS = [
    "rule_baseline",
    "vanilla_transformer",
    "proposed_constraint_guided_transformer",
    "transformer_no_constraints",
    "without_pcset_constraints",
    "without_serial_constraints",
    "without_rhythm_constraints",
    "without_gesture_constraints",
    "serial_only",
    "pcset_only",
    "rhythm_only",
    "gesture_only",
    "no_constraints",
]

EXPECTED_NEURAL_RUNS = [
    name for name in EXPECTED_EXPERIMENTS if name != "rule_baseline"
]

REQUIRED_METRIC_COLUMNS = {
    "token_accuracy",
    "target_pcset_coverage",
    "interval_vector_distance",
    "row_order_accuracy",
    "aggregate_completion_rate",
    "serial_transformation_accuracy",
    "rhythmic_profile_distance",
    "density_curve_error",
    "gesture_consistency_score",
    "range_violation_rate",
    "musicxml_export_success_rate",
}

RULE_OPTIONAL_NUMERIC_METRICS = {"token_accuracy"}


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _read_json_object(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _read_run_incidents(
    path: str | Path | None,
) -> tuple[list[dict[str, str]], list[str]]:
    if path is None:
        return [], []
    incident_path = Path(path)
    if not incident_path.exists():
        return [], []
    try:
        payload = _read_json_object(incident_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return [], [f"{incident_path.as_posix()}: {type(exc).__name__}: {exc}"]
    raw_incidents = payload.get("incidents", [])
    if not isinstance(raw_incidents, list):
        return [], [f"{incident_path.as_posix()}: incidents must be a list"]

    required_fields = ("stage", "status", "failure", "recovery", "evidence")
    incidents: list[dict[str, str]] = []
    errors: list[str] = []
    for index, item in enumerate(raw_incidents, start=1):
        if not isinstance(item, dict):
            errors.append(
                f"{incident_path.as_posix()}: incident {index} must be an object"
            )
            continue
        missing = [
            field
            for field in required_fields
            if not str(item.get(field, "")).strip()
        ]
        if missing:
            errors.append(
                f"{incident_path.as_posix()}: incident {index} missing "
                + ", ".join(missing)
            )
            continue
        incidents.append({str(key): str(value) for key, value in item.items()})
    return incidents, errors


def _metric_row_errors(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        experiment = str(row.get("experiment") or "<missing>")
        if row.get("split") != "test":
            errors.append(f"{experiment}: split must be test")
        try:
            if int(row.get("num_samples") or 0) != 2000:
                errors.append(f"{experiment}: num_samples must be 2000")
        except ValueError:
            errors.append(f"{experiment}: num_samples is not an integer")
        required = REQUIRED_METRIC_COLUMNS - (
            RULE_OPTIONAL_NUMERIC_METRICS if experiment == "rule_baseline" else set()
        )
        for field in sorted(required):
            value = row.get(field)
            try:
                numeric = float(value) if value not in {None, ""} else float("nan")
            except ValueError:
                numeric = float("nan")
            if not math.isfinite(numeric):
                errors.append(f"{experiment}: {field} is missing or non-finite")
    return errors


def env_check() -> None:
    print("Python", sys.version)
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("cuda_device_count", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("cuda_device", torch.cuda.get_device_name(0))
    if sys.version_info[:2] not in {(3, 10), (3, 11)}:
        raise SystemExit("Python 3.10 or 3.11 is required for the full run.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Install a CUDA-enabled PyTorch build and NVIDIA driver before running the full experiment.")


def write_split_summary(config_path: str, output: str) -> None:
    config = load_yaml(config_path)
    maybe_generate_data(config)
    data_path = Path(config["data_path"])
    payload = torch.load(data_path, map_location="cpu", weights_only=False)
    actual_counts = describe_splits(data_path)
    lengths = [len(sample["token_ids"]) for sample in payload.get("samples", [])]
    summary = {
        "train_count": actual_counts.get("train", 0),
        "val_count": actual_counts.get("val", 0),
        "test_count": actual_counts.get("test", 0),
        "random_seed": config.get("seed"),
        "generation_config_path": config_path,
        "processed_data_path": config.get("data_path"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "smoke": False,
        "split_source": "explicit corpus train_samples/val_samples/test_samples saved by generate_corpus.py",
        "corpus_format": payload.get("format"),
        "sequence_training_strategy": config.get("training", {}).get("sequence_mode", "truncate"),
        "target_tokens_per_window": config.get("training", {}).get("target_tokens_per_window"),
        "raw_sequence_length_median": statistics.median(lengths) if lengths else 0,
        "raw_sequence_length_max": max(lengths, default=0),
        "raw_sequences_over_model_context": sum(
            length > int(config.get("model", {}).get("max_seq_len", 256)) + 1
            for length in lengths
        ),
        "raw_sequence_tokens_discarded_by_training": 0
        if config.get("training", {}).get("sequence_mode") in {"rotating", "coverage_cycle", "all"}
        else "LEGACY_TRUNCATION_MODE",
    }
    payload_seed = payload.get("seed")
    if payload_seed != config.get("seed"):
        raise SystemExit(
            f"Corpus seed {payload_seed!r} does not match config seed {config.get('seed')!r}."
        )
    summary["data_sha256"] = _file_sha256(data_path)
    summary["vocab_sha256"] = _file_sha256(config["vocab_path"])
    save_json(summary, output)
    print("full_split_summary", summary)
    if summary["train_count"] != 20000 or summary["val_count"] != 2000 or summary["test_count"] != 2000:
        raise SystemExit("Full split counts are not 20000/2000/2000.")


def _read_command_log(path: str | Path) -> list[str]:
    log_path = Path(path)
    if not log_path.exists():
        return []
    raw = log_path.read_bytes()
    # Windows PowerShell 5.1 Tee-Object writes UTF-16 with a BOM; pwsh and
    # Python-generated logs are normally UTF-8.
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
    text = raw.decode(encoding, errors="replace")
    return [
        line.removeprefix("COMMAND ").strip()
        for line in text.splitlines()
        if line.startswith("COMMAND ")
    ]


def promote_generation_examples(
    source: str,
    output: str,
    source_root: str = "results/eval_musicxml_v3",
    destination_root: str = "results/eval_musicxml",
) -> None:
    payload = json.loads(Path(source).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list: {source}")

    normalized_source = source_root.replace("\\", "/").rstrip("/")
    normalized_destination = destination_root.replace("\\", "/").rstrip("/")
    rewritten = 0
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(f"Expected an object at item {index}: {source}")
        for field in ("musicxml", "analysis_report"):
            value = record.get(field)
            if not isinstance(value, str):
                raise ValueError(
                    f"Missing string field {field!r} at item {index}: {source}"
                )
            normalized_value = value.replace("\\", "/")
            prefix = f"{normalized_source}/"
            if normalized_value.startswith(prefix):
                record[field] = (
                    f"{normalized_destination}/{normalized_value[len(prefix):]}"
                )
                rewritten += 1

    if rewritten == 0:
        raise ValueError(
            f"No artifact paths under {source_root!r} were found in {source}"
        )
    save_json(payload, output)
    print(
        "generation_examples_promoted",
        {"source": source, "output": output, "rewritten_paths": rewritten},
    )


def _generation_example_errors(path: str | Path) -> list[str]:
    examples_path = Path(path)
    if not examples_path.is_file():
        return [f"{examples_path.as_posix()}: file is missing"]
    try:
        payload = json.loads(examples_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{examples_path.as_posix()}: {type(exc).__name__}: {exc}"]
    if not isinstance(payload, list):
        return [f"{examples_path.as_posix()}: expected a JSON list"]

    errors: list[str] = []
    counts = {name: 0 for name in EXPECTED_EXPERIMENTS}
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            errors.append(f"generation example {index}: expected an object")
            continue
        experiment = str(record.get("experiment", ""))
        if experiment in counts:
            counts[experiment] += 1
        else:
            errors.append(
                f"generation example {index}: unexpected experiment {experiment!r}"
            )
        xml_path = Path(str(record.get("musicxml", "")))
        report_path = Path(str(record.get("analysis_report", "")))
        if not xml_path.is_file():
            errors.append(f"generation example {index}: missing {xml_path.as_posix()}")
        else:
            try:
                if not ElementTree.parse(xml_path).getroot().tag.endswith(
                    "score-partwise"
                ):
                    errors.append(
                        f"generation example {index}: not score-partwise "
                        f"{xml_path.as_posix()}"
                    )
            except (ElementTree.ParseError, OSError) as exc:
                errors.append(
                    f"generation example {index}: invalid MusicXML "
                    f"{xml_path.as_posix()}: {exc}"
                )
        if not report_path.is_file():
            errors.append(
                f"generation example {index}: missing {report_path.as_posix()}"
            )
        else:
            try:
                _read_json_object(report_path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(
                    f"generation example {index}: invalid report "
                    f"{report_path.as_posix()}: {exc}"
                )
    for experiment, count in counts.items():
        if count != 20:
            errors.append(
                f"{experiment}: expected 20 generation examples, found {count}"
            )
    return errors


def _manifest_matches_package(
    manifest: dict[str, object],
    xml_paths: list[Path],
    expert_root: Path,
) -> bool:
    items = manifest.get("items")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        return False
    expected_ids = {path.stem for path in xml_paths}
    expected_xml = {path.resolve() for path in xml_paths}
    expected_reports = {
        (expert_root / "analysis_reports" / f"{path.stem}.json").resolve()
        for path in xml_paths
    }
    manifest_ids = {str(item.get("id")) for item in items}
    manifest_xml = {
        Path(str(item.get("musicxml", ""))).resolve()
        for item in items
    }
    manifest_reports = {
        Path(str(item.get("analysis_report", ""))).resolve()
        for item in items
    }
    return (
        manifest.get("count") == len(items)
        and len(items) >= 20
        and manifest_ids == expected_ids
        and manifest_xml == expected_xml
        and manifest_reports == expected_reports
    )


def write_report(
    output: str,
    metrics: str = "results/project2_metrics.csv",
    split_summary: str = "results/project2_full_split_summary.json",
    expert_dir: str = "expert_eval/project2",
    constraints: str = "results/project2_constraints.csv",
    examples: str = "results/project2_generation_examples.json",
    run_root: str = "runs",
    log_path: str = "logs/project2_full_run.log",
    main_table: str = "paper/tables/project2_main_results.tex",
    ablation_table: str = "paper/tables/project2_ablation_results.tex",
    incidents: str | None = None,
) -> None:
    metrics_path = Path(metrics)
    rows = _read_csv_rows(metrics_path)
    expert_root = Path(expert_dir)
    xml_paths = sorted((expert_root / "musicxml").glob("*.musicxml"))
    xml_ok = 0
    measure_ok = 0
    voice_ok = 0
    xml_parse_errors: list[str] = []
    for path in xml_paths:
        try:
            root = ElementTree.parse(path).getroot()
        except (ElementTree.ParseError, OSError) as exc:
            xml_parse_errors.append(f"{path.as_posix()}: {type(exc).__name__}: {exc}")
            continue
        if root.tag.endswith("score-partwise"):
            xml_ok += 1
        report_path = expert_root / "analysis_reports" / f"{path.stem}.json"
        if report_path.exists():
            try:
                metadata = _read_json_object(report_path).get("metadata", {})
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                xml_parse_errors.append(
                    f"{report_path.as_posix()}: {type(exc).__name__}: {exc}"
                )
                continue
            requested = int(metadata.get("measures", 0))
            requested_voices = int(metadata.get("voices", 0))
            counts = [len(part.findall("./measure")) for part in root.findall("./part")]
            measure_ok += int(bool(counts) and all(count == requested for count in counts))
            voice_ok += int(len(root.findall("./part")) == requested_voices)

    summary_path = Path(split_summary)
    summary = _read_json_object(summary_path) if summary_path.exists() else {}
    completed = [
        str(row.get("experiment"))
        for row in rows
        if row.get("experiment") and row.get("split") == "test"
    ]
    row_counts = {name: completed.count(name) for name in EXPECTED_EXPERIMENTS}
    missing = [name for name in EXPECTED_EXPERIMENTS if row_counts[name] == 0]
    duplicate_rows = [name for name in EXPECTED_EXPERIMENTS if row_counts[name] > 1]
    training_summaries: list[dict[str, object]] = []
    for summary_file in sorted(Path(run_root).glob("*/train_summary.json")):
        try:
            payload = _read_json_object(summary_file)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        training_summaries.append({"path": summary_file.as_posix(), **payload})
    training_by_run = {
        Path(str(item["path"])).parent.name: item
        for item in training_summaries
    }
    incomplete_training = []
    for run_name in EXPECTED_NEURAL_RUNS:
        summary_item = training_by_run.get(run_name)
        checkpoint_path = Path(run_root) / run_name / "checkpoint.pt"
        checkpoint_hash_matches = (
            checkpoint_path.is_file()
            and summary_item is not None
            and summary_item.get("checkpoint_sha256") == _file_sha256(checkpoint_path)
        )
        if (
            summary_item is None
            or summary_item.get("completed") is not True
            or not checkpoint_hash_matches
        ):
            incomplete_training.append(run_name)
    oom_adjustments = [
        f"{item['path']}: {item['oom_adjustment']}"
        for item in training_summaries
        if item.get("oom_adjustment")
    ]
    peak_ram = max(
        (
            float(item["peak_process_ram_gib"])
            for item in training_summaries
            if item.get("peak_process_ram_gib") is not None
        ),
        default=None,
    )
    peak_vram = max(
        (
            float(item["peak_cuda_memory_allocated_gib"])
            for item in training_summaries
            if item.get("peak_cuda_memory_allocated_gib") is not None
        ),
        default=None,
    )
    commands = _read_command_log(log_path)
    run_incidents, incident_errors = _read_run_incidents(incidents)
    unresolved_incidents = [
        item
        for item in run_incidents
        if item.get("status", "").strip().lower() not in {"recovered", "resolved"}
    ]
    metrics_columns = set(rows[0]) if rows else set()
    metric_row_errors = _metric_row_errors(rows)
    generation_example_errors = _generation_example_errors(examples)
    rows_complete = (
        not missing
        and not duplicate_rows
        and len(rows) == len(EXPECTED_EXPERIMENTS)
        and REQUIRED_METRIC_COLUMNS.issubset(metrics_columns)
        and not metric_row_errors
    )
    split_complete = (
        summary.get("train_count") == 20000
        and summary.get("val_count") == 2000
        and summary.get("test_count") == 2000
        and summary.get("smoke") is False
        and summary.get("random_seed") == 42
        and bool(summary.get("generation_config_path"))
    )
    manifest_ok = False
    manifest_path = expert_root / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = _read_json_object(manifest_path)
            manifest_ok = _manifest_matches_package(
                manifest,
                xml_paths,
                expert_root,
            )
        except (OSError, json.JSONDecodeError, ValueError, AttributeError):
            manifest_ok = False
    package_complete = (
        len(xml_paths) >= 20
        and xml_ok == len(xml_paths)
        and measure_ok == len(xml_paths)
        and voice_ok == len(xml_paths)
        and not xml_parse_errors
        and (expert_root / "blind_rating_form_project2.md").is_file()
        and (expert_root / "blind_rating_form_project2.csv").is_file()
        and manifest_ok
    )
    expert_complete = package_complete
    tables_complete = (
        Path(main_table).is_file()
        and Path(main_table).stat().st_size > 0
        and Path(ablation_table).is_file()
        and Path(ablation_table).stat().st_size > 0
    )
    supporting_outputs_complete = (
        len(_read_csv_rows(constraints)) == len(EXPECTED_EXPERIMENTS)
        and not generation_example_errors
        and tables_complete
    )
    completion_status = (
        rows_complete
        and split_complete
        and package_complete
        and supporting_outputs_complete
        and not incomplete_training
        and not incident_errors
        and not unresolved_incidents
    )
    report = [
        "# Project 2 Corrected Full Run Report",
        "",
        f"- Completion gate: {'PASS' if completion_status else 'INCOMPLETE'}",
        "",
        "## Commands Executed",
        *(f"- `{command}`" for command in commands),
        *([] if commands else ["- No `COMMAND` records were found in the supplied log."]),
        "",
        "## Environment Information",
        f"- Python: {sys.version.split()[0]}",
        f"- PyTorch: {torch.__version__}",
        f"- CUDA available: {torch.cuda.is_available()}",
        f"- CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}",
        f"- Peak process RAM across completed training runs: {peak_ram:.3f} GiB" if peak_ram is not None else "- Peak process RAM: unavailable",
        f"- Peak allocated CUDA memory across completed training runs: {peak_vram:.3f} GiB" if peak_vram is not None else "- Peak allocated CUDA memory: unavailable",
        "",
        "## CUDA Check Output",
        f"- cuda_available: {torch.cuda.is_available()}",
        f"- cuda_device_count: {torch.cuda.device_count()}",
        f"- cuda_device_name_0: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}",
        "",
        "## Corpus Split Counts",
        f"- Train: {summary.get('train_count', 'PENDING_REAL_EXPERIMENT')}",
        f"- Validation: {summary.get('val_count', 'PENDING_REAL_EXPERIMENT')}",
        f"- Test: {summary.get('test_count', 'PENDING_REAL_EXPERIMENT')}",
        f"- Smoke: {summary.get('smoke', 'PENDING_REAL_EXPERIMENT')}",
        f"- Corpus format: {summary.get('corpus_format', 'PENDING_REAL_EXPERIMENT')}",
        f"- Sequence strategy: {summary.get('sequence_training_strategy', 'PENDING_REAL_EXPERIMENT')}",
        f"- Raw score-body tokens discarded by training: {summary.get('raw_sequence_tokens_discarded_by_training', 'PENDING_REAL_EXPERIMENT')}",
        "",
        "## Experiment Configs Completed",
        *(f"- {name}" for name in EXPECTED_EXPERIMENTS if row_counts[name] == 1),
        "",
        "## Neural Checkpoints Completed",
        *(f"- {name}" for name in EXPECTED_NEURAL_RUNS if name not in incomplete_training),
        "",
        "## Neural Checkpoint Details",
        *(
            (
                f"- {name}: epochs={training_by_run[name].get('epochs_ran')}, "
                f"best_epoch={training_by_run[name].get('best_epoch')}, "
                f"stop_reason={training_by_run[name].get('stop_reason')}, "
                f"checkpoint={(Path(run_root) / name / 'checkpoint.pt').as_posix()}, "
                f"sha256={training_by_run[name].get('checkpoint_sha256')}"
            )
            for name in EXPECTED_NEURAL_RUNS
            if name not in incomplete_training
        ),
        *(
            ["- None."]
            if len(incomplete_training) == len(EXPECTED_NEURAL_RUNS)
            else []
        ),
        "",
        "## Pending Evaluation Rows",
        *(f"- Missing test metric row: {name}" for name in missing),
        *(f"- Duplicate or invalid metric-row count: {name}" for name in duplicate_rows),
        *(f"- Invalid metric row: {error}" for error in metric_row_errors),
        *([] if missing or duplicate_rows or metric_row_errors else ["- None."]),
        "",
        "## Missing or Incomplete Neural Checkpoints",
        *(f"- Missing or incomplete checkpoint: {name}" for name in incomplete_training),
        *([] if incomplete_training else ["- None."]),
        "",
        "## Failed or Retried Stages",
        *(
            (
                f"- {item['stage']} [{item['status']}]: {item['failure']} "
                f"Recovery: {item['recovery']} Evidence: {item['evidence']}"
            )
            for item in run_incidents
        ),
        *(
            ["- No failed or retried stage was recorded in the supplied incident file."]
            if not run_incidents and not incident_errors
            else []
        ),
        *(f"- Invalid incident record: {error}" for error in incident_errors),
        *(f"- Invalid MusicXML/report artifact: {error}" for error in xml_parse_errors),
        *(
            f"- Invalid generation example artifact: {error}"
            for error in generation_example_errors
        ),
        "",
        "## OOM Adjustments",
        *oom_adjustments,
        *([] if oom_adjustments else ["- None recorded in completed training summaries."]),
        "",
        "## Final Metrics File Paths",
        f"- {metrics_path.as_posix()}",
        f"- {Path(constraints).as_posix()}",
        f"- {Path(examples).as_posix()}",
        f"- {summary_path.as_posix()}",
        "",
        "## Generated MusicXML Examples Path",
        f"- {(expert_root / 'musicxml').as_posix()}/ ({xml_ok}/{len(xml_paths)} structurally score-partwise; {measure_ok}/{len(xml_paths)} requested-span adherent; {voice_ok}/{len(xml_paths)} requested-voice adherent)",
        "",
        "## Paper Tables Path",
        f"- {Path(main_table).as_posix()} ({'present' if Path(main_table).is_file() else 'missing'})",
        f"- {Path(ablation_table).as_posix()} ({'present' if Path(ablation_table).is_file() else 'missing'})",
        "",
        "## Remaining TODOs",
        *(
            [f"- Complete neural training for: {', '.join(incomplete_training)}."]
            if incomplete_training
            else []
        ),
        *(
            [f"- Produce exactly one 2,000-sample test metric row for: {', '.join(missing)}."]
            if missing
            else []
        ),
        *(
            ["- Resolve duplicate or invalid metric rows before table generation."]
            if duplicate_rows
            else []
        ),
        *(
            ["- Resolve missing or non-finite metric values before table generation."]
            if metric_row_errors
            else []
        ),
        *(
            ["- Generate the corrected main and ablation tables from complete v3 CSV files."]
            if not tables_complete
            else []
        ),
        *(
            ["- Generate and validate at least 20 full-run MusicXML examples and paired analysis reports."]
            if not expert_complete
            else []
        ),
        *(
            ["- Resolve or explicitly close the recorded run incident(s)."]
            if unresolved_incidents or incident_errors
            else []
        ),
        "- Add blind expert ratings after human evaluation.",
        "- Add independent, legally supplied MusicXML validation examples when available.",
        "- Complete author metadata, declarations, and live journal-portal checks.",
    ]
    Path(output).write_text("\n".join(report) + "\n", encoding="utf-8")
    print("full_run_report_written", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("env-check")
    split_parser = subparsers.add_parser("write-split-summary")
    split_parser.add_argument("--config", default="configs/post_tonal_main.yaml")
    split_parser.add_argument("--output", default="results/project2_full_split_summary.json")
    promote_parser = subparsers.add_parser("promote-generation-examples")
    promote_parser.add_argument("--source", required=True)
    promote_parser.add_argument("--output", required=True)
    promote_parser.add_argument(
        "--source-root", default="results/eval_musicxml_v3"
    )
    promote_parser.add_argument(
        "--destination-root", default="results/eval_musicxml"
    )
    report_parser = subparsers.add_parser("write-report")
    report_parser.add_argument("--output", default="results/project2_full_run_report.md")
    report_parser.add_argument("--metrics", default="results/project2_metrics.csv")
    report_parser.add_argument("--split-summary", default="results/project2_full_split_summary.json")
    report_parser.add_argument("--expert-dir", default="expert_eval/project2")
    report_parser.add_argument("--constraints", default="results/project2_constraints.csv")
    report_parser.add_argument("--examples", default="results/project2_generation_examples.json")
    report_parser.add_argument("--run-root", default="runs")
    report_parser.add_argument("--log-path", default="logs/project2_full_run.log")
    report_parser.add_argument("--main-table", default="paper/tables/project2_main_results.tex")
    report_parser.add_argument("--ablation-table", default="paper/tables/project2_ablation_results.tex")
    report_parser.add_argument("--incidents")
    args = parser.parse_args()
    if args.command == "env-check":
        env_check()
    elif args.command == "write-split-summary":
        write_split_summary(args.config, args.output)
    elif args.command == "promote-generation-examples":
        promote_generation_examples(
            args.source,
            args.output,
            args.source_root,
            args.destination_root,
        )
    elif args.command == "write-report":
        write_report(
            args.output,
            args.metrics,
            args.split_summary,
            args.expert_dir,
            args.constraints,
            args.examples,
            args.run_root,
            args.log_path,
            args.main_table,
            args.ablation_table,
            args.incidents,
        )


if __name__ == "__main__":
    main()
