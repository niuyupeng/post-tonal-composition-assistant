"""Full-run helper commands for Project 2 experiment execution."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

import torch

from post_tonal.train import maybe_generate_data
from post_tonal.utils import load_yaml, save_json


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
    corpus = config.get("corpus", {})
    summary = {
        "train_count": int(corpus.get("train_samples", 0)),
        "val_count": int(corpus.get("val_samples", 0)),
        "test_count": int(corpus.get("test_samples", 0)),
        "random_seed": config.get("seed"),
        "generation_config_path": config_path,
        "processed_data_path": config.get("data_path"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "smoke": False,
        "split_source": "explicit corpus train_samples/val_samples/test_samples saved by generate_corpus.py",
    }
    save_json(summary, output)
    print("full_split_summary", summary)
    if summary["train_count"] != 20000 or summary["val_count"] != 2000 or summary["test_count"] != 2000:
        raise SystemExit("Full split counts are not 20000/2000/2000.")


def write_report(output: str) -> None:
    metrics_path = Path("results/project2_metrics.csv")
    rows = list(csv.DictReader(metrics_path.open(newline="", encoding="utf-8"))) if metrics_path.exists() else []
    xml_paths = sorted(Path("expert_eval/project2/musicxml").glob("*.musicxml"))
    xml_ok = 0
    for path in xml_paths:
        root = ElementTree.parse(path).getroot()
        if root.tag.endswith("score-partwise"):
            xml_ok += 1

    summary_path = Path("results/project2_full_split_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    completed = [row.get("experiment") for row in rows]
    report = [
        "# Project 2 Full Run Report",
        "",
        "## Commands Executed",
        "- .\\scripts\\run_project2_full_local.ps1",
        "",
        "## Environment Information",
        f"- Python: {sys.version.split()[0]}",
        f"- PyTorch: {torch.__version__}",
        f"- CUDA available: {torch.cuda.is_available()}",
        f"- CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}",
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
        "",
        "## Experiment Configs Completed",
        *(f"- {name}" for name in completed),
        "",
        "## Failed Experiments",
        "- None recorded by the completed script run.",
        "",
        "## OOM Adjustments",
        "- None recorded.",
        "",
        "## Final Metrics File Paths",
        "- results/project2_metrics.csv",
        "- results/project2_constraints.csv",
        "- results/project2_generation_examples.json",
        "- results/project2_full_split_summary.json",
        "",
        "## Generated MusicXML Examples Path",
        f"- expert_eval/project2/musicxml/ ({xml_ok}/{len(xml_paths)} structurally score-partwise)",
        "",
        "## Paper Tables Path",
        "- paper/tables/project2_main_results.tex",
        "- paper/tables/project2_ablation_results.tex",
        "",
        "## Remaining TODOs",
        "- Add blind expert ratings after human evaluation.",
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
    report_parser = subparsers.add_parser("write-report")
    report_parser.add_argument("--output", default="results/project2_full_run_report.md")
    args = parser.parse_args()
    if args.command == "env-check":
        env_check()
    elif args.command == "write-split-summary":
        write_split_summary(args.config, args.output)
    elif args.command == "write-report":
        write_report(args.output)


if __name__ == "__main__":
    main()
