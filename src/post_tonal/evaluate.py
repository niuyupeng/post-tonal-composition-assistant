"""Evaluate token and constraint metrics for post-tonal symbolic generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any
from xml.etree import ElementTree

import torch
from torch import nn
from torch.utils.data import DataLoader

from post_tonal.data.post_tonal_dataset import PostTonalDataset, collate_batch
from post_tonal.data.score_tokenizer import ScoreTokenizer
from post_tonal.export_musicxml import export_musicxml
from post_tonal.models.transformer import PostTonalTransformer
from post_tonal.theory.analysis_report import analyze_events
from post_tonal.generate import candidate_loss
from post_tonal.train import maybe_generate_data, token_accuracy
from post_tonal.utils import ensure_dir, get_device, load_yaml, save_json, set_seed


def _mean(values: list[float | None]) -> float | None:
    filtered = [float(v) for v in values if v is not None]
    return None if not filtered else float(mean(filtered))


METRIC_FIELDS = [
    "experiment",
    "split",
    "num_samples",
    "token_accuracy",
    "model_loss",
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
]

CONSTRAINT_FIELDS = [
    "experiment",
    "split",
    "target_pcset_coverage",
    "interval_vector_distance",
    "row_order_accuracy",
    "aggregate_completion_rate",
    "serial_transformation_accuracy",
    "rhythmic_profile_distance",
    "density_curve_error",
    "gesture_consistency_score",
    "range_violation_rate",
]


def load_model(checkpoint: str | Path, tokenizer: ScoreTokenizer) -> tuple[PostTonalTransformer, dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = PostTonalTransformer(vocab_size=tokenizer.vocab_size, **cfg.get("model", {}))
    model.load_state_dict(ckpt["model_state"])
    return model, cfg


def musicxml_structurally_valid(path: str | Path) -> bool:
    try:
        root = ElementTree.parse(path).getroot()
    except Exception:
        return False
    return root.tag.endswith("score-partwise") or root.tag.endswith("score-timewise")


def generated_events_from_model(
    model: PostTonalTransformer,
    tokenizer: ScoreTokenizer,
    metadata: dict[str, Any],
    attempts: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    prefix_tokens = tokenizer.condition_tokens(metadata)
    prefix_ids = tokenizer.encode(prefix_tokens)
    best_events: list[dict[str, Any]] = []
    best_loss = float("inf")
    weights = {
        "pcset": 1.0,
        "interval_vector": 0.05,
        "row_order": 1.0,
        "aggregate": 0.5,
        "rhythm": 0.5,
        "gesture": 0.5,
        "range": 2.0,
    }
    for _ in range(max(1, attempts)):
        ids = model.sample(prefix_ids, tokenizer.eos_id, max_new_tokens=max_new_tokens, temperature=1.0, top_k=20)
        tokens = tokenizer.decode(ids)
        events = tokenizer.tokens_to_events(tokens)
        for event in events:
            event.setdefault("instrument", metadata.get("instrument", "generic_voice"))
            event.setdefault("gesture", metadata.get("gesture", "fragmented"))
            event.setdefault("rhythm_profile", metadata.get("rhythm_profile", "medium"))
        report = analyze_events(events, metadata)
        loss = candidate_loss(report, weights)
        if loss < best_loss:
            best_events = events
            best_loss = loss
    return best_events


def _events_sha256(events: list[dict[str, Any]]) -> str:
    payload = json.dumps(events, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generated_events_from_model_batch(
    model: PostTonalTransformer,
    tokenizer: ScoreTokenizer,
    metadatas: list[dict[str, Any]],
    seeds: list[int],
    attempts: int,
    max_new_tokens: int,
    batch_size: int,
    use_amp: bool,
    progress: bool = False,
) -> tuple[list[list[dict[str, Any]]], list[str]]:
    """Generate and rerank candidates while preserving an RNG stream per sample."""
    if len(metadatas) != len(seeds):
        raise ValueError("Expected one deterministic generation seed per metadata record.")
    if not metadatas:
        return [], []
    batch_size = max(1, int(batch_size))
    attempts = max(1, int(attempts))
    device = next(model.parameters()).device
    generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]
    prefixes = [tokenizer.encode(tokenizer.condition_tokens(metadata)) for metadata in metadatas]
    best_events: list[list[dict[str, Any]]] = [[] for _ in metadatas]
    best_losses = [float("inf")] * len(metadatas)
    first_candidate_hashes = [""] * len(metadatas)
    weights = {
        "pcset": 1.0,
        "interval_vector": 0.05,
        "row_order": 1.0,
        "aggregate": 0.5,
        "rhythm": 0.5,
        "gesture": 0.5,
        "range": 2.0,
    }

    for attempt_index in range(attempts):
        for start in range(0, len(metadatas), batch_size):
            end = min(start + batch_size, len(metadatas))
            sampled = model.sample_batch(
                prefixes[start:end],
                tokenizer.eos_id,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_k=20,
                generators=generators[start:end],
                use_amp=use_amp,
            )
            for local_index, ids in enumerate(sampled):
                sample_index = start + local_index
                metadata = metadatas[sample_index]
                events = tokenizer.tokens_to_events(tokenizer.decode(ids))
                for event in events:
                    event.setdefault("instrument", metadata.get("instrument", "generic_voice"))
                    event.setdefault("gesture", metadata.get("gesture", "fragmented"))
                    event.setdefault("rhythm_profile", metadata.get("rhythm_profile", "medium"))
                if attempt_index == 0:
                    first_candidate_hashes[sample_index] = _events_sha256(events)
                report = analyze_events(events, metadata)
                loss = candidate_loss(report, weights)
                if loss < best_losses[sample_index]:
                    best_events[sample_index] = events
                    best_losses[sample_index] = loss
            if progress and (end == len(metadatas) or end % (batch_size * 10) == 0):
                print(
                    {
                        "generation_attempt": attempt_index + 1,
                        "generation_attempts": attempts,
                        "processed_samples": end,
                        "total_samples": len(metadatas),
                    },
                    flush=True,
                )
    return best_events, first_candidate_hashes


def append_csv_row(path: str | Path, row: dict[str, Any], fields: list[str]) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    write_header = not path_obj.exists()
    with open(path_obj, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field) for field in fields})


def append_examples(path: str | Path, examples: list[dict[str, Any]]) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    existing: list[dict[str, Any]] = []
    if path_obj.exists():
        try:
            existing = json.loads(path_obj.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []
    existing.extend(examples)
    path_obj.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")


def evaluate(
    config_path: str | Path,
    checkpoint: str | Path | None,
    output: str | Path,
    table_output: str | Path | None = None,
    split: str = "test",
    experiment_name: str | None = None,
    metrics_csv: str | Path | None = None,
    constraints_csv: str | Path | None = None,
    examples_output: str | Path | None = None,
    per_sample_output: str | Path | None = None,
    main_table_output: str | Path | None = None,
    ablation_table_output: str | Path | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    eval_cfg = config.get("evaluation", {})
    evaluation_seed = int(eval_cfg.get("seed", config.get("seed", 0)))
    set_seed(evaluation_seed)
    maybe_generate_data(config)
    tokenizer = ScoreTokenizer.load(config["vocab_path"])
    dataset = PostTonalDataset(config["data_path"], max_seq_len=int(config.get("model", {}).get("max_seq_len", 256)), split=split)
    experiment_name = experiment_name or str(config.get("experiment_name", Path(config_path).stem))

    model_metrics: dict[str, float | None] = {"token_accuracy": None, "loss": None}
    model: PostTonalTransformer | None = None
    if checkpoint is not None and Path(checkpoint).exists():
        device = get_device(config.get("device"))
        model, _ = load_model(checkpoint, tokenizer)
        model.to(device)
        model.eval()
        loader = DataLoader(
            dataset,
            batch_size=int(config.get("training", {}).get("batch_size", 16)),
            shuffle=False,
            collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_id),
        )
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        losses: list[float] = []
        accuracies: list[float] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
                losses.append(float(loss.item()))
                accuracies.append(token_accuracy(logits, labels))
                max_batches = eval_cfg.get("max_batches")
                if max_batches is not None and batch_idx + 1 >= int(max_batches):
                    break
        model_metrics = {"token_accuracy": _mean(accuracies), "loss": _mean(losses)}

    reports: list[dict[str, Any]] = []
    per_sample_records: list[dict[str, Any]] = []
    xml_success = 0
    examples: list[dict[str, Any]] = []
    export_examples = int(eval_cfg.get("export_examples", 0) or 0)
    generation_count = int(eval_cfg.get("generation_examples", export_examples) or export_examples)
    export_dir = ensure_dir(Path(output).parent / "eval_musicxml" / experiment_name)
    use_model_generation = bool(eval_cfg.get("model_generation", False)) and model is not None
    attempts = int(eval_cfg.get("constraint_guided_attempts", 1 if not eval_cfg.get("constraint_guided_decoding", False) else 4))
    max_new_tokens = int(eval_cfg.get("max_new_tokens", config.get("model", {}).get("max_seq_len", 256)))
    generation_batch_size = max(1, int(eval_cfg.get("generation_batch_size", 1)))
    generation_fp16 = bool(eval_cfg.get("generation_fp16", False))
    sampling_protocol = (
        "per_sample_generator_batch_v1"
        if use_model_generation and generation_batch_size > 1
        else "legacy_global_generator_v1"
        if use_model_generation
        else "target_events"
    )
    metric_samples = eval_cfg.get("constraint_metric_samples")
    if metric_samples is None:
        metric_samples = generation_count if use_model_generation else len(dataset.samples)
    metric_samples = min(len(dataset.samples), max(1, int(metric_samples)))
    eval_samples = dataset.samples[:metric_samples]

    generated_events_cache: list[list[dict[str, Any]]] | None = None
    first_candidate_hashes: list[str] | None = None
    if use_model_generation and generation_batch_size > 1:
        generation_total = min(len(dataset.samples), max(metric_samples, generation_count))
        generation_samples = dataset.samples[:generation_total]
        generated_events_cache, first_candidate_hashes = generated_events_from_model_batch(
            model,
            tokenizer,
            [sample.get("metadata", {}) for sample in generation_samples],
            [evaluation_seed + idx for idx in range(generation_total)],
            attempts=attempts,
            max_new_tokens=max_new_tokens,
            batch_size=generation_batch_size,
            use_amp=generation_fp16,
            progress=bool(eval_cfg.get("generation_progress", False)),
        )

    for idx, sample in enumerate(eval_samples):
        metadata = sample.get("metadata", {})
        first_candidate_sha256: str | None = None
        if generated_events_cache is not None:
            events = generated_events_cache[idx]
            first_candidate_sha256 = first_candidate_hashes[idx] if first_candidate_hashes else None
        elif use_model_generation:
            set_seed(evaluation_seed + idx)
            events = generated_events_from_model(model, tokenizer, metadata, attempts=attempts, max_new_tokens=max_new_tokens)
            if attempts == 1:
                first_candidate_sha256 = _events_sha256(events)
        else:
            events = sample.get("events", [])
        report = analyze_events(events, metadata)
        reports.append(report)
        per_sample_records.append(
            {
                "experiment": experiment_name,
                "split": split,
                "sample_index": idx,
                "sample_id": sample.get("id"),
                "evaluation_seed": evaluation_seed + idx,
                "candidate_attempts": attempts if use_model_generation else 0,
                "generation_batch_size": generation_batch_size if use_model_generation else 0,
                "sampling_protocol": sampling_protocol,
                "first_candidate_sha256": first_candidate_sha256,
                "metadata": metadata,
                "analysis": report,
            }
        )
        if idx < export_examples:
            out_path = export_dir / f"{experiment_name}_{split}_{idx:03d}.musicxml"
            report_path = export_dir / f"{experiment_name}_{split}_{idx:03d}.json"
            structural_ok = False
            try:
                export_musicxml(events, out_path, metadata)
                structural_ok = musicxml_structurally_valid(out_path)
                if structural_ok:
                    xml_success += 1
                save_json({"metadata": metadata, "analysis": report}, report_path)
            except Exception:
                structural_ok = False
            examples.append(
                {
                    "experiment": experiment_name,
                    "split": split,
                    "sample_id": sample.get("id"),
                    "musicxml": str(out_path),
                    "analysis_report": str(report_path),
                    "musicxml_structurally_valid": structural_ok,
                    "analysis": report,
                }
            )

    # Ensure generation examples can be larger than exported evaluation count by
    # writing additional MusicXML files if requested.
    for extra_idx, sample in enumerate(dataset.samples[export_examples:generation_count], start=export_examples):
        metadata = sample.get("metadata", {})
        if generated_events_cache is not None:
            events = generated_events_cache[extra_idx]
        elif use_model_generation:
            set_seed(evaluation_seed + extra_idx)
            events = generated_events_from_model(model, tokenizer, metadata, attempts=attempts, max_new_tokens=max_new_tokens)
        else:
            events = sample.get("events", [])
        report = analyze_events(events, metadata)
        out_path = export_dir / f"{experiment_name}_{split}_{extra_idx:03d}.musicxml"
        report_path = export_dir / f"{experiment_name}_{split}_{extra_idx:03d}.json"
        try:
            export_musicxml(events, out_path, metadata)
            structural_ok = musicxml_structurally_valid(out_path)
            save_json({"metadata": metadata, "analysis": report}, report_path)
            examples.append(
                {
                    "experiment": experiment_name,
                    "split": split,
                    "sample_id": sample.get("id"),
                    "musicxml": str(out_path),
                    "analysis_report": str(report_path),
                    "musicxml_structurally_valid": structural_ok,
                    "analysis": report,
                }
            )
        except Exception:
            pass

    metrics = {
        "experiment": experiment_name,
        "split": split,
        "num_samples": len(eval_samples),
        "token_accuracy": model_metrics["token_accuracy"],
        "model_loss": model_metrics["loss"],
        "target_pcset_coverage": _mean([report["pcset_coverage"] for report in reports]),
        "pcset_coverage": _mean([report["pcset_coverage"] for report in reports]),
        "interval_vector_distance": _mean([report["interval_vector_distance"] for report in reports]),
        "row_order_accuracy": _mean([report["row_order_accuracy"] for report in reports]),
        "serial_row_order_accuracy": _mean([report["row_order_accuracy"] for report in reports]),
        "aggregate_completion_rate": _mean([report["aggregate_completion_rate"] for report in reports]),
        "serial_transformation_accuracy": _mean([report["serial_transformation_accuracy"] for report in reports]),
        "rhythmic_profile_distance": _mean([report["rhythmic_profile_distance"] for report in reports]),
        "density_curve_error": _mean([report["density_curve_error"] for report in reports]),
        "gesture_consistency_score": _mean([report["gesture_consistency_score"] for report in reports]),
        "range_violation_rate": _mean([report["range_violation_rate"] for report in reports]),
        "instrument_range_violation_rate": _mean([report["instrument_range_violation_rate"] for report in reports]),
        "musicxml_export_success_rate": 1.0 if export_examples == 0 else xml_success / export_examples,
        "generation_batch_size": generation_batch_size if use_model_generation else 0,
        "sampling_protocol": sampling_protocol,
    }
    save_json(metrics, output)
    if metrics_csv is not None:
        append_csv_row(metrics_csv, metrics, METRIC_FIELDS)
    if constraints_csv is not None:
        append_csv_row(constraints_csv, metrics, CONSTRAINT_FIELDS)
    if examples_output is not None:
        append_examples(examples_output, examples)
    if per_sample_output is not None:
        save_json(
            {
                "experiment": experiment_name,
                "split": split,
                "evaluation_seed": evaluation_seed,
                "candidate_attempts": attempts if use_model_generation else 0,
                "generation_batch_size": generation_batch_size if use_model_generation else 0,
                "sampling_protocol": sampling_protocol,
                "num_samples": len(per_sample_records),
                "samples": per_sample_records,
            },
            per_sample_output,
        )
    if table_output is not None:
        write_latex_table(metrics, table_output)
    if main_table_output is not None:
        write_latex_table(metrics, main_table_output)
    if ablation_table_output is not None:
        write_latex_table(metrics, ablation_table_output)
    return metrics


def write_latex_table(metrics: dict[str, Any], path: str | Path) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    rows = []
    for key, value in metrics.items():
        if isinstance(value, float):
            display = f"{value:.4f}"
        elif value is None:
            display = "PENDING_REAL_EXPERIMENT"
        else:
            display = str(value)
        rows.append(f"{key.replace('_', ' ')} & {display} \\\\")
    content = "\\begin{tabular}{ll}\nMetric & Value \\\\\n\\hline\n" + "\n".join(rows) + "\n\\end{tabular}\n"
    path_obj.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="results/metrics.json")
    parser.add_argument("--table-output", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--metrics-csv", default=None)
    parser.add_argument("--constraints-csv", default=None)
    parser.add_argument("--examples-output", default=None)
    parser.add_argument("--per-sample-output", default=None)
    parser.add_argument("--main-table-output", default=None)
    parser.add_argument("--ablation-table-output", default=None)
    args = parser.parse_args()
    metrics = evaluate(
        args.config,
        args.checkpoint,
        args.output,
        args.table_output,
        split=args.split,
        experiment_name=args.experiment_name,
        metrics_csv=args.metrics_csv,
        constraints_csv=args.constraints_csv,
        examples_output=args.examples_output,
        per_sample_output=args.per_sample_output,
        main_table_output=args.main_table_output,
        ablation_table_output=args.ablation_table_output,
    )
    print(metrics)


if __name__ == "__main__":
    main()
