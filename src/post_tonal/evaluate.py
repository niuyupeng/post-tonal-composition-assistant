"""Evaluate token and constraint metrics for post-tonal symbolic generation."""

from __future__ import annotations

import argparse
import copy
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
from post_tonal.models.rule_generator import RuleGenerator
from post_tonal.models.transformer import PostTonalTransformer
from post_tonal.theory.analysis_report import analyze_events
from post_tonal.generate import candidate_loss
from post_tonal.train import maybe_generate_data
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
    "pcset_precision",
    "pcset_jaccard",
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
    "musicxml_voice_adherence_rate",
]

CONSTRAINT_FIELDS = [
    "experiment",
    "split",
    "target_pcset_coverage",
    "pcset_precision",
    "pcset_jaccard",
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


def musicxml_structure_summary(path: str | Path) -> dict[str, Any]:
    root = ElementTree.parse(path).getroot()
    parts = root.findall("./part")
    measure_counts = [len(part.findall("./measure")) for part in parts]
    return {
        "root_tag": root.tag.rsplit("}", 1)[-1],
        "part_count": len(parts),
        "measure_counts": measure_counts,
    }


def musicxml_matches_request(path: str | Path, metadata: dict[str, Any]) -> tuple[bool, bool]:
    summary = musicxml_structure_summary(path)
    requested_measures = max(1, int(metadata.get("measures", 4)))
    requested_voices = max(1, int(metadata.get("voices", 1)))
    measure_ok = bool(summary["measure_counts"]) and all(
        count == requested_measures for count in summary["measure_counts"]
    )
    voice_ok = summary["part_count"] == requested_voices
    return measure_ok, voice_ok


def generated_events_from_model(
    model: PostTonalTransformer,
    tokenizer: ScoreTokenizer,
    metadata: dict[str, Any],
    attempts: int,
    max_new_tokens: int,
    grammar_constrained: bool = True,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    prefix_tokens = tokenizer.condition_tokens(metadata)
    prefix_ids = tokenizer.encode(prefix_tokens)
    best_events: list[dict[str, Any]] = []
    best_loss = float("inf")
    weights = weights or {
        "pcset": 1.0,
        "pcset_precision": 0.5,
        "interval_vector": 0.05,
        "row_order": 1.0,
        "serial_transformation": 0.5,
        "aggregate": 0.5,
        "rhythm": 0.5,
        "gesture": 0.5,
        "range": 2.0,
        "content_span": 1.0,
        "voice_count": 0.5,
    }
    for _ in range(max(1, attempts)):
        grammar = (
            (lambda ids: tokenizer.allowed_next_token_ids(ids, metadata))
            if grammar_constrained
            else None
        )
        ids = model.sample(
            prefix_ids,
            tokenizer.eos_id,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=20,
            allowed_token_ids_fn=grammar,
        )
        tokens = tokenizer.decode(ids)
        events = tokenizer.tokens_to_events(tokens, metadata)
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


def constraint_weights(eval_cfg: dict[str, Any] | None = None) -> dict[str, float]:
    configured = (eval_cfg or {}).get("constraint_weights", {})
    defaults = {
        "pcset": 1.0,
        "pcset_precision": 0.5,
        "interval_vector": 0.05,
        "row_order": 1.0,
        "serial_transformation": 0.5,
        "aggregate": 0.5,
        "rhythm": 0.5,
        "gesture": 0.5,
        "range": 2.0,
        "content_span": 1.0,
        "voice_count": 0.5,
    }
    return {
        key: float(configured.get(key, value))
        for key, value in defaults.items()
    }


def _events_sha256(events: list[dict[str, Any]]) -> str:
    payload = json.dumps(events, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generated_events_from_model_batch(
    model: PostTonalTransformer,
    tokenizer: ScoreTokenizer,
    metadatas: list[dict[str, Any]],
    seeds: list[int],
    attempts: int,
    max_new_tokens: int | list[int],
    batch_size: int,
    use_amp: bool,
    grammar_constrained: bool = True,
    progress: bool = False,
    weights: dict[str, float] | None = None,
) -> tuple[list[list[dict[str, Any]]], list[str]]:
    """Generate and rerank candidates while preserving an RNG stream per sample."""
    if len(metadatas) != len(seeds):
        raise ValueError("Expected one deterministic generation seed per metadata record.")
    if not metadatas:
        return [], []
    batch_size = max(1, int(batch_size))
    attempts = max(1, int(attempts))
    token_budgets = (
        [max(0, int(max_new_tokens))] * len(metadatas)
        if isinstance(max_new_tokens, int)
        else [max(0, int(value)) for value in max_new_tokens]
    )
    if len(token_budgets) != len(metadatas):
        raise ValueError("Expected one generation token budget per metadata record.")
    device = next(model.parameters()).device
    generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]
    prefixes = [tokenizer.encode(tokenizer.condition_tokens(metadata)) for metadata in metadatas]
    grammar_callbacks = [
        (lambda ids, metadata=metadata: tokenizer.allowed_next_token_ids(ids, metadata))
        if grammar_constrained
        else None
        for metadata in metadatas
    ]
    best_events: list[list[dict[str, Any]]] = [[] for _ in metadatas]
    best_losses = [float("inf")] * len(metadatas)
    first_candidate_hashes = [""] * len(metadatas)
    weights = weights or constraint_weights()

    for attempt_index in range(attempts):
        for start in range(0, len(metadatas), batch_size):
            end = min(start + batch_size, len(metadatas))
            sampled = model.sample_batch(
                prefixes[start:end],
                tokenizer.eos_id,
                max_new_tokens=token_budgets[start:end],
                temperature=1.0,
                top_k=20,
                generators=generators[start:end],
                use_amp=use_amp,
                allowed_token_ids_fns=grammar_callbacks[start:end],
            )
            for local_index, ids in enumerate(sampled):
                sample_index = start + local_index
                metadata = metadatas[sample_index]
                events = tokenizer.tokens_to_events(tokenizer.decode(ids), metadata)
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


def generated_events_from_rule(
    metadata: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    """Generate an independent deterministic rule-baseline realization."""

    return RuleGenerator(seed=int(seed)).generate(copy.deepcopy(metadata))


def generation_token_budget(
    metadata: dict[str, Any],
    base_tokens: int,
    tokens_per_measure: int,
    token_cap: int,
) -> int:
    requested = max(1, int(metadata.get("measures", 4)))
    return min(max(int(base_tokens), requested * int(tokens_per_measure)), int(token_cap))


def append_csv_row(path: str | Path, row: dict[str, Any], fields: list[str]) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    existing: list[dict[str, Any]] = []
    if path_obj.exists():
        with path_obj.open(newline="", encoding="utf-8-sig") as handle:
            existing = list(csv.DictReader(handle))
    key = (str(row.get("experiment")), str(row.get("split")))
    existing = [
        item
        for item in existing
        if (str(item.get("experiment")), str(item.get("split"))) != key
    ]
    existing.append(row)
    with open(path_obj, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for item in existing:
            writer.writerow({field: item.get(field) for field in fields})


def append_examples(path: str | Path, examples: list[dict[str, Any]]) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    existing: list[dict[str, Any]] = []
    if path_obj.exists():
        try:
            existing = json.loads(path_obj.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []
    replacement_keys = {
        (str(item.get("experiment")), str(item.get("split")))
        for item in examples
    }
    existing = [
        item
        for item in existing
        if (str(item.get("experiment")), str(item.get("split"))) not in replacement_keys
    ]
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
    export_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    eval_cfg = config.get("evaluation", {})
    evaluation_seed = int(eval_cfg.get("seed", config.get("seed", 0)))
    set_seed(evaluation_seed)
    maybe_generate_data(config)
    data_path = Path(config["data_path"])
    vocab_path = Path(config["vocab_path"])
    tokenizer = ScoreTokenizer.load(vocab_path)
    training_cfg = config.get("training", {})
    configured_sequence_mode = str(training_cfg.get("sequence_mode", "truncate"))
    evaluation_sequence_mode = str(
        eval_cfg.get(
            "teacher_forced_sequence_mode",
            "all" if configured_sequence_mode != "truncate" else "truncate",
        )
    )
    dataset = PostTonalDataset(
        data_path,
        max_seq_len=int(config.get("model", {}).get("max_seq_len", 256)),
        split=split,
        sep_id=tokenizer.token_to_id["SEP"],
        sequence_mode=evaluation_sequence_mode,
        target_tokens_per_window=int(training_cfg.get("target_tokens_per_window", 128)),
        seed=int(config.get("seed", 0)),
        tokenizer=tokenizer,
        condition_ablation=config.get("condition_ablation"),
        coverage_cycle_epochs=int(training_cfg.get("coverage_cycle_epochs", 10)),
    )
    required_split_samples = eval_cfg.get("required_split_samples")
    if required_split_samples is not None and len(dataset.samples) != int(required_split_samples):
        raise ValueError(
            f"Split {split!r} contains {len(dataset.samples)} samples; "
            f"the evaluation contract requires exactly {int(required_split_samples)}."
        )
    experiment_name = experiment_name or str(config.get("experiment_name", Path(config_path).stem))

    model_metrics: dict[str, float | None] = {"token_accuracy": None, "loss": None}
    model: PostTonalTransformer | None = None
    checkpoint_path: Path | None = None
    checkpoint_sha256: str | None = None
    checkpoint_training_seed: int | None = None
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        checkpoint_sha256 = _file_sha256(checkpoint_path)
        device = get_device(config.get("device"))
        model, checkpoint_config = load_model(checkpoint_path, tokenizer)
        if checkpoint_config.get("seed") is not None:
            checkpoint_training_seed = int(checkpoint_config["seed"])
        model.to(device)
        model.eval()
        loader = DataLoader(
            dataset,
            batch_size=int(config.get("training", {}).get("batch_size", 16)),
            shuffle=False,
            collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_id),
        )
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
                mask = labels != -100
                total_loss += float(loss.item())
                total_correct += int((torch.argmax(logits, dim=-1)[mask] == labels[mask]).sum().item())
                total_tokens += int(mask.sum().item())
                max_batches = eval_cfg.get("max_batches")
                if max_batches is not None and batch_idx + 1 >= int(max_batches):
                    break
        model_metrics = {
            "token_accuracy": total_correct / max(1, total_tokens),
            "loss": total_loss / max(1, total_tokens),
            "evaluated_tokens": total_tokens,
        }

    reports: list[dict[str, Any]] = []
    per_sample_records: list[dict[str, Any]] = []
    xml_success = 0
    xml_measure_success = 0
    xml_voice_success = 0
    examples: list[dict[str, Any]] = []
    export_examples = int(eval_cfg.get("export_examples", 0) or 0)
    generation_count = int(eval_cfg.get("generation_examples", export_examples) or export_examples)
    export_root = ensure_dir(
        Path(export_dir)
        if export_dir is not None
        else Path(output).parent / "eval_musicxml" / experiment_name
    )
    use_model_generation = bool(eval_cfg.get("model_generation", False)) and model is not None
    use_rule_generation = bool(eval_cfg.get("rule_generation", False))
    if bool(eval_cfg.get("model_generation", False)) and model is None:
        raise ValueError("Model generation was requested, but no checkpoint was loaded.")
    if use_model_generation and use_rule_generation:
        raise ValueError("Choose either model generation or rule generation, not both.")
    attempts = int(eval_cfg.get("constraint_guided_attempts", 1 if not eval_cfg.get("constraint_guided_decoding", False) else 4))
    max_new_tokens = int(eval_cfg.get("max_new_tokens", config.get("model", {}).get("max_seq_len", 256)))
    max_new_tokens_per_measure = int(eval_cfg.get("max_new_tokens_per_measure", 32))
    max_new_tokens_cap = int(eval_cfg.get("max_new_tokens_cap", 768))
    grammar_constrained = bool(eval_cfg.get("grammar_constrained_decoding", True))
    weights = constraint_weights(eval_cfg)
    generation_batch_size = max(1, int(eval_cfg.get("generation_batch_size", 1)))
    generation_fp16 = bool(eval_cfg.get("generation_fp16", False))
    sampling_protocol = (
        "per_sample_generator_batch_v1"
        if use_model_generation and generation_batch_size > 1
        else "per_sample_seed_serial_v2"
        if use_model_generation
        else "per_sample_rule_generator_v1"
        if use_rule_generation
        else "target_events"
    )
    metric_samples = eval_cfg.get("constraint_metric_samples")
    if metric_samples is None:
        metric_samples = generation_count if use_model_generation else len(dataset.samples)
    metric_samples = max(1, int(metric_samples))
    if metric_samples > len(dataset.samples):
        raise ValueError(
            f"Requested {metric_samples} constraint-metric samples from split {split!r}, "
            f"but only {len(dataset.samples)} are available."
        )
    eval_samples = dataset.samples[:metric_samples]

    generated_events_cache: list[list[dict[str, Any]]] | None = None
    first_candidate_hashes: list[str] | None = None
    if use_model_generation and generation_batch_size > 1:
        generation_total = min(len(dataset.samples), max(metric_samples, generation_count))
        generation_samples = dataset.samples[:generation_total]
        token_budgets = [
            generation_token_budget(
                sample.get("metadata", {}),
                max_new_tokens,
                max_new_tokens_per_measure,
                max_new_tokens_cap,
            )
            for sample in generation_samples
        ]
        generated_events_cache, first_candidate_hashes = generated_events_from_model_batch(
            model,
            tokenizer,
            [sample.get("metadata", {}) for sample in generation_samples],
            [evaluation_seed + idx for idx in range(generation_total)],
            attempts=attempts,
            max_new_tokens=token_budgets,
            batch_size=generation_batch_size,
            use_amp=generation_fp16,
            grammar_constrained=grammar_constrained,
            progress=bool(eval_cfg.get("generation_progress", False)),
            weights=weights,
        )

    for idx, sample in enumerate(eval_samples):
        condition_metadata = sample.get("metadata", {})
        target_metadata = sample.get(
            "_target_metadata",
            sample.get("target_metadata", condition_metadata),
        )
        first_candidate_sha256: str | None = None
        if generated_events_cache is not None:
            events = generated_events_cache[idx]
            first_candidate_sha256 = first_candidate_hashes[idx] if first_candidate_hashes else None
        elif use_model_generation:
            set_seed(evaluation_seed + idx)
            events = generated_events_from_model(
                model,
                tokenizer,
                condition_metadata,
                attempts=attempts,
                max_new_tokens=generation_token_budget(
                    condition_metadata,
                    max_new_tokens,
                    max_new_tokens_per_measure,
                    max_new_tokens_cap,
                ),
                grammar_constrained=grammar_constrained,
                weights=weights,
            )
            if attempts == 1:
                first_candidate_sha256 = _events_sha256(events)
        elif use_rule_generation:
            events = generated_events_from_rule(condition_metadata, evaluation_seed + idx)
            first_candidate_sha256 = _events_sha256(events)
        else:
            events = sample.get("events", [])
        report = analyze_events(events, target_metadata)
        reports.append(report)
        per_sample_records.append(
            {
                "experiment": experiment_name,
                "split": split,
                "sample_index": idx,
                "sample_id": sample.get("id"),
                "evaluation_seed": evaluation_seed + idx,
                "candidate_attempts": attempts if use_model_generation else 1 if use_rule_generation else 0,
                "generation_batch_size": generation_batch_size if use_model_generation else 0,
                "sampling_protocol": sampling_protocol,
                "first_candidate_sha256": first_candidate_sha256,
                "metadata": target_metadata,
                "condition_metadata": condition_metadata,
                "analysis": report,
            }
        )
        if idx < export_examples:
            out_path = export_root / f"{experiment_name}_{split}_{idx:03d}.musicxml"
            report_path = export_root / f"{experiment_name}_{split}_{idx:03d}.json"
            structural_ok = False
            measure_ok = False
            voice_ok = False
            export_error: str | None = None
            try:
                export_musicxml(events, out_path, target_metadata)
                structural_ok = musicxml_structurally_valid(out_path)
                if structural_ok:
                    xml_success += 1
                    measure_ok, voice_ok = musicxml_matches_request(out_path, target_metadata)
                    xml_measure_success += int(measure_ok)
                    xml_voice_success += int(voice_ok)
                save_json(
                    {
                        "metadata": target_metadata,
                        "condition_metadata": condition_metadata,
                        "analysis": report,
                        "export_validation": {
                            "structurally_valid": structural_ok,
                            "measure_count_adherent": measure_ok,
                            "voice_count_adherent": voice_ok,
                        },
                    },
                    report_path,
                )
            except Exception as exc:
                structural_ok = False
                export_error = f"{type(exc).__name__}: {exc}"
            examples.append(
                {
                    "experiment": experiment_name,
                    "split": split,
                    "sample_id": sample.get("id"),
                    "musicxml": str(out_path),
                    "analysis_report": str(report_path),
                    "musicxml_structurally_valid": structural_ok,
                    "musicxml_measure_count_adherent": measure_ok,
                    "musicxml_voice_count_adherent": voice_ok,
                    "metadata": target_metadata,
                    "condition_metadata": condition_metadata,
                    "analysis": report,
                    "export_error": export_error,
                }
            )

    # Ensure generation examples can be larger than exported evaluation count by
    # writing additional MusicXML files if requested.
    for extra_idx, sample in enumerate(dataset.samples[export_examples:generation_count], start=export_examples):
        condition_metadata = sample.get("metadata", {})
        target_metadata = sample.get(
            "_target_metadata",
            sample.get("target_metadata", condition_metadata),
        )
        if generated_events_cache is not None:
            events = generated_events_cache[extra_idx]
        elif use_model_generation:
            set_seed(evaluation_seed + extra_idx)
            events = generated_events_from_model(
                model,
                tokenizer,
                condition_metadata,
                attempts=attempts,
                max_new_tokens=generation_token_budget(
                    condition_metadata,
                    max_new_tokens,
                    max_new_tokens_per_measure,
                    max_new_tokens_cap,
                ),
                grammar_constrained=grammar_constrained,
                weights=weights,
            )
        elif use_rule_generation:
            events = generated_events_from_rule(
                condition_metadata,
                evaluation_seed + extra_idx,
            )
        else:
            events = sample.get("events", [])
        report = analyze_events(events, target_metadata)
        out_path = export_root / f"{experiment_name}_{split}_{extra_idx:03d}.musicxml"
        report_path = export_root / f"{experiment_name}_{split}_{extra_idx:03d}.json"
        try:
            export_musicxml(events, out_path, target_metadata)
            structural_ok = musicxml_structurally_valid(out_path)
            measure_ok, voice_ok = (
                musicxml_matches_request(out_path, target_metadata)
                if structural_ok
                else (False, False)
            )
            save_json(
                {
                    "metadata": target_metadata,
                    "condition_metadata": condition_metadata,
                    "analysis": report,
                    "export_validation": {
                        "structurally_valid": structural_ok,
                        "measure_count_adherent": measure_ok,
                        "voice_count_adherent": voice_ok,
                    },
                },
                report_path,
            )
            examples.append(
                {
                    "experiment": experiment_name,
                    "split": split,
                    "sample_id": sample.get("id"),
                    "musicxml": str(out_path),
                    "analysis_report": str(report_path),
                    "musicxml_structurally_valid": structural_ok,
                    "musicxml_measure_count_adherent": measure_ok,
                    "musicxml_voice_count_adherent": voice_ok,
                    "metadata": target_metadata,
                    "condition_metadata": condition_metadata,
                    "analysis": report,
                    "export_error": None,
                }
            )
        except Exception as exc:
            examples.append(
                {
                    "experiment": experiment_name,
                    "split": split,
                    "sample_id": sample.get("id"),
                    "musicxml": str(out_path),
                    "analysis_report": str(report_path),
                    "musicxml_structurally_valid": False,
                    "musicxml_measure_count_adherent": False,
                    "musicxml_voice_count_adherent": False,
                    "metadata": target_metadata,
                    "condition_metadata": condition_metadata,
                    "analysis": report,
                    "export_error": f"{type(exc).__name__}: {exc}",
                }
            )

    metrics = {
        "experiment": experiment_name,
        "split": split,
        "num_samples": len(eval_samples),
        "token_accuracy": model_metrics["token_accuracy"],
        "model_loss": model_metrics["loss"],
        "target_pcset_coverage": _mean([report["pcset_coverage"] for report in reports]),
        "pcset_coverage": _mean([report["pcset_coverage"] for report in reports]),
        "pcset_precision": _mean([report["pcset_precision"] for report in reports]),
        "pcset_jaccard": _mean([report["pcset_jaccard"] for report in reports]),
        "interval_vector_distance": _mean([report["interval_vector_distance"] for report in reports]),
        "row_order_accuracy": _mean([report["row_order_accuracy"] for report in reports]),
        "serial_row_order_accuracy": _mean([report["row_order_accuracy"] for report in reports]),
        "aggregate_completion_rate": _mean(
            [
                report["aggregate_completion_rate"]
                if report.get("aggregate_target_applicable", False)
                else None
                for report in reports
            ]
        ),
        "serial_transformation_accuracy": _mean([report["serial_transformation_accuracy"] for report in reports]),
        "rhythmic_profile_distance": _mean([report["rhythmic_profile_distance"] for report in reports]),
        "density_curve_error": _mean([report["density_curve_error"] for report in reports]),
        "gesture_consistency_score": _mean([report["gesture_consistency_score"] for report in reports]),
        "range_violation_rate": _mean([report["range_violation_rate"] for report in reports]),
        "instrument_range_violation_rate": _mean([report["instrument_range_violation_rate"] for report in reports]),
        "content_span_ratio": _mean([report["content_span_ratio"] for report in reports]),
        "voice_count_adherence": _mean([report["voice_count_adherence"] for report in reports]),
        "musicxml_export_success_rate": 1.0 if export_examples == 0 else xml_success / export_examples,
        "musicxml_measure_adherence_rate": 1.0 if export_examples == 0 else xml_measure_success / export_examples,
        "musicxml_voice_adherence_rate": 1.0 if export_examples == 0 else xml_voice_success / export_examples,
        "generation_batch_size": generation_batch_size if use_model_generation else 0,
        "sampling_protocol": sampling_protocol,
        "grammar_constrained_decoding": grammar_constrained if use_model_generation else False,
        "generation_source": (
            "transformer"
            if use_model_generation
            else "rule_generator"
            if use_rule_generation
            else "stored_target_events"
        ),
        "teacher_forced_evaluated_tokens": model_metrics.get("evaluated_tokens"),
    }
    provenance = {
        "config_path": Path(config_path).resolve().as_posix(),
        "config_sha256": _file_sha256(config_path),
        "data_path": data_path.resolve().as_posix(),
        "data_sha256": _file_sha256(data_path),
        "vocab_path": vocab_path.resolve().as_posix(),
        "vocab_sha256": _file_sha256(vocab_path),
        "dataset_split": split,
        "dataset_split_size": len(dataset.samples),
        "checkpoint_path": None if checkpoint_path is None else checkpoint_path.resolve().as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_training_seed": checkpoint_training_seed,
    }
    save_json({**metrics, "provenance": provenance}, output)
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
                "candidate_attempts": attempts if use_model_generation else 1 if use_rule_generation else 0,
                "generation_batch_size": generation_batch_size if use_model_generation else 0,
                "sampling_protocol": sampling_protocol,
                "num_samples": len(per_sample_records),
                "provenance": provenance,
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
    parser.add_argument("--export-dir", default=None)
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
        export_dir=args.export_dir,
    )
    print(metrics)


if __name__ == "__main__":
    main()
