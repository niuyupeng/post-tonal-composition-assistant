"""Train a post-tonal Transformer on synthetic symbolic score tokens."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from post_tonal.data.generate_corpus import derive_corpus, generate_samples
from post_tonal.data.post_tonal_dataset import PostTonalDataset, collate_batch, train_val_split
from post_tonal.data.score_tokenizer import ScoreTokenizer
from post_tonal.models.transformer import PostTonalTransformer
from post_tonal.utils import ensure_dir, get_device, load_json, load_yaml, save_json, set_seed


def maybe_generate_data(config: dict[str, Any]) -> None:
    data_path = Path(config["data_path"])
    vocab_path = Path(config["vocab_path"])
    if data_path.exists() and vocab_path.exists() and not config.get("force_generate_data", False):
        return
    if not config.get("generate_data", True) and not config.get("force_generate_data", False):
        missing = [str(path) for path in (data_path, vocab_path) if not path.exists()]
        raise FileNotFoundError(
            "Synthetic-data generation is disabled and required inputs are missing: "
            + ", ".join(missing)
        )
    corpus = config.get("corpus", {})
    derive_from = corpus.get("derive_from")
    if derive_from:
        source_path = Path(derive_from)
        if not source_path.exists():
            raise FileNotFoundError(
                f"Derived corpus source does not exist: {source_path}. "
                "Generate the base corpus before its condition ablations."
            )
        ablation = corpus.get("ablation")
        if not ablation:
            raise ValueError("A derived corpus requires corpus.ablation.")
        derive_corpus(
            source=source_path,
            output=data_path,
            vocab_output=vocab_path,
            ablation=str(ablation),
        )
        return
    generate_samples(
        num_samples=int(corpus.get("num_samples", 128)),
        output=data_path,
        vocab_output=vocab_path,
        seed=int(config.get("seed", 0)),
        min_measures=int(corpus.get("min_measures", 4)),
        max_measures=int(corpus.get("max_measures", 16)),
        min_voices=int(corpus.get("min_voices", 2)),
        max_voices=int(corpus.get("max_voices", 8)),
        export_musicxml_flag=bool(corpus.get("export_musicxml", False)),
        musicxml_dir=corpus.get("musicxml_dir", "data/generated"),
        focus=corpus.get("focus"),
        ablation=corpus.get("ablation"),
        train_samples=corpus.get("train_samples"),
        val_samples=corpus.get("val_samples"),
        test_samples=corpus.get("test_samples"),
    )


def token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    mask = labels != -100
    if mask.sum().item() == 0:
        return 0.0
    return float((preds[mask] == labels[mask]).float().mean().item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    fast_dev_run: bool = False,
    use_amp: bool = False,
    gradient_accumulation_steps: int = 1,
    gradient_clip_norm: float | None = 1.0,
) -> dict[str, float]:
    is_train = optimizer is not None
    accumulation_steps = max(1, int(gradient_accumulation_steps))
    model.train(is_train)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    batches = 0
    iterator = tqdm(loader, desc="train" if is_train else "valid", leave=False)
    if is_train:
        optimizer.zero_grad(set_to_none=True)
    for batch_idx, batch in enumerate(iterator):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(input_ids, attention_mask)
            loss_sum = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            valid_tokens = int((labels != -100).sum().item())
            loss = loss_sum / max(1, valid_tokens)
        if is_train:
            backward_loss = loss / accumulation_steps
            if scaler is not None and use_amp:
                scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()

            is_last_batch = batch_idx + 1 == len(loader)
            is_fast_dev_last = fast_dev_run and batch_idx + 1 >= 2
            should_step = (batch_idx + 1) % accumulation_steps == 0 or is_last_batch or is_fast_dev_last
            if should_step:
                if scaler is not None and use_amp:
                    scaler.unscale_(optimizer)
                if gradient_clip_norm is not None and gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                if scaler is not None and use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        predictions = torch.argmax(logits.detach(), dim=-1)
        valid_mask = labels != -100
        total_loss += float(loss_sum.item())
        total_correct += int((predictions[valid_mask] == labels[valid_mask]).sum().item())
        total_tokens += valid_tokens
        batches += 1
        if fast_dev_run and batches >= 2:
            break
    return {
        "loss": total_loss / max(1, total_tokens),
        "token_accuracy": total_correct / max(1, total_tokens),
        "evaluated_tokens": float(total_tokens),
    }


def _peak_process_memory_bytes() -> int | None:
    if os.name == "nt":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        success = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        return int(counters.PeakWorkingSetSize) if success else None
    try:
        import resource

        usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return usage if sys.platform == "darwin" else usage * 1024
    except (ImportError, AttributeError):
        return None


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def _merge_resume_provenance(
    checkpoint: dict[str, Any],
    summary: dict[str, Any] | None,
) -> dict[str, Any]:
    summary = summary or {}
    return {
        "resume_count": max(
            int(checkpoint.get("resume_count", 0)),
            int(summary.get("resume_count", 0)),
        )
        + 1,
        "elapsed_seconds": max(
            float(checkpoint.get("elapsed_seconds", 0.0)),
            float(summary.get("elapsed_seconds", 0.0)),
        ),
        "peak_process_ram_gib": max(
            float(checkpoint.get("peak_process_ram_gib", 0.0) or 0.0),
            float(summary.get("peak_process_ram_gib", 0.0) or 0.0),
        ),
        "peak_cuda_memory_allocated_gib": max(
            float(checkpoint.get("peak_cuda_memory_allocated_gib", 0.0) or 0.0),
            float(summary.get("peak_cuda_memory_allocated_gib", 0.0) or 0.0),
        ),
        "started_at": str(
            summary.get("started_at")
            or checkpoint.get("started_at")
            or datetime.now(timezone.utc).isoformat()
        ),
    }


def train(
    config_path: str | Path,
    batch_size_override: int | None = None,
    gradient_accumulation_steps_override: int | None = None,
    oom_adjustment: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    training_started = time.perf_counter()
    config_path = Path(config_path)
    config = load_yaml(config_path)
    train_cfg = config.setdefault("training", {})
    if batch_size_override is not None:
        train_cfg["batch_size"] = int(batch_size_override)
        train_cfg["validation_batch_size"] = min(
            int(train_cfg.get("validation_batch_size", 64)),
            max(16, int(batch_size_override) * 4),
        )
    if gradient_accumulation_steps_override is not None:
        train_cfg["gradient_accumulation_steps"] = int(gradient_accumulation_steps_override)
    set_seed(int(config.get("seed", 0)))
    maybe_generate_data(config)
    run_dir = ensure_dir(config.get("run_dir", "runs/default"))
    config_sha256 = _file_sha256(config_path)
    data_sha256 = _file_sha256(config["data_path"])
    vocab_sha256 = _file_sha256(config["vocab_path"])
    tokenizer = ScoreTokenizer.load(config["vocab_path"])
    model_cfg = config.get("model", {})
    max_seq_len = int(model_cfg.get("max_seq_len", 256))
    sequence_mode = str(train_cfg.get("sequence_mode", "truncate"))
    validation_sequence_mode = str(
        train_cfg.get(
            "validation_sequence_mode",
            "all" if sequence_mode != "truncate" else "truncate",
        )
    )
    target_tokens_per_window = int(train_cfg.get("target_tokens_per_window", 128))
    dataset_kwargs = {
        "max_seq_len": max_seq_len,
        "sep_id": tokenizer.token_to_id["SEP"],
        "target_tokens_per_window": target_tokens_per_window,
        "seed": int(config.get("seed", 0)),
        "tokenizer": tokenizer,
        "condition_ablation": config.get("condition_ablation"),
        "coverage_cycle_epochs": int(train_cfg.get("coverage_cycle_epochs", 10)),
    }
    try:
        train_ds = PostTonalDataset(
            config["data_path"],
            split="train",
            sequence_mode=sequence_mode,
            **dataset_kwargs,
        )
        val_ds = PostTonalDataset(
            config["data_path"],
            split="val",
            sequence_mode=validation_sequence_mode,
            **dataset_kwargs,
        )
    except ValueError:
        dataset = PostTonalDataset(
            config["data_path"],
            max_seq_len=max_seq_len,
            sequence_mode="truncate",
            tokenizer=tokenizer,
            condition_ablation=config.get("condition_ablation"),
        )
        train_ds, val_ds = train_val_split(dataset, float(train_cfg.get("val_fraction", 0.1)), int(config.get("seed", 0)))
    collate = lambda batch: collate_batch(batch, tokenizer.pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 16)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate,
    )
    validation_batch_size = int(
        train_cfg.get("validation_batch_size", train_cfg.get("batch_size", 16))
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=validation_batch_size,
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate,
    )
    device = get_device(config.get("device"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = PostTonalTransformer(vocab_size=tokenizer.vocab_size, **model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 3e-4)), weight_decay=float(train_cfg.get("weight_decay", 0.01)))
    use_amp = device.type == "cuda" and bool(train_cfg.get("fp16", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    accumulation_steps = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
    gradient_clip_norm = (
        float(train_cfg.get("gradient_clip_norm", 1.0))
        if bool(train_cfg.get("gradient_clipping", True))
        else None
    )

    metrics_path = Path(run_dir) / "metrics.csv"
    checkpoint_path = Path(run_dir) / "checkpoint.pt"
    last_checkpoint_path = Path(run_dir) / "last_checkpoint.pt"
    summary_path = Path(run_dir) / "train_summary.json"
    best_val = float("inf")
    best_epoch: int | None = None
    patience = int(train_cfg.get("early_stopping_patience", 5))
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    start_epoch = 1
    resume_count = 0
    elapsed_before_resume = 0.0
    previous_peak_process_ram_gib = 0.0
    previous_peak_cuda_memory_gib = 0.0
    started_at = datetime.now(timezone.utc).isoformat()

    if resume and last_checkpoint_path.exists():
        resume_state = torch.load(
            last_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        expected_hashes = {
            "config_sha256": config_sha256,
            "data_sha256": data_sha256,
            "vocab_sha256": vocab_sha256,
        }
        for key, expected in expected_hashes.items():
            observed = resume_state.get(key)
            if observed != expected:
                raise ValueError(
                    f"Cannot resume {run_dir}: {key} changed "
                    f"(checkpoint={observed}, current={expected})."
                )
        previous_summary: dict[str, Any] | None = None
        if summary_path.exists():
            candidate_summary = load_json(summary_path)
            if isinstance(candidate_summary, dict) and all(
                candidate_summary.get(key) == expected
                for key, expected in expected_hashes.items()
            ):
                previous_summary = candidate_summary
        resume_provenance = _merge_resume_provenance(
            resume_state,
            previous_summary,
        )
        model.load_state_dict(resume_state["model_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        if use_amp and resume_state.get("scaler_state"):
            scaler.load_state_dict(resume_state["scaler_state"])
        best_val = float(resume_state.get("best_val", best_val))
        best_epoch_value = resume_state.get("best_epoch")
        best_epoch = None if best_epoch_value is None else int(best_epoch_value)
        bad_epochs = int(resume_state.get("bad_epochs", 0))
        history = list(resume_state.get("history", []))
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        resume_count = int(resume_provenance["resume_count"])
        elapsed_before_resume = float(resume_provenance["elapsed_seconds"])
        previous_peak_process_ram_gib = float(
            resume_provenance["peak_process_ram_gib"]
        )
        previous_peak_cuda_memory_gib = float(
            resume_provenance["peak_cuda_memory_allocated_gib"]
        )
        started_at = str(resume_provenance["started_at"])
        _restore_rng_state(resume_state.get("rng_state"))

    stop_reason = "max_epochs"
    max_epochs = int(train_cfg.get("epochs", 1))
    if bad_epochs >= patience:
        stop_reason = "early_stopping"

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_token_accuracy", "val_loss", "val_token_accuracy", "device"])
        writer.writeheader()
        for existing_row in history:
            writer.writerow(existing_row)
        f.flush()
        for epoch in range(start_epoch, max_epochs + 1):
            if bad_epochs >= patience:
                stop_reason = "early_stopping"
                break
            if hasattr(train_ds, "set_epoch"):
                train_ds.set_epoch(epoch - 1)
            if hasattr(val_ds, "set_epoch"):
                val_ds.set_epoch(epoch - 1)
            train_metrics = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                scaler,
                fast_dev_run=bool(train_cfg.get("fast_dev_run", False)),
                use_amp=use_amp,
                gradient_accumulation_steps=accumulation_steps,
                gradient_clip_norm=gradient_clip_norm,
            )
            with torch.no_grad():
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    None,
                    device,
                    None,
                    fast_dev_run=bool(train_cfg.get("fast_dev_run", False)),
                    use_amp=False,
                    gradient_clip_norm=gradient_clip_norm,
                )
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_token_accuracy": train_metrics["token_accuracy"],
                "val_loss": val_metrics["loss"],
                "val_token_accuracy": val_metrics["token_accuracy"],
                "device": str(device),
            }
            writer.writerow(row)
            f.flush()
            history.append(row)
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                best_epoch = epoch
                bad_epochs = 0
                _atomic_torch_save(
                    {
                        "model_state": model.state_dict(),
                        "config": config,
                        "vocab_path": str(config["vocab_path"]),
                        "vocab_size": tokenizer.vocab_size,
                        "best_epoch": best_epoch,
                        "best_val_loss": best_val,
                        "config_sha256": config_sha256,
                        "data_sha256": data_sha256,
                        "vocab_sha256": vocab_sha256,
                    },
                    checkpoint_path,
                )
            else:
                bad_epochs += 1
            elapsed_seconds = elapsed_before_resume + (time.perf_counter() - training_started)
            process_peak_bytes = _peak_process_memory_bytes()
            process_peak_gib = (
                0.0
                if process_peak_bytes is None
                else process_peak_bytes / 1024**3
            )
            cuda_peak_gib = (
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else 0.0
            )
            previous_peak_process_ram_gib = max(
                previous_peak_process_ram_gib,
                process_peak_gib,
            )
            previous_peak_cuda_memory_gib = max(
                previous_peak_cuda_memory_gib,
                cuda_peak_gib,
            )
            _atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict() if use_amp else None,
                    "config": config,
                    "epoch": epoch,
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                    "bad_epochs": bad_epochs,
                    "history": history,
                    "resume_count": resume_count,
                    "elapsed_seconds": elapsed_seconds,
                    "peak_process_ram_gib": previous_peak_process_ram_gib,
                    "peak_cuda_memory_allocated_gib": previous_peak_cuda_memory_gib,
                    "started_at": started_at,
                    "rng_state": _capture_rng_state(),
                    "config_sha256": config_sha256,
                    "data_sha256": data_sha256,
                    "vocab_sha256": vocab_sha256,
                },
                last_checkpoint_path,
            )
            save_json(
                {
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                    "epochs_ran": len(history),
                    "completed": False,
                    "stop_reason": "in_progress",
                    "last_completed_epoch": epoch,
                    "resume_count": resume_count,
                    "oom_adjustment": oom_adjustment,
                    "config_path": str(config_path),
                    "config_sha256": config_sha256,
                    "data_sha256": data_sha256,
                    "vocab_sha256": vocab_sha256,
                    "history": history,
                },
                summary_path,
            )
            if bad_epochs >= patience:
                stop_reason = "early_stopping"
                break

    batch_size = int(train_cfg.get("batch_size", 16))
    peak_process_memory = _peak_process_memory_bytes()
    current_peak_process_gib = (
        0.0
        if peak_process_memory is None
        else peak_process_memory / 1024**3
    )
    current_peak_cuda_gib = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    elapsed_seconds = elapsed_before_resume + (time.perf_counter() - training_started)
    summary = {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "last_completed_epoch": int(history[-1]["epoch"]) if history else 0,
        "device": str(device),
        "batch_size": batch_size,
        "validation_batch_size": validation_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": batch_size * accumulation_steps,
        "sequence_mode": sequence_mode,
        "validation_sequence_mode": validation_sequence_mode,
        "target_tokens_per_window": target_tokens_per_window,
        "train_dataset_items_per_epoch": len(train_ds),
        "validation_windows": len(val_ds),
        "gradient_clip_norm": gradient_clip_norm,
        "oom_adjustment": oom_adjustment,
        "resume_count": resume_count,
        "stop_reason": stop_reason,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "data_sha256": data_sha256,
        "vocab_sha256": vocab_sha256,
        "checkpoint_sha256": _file_sha256(checkpoint_path) if checkpoint_path.exists() else None,
        "last_checkpoint_path": str(last_checkpoint_path),
        "peak_process_ram_gib": (
            max(previous_peak_process_ram_gib, current_peak_process_gib)
            if peak_process_memory is not None or previous_peak_process_ram_gib > 0
            else None
        ),
        "peak_cuda_memory_allocated_gib": (
            max(previous_peak_cuda_memory_gib, current_peak_cuda_gib)
            if device.type == "cuda"
            else None
        ),
        "completed": True,
        "history": history,
    }
    _atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if use_amp else None,
            "config": config,
            "epoch": summary["last_completed_epoch"],
            "best_val": best_val,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "history": history,
            "resume_count": resume_count,
            "elapsed_seconds": elapsed_seconds,
            "peak_process_ram_gib": summary["peak_process_ram_gib"],
            "peak_cuda_memory_allocated_gib": (
                summary["peak_cuda_memory_allocated_gib"] or 0.0
            ),
            "started_at": started_at,
            "rng_state": _capture_rng_state(),
            "oom_adjustment": oom_adjustment,
            "config_sha256": config_sha256,
            "data_sha256": data_sha256,
            "vocab_sha256": vocab_sha256,
        },
        last_checkpoint_path,
    )
    post_save_peak = _peak_process_memory_bytes()
    if post_save_peak is not None:
        summary["peak_process_ram_gib"] = max(
            float(summary["peak_process_ram_gib"] or 0.0),
            post_save_peak / 1024**3,
        )
    save_json(summary, summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--auto-oom-retry", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    attempts = [
        (
            args.batch_size,
            args.gradient_accumulation_steps,
            None,
        )
    ]
    if args.auto_oom_retry and args.batch_size is None:
        attempts.extend(
            [
                (8, 2, "CUDA OOM retry: batch_size=8, gradient_accumulation_steps=2"),
                (4, 4, "CUDA OOM retry: batch_size=4, gradient_accumulation_steps=4"),
            ]
        )
    last_error: Exception | None = None
    for attempt_index, (batch_size, accumulation_steps, adjustment) in enumerate(attempts):
        try:
            summary = train(
                args.config,
                batch_size_override=batch_size,
                gradient_accumulation_steps_override=accumulation_steps,
                oom_adjustment=adjustment,
                resume=args.resume or attempt_index > 0,
            )
            break
        except Exception as exc:
            last_error = exc
            if not args.auto_oom_retry or not _is_cuda_oom(exc) or (batch_size, accumulation_steps, adjustment) == attempts[-1]:
                raise
            next_batch, next_accumulation, next_adjustment = attempts[attempt_index + 1]
            print(
                {
                    "cuda_oom": str(exc),
                    "next_batch_size": next_batch,
                    "next_gradient_accumulation_steps": next_accumulation,
                    "next_retry": next_adjustment,
                },
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        assert last_error is not None
        raise last_error
    print(summary)


if __name__ == "__main__":
    main()
