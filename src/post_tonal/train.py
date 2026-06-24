"""Train a post-tonal Transformer on synthetic symbolic score tokens."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from post_tonal.data.generate_corpus import generate_samples
from post_tonal.data.post_tonal_dataset import PostTonalDataset, collate_batch, train_val_split
from post_tonal.data.score_tokenizer import ScoreTokenizer
from post_tonal.models.transformer import PostTonalTransformer
from post_tonal.utils import ensure_dir, get_device, load_yaml, save_json, set_seed


def maybe_generate_data(config: dict[str, Any]) -> None:
    data_path = Path(config["data_path"])
    vocab_path = Path(config["vocab_path"])
    if data_path.exists() and vocab_path.exists() and not config.get("force_generate_data", False):
        return
    corpus = config.get("corpus", {})
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
    scaler: torch.cuda.amp.GradScaler | None,
    fast_dev_run: bool = False,
    use_amp: bool = False,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    total_acc = 0.0
    batches = 0
    iterator = tqdm(loader, desc="train" if is_train else "valid", leave=False)
    for batch in iterator:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        if is_train:
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += float(loss.item())
        total_acc += token_accuracy(logits.detach(), labels)
        batches += 1
        if fast_dev_run and batches >= 2:
            break
    return {"loss": total_loss / max(1, batches), "token_accuracy": total_acc / max(1, batches)}


def train(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    set_seed(int(config.get("seed", 0)))
    maybe_generate_data(config)
    run_dir = ensure_dir(config.get("run_dir", "runs/default"))
    tokenizer = ScoreTokenizer.load(config["vocab_path"])
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    max_seq_len = int(model_cfg.get("max_seq_len", 256))
    try:
        train_ds = PostTonalDataset(config["data_path"], max_seq_len=max_seq_len, split="train")
        val_ds = PostTonalDataset(config["data_path"], max_seq_len=max_seq_len, split="val")
    except ValueError:
        dataset = PostTonalDataset(config["data_path"], max_seq_len=max_seq_len)
        train_ds, val_ds = train_val_split(dataset, float(train_cfg.get("val_fraction", 0.1)), int(config.get("seed", 0)))
    collate = lambda batch: collate_batch(batch, tokenizer.pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 16)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.get("batch_size", 16)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate,
    )
    device = get_device(config.get("device"))
    model = PostTonalTransformer(vocab_size=tokenizer.vocab_size, **model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 3e-4)), weight_decay=float(train_cfg.get("weight_decay", 0.01)))
    use_amp = device.type == "cuda" and bool(train_cfg.get("fp16", True))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    metrics_path = Path(run_dir) / "metrics.csv"
    best_val = float("inf")
    patience = int(train_cfg.get("early_stopping_patience", 5))
    bad_epochs = 0
    history: list[dict[str, Any]] = []

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_token_accuracy", "val_loss", "val_token_accuracy", "device"])
        writer.writeheader()
        for epoch in range(1, int(train_cfg.get("epochs", 1)) + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                scaler,
                fast_dev_run=bool(train_cfg.get("fast_dev_run", False)),
                use_amp=use_amp,
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
                bad_epochs = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": config,
                        "vocab_path": str(config["vocab_path"]),
                        "vocab_size": tokenizer.vocab_size,
                    },
                    Path(run_dir) / "checkpoint.pt",
                )
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

    summary = {"best_val_loss": best_val, "epochs_ran": len(history), "device": str(device), "history": history}
    save_json(summary, Path(run_dir) / "train_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summary = train(args.config)
    print(summary)


if __name__ == "__main__":
    main()
