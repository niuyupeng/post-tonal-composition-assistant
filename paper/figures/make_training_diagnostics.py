"""Render training diagnostics from the saved primary-run summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def render(summary_path: Path, output_stem: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history = summary["history"]
    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]
    train_accuracy = [float(row["train_token_accuracy"]) for row in history]
    val_accuracy = [float(row["val_token_accuracy"]) for row in history]
    best_index = min(range(len(val_loss)), key=val_loss.__getitem__)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), constrained_layout=True)
    colors = {"train": "#0072B2", "validation": "#D55E00"}

    axes[0].plot(epochs, train_loss, color=colors["train"], linewidth=1.5, label="Training")
    axes[0].plot(epochs, val_loss, color=colors["validation"], linewidth=1.5, label="Validation")
    axes[0].scatter(
        [epochs[best_index]],
        [val_loss[best_index]],
        color="black",
        s=18,
        zorder=3,
        label=f"Best epoch ({epochs[best_index]})",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[0].legend(frameon=False)
    axes[0].text(-0.16, 1.03, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=10)

    axes[1].plot(epochs, train_accuracy, color=colors["train"], linewidth=1.5, label="Training")
    axes[1].plot(epochs, val_accuracy, color=colors["validation"], linewidth=1.5, label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Full-sequence token accuracy")
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[1].legend(frameon=False)
    axes[1].text(-0.16, 1.03, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=10)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("runs/proposed_constraint_guided_transformer/train_summary.json"),
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path("paper/figures/training_diagnostics"),
    )
    args = parser.parse_args()
    render(args.summary, args.output_stem)


if __name__ == "__main__":
    main()
