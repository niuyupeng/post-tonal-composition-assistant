"""Build the Chinese manuscript result figure from canonical full-run metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

from make_project2_full_results import ABLATIONS, COLORS, FULL, VANILLA, limits, read_rows


ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = ROOT / "paper" / "figures" / "project2_full_results_zh.pdf"
SVG_PATH = ROOT / "paper" / "figures" / "project2_full_results_zh.svg"
PNG_PATH = ROOT / "paper" / "figures" / "project2_full_results_zh.png"
FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")

METRICS_ZH = [
    ("target_pcset_coverage", "音级集合\n覆盖率", "higher"),
    ("interval_vector_distance", "音程向量\n距离", "lower"),
    ("row_order_accuracy", "序列顺序\n准确率", "higher"),
    ("rhythmic_profile_distance", "节奏轮廓\n距离", "lower"),
    ("gesture_consistency_score", "姿态\n一致性", "higher"),
]


def add_pair(
    ax: plt.Axes,
    values: list[float],
    labels: list[str],
    title: str,
    direction: str,
) -> None:
    x = [0, 1]
    ax.plot(x, values, color="#444444", linewidth=1.1, zorder=1)
    ax.scatter(
        x,
        values,
        s=54,
        c=[COLORS["reference"], COLORS["comparison"]],
        edgecolors="white",
        linewidths=0.8,
        zorder=2,
    )
    ax.set_xlim(-0.42, 1.42)
    ax.set_ylim(*limits(values))
    ax.set_xticks(x, labels)
    direction_label = "↑ 越高越好" if direction == "higher" else "↓ 越低越好"
    ax.set_title(f"{title}\n{direction_label}", fontsize=7.4, pad=4, color="#333333")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.tick_params(axis="both", labelsize=7, length=2.5, width=0.6)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.6)
    value_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for xi, value in zip(x, values):
        if 0.99 <= value < 1:
            label = f"{value:.4f}"
        elif value < 1:
            label = f"{value:.3f}"
        else:
            label = f"{value:.2f}"
        ax.text(
            xi,
            value + 0.045 * value_span,
            label,
            ha="center",
            va="bottom",
            fontsize=6.8,
            color="#202020",
        )


def build_figure() -> None:
    if not FONT_PATH.exists():
        raise RuntimeError(f"Chinese font not found: {FONT_PATH}")
    font_manager.fontManager.addfont(str(FONT_PATH))
    family = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 8,
            "axes.unicode_minus": False,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    rows = read_rows()
    fig, axes = plt.subplots(2, 5, figsize=(7.1, 4.05), constrained_layout=False)
    fig.subplots_adjust(left=0.072, right=0.995, top=0.80, bottom=0.13, wspace=0.62, hspace=1.05)

    short_labels = {
        "without_pcset_constraints": "去音级集",
        "without_serial_constraints": "去序列",
        "without_rhythm_constraints": "去节奏",
        "without_gesture_constraints": "去姿态",
    }
    for col, (metric, title, direction) in enumerate(METRICS_ZH):
        add_pair(
            axes[0, col],
            [float(rows[VANILLA][metric]), float(rows[FULL][metric])],
            ["K=1", "K=4"],
            title,
            direction,
        )
        ablation = ABLATIONS[metric]
        add_pair(
            axes[1, col],
            [float(rows[VANILLA][metric]), float(rows[ablation][metric])],
            ["完整 K=1", short_labels[ablation]],
            title,
            direction,
        )

    fig.text(0.018, 0.965, "a", fontsize=11, fontweight="bold", va="top")
    fig.text(0.047, 0.963, "引导式候选选择：共享生成器", fontsize=8.3, fontweight="bold", va="top")
    fig.text(0.018, 0.505, "b", fontsize=11, fontweight="bold", va="top")
    fig.text(0.047, 0.503, "条件移除：分别训练的 K=1 解码器", fontsize=8.3, fontweight="bold", va="top")
    fig.text(
        0.5,
        0.035,
        "自动测试集指标（每个实验 2,000 个条件）",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color="#444444",
    )

    fig.savefig(PDF_PATH, bbox_inches="tight")
    fig.savefig(SVG_PATH, bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
    print(f"Wrote {PDF_PATH.relative_to(ROOT)}")
    print(f"Wrote {SVG_PATH.relative_to(ROOT)}")
    print(f"Wrote {PNG_PATH.relative_to(ROOT)}")
