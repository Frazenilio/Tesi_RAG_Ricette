from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_results(
    maxs: list[float],
    means: list[float],
    stddevs: list[float],
    title: str = "",
    save_path: Path | None = None,
) -> None:
    # Use a modern style if available
    try:
        plt.style.use("seaborn-v0_8-muted")
    except (OSError, ValueError):
        plt.style.use("ggplot")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    # Convert to numpy for easier handling
    maxs_arr = np.array(maxs)
    means_arr = np.array(means)

    # 1. Scatter Plot (Max IoU per Query)
    axes[0, 0].scatter(
        range(len(maxs)), maxs, alpha=0.6, s=40, label="Max IoU", color="tab:blue"
    )
    axes[0, 0].axhline(
        y=1.0, color="tab:green", linestyle="--", alpha=0.5, label="Perfect (1.0)"
    )
    axes[0, 0].axhline(
        y=0.95,
        color="tab:orange",
        linestyle=":",
        alpha=0.5,
        label="Near Perfect (0.95)",
    )
    axes[0, 0].set_title("Performance per Query", fontsize=13, pad=10)
    axes[0, 0].set_ylabel("IoU Score")
    axes[0, 0].set_xlabel("Query Index")
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].legend(loc="lower left", fontsize="small", frameon=True)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Distribution (Histogram)
    axes[0, 1].hist(
        maxs_arr, bins=15, range=(0, 1), alpha=0.7, color="tab:blue", edgecolor="white"
    )
    axes[0, 1].set_title("IoU Distribution (Max)", fontsize=13, pad=10)
    axes[0, 1].set_xlabel("IoU Score")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].grid(axis="y", alpha=0.3)

    # 3. Violin + Boxplot (Summary)
    data_to_plot = [maxs_arr, means_arr]
    # Filter out empty arrays to avoid errors in violinplot
    if all(len(d) > 0 for d in data_to_plot):
        parts = axes[1, 0].violinplot(
            data_to_plot, showmeans=False, showmedians=False, showextrema=False
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("tab:blue")
            pc.set_alpha(0.2)

    axes[1, 0].boxplot(
        data_to_plot,
        labels=["Max IoU", "Mean IoU"],
        patch_artist=True,
        boxprops=dict(facecolor="white", color="tab:blue", alpha=0.8),
        medianprops=dict(color="tab:red", linewidth=2),
    )
    axes[1, 0].set_title("Statistical Summary", fontsize=13, pad=10)
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].grid(axis="y", alpha=0.3)

    # 4. Success Rates & Stats (Text Panel)
    pct_perfect = np.mean(maxs_arr == 1.0) if len(maxs_arr) > 0 else 0
    pct_near = np.mean(maxs_arr >= 0.95) if len(maxs_arr) > 0 else 0
    avg_max = np.mean(maxs_arr) if len(maxs_arr) > 0 else 0
    avg_mean = np.mean(means_arr) if len(means_arr) > 0 else 0

    stats_text = (
        f"Sample Size: {len(maxs)}\n\n"
        f"Perfect Matches (1.0):\n{pct_perfect: >18.1%}\n\n"
        f"Near Perfect (≥0.95):\n{pct_near: >18.1%}\n\n"
        f"Avg Max IoU: {avg_max: >11.3f}\n"
        f"Avg Mean IoU: {avg_mean: >10.3f}"
    )
    axes[1, 1].text(
        0.5,
        0.5,
        stats_text,
        ha="center",
        va="center",
        fontsize=13,
        family="monospace",
        bbox=dict(facecolor="tab:gray", alpha=0.05, boxstyle="round,pad=1.5"),
    )
    axes[1, 1].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)

    print(f"% max IoU == 1.00 => {pct_perfect:.2%}")
    print(f"% max IoU >= 0.95 => {pct_near:.2%}")


def plot_divergence_results(
    scores: list[float],
    title: str = "",
    save_path: Path | None = None,
) -> None:
    """Plot per il Test 3: un unico score di divergenza per query (RAG vs LLM-only).

    A differenza di plot_results, qui non esiste la distinzione max/mean sui chunk
    di riferimento: ogni query produce un solo valore IoU di divergenza.
    """
    try:
        plt.style.use("seaborn-v0_8-muted")
    except (OSError, ValueError):
        plt.style.use("ggplot")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

    scores_arr = np.array(scores)

    # 1. Scatter
    axes[0].scatter(range(len(scores)), scores, alpha=0.6, s=40, color="tab:purple")
    axes[0].axhline(y=1.0, color="tab:green", linestyle="--", alpha=0.5, label="1.0")
    axes[0].axhline(y=0.95, color="tab:orange", linestyle=":", alpha=0.5, label="0.95")
    axes[0].set_title("Divergence per Query", fontsize=13, pad=10)
    axes[0].set_ylabel("IoU (RAG vs LLM-only)")
    axes[0].set_xlabel("Query Index")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize="small")
    axes[0].grid(True, alpha=0.3)

    # 2. Histogram
    axes[1].hist(
        scores_arr,
        bins=15,
        range=(0, 1),
        alpha=0.7,
        color="tab:purple",
        edgecolor="white",
    )
    axes[1].set_title("Divergence Distribution", fontsize=13, pad=10)
    axes[1].set_xlabel("IoU Score")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(axis="y", alpha=0.3)

    # 3. Stats
    pct_identical = np.mean(scores_arr == 1.0) if len(scores_arr) > 0 else 0
    pct_near = np.mean(scores_arr >= 0.95) if len(scores_arr) > 0 else 0
    avg_div = np.mean(scores_arr) if len(scores_arr) > 0 else 0
    stats_text = (
        f"Sample Size: {len(scores)}\n\n"
        f"Identical responses (1.0):\n{pct_identical: >18.1%}\n\n"
        f"Near-identical (≥0.95):\n{pct_near: >18.1%}\n\n"
        f"Avg Divergence IoU: {avg_div: >6.3f}"
    )
    axes[2].text(
        0.5,
        0.5,
        stats_text,
        ha="center",
        va="center",
        fontsize=13,
        family="monospace",
        bbox=dict(facecolor="tab:gray", alpha=0.05, boxstyle="round,pad=1.5"),
    )
    axes[2].axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)

    print(f"% identical responses (IoU==1.0) => {pct_identical:.2%}")
    print(f"% near-identical (IoU>=0.95)     => {pct_near:.2%}")
    print(f"Avg divergence IoU               => {avg_div:.3f}")
