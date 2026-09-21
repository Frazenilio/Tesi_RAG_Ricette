import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.metrics import compute_iou_stats

DEFAULT_DIRS = [
    PROJECT_ROOT / "results" / "2026-07-15T11-32-35_START_DS",
    PROJECT_ROOT / "results" / "2026-07-17T10-32-24",
]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "Plots" / "SavedPlots" / "RAG_Evaluation"

# Color Palette for Thesis Plots
COLOR_IOU = "#2b5c8f"        # Deep Slate Blue
COLOR_HUMAN = "#27ae60"      # Emerald Green
COLOR_IOU_LIGHT = "#5dade2"
COLOR_HUMAN_LIGHT = "#58d68d"

MODEL_ORDER = [
    "Llama-3.2-3B-Instruct-GGUF",
    "gemma3:1b",
    "granite4.1:3b",
    "ministral-3:3b",
    "qwen3.5:4b",
]

MODEL_LABELS = {
    "Llama-3.2-3B-Instruct-GGUF": "Llama 3.2 3B",
    "gemma3:1b": "Gemma 3 1B",
    "granite4.1:3b": "Granite 4.1 3B",
    "ministral-3:3b": "Ministral 3 3B",
    "qwen3.5:4b": "Qwen 3.5 4B",
}


def setup_style():
    """Sets consistent publication-ready plot style."""
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 15,
        "figure.titleweight": "bold",
    })


def load_rag_data(directories: list[Path]) -> pd.DataFrame:
    """Loads all RAG generation results and computes IoU and Human scores."""
    rows = []
    for d in directories:
        d = Path(d)
        if not d.exists():
            print(f"Warning: Directory not found: {d}")
            continue
        json_files = sorted(d.glob("**/*.json"))
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {jf}: {e}")
                continue

            gen_model = data.get("model", "unknown")
            gen_model_short = gen_model.split("/")[-1]
            target_type = data.get("target_type", "unknown")
            qa_pairs = data.get("qa_pairs", [])

            for q in qa_pairs:
                orig = q.get("original_answer", "")
                refs = q.get("correct_answers", [])
                max_iou, mean_iou, _ = compute_iou_stats(orig, refs)

                human_score = float(q.get("human_score")) if q.get("human_score") is not None else np.nan

                rows.append({
                    "model_raw": gen_model,
                    "model": gen_model_short,
                    "task": target_type,
                    "recipe_name": q.get("recipe_name", ""),
                    "iou_max": max_iou * 100.0,
                    "iou_mean": mean_iou * 100.0,
                    "human_score": human_score,
                })

    df = pd.DataFrame(rows)
    return df


def plot_rag_correctness_by_model_and_task(df: pd.DataFrame, output_dir: Path, dpi: int = 300) -> Path:
    """Plot 1: Grouped bar chart comparing IoU Score vs Human Evaluation for each LLM, divided by Ingredients and Directions."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    tasks = [("ingredients", "Ingredients"), ("directions", "Directions")]
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    model_display_names = [MODEL_LABELS.get(m, m) for m in models]

    x = np.arange(len(models))
    width = 0.35

    for ax_idx, (t_key, t_title) in enumerate(tasks):
        ax = axes[ax_idx]
        t_data = df[df["task"] == t_key]

        iou_means, iou_sems = [], []
        human_means, human_sems = [], []

        for m in models:
            sub = t_data[t_data["model"] == m]
            iou_means.append(sub["iou_max"].mean())
            iou_sems.append(sub["iou_max"].sem())
            human_means.append(sub["human_score"].mean())
            human_sems.append(sub["human_score"].sem())

        r_iou = ax.bar(
            x - width / 2,
            iou_means,
            width,
            yerr=iou_sems,
            capsize=4,
            label="IoU Score (%)",
            color=COLOR_IOU,
            alpha=0.92,
            edgecolor="white",
            linewidth=1.2,
        )

        r_human = ax.bar(
            x + width / 2,
            human_means,
            width,
            yerr=human_sems,
            capsize=4,
            label="Human Evaluation Score",
            color=COLOR_HUMAN,
            alpha=0.92,
            edgecolor="white",
            linewidth=1.2,
        )

        # Value annotations cleanly floating above error bars
        for rects, sems in [(r_iou, iou_sems), (r_human, human_sems)]:
            for rect, sem in zip(rects, sems):
                h = rect.get_height()
                if h > 0:
                    y_pos = h + (sem if not np.isnan(sem) else 0) + 1.8
                    ax.annotate(
                        f"{h:.1f}",
                        xy=(rect.get_x() + rect.get_width() / 2, y_pos),
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                        color="#2c3e50",
                    )

        ax.set_title(f"Task: {t_title}", fontsize=13, pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(model_display_names, rotation=15, ha="right", fontsize=10.5)
        ax.set_ylabel("Score (0 – 100 Scale / % Overlap)" if ax_idx == 0 else "", fontsize=11)
        ax.set_ylim(0, 118)
        ax.axhline(100, color="gray", linestyle="--", alpha=0.3, linewidth=1)
        ax.legend(loc="upper right", frameon=True, fontsize=10)

    fig.suptitle(
        "RAG Answer Correctness: IoU Score vs. Human Evaluation\nDivided by Model and Task (Ingredients vs. Directions)",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    out_path = output_dir / "rag_correctness_by_model_and_task.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_rag_distributions_boxplots(df: pd.DataFrame, output_dir: Path, dpi: int = 300) -> Path:
    """Plot 2: Boxplots showing the distribution of IoU and Human scores per model and task."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 11), sharex=True)

    tasks = [("ingredients", "Ingredients"), ("directions", "Directions")]
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]

    # Format dataframe for seaborn boxplot
    melted = df.copy()
    melted["model_display"] = melted["model"].map(MODEL_LABELS)
    melted_iou = melted[["model_display", "task", "iou_max"]].copy()
    melted_iou["Metric"] = "IoU Score (%)"
    melted_iou.rename(columns={"iou_max": "Score"}, inplace=True)

    melted_human = melted[["model_display", "task", "human_score"]].copy()
    melted_human["Metric"] = "Human Evaluation"
    melted_human.rename(columns={"human_score": "Score"}, inplace=True)

    plot_df = pd.concat([melted_iou, melted_human], ignore_index=True)
    palette = {"IoU Score (%)": COLOR_IOU, "Human Evaluation": COLOR_HUMAN}

    for ax_idx, (t_key, t_title) in enumerate(tasks):
        ax = axes[ax_idx]
        sub = plot_df[plot_df["task"] == t_key]

        ordered_labels = [MODEL_LABELS[m] for m in models]
        sns.boxplot(
            data=sub,
            x="model_display",
            y="Score",
            hue="Metric",
            order=ordered_labels,
            palette=palette,
            ax=ax,
            boxprops=dict(alpha=0.85),
            medianprops=dict(color="#c0392b", linewidth=2.2),
            showmeans=True,
            meanprops=dict(marker="o", markeredgecolor="black", markerfacecolor="white", markersize=6),
        )

        ax.set_title(f"Score Distribution: {t_title} (Circle = Mean, Line = Median)", fontsize=13, pad=10)
        ax.set_ylabel("Score (0 – 100)", fontsize=11)
        ax.set_xlabel("" if ax_idx == 0 else "LLM Model", fontsize=11)
        ax.set_ylim(-5, 108)
        ax.axhline(100, color="gray", linestyle="--", alpha=0.3)
        ax.legend(loc="lower left", frameon=True, fontsize=10)

    fig.suptitle(
        "RAG Correctness Distribution & Variance per Model\n(Comparing IoU Lexical Score and Human Evaluation)",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    out_path = output_dir / "rag_score_distributions_by_model.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_rag_success_rate_thresholds(
    df: pd.DataFrame,
    output_dir: Path,
    human_threshold: float = 95.0,
    iou_threshold: float = 95.0,
    dpi: int = 300,
) -> Path:
    """Plot 3: Percentage of High-Quality Answers (Human Score >= threshold and IoU >= threshold) across models and tasks."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    tasks = [("ingredients", "Ingredients"), ("directions", "Directions")]
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    model_display_names = [MODEL_LABELS.get(m, m) for m in models]

    x = np.arange(len(models))
    width = 0.35

    for ax_idx, (t_key, t_title) in enumerate(tasks):
        ax = axes[ax_idx]
        t_data = df[df["task"] == t_key]

        pct_human_high = []
        pct_iou_high = []

        for m in models:
            sub = t_data[t_data["model"] == m]
            # Human >= human_threshold
            h_rate = (sub["human_score"] >= human_threshold).mean() * 100.0
            # IoU >= iou_threshold
            i_rate = (sub["iou_max"] >= iou_threshold).mean() * 100.0
            pct_human_high.append(h_rate)
            pct_iou_high.append(i_rate)

        r1 = ax.bar(
            x - width / 2,
            pct_iou_high,
            width,
            label=f"IoU High Overlap (≥{int(iou_threshold)}%)",
            color=COLOR_IOU,
            alpha=0.92,
            edgecolor="white",
            linewidth=1.2,
        )

        r2 = ax.bar(
            x + width / 2,
            pct_human_high,
            width,
            label=f"Human Ground Truth Acceptable (≥{int(human_threshold)})",
            color=COLOR_HUMAN,
            alpha=0.92,
            edgecolor="white",
            linewidth=1.2,
        )

        for rects, vals in [(r1, pct_iou_high), (r2, pct_human_high)]:
            for rect, val in zip(rects, vals):
                h = rect.get_height()
                ax.annotate(
                    f"{val:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, h + 1.5),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color="#2c3e50",
                )

        ax.set_title(f"Task: {t_title}", fontsize=13, pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(model_display_names, rotation=15, ha="right", fontsize=10.5)
        ax.set_ylabel("% of Correct / High-Quality Answers" if ax_idx == 0 else "", fontsize=11)
        ax.set_ylim(0, 115)
        ax.axhline(100, color="gray", linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", frameon=True, fontsize=10)

    fig.suptitle(
        f"RAG Success Rate: Percentage of High-Quality Answers\n(Human Score ≥ {int(human_threshold)} vs. IoU Score ≥ {int(iou_threshold)}%)",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    out_path = output_dir / "rag_success_rate_by_threshold.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_rag_scatter_iou_vs_human(df: pd.DataFrame, output_dir: Path, dpi: int = 300) -> Path:
    """Plot 4: Scatter plot of IoU Score vs Human Evaluation for each recipe, colored by model, showing alignment and outliers."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    tasks = [("ingredients", "Ingredients"), ("directions", "Directions")]
    model_colors = {
        "Llama-3.2-3B-Instruct-GGUF": "#2980b9",
        "gemma3:1b": "#d35400",
        "granite4.1:3b": "#7f8c8d",
        "ministral-3:3b": "#c0392b",
        "qwen3.5:4b": "#16a085",
    }

    for ax_idx, (t_key, t_title) in enumerate(tasks):
        ax = axes[ax_idx]
        t_data = df[df["task"] == t_key]

        for m in MODEL_ORDER:
            if m not in t_data["model"].unique():
                continue
            sub = t_data[t_data["model"] == m]
            ax.scatter(
                sub["iou_max"],
                sub["human_score"],
                label=MODEL_LABELS.get(m, m),
                color=model_colors.get(m, "#333333"),
                alpha=0.65,
                s=48,
                edgecolors="none",
            )

        # Overall trendline
        valid = t_data.dropna(subset=["iou_max", "human_score"])
        if len(valid) > 1:
            z = np.polyfit(valid["iou_max"], valid["human_score"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid["iou_max"].min(), valid["iou_max"].max(), 100)
            y_line = np.clip(p(x_line), 0, 100)
            corr = valid["iou_max"].corr(valid["human_score"])
            ax.plot(x_line, y_line, color="#2c3e50", linestyle="--", linewidth=2, label=f"Trendline (r = {corr:.2f})")

        ax.set_title(f"Task: {t_title}", fontsize=13, pad=12)
        ax.set_xlabel("Max IoU Score (%)", fontsize=11)
        ax.set_ylabel("Human Evaluation Score (0 – 100)" if ax_idx == 0 else "", fontsize=11)
        ax.set_xlim(15, 105)
        ax.set_ylim(-5, 110)
        ax.legend(loc="upper left", frameon=True, fontsize=9.5)

    fig.suptitle(
        "Correlation: IoU Lexical Overlap vs. Human Quality Score per Recipe\n(Divided by Ingredients and Directions)",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    out_path = output_dir / "rag_iou_vs_human_scatter.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready plots for RAG answer correctness (IoU vs Human Evaluation).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for plots")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figures")
    parser.add_argument("--threshold", type=float, default=95.0, help="Correctness threshold for both Human and IoU (default: 95.0)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    print(f"Loading RAG evaluation results from {len(DEFAULT_DIRS)} source directories...")
    df = load_rag_data(DEFAULT_DIRS)
    print(f"Loaded {len(df)} total evaluated recipe instances across {df['model'].nunique()} models.")

    print("\n--- Generating RAG Correctness Plots ---")
    p1 = plot_rag_correctness_by_model_and_task(df, args.output_dir, args.dpi)
    p2 = plot_rag_distributions_boxplots(df, args.output_dir, args.dpi)
    p3 = plot_rag_success_rate_thresholds(
        df,
        args.output_dir,
        human_threshold=args.threshold,
        iou_threshold=args.threshold,
        dpi=args.dpi,
    )
    p4 = plot_rag_scatter_iou_vs_human(df, args.output_dir, args.dpi)

    print("\nAll RAG evaluation plots successfully generated in:", args.output_dir)


if __name__ == "__main__":
    main()
