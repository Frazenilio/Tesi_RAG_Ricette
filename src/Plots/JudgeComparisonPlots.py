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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "Plots" / "SavedPlots" / "Judges"

# Visual styling tokens
COLOR_IOU = "#2b5c8f"        # Deep Slate Blue
COLOR_HUMAN = "#27ae60"      # Emerald Green
COLOR_AVG_JUDGE = "#8e44ad"  # Royal Purple

COLOR_LLAMA = "#2980b9"      # Ocean Blue
COLOR_GEMMA = "#d35400"      # Burnt Orange
COLOR_MINISTRAL = "#c0392b"  # Crimson Red
COLOR_QWEN = "#16a085"       # Teal
COLOR_GRANITE = "#7f8c8d"    # Gray

JUDGE_COLORS = {
    "Llama-3.2-3B-Instruct-GGUF": COLOR_LLAMA,
    "gemma3:1b": COLOR_GEMMA,
    "ministral-3:3b": COLOR_MINISTRAL,
    "qwen3.5:2b": COLOR_QWEN,
}

JUDGE_LABELS = {
    "Llama-3.2-3B-Instruct-GGUF": "Llama 3.2 3B",
    "gemma3:1b": "Gemma 3 1B",
    "ministral-3:3b": "Ministral 3 3B",
    "qwen3.5:2b": "Qwen 3.5 2B",
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


def load_and_extract_data(directories: list[Path]) -> pd.DataFrame:
    """Loads all JSON evaluation runs from specified directories and extracts QA pairs with IoU, Human, and Judge scores."""
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
                round_1 = q.get("round_1", {})
                r1_avg = float(round_1.get("average_score")) if round_1.get("average_score") is not None else np.nan

                row = {
                    "source_file": jf.name,
                    "gen_model": gen_model,
                    "gen_model_short": gen_model_short,
                    "target_type": target_type,
                    "recipe_name": q.get("recipe_name", ""),
                    "iou_max": max_iou * 100.0,
                    "iou_mean": mean_iou * 100.0,
                    "human_score": human_score,
                    "r1_avg_judge": r1_avg,
                }

                judges_dict = round_1.get("judges", {})
                for jname, jinfo in judges_dict.items():
                    j_short = jname.split("/")[-1]
                    if jinfo.get("score") is not None:
                        row[f"judge_{j_short}"] = float(jinfo["score"])

                rows.append(row)

    df = pd.DataFrame(rows)
    return df


def plot_overall_task_comparison(df: pd.DataFrame, output_dir: Path, dpi: int = 300) -> Path:
    """Plot 1: Overall Average Scores (IoU, Human, Consensus Judges, Individual Judges) divided by task (Ingredients vs Directions)."""
    fig, ax = plt.subplots(figsize=(13, 7.5))

    tasks = ["ingredients", "directions"]
    task_labels = ["Ingredients", "Directions"]

    judge_cols = [c for c in df.columns if c.startswith("judge_")]
    # Clean order of metrics to plot
    metric_keys = ["iou_max", "human_score", "r1_avg_judge"] + judge_cols
    metric_labels = ["Max IoU (%)", "Human Score", "Round 1 Avg Judges"] + [
        JUDGE_LABELS.get(c.replace("judge_", ""), c.replace("judge_", "")) for c in judge_cols
    ]
    colors = [COLOR_IOU, COLOR_HUMAN, COLOR_AVG_JUDGE] + [
        JUDGE_COLORS.get(c.replace("judge_", ""), "#7f8c8d") for c in judge_cols
    ]

    n_tasks = len(tasks)
    n_metrics = len(metric_keys)
    x = np.arange(n_tasks)
    width = 0.8 / n_metrics

    for i, (mkey, mlabel, col) in enumerate(zip(metric_keys, metric_labels, colors)):
        means = []
        sems = []
        for t in tasks:
            sub = df[df["target_type"] == t][mkey].dropna()
            means.append(sub.mean() if len(sub) > 0 else 0)
            sems.append(sub.sem() if len(sub) > 1 else 0)

        offset = (i - (n_metrics - 1) / 2) * width
        rects = ax.bar(
            x + offset,
            means,
            width,
            yerr=sems,
            capsize=4,
            label=mlabel,
            color=col,
            edgecolor="white",
            linewidth=1.2,
            alpha=0.92,
        )

        for rect, mean, sem in zip(rects, means, sems):
            if mean > 0:
                y_pos = mean + (sem if not np.isnan(sem) else 0) + 1.5
                ax.annotate(
                    f"{mean:.1f}",
                    xy=(rect.get_x() + rect.get_width() / 2, y_pos),
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#2c3e50",
                )

    ax.set_title(
        "Evaluation Benchmark: IoU Score, Human Score, and LLM Judges\nDivided by Task (Ingredients vs. Directions)",
        pad=18,
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=12, fontweight="bold")
    ax.set_ylabel("Score (0 – 100 Scale / % Overlap)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.axhline(100, color="gray", linestyle="--", alpha=0.3, linewidth=1)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        frameon=True,
        facecolor="white",
        edgecolor="#bdc3c7",
        fontsize=9.5,
    )

    plt.tight_layout()
    out_path = output_dir / "iou_human_judges_by_task.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_per_judge_model_breakdown(df: pd.DataFrame, output_dir: Path, dpi: int = 300) -> Path:
    """Plot 2: Dedicated subplots for EACH judge model (2x2 grid), comparing Judge vs Human vs IoU on the evaluated instances."""
    judge_cols = [c for c in df.columns if c.startswith("judge_")]
    n_judges = len(judge_cols)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharey=True)
    axes = axes.flatten()

    tasks = ["ingredients", "directions"]
    task_labels = ["Ingredients", "Directions"]

    for idx, jcol in enumerate(judge_cols):
        ax = axes[idx]
        j_key = jcol.replace("judge_", "")
        j_label = JUDGE_LABELS.get(j_key, j_key)
        j_color = JUDGE_COLORS.get(j_key, "#2980b9")

        # Filter rows where this judge evaluated
        sub_df = df.dropna(subset=[jcol]).copy()

        x = np.arange(len(tasks))
        width = 0.24

        # Metrics for this judge's subset: Judge Score, Human Score, IoU Max
        judge_means, judge_sems = [], []
        human_means, human_sems = [], []
        iou_means, iou_sems = [], []

        for t in tasks:
            t_data = sub_df[sub_df["target_type"] == t]
            judge_means.append(t_data[jcol].mean())
            judge_sems.append(t_data[jcol].sem())
            human_means.append(t_data["human_score"].mean())
            human_sems.append(t_data["human_score"].sem())
            iou_means.append(t_data["iou_max"].mean())
            iou_sems.append(t_data["iou_max"].sem())

        r_judge = ax.bar(
            x - width,
            judge_means,
            width,
            yerr=judge_sems,
            capsize=4,
            label=f"Judge ({j_label})",
            color=j_color,
            alpha=0.9,
            edgecolor="white",
        )
        r_human = ax.bar(
            x,
            human_means,
            width,
            yerr=human_sems,
            capsize=4,
            label="Human Ground Truth",
            color=COLOR_HUMAN,
            alpha=0.9,
            edgecolor="white",
        )
        r_iou = ax.bar(
            x + width,
            iou_means,
            width,
            yerr=iou_sems,
            capsize=4,
            label="IoU Score (%)",
            color=COLOR_IOU,
            alpha=0.9,
            edgecolor="white",
        )

        # Annotate numbers cleanly above whiskers
        for rects, sems_list in [(r_judge, judge_sems), (r_human, human_sems), (r_iou, iou_sems)]:
            for r, sem in zip(rects, sems_list):
                h = r.get_height()
                if h > 0:
                    y_pos = h + (sem if not np.isnan(sem) else 0) + 1.8
                    ax.annotate(
                        f"{h:.1f}",
                        xy=(r.get_x() + r.get_width() / 2, y_pos),
                        ha="center",
                        va="bottom",
                        fontsize=8.5,
                        fontweight="bold",
                    )

        # Delta callouts (Judge - Human)
        for t_i, (jm, hm) in enumerate(zip(judge_means, human_means)):
            delta = jm - hm
            sign = "+" if delta >= 0 else ""
            d_color = "#c0392b" if delta < -5 else ("#e67e22" if delta > 5 else "#27ae60")
            ax.text(
                t_i,
                105,
                f"Δ(Judge - Human) = {sign}{delta:.1f}",
                ha="center",
                va="center",
                fontsize=9.5,
                fontweight="bold",
                color=d_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9f9", edgecolor=d_color, alpha=0.95),
            )

        n_evals = len(sub_df)
        ax.set_title(f"Judge: {j_label} (Evaluated n={n_evals})", fontsize=12, pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(task_labels, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 118)
        ax.axhline(100, color="gray", linestyle="--", alpha=0.3)
        ax.legend(loc="lower left", fontsize=8.5, frameon=True)

    fig.suptitle(
        "Individual Judge Model Analysis vs. Human Ground Truth & IoU Score\n(Divided by Ingredients and Directions)",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    out_path = output_dir / "judge_vs_human_iou_by_judge_model.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_generator_models_breakdown(df: pd.DataFrame, output_dir: Path, dpi: int = 300) -> Path:
    """Plot 3: Compares the 5 Generator Models on IoU, Human Score, and Consensus Judges for Ingredients & Directions."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    gen_models = sorted(df["gen_model_short"].unique())
    model_labels = {
        "Llama-3.2-3B-Instruct-GGUF": "Llama 3.2 3B",
        "gemma3:1b": "Gemma 3 1B",
        "granite4.1:3b": "Granite 4.1 3B",
        "ministral-3:3b": "Ministral 3 3B",
        "qwen3.5:4b": "Qwen 3.5 4B",
    }

    x = np.arange(len(gen_models))
    width = 0.26

    tasks = [("ingredients", "Ingredients"), ("directions", "Directions")]

    for ax_idx, (t_key, t_title) in enumerate(tasks):
        ax = axes[ax_idx]
        t_data = df[df["target_type"] == t_key]

        iou_vals, human_vals, judge_vals = [], [], []
        iou_err, human_err, judge_err = [], [], []

        for gm in gen_models:
            sub = t_data[t_data["gen_model_short"] == gm]
            iou_vals.append(sub["iou_max"].mean())
            iou_err.append(sub["iou_max"].sem())
            human_vals.append(sub["human_score"].mean())
            human_err.append(sub["human_score"].sem())
            judge_vals.append(sub["r1_avg_judge"].mean())
            judge_err.append(sub["r1_avg_judge"].sem())

        r1 = ax.bar(
            x - width,
            iou_vals,
            width,
            yerr=iou_err,
            capsize=3,
            label="IoU Score (%)",
            color=COLOR_IOU,
            alpha=0.9,
            edgecolor="white",
        )
        r2 = ax.bar(
            x,
            human_vals,
            width,
            yerr=human_err,
            capsize=3,
            label="Human Score",
            color=COLOR_HUMAN,
            alpha=0.9,
            edgecolor="white",
        )
        r3 = ax.bar(
            x + width,
            judge_vals,
            width,
            yerr=judge_err,
            capsize=3,
            label="Round 1 Avg Judges",
            color=COLOR_AVG_JUDGE,
            alpha=0.9,
            edgecolor="white",
        )

        for rects, errs in [(r1, iou_err), (r2, human_err), (r3, judge_err)]:
            for r, err in zip(rects, errs):
                h = r.get_height()
                if h > 0:
                    y_pos = h + (err if not np.isnan(err) else 0) + 1.8
                    ax.annotate(
                        f"{h:.1f}",
                        xy=(r.get_x() + r.get_width() / 2, y_pos),
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                        fontweight="bold",
                    )

        ax.set_title(f"Task: {t_title}", fontsize=13, pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels([model_labels.get(m, m) for m in gen_models], rotation=15, ha="right", fontsize=10.5)
        ax.set_ylabel("Score (0 – 100)" if ax_idx == 0 else "", fontsize=11)
        ax.set_ylim(0, 118)
        ax.axhline(100, color="gray", linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=9.5, frameon=True)

    fig.suptitle(
        "Performance Across Generator Models: IoU vs. Human vs. Consensus Judges",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    out_path = output_dir / "generator_models_iou_human_judges.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_correlation_heatmaps(df: pd.DataFrame, output_dir: Path, dpi: int = 300) -> Path:
    """Plot 4: Correlation heatmaps showing alignment between IoU, Human Score, and LLM Judges for each task."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    tasks = [("ingredients", "Ingredients"), ("directions", "Directions")]
    judge_cols = [c for c in df.columns if c.startswith("judge_")]

    cols_to_corr = ["human_score", "iou_max", "r1_avg_judge"] + judge_cols
    col_labels = ["Human", "IoU Max", "Avg Judges"] + [
        JUDGE_LABELS.get(c.replace("judge_", ""), c.replace("judge_", "")) for c in judge_cols
    ]

    for ax_idx, (t_key, t_title) in enumerate(tasks):
        ax = axes[ax_idx]
        sub = df[df["target_type"] == t_key][cols_to_corr].copy()
        sub.columns = col_labels

        corr = sub.corr(method="pearson")

        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            vmin=-0.5,
            vmax=1.0,
            center=0.0,
            linewidths=0.8,
            cbar=ax_idx == 1,
            cbar_kws={"label": "Pearson Correlation (r)"},
            ax=ax,
        )
        ax.set_title(f"Metric Alignment Matrix: {t_title}", fontsize=13, pad=12)

    fig.suptitle(
        "Correlation & Alignment: Human Ground Truth vs. IoU vs. LLM Judges",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    out_path = output_dir / "iou_judges_human_correlation_matrix.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready plots for IoU, Human and LLM Judge scores.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for plots")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figures")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    print(f"Loading evaluation results from {len(DEFAULT_DIRS)} source directories...")
    df = load_and_extract_data(DEFAULT_DIRS)
    print(f"Loaded {len(df)} total evaluated recipe instances across {df['gen_model'].nunique()} models.")

    print("\n--- Generating Publication Plots ---")
    p1 = plot_overall_task_comparison(df, args.output_dir, args.dpi)
    p2 = plot_per_judge_model_breakdown(df, args.output_dir, args.dpi)
    p3 = plot_generator_models_breakdown(df, args.output_dir, args.dpi)
    p4 = plot_correlation_heatmaps(df, args.output_dir, args.dpi)

    print("\nAll plots successfully generated in:", args.output_dir)


if __name__ == "__main__":
    main()
