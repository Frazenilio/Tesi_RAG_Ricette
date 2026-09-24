import argparse
import json
import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "FinalJson_RERUN"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "Plots" / "SavedPlots" / "ReEvaluation" / "Human_Comparison"

# Generator model order and display mapping
CANONICAL_MODEL_ORDER = [
    "Gemma 3 1B",
    "Granite 4.1 3B",
    "Llama 3.2 3B",
    "Ministral 3 3B",
    "Qwen 3.5 4B",
]

GENERATOR_DISPLAY_NAMES = {
    "gemma3_1b": "Gemma 3 1B",
    "gemma3:1b": "Gemma 3 1B",
    "granite4.1_3b": "Granite 4.1 3B",
    "granite4.1:3b": "Granite 4.1 3B",
    "hf.co_bartowski_Llama-3.2-3B-Instruct-GGUF": "Llama 3.2 3B",
    "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF": "Llama 3.2 3B",
    "ministral-3_3b": "Ministral 3 3B",
    "ministral-3:3b": "Ministral 3 3B",
    "qwen3.5_4b": "Qwen 3.5 4B",
    "qwen3.5:4b": "Qwen 3.5 4B",
}

MODEL_COLORS = {
    "Llama 3.2 3B": "#2980b9",
    "Gemma 3 1B": "#d35400",
    "Granite 4.1 3B": "#7f8c8d",
    "Ministral 3 3B": "#c0392b",
    "Qwen 3.5 4B": "#16a085",
}

# Evaluator colors
COLOR_R1 = "#d35400"      # Rust / Burnt Orange (Judge Round 1)
COLOR_R2 = "#f39c12"      # Amber / Golden Orange (Judge Round 2)
COLOR_HUMAN = "#27ae60"   # Emerald Green (Human Ground Truth)

TASKS = [("ingredients", "Ingredients"), ("directions", "Directions")]


def setup_style():
    """Sets consistent publication-ready plot style matching project standards."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
    })


def load_reevaluation_human_data(results_dir: Path = DEFAULT_RESULTS_DIR) -> pd.DataFrame:
    """Loads all JSON files from the results directory and constructs a comprehensive DataFrame."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    json_files = sorted(results_dir.glob("*/*.json"))
    if not json_files:
        raise ValueError(f"No JSON result files found in {results_dir}")

    rows = []
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {jf}: {e}")
            continue

        raw_model = data.get("model", jf.parent.name)
        model_name = GENERATOR_DISPLAY_NAMES.get(raw_model, GENERATOR_DISPLAY_NAMES.get(jf.parent.name, raw_model))
        target_type = data.get("target_type", "").lower()
        qa_pairs = data.get("qa_pairs", [])

        for qa in qa_pairs:
            recipe = qa.get("recipe_name", "Unknown")
            human_score = float(qa.get("human_score")) if qa.get("human_score") is not None else np.nan

            r1_raw = qa.get("round_1", {}).get("average_score")
            r2_raw = qa.get("round_2", {}).get("average_score")
            judge_r1 = float(r1_raw) if r1_raw is not None else np.nan
            judge_r2 = float(r2_raw) if r2_raw is not None else np.nan

            mae_r1 = abs(judge_r1 - human_score) if not (np.isnan(judge_r1) or np.isnan(human_score)) else np.nan
            mae_r2 = abs(judge_r2 - human_score) if not (np.isnan(judge_r2) or np.isnan(human_score)) else np.nan

            rows.append({
                "folder": jf.parent.name,
                "model_raw": raw_model,
                "model": model_name,
                "task": target_type,
                "recipe": recipe,
                "human_score": human_score,
                "judge_r1": judge_r1,
                "judge_r2": judge_r2,
                "delta_judge": judge_r2 - judge_r1 if not (np.isnan(judge_r1) or np.isnan(judge_r2)) else np.nan,
                "mae_r1": mae_r1,
                "mae_r2": mae_r2,
                "delta_mae": mae_r2 - mae_r1 if not (np.isnan(mae_r1) or np.isnan(mae_r2)) else np.nan,
            })

    return pd.DataFrame(rows)


def _annotate_bars(ax, bars, sems, offset: float = 1.5, fontsize: float = 9.5):
    """Adds cleanly formatted numeric values above bars with optional SEM."""
    for bar, sem in zip(bars, sems):
        h = bar.get_height()
        if not np.isnan(h) and h > 0:
            err = sem if sem is not None and not np.isnan(sem) else 0.0
            y_pos = h + err + offset
            ax.annotate(
                f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                ha="center",
                va="bottom",
                fontsize=fontsize,
                fontweight="bold",
                color="#2c3e50",
            )


def plot_correctness_split(df: pd.DataFrame, output_dir: Path, dpi: int = 300) -> list[Path]:
    """Generates bar charts comparing Mean Scores: Round 1 vs Round 2 vs Human Evaluation."""
    saved_paths = []
    models = [m for m in CANONICAL_MODEL_ORDER if m in df["model"].unique()]
    x = np.arange(len(models))
    bar_w = 0.26

    # 1. Dedicated plot for each task
    for t_key, t_title in TASKS:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        t_data = df[df["task"] == t_key]

        r1_means = [t_data[t_data["model"] == m]["judge_r1"].mean() for m in models]
        r1_sems = [t_data[t_data["model"] == m]["judge_r1"].sem() for m in models]

        r2_means = [t_data[t_data["model"] == m]["judge_r2"].mean() for m in models]
        r2_sems = [t_data[t_data["model"] == m]["judge_r2"].sem() for m in models]

        human_means = [t_data[t_data["model"] == m]["human_score"].mean() for m in models]
        human_sems = [t_data[t_data["model"] == m]["human_score"].sem() for m in models]

        b_r1 = ax.bar(
            x - bar_w,
            r1_means,
            bar_w,
            yerr=r1_sems,
            capsize=4,
            label="Judge Score (Round 1)",
            color=COLOR_R1,
            alpha=0.92,
            edgecolor="white",
            linewidth=1.2,
        )
        b_r2 = ax.bar(
            x,
            r2_means,
            bar_w,
            yerr=r2_sems,
            capsize=4,
            label="Judge Score (Round 2)",
            color=COLOR_R2,
            alpha=0.92,
            edgecolor="white",
            linewidth=1.2,
        )
        b_human = ax.bar(
            x + bar_w,
            human_means,
            bar_w,
            yerr=human_sems,
            capsize=4,
            label="Human Ground Truth",
            color=COLOR_HUMAN,
            alpha=0.92,
            edgecolor="white",
            linewidth=1.2,
        )

        _annotate_bars(ax, b_r1, r1_sems, offset=1.5, fontsize=9.5)
        _annotate_bars(ax, b_r2, r2_sems, offset=1.5, fontsize=9.5)
        _annotate_bars(ax, b_human, human_sems, offset=1.5, fontsize=9.5)

        ax.set_title(
            f"RAG Re-Evaluation Correctness: {t_title}\nRound 1 vs. Round 2 vs. Human Ground Truth Across Models",
            pad=14,
        )
        ax.set_ylabel("Average Score (0–100 Scale)", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 118)
        ax.axhline(100, color="gray", linestyle="--", alpha=0.35, linewidth=1)
        ax.legend(loc="upper right", frameon=True, fontsize=11)

        plt.tight_layout()
        out_path = output_dir / f"reevaluation_human_correctness_{t_key}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {out_path.name}")
        saved_paths.append(out_path)

    # 2. Combined 1x2 panel overview
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), sharey=True)
    for ax_idx, (t_key, t_title) in enumerate(TASKS):
        ax = axes[ax_idx]
        t_data = df[df["task"] == t_key]

        r1_means = [t_data[t_data["model"] == m]["judge_r1"].mean() for m in models]
        r1_sems = [t_data[t_data["model"] == m]["judge_r1"].sem() for m in models]
        r2_means = [t_data[t_data["model"] == m]["judge_r2"].mean() for m in models]
        r2_sems = [t_data[t_data["model"] == m]["judge_r2"].sem() for m in models]
        human_means = [t_data[t_data["model"] == m]["human_score"].mean() for m in models]
        human_sems = [t_data[t_data["model"] == m]["human_score"].sem() for m in models]

        b_r1 = ax.bar(x - bar_w, r1_means, bar_w, yerr=r1_sems, capsize=3.5, label="Judge Score (Round 1)", color=COLOR_R1, alpha=0.92, edgecolor="white", linewidth=1.2)
        b_r2 = ax.bar(x, r2_means, bar_w, yerr=r2_sems, capsize=3.5, label="Judge Score (Round 2)", color=COLOR_R2, alpha=0.92, edgecolor="white", linewidth=1.2)
        b_human = ax.bar(x + bar_w, human_means, bar_w, yerr=human_sems, capsize=3.5, label="Human Ground Truth", color=COLOR_HUMAN, alpha=0.92, edgecolor="white", linewidth=1.2)

        _annotate_bars(ax, b_r1, r1_sems, offset=1.5, fontsize=8.5)
        _annotate_bars(ax, b_r2, r2_sems, offset=1.5, fontsize=8.5)
        _annotate_bars(ax, b_human, human_sems, offset=1.5, fontsize=8.5)

        ax.set_title(f"Task: {t_title}", fontsize=13, pad=12)
        if ax_idx == 0:
            ax.set_ylabel("Average Score (0–100 Scale)", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10.5, fontweight="bold")
        ax.set_ylim(0, 118)
        ax.axhline(100, color="gray", linestyle="--", alpha=0.35, linewidth=1)
        ax.legend(loc="upper right", frameon=True, fontsize=10)

    fig.suptitle(
        "RAG Re-Evaluation Correctness: Round 1 vs. Round 2 vs. Human Ground Truth\nDivided by Model and Task (Ingredients vs. Directions)",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    combined_path = output_dir / "reevaluation_human_correctness_by_model_and_task.png"
    fig.savefig(combined_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {combined_path.name}")
    saved_paths.append(combined_path)

    # 3. Macro Task-level aggregated overview (Ingredients vs Directions overall)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    x_task = np.arange(len(TASKS))
    task_labels = [t[1] for t in TASKS]

    t_r1_means = [df[df["task"] == t[0]]["judge_r1"].mean() for t in TASKS]
    t_r1_sems = [df[df["task"] == t[0]]["judge_r1"].sem() for t in TASKS]
    t_r2_means = [df[df["task"] == t[0]]["judge_r2"].mean() for t in TASKS]
    t_r2_sems = [df[df["task"] == t[0]]["judge_r2"].sem() for t in TASKS]
    t_human_means = [df[df["task"] == t[0]]["human_score"].mean() for t in TASKS]
    t_human_sems = [df[df["task"] == t[0]]["human_score"].sem() for t in TASKS]

    b_r1 = ax.bar(x_task - bar_w, t_r1_means, bar_w, yerr=t_r1_sems, capsize=5, label="Judge Score (Round 1)", color=COLOR_R1, alpha=0.92, edgecolor="white", linewidth=1.2)
    b_r2 = ax.bar(x_task, t_r2_means, bar_w, yerr=t_r2_sems, capsize=5, label="Judge Score (Round 2)", color=COLOR_R2, alpha=0.92, edgecolor="white", linewidth=1.2)
    b_human = ax.bar(x_task + bar_w, t_human_means, bar_w, yerr=t_human_sems, capsize=5, label="Human Ground Truth", color=COLOR_HUMAN, alpha=0.92, edgecolor="white", linewidth=1.2)

    _annotate_bars(ax, b_r1, t_r1_sems, offset=1.5, fontsize=10.5)
    _annotate_bars(ax, b_r2, t_r2_sems, offset=1.5, fontsize=10.5)
    _annotate_bars(ax, b_human, t_human_sems, offset=1.5, fontsize=10.5)

    ax.set_title("Overall Re-Evaluation Performance: Task Summary\nRound 1 vs. Round 2 vs. Human Ground Truth", pad=14)
    ax.set_ylabel("Average Score (0–100 Scale)", fontsize=12, fontweight="bold")
    ax.set_xticks(x_task)
    ax.set_xticklabels(task_labels, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.axhline(100, color="gray", linestyle="--", alpha=0.35, linewidth=1)
    ax.legend(loc="upper right", frameon=True, fontsize=11)

    plt.tight_layout()
    overview_path = output_dir / "reevaluation_human_correctness_task_overview.png"
    fig.savefig(overview_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {overview_path.name}")
    saved_paths.append(overview_path)

    return saved_paths


def plot_distributions_split(df: pd.DataFrame, output_dir: Path, dpi: int = 300) -> list[Path]:
    """Generates boxplots showing the score distribution and dispersion: Round 1 vs Round 2 vs Human."""
    saved_paths = []
    models = [m for m in CANONICAL_MODEL_ORDER if m in df["model"].unique()]

    # Prepare melted dataframe for seaborn
    melt_list = []
    for _, row in df.iterrows():
        melt_list.append({"model": row["model"], "task": row["task"], "Score": row["judge_r1"], "Evaluator": "Judge (Round 1)"})
        melt_list.append({"model": row["model"], "task": row["task"], "Score": row["judge_r2"], "Evaluator": "Judge (Round 2)"})
        melt_list.append({"model": row["model"], "task": row["task"], "Score": row["human_score"], "Evaluator": "Human Ground Truth"})

    plot_df = pd.DataFrame(melt_list)
    palette = {
        "Judge (Round 1)": COLOR_R1,
        "Judge (Round 2)": COLOR_R2,
        "Human Ground Truth": COLOR_HUMAN,
    }

    # 1. Dedicated boxplots per task
    for t_key, t_title in TASKS:
        fig, ax = plt.subplots(figsize=(11.5, 6.8))
        sub = plot_df[plot_df["task"] == t_key]

        sns.boxplot(
            data=sub,
            x="model",
            y="Score",
            hue="Evaluator",
            order=models,
            palette=palette,
            ax=ax,
            boxprops=dict(alpha=0.88),
            medianprops=dict(color="#1a252f", linewidth=2.2),
            showmeans=True,
            meanprops=dict(marker="o", markeredgecolor="black", markerfacecolor="white", markersize=6.5),
        )

        ax.set_title(
            f"Score Distribution & Dispersion: {t_title}\nRound 1 vs. Round 2 vs. Human (Circle = Mean, Dark Line = Median)",
            pad=14,
        )
        ax.set_ylabel("Score (0–100 Scale)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Generator LLM", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=11, fontweight="bold")
        ax.set_ylim(-5, 110)
        ax.axhline(100, color="gray", linestyle="--", alpha=0.3)
        ax.legend(loc="lower left", frameon=True, fontsize=11)

        plt.tight_layout()
        out_path = output_dir / f"reevaluation_human_distributions_{t_key}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {out_path.name}")
        saved_paths.append(out_path)

    # 2. Combined 2x1 stacked subplot overview
    fig, axes = plt.subplots(2, 1, figsize=(15, 11), sharex=True)
    for ax_idx, (t_key, t_title) in enumerate(TASKS):
        ax = axes[ax_idx]
        sub = plot_df[plot_df["task"] == t_key]

        sns.boxplot(
            data=sub,
            x="model",
            y="Score",
            hue="Evaluator",
            order=models,
            palette=palette,
            ax=ax,
            boxprops=dict(alpha=0.88),
            medianprops=dict(color="#1a252f", linewidth=2.0),
            showmeans=True,
            meanprops=dict(marker="o", markeredgecolor="black", markerfacecolor="white", markersize=6),
        )

        ax.set_title(f"Task: {t_title} (Circle = Mean, Dark Line = Median)", fontsize=13, pad=10)
        ax.set_ylabel("Score (0–100)", fontsize=11, fontweight="bold")
        ax.set_xlabel("" if ax_idx == 0 else "Generator LLM", fontsize=11, fontweight="bold")
        ax.set_ylim(-5, 110)
        ax.axhline(100, color="gray", linestyle="--", alpha=0.3)
        ax.legend(loc="lower left", frameon=True, fontsize=10)

    fig.suptitle(
        "Score Distributions Across Models: Round 1 vs. Round 2 vs. Human Ground Truth\n(Divided by Ingredients and Directions)",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    combined_path = output_dir / "reevaluation_human_distributions_by_model.png"
    fig.savefig(combined_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {combined_path.name}")
    saved_paths.append(combined_path)

    return saved_paths


def plot_human_correlation_split(df: pd.DataFrame, output_dir: Path, dpi: int = 300) -> list[Path]:
    """Generates scatter plots comparing Round 1 vs Human and Round 2 vs Human agreement."""
    saved_paths = []
    models = [m for m in CANONICAL_MODEL_ORDER if m in df["model"].unique()]

    # 1. Dedicated 1x2 panel for each task
    for t_key, t_title in TASKS:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
        t_data = df[df["task"] == t_key]

        for idx, (round_col, round_title, color_line) in enumerate([
            ("judge_r1", "Round 1 Judge Score", COLOR_R1),
            ("judge_r2", "Round 2 Judge Score", COLOR_R2),
        ]):
            ax = axes[idx]
            for m in models:
                sub = t_data[t_data["model"] == m]
                ax.scatter(
                    sub[round_col],
                    sub["human_score"],
                    label=m,
                    color=MODEL_COLORS.get(m, "#333333"),
                    alpha=0.68,
                    s=55,
                    edgecolors="none",
                )

            valid = t_data.dropna(subset=[round_col, "human_score"])
            if len(valid) > 1:
                corr = valid[round_col].corr(valid["human_score"])
                z = np.polyfit(valid[round_col], valid["human_score"], 1)
                p = np.poly1d(z)
                x_vals = np.linspace(valid[round_col].min(), valid[round_col].max(), 100)
                ax.plot(
                    x_vals,
                    np.clip(p(x_vals), 0, 100),
                    color="#2c3e50",
                    linestyle="--",
                    linewidth=2.5,
                    label=f"Trendline (r = {corr:.2f})",
                )

            ax.set_title(f"{round_title} vs. Human Ground Truth", fontsize=13, pad=12)
            ax.set_xlabel(f"{round_title} (0–100)", fontsize=11, fontweight="bold")
            if idx == 0:
                ax.set_ylabel("Human Ground Truth Score (0–100)", fontsize=11, fontweight="bold")
            ax.set_xlim(-5, 105)
            ax.set_ylim(-5, 108)
            ax.axline((0, 0), slope=1, color="gray", linestyle=":", alpha=0.45, label="Perfect Agreement (y=x)")
            ax.legend(loc="upper left", frameon=True, fontsize=9.5)

        fig.suptitle(f"Alignment with Human Ground Truth: {t_title}\nComparing Judge Round 1 vs. Judge Round 2", fontsize=15, fontweight="bold", y=0.99)
        plt.tight_layout()
        out_path = output_dir / f"reevaluation_human_correlation_{t_key}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {out_path.name}")
        saved_paths.append(out_path)

    # 2. 2x2 grid overview (Tasks x Rounds)
    fig, axes = plt.subplots(2, 2, figsize=(16, 13), sharey=True)
    for row_idx, (t_key, t_title) in enumerate(TASKS):
        t_data = df[df["task"] == t_key]
        for col_idx, (round_col, round_title) in enumerate([
            ("judge_r1", "Round 1 Judge Score"),
            ("judge_r2", "Round 2 Judge Score"),
        ]):
            ax = axes[row_idx, col_idx]
            for m in models:
                sub = t_data[t_data["model"] == m]
                ax.scatter(sub[round_col], sub["human_score"], label=m, color=MODEL_COLORS.get(m, "#333333"), alpha=0.65, s=48, edgecolors="none")

            valid = t_data.dropna(subset=[round_col, "human_score"])
            if len(valid) > 1:
                corr = valid[round_col].corr(valid["human_score"])
                z = np.polyfit(valid[round_col], valid["human_score"], 1)
                p = np.poly1d(z)
                x_vals = np.linspace(valid[round_col].min(), valid[round_col].max(), 100)
                ax.plot(x_vals, np.clip(p(x_vals), 0, 100), color="#2c3e50", linestyle="--", linewidth=2.2, label=f"Trendline (r = {corr:.2f})")

            ax.set_title(f"{t_title} — {round_title} vs. Human", fontsize=12, pad=10)
            ax.set_xlabel(f"{round_title} (0–100)", fontsize=10.5, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel("Human Score (0–100)", fontsize=10.5, fontweight="bold")
            ax.set_xlim(-5, 105)
            ax.set_ylim(-5, 108)
            ax.axline((0, 0), slope=1, color="gray", linestyle=":", alpha=0.4, label="Ideal (y=x)")
            ax.legend(loc="upper left", frameon=True, fontsize=9)

    fig.suptitle("Correlation with Human Ground Truth: Round 1 vs. Round 2 Across Tasks", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    overview_path = output_dir / "reevaluation_human_correlation_overview.png"
    fig.savefig(overview_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {overview_path.name}")
    saved_paths.append(overview_path)

    return saved_paths


def plot_mae_gap(df: pd.DataFrame, output_dir: Path, dpi: int = 300) -> list[Path]:
    """Generates bar plots showing Mean Absolute Error from Human Ground Truth: |Judge - Human|."""
    saved_paths = []
    models = [m for m in CANONICAL_MODEL_ORDER if m in df["model"].unique()]
    x = np.arange(len(models))
    bar_w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)

    for ax_idx, (t_key, t_title) in enumerate(TASKS):
        ax = axes[ax_idx]
        t_data = df[df["task"] == t_key]

        mae_r1_means = [t_data[t_data["model"] == m]["mae_r1"].mean() for m in models]
        mae_r1_sems = [t_data[t_data["model"] == m]["mae_r1"].sem() for m in models]

        mae_r2_means = [t_data[t_data["model"] == m]["mae_r2"].mean() for m in models]
        mae_r2_sems = [t_data[t_data["model"] == m]["mae_r2"].sem() for m in models]

        b1 = ax.bar(x - bar_w / 2, mae_r1_means, bar_w, yerr=mae_r1_sems, capsize=4, label="Round 1 MAE from Human", color=COLOR_R1, alpha=0.92, edgecolor="white", linewidth=1.2)
        b2 = ax.bar(x + bar_w / 2, mae_r2_means, bar_w, yerr=mae_r2_sems, capsize=4, label="Round 2 MAE from Human", color=COLOR_R2, alpha=0.92, edgecolor="white", linewidth=1.2)

        _annotate_bars(ax, b1, mae_r1_sems, offset=1.0, fontsize=9.5)
        _annotate_bars(ax, b2, mae_r2_sems, offset=1.0, fontsize=9.5)

        ax.set_title(f"Task: {t_title}", fontsize=13, pad=12)
        if ax_idx == 0:
            ax.set_ylabel("Mean Absolute Error (Points Gap from Human)", fontsize=11.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10.5, fontweight="bold")
        ax.set_ylim(0, max(max(mae_r1_means), max(mae_r2_means)) * 1.35)
        ax.legend(loc="upper right", frameon=True, fontsize=10.5)

    fig.suptitle("Discrepancy with Human Ground Truth: Mean Absolute Error (MAE)\nLower Score = Closer Alignment to Human Grading", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path = output_dir / "reevaluation_human_mae_gap.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path.name}")
    saved_paths.append(out_path)

    return saved_paths


def plot_high_human_degradation(
    df: pd.DataFrame,
    output_dir: Path,
    human_threshold: float = 95.0,
    dpi: int = 300,
) -> list[Path]:
    """Generates plots specifically analyzing high-quality answers (Human Score >= threshold).
    Compares Round 1 vs. Round 2 judge scores to evaluate whether answers degraded.
    """
    saved_paths = []
    models = [m for m in CANONICAL_MODEL_ORDER if m in df["model"].unique()]
    high_df = df[df["human_score"] >= human_threshold].copy()
    bar_w = 0.35
    x = np.arange(len(models))

    # 1. Dedicated standalone plots per task
    for t_key, t_title in TASKS:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        t_data = high_df[high_df["task"] == t_key]

        counts = [len(t_data[t_data["model"] == m]) for m in models]
        r1_means = [t_data[t_data["model"] == m]["judge_r1"].mean() if len(t_data[t_data["model"] == m]) > 0 else 0.0 for m in models]
        r1_sems = [t_data[t_data["model"] == m]["judge_r1"].sem() if len(t_data[t_data["model"] == m]) > 1 else 0.0 for m in models]
        r2_means = [t_data[t_data["model"] == m]["judge_r2"].mean() if len(t_data[t_data["model"] == m]) > 0 else 0.0 for m in models]
        r2_sems = [t_data[t_data["model"] == m]["judge_r2"].sem() if len(t_data[t_data["model"] == m]) > 1 else 0.0 for m in models]

        b1 = ax.bar(x - bar_w / 2, r1_means, bar_w, yerr=r1_sems, capsize=4, label="Judge Score (Round 1)", color=COLOR_R1, alpha=0.92, edgecolor="white", linewidth=1.2)
        b2 = ax.bar(x + bar_w / 2, r2_means, bar_w, yerr=r2_sems, capsize=4, label="Judge Score (Round 2)", color=COLOR_R2, alpha=0.92, edgecolor="white", linewidth=1.2)

        _annotate_bars(ax, b1, r1_sems, offset=1.5, fontsize=9.5)
        _annotate_bars(ax, b2, r2_sems, offset=1.5, fontsize=9.5)

        x_labels = [f"{m}\n(N={c})" for m, c in zip(models, counts)]
        ax.set_title(
            f"High-Quality Answer Degradation: {t_title}\nAverage Judge Scores for Answers with Human Score ≥ {int(human_threshold)}",
            pad=14,
        )
        ax.set_ylabel("Average Judge Score (0–100 Scale)", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=10.5, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.axhline(100, color="gray", linestyle="--", alpha=0.35, linewidth=1)
        ax.legend(loc="upper right", frameon=True, fontsize=11)

        plt.tight_layout()
        out_path = output_dir / f"reevaluation_high_human_degradation_{t_key}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {out_path.name}")
        saved_paths.append(out_path)

    # 2. Combined 1x2 panel overview
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), sharey=True)
    for ax_idx, (t_key, t_title) in enumerate(TASKS):
        ax = axes[ax_idx]
        t_data = high_df[high_df["task"] == t_key]

        counts = [len(t_data[t_data["model"] == m]) for m in models]
        r1_means = [t_data[t_data["model"] == m]["judge_r1"].mean() if len(t_data[t_data["model"] == m]) > 0 else 0.0 for m in models]
        r1_sems = [t_data[t_data["model"] == m]["judge_r1"].sem() if len(t_data[t_data["model"] == m]) > 1 else 0.0 for m in models]
        r2_means = [t_data[t_data["model"] == m]["judge_r2"].mean() if len(t_data[t_data["model"] == m]) > 0 else 0.0 for m in models]
        r2_sems = [t_data[t_data["model"] == m]["judge_r2"].sem() if len(t_data[t_data["model"] == m]) > 1 else 0.0 for m in models]

        b1 = ax.bar(x - bar_w / 2, r1_means, bar_w, yerr=r1_sems, capsize=3.5, label="Judge Score (Round 1)", color=COLOR_R1, alpha=0.92, edgecolor="white", linewidth=1.2)
        b2 = ax.bar(x + bar_w / 2, r2_means, bar_w, yerr=r2_sems, capsize=3.5, label="Judge Score (Round 2)", color=COLOR_R2, alpha=0.92, edgecolor="white", linewidth=1.2)

        _annotate_bars(ax, b1, r1_sems, offset=1.5, fontsize=8.5)
        _annotate_bars(ax, b2, r2_sems, offset=1.5, fontsize=8.5)

        x_labels = [f"{m}\n(N={c})" for m, c in zip(models, counts)]
        ax.set_title(f"Task: {t_title}", fontsize=13, pad=12)
        if ax_idx == 0:
            ax.set_ylabel("Average Judge Score (0–100 Scale)", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=15, ha="right", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.axhline(100, color="gray", linestyle="--", alpha=0.35, linewidth=1)
        ax.legend(loc="upper right", frameon=True, fontsize=10)

    fig.suptitle(
        f"Answer Degradation Analysis: Answers with Human Ground Truth Score ≥ {int(human_threshold)}\n"
        "Comparing Round 1 vs. Round 2 Judge Scores Across Models",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    comb_path = output_dir / "reevaluation_high_human_degradation_by_model.png"
    fig.savefig(comb_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {comb_path.name}")
    saved_paths.append(comb_path)

    # 3. Delta Bar Chart (Net Change Δ for High Human Score >= 95)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax_idx, (t_key, t_title) in enumerate(TASKS):
        ax = axes[ax_idx]
        t_data = high_df[high_df["task"] == t_key]

        counts = [len(t_data[t_data["model"] == m]) for m in models]
        deltas = [
            t_data[t_data["model"] == m]["delta_judge"].mean() if len(t_data[t_data["model"] == m]) > 0 else 0.0
            for m in models
        ]
        delta_sems = [
            t_data[t_data["model"] == m]["delta_judge"].sem() if len(t_data[t_data["model"] == m]) > 1 else 0.0
            for m in models
        ]

        bar_colors = ["#c0392b" if d < -0.05 else ("#27ae60" if d > 0.05 else "#7f8c8d") for d in deltas]
        bars = ax.bar(x, deltas, width=0.5, yerr=delta_sems, capsize=4, color=bar_colors, alpha=0.88, edgecolor="white", linewidth=1.2)

        for bar, d, sem in zip(bars, deltas, delta_sems):
            err = sem if sem is not None and not np.isnan(sem) else 0.0
            if d >= 0:
                y_pos = d + err + 0.6
                va = "bottom"
            else:
                y_pos = d - err - 0.6
                va = "top"
            ax.annotate(f"{d:+.1f}", xy=(bar.get_x() + bar.get_width() / 2, y_pos), ha="center", va=va, fontsize=9.5, fontweight="bold", color="#2c3e50")

        x_labels = [f"{m}\n(N={c})" for m, c in zip(models, counts)]
        ax.set_title(f"Task: {t_title}", fontsize=13, pad=12)
        if ax_idx == 0:
            ax.set_ylabel("Score Change: Round 2 – Round 1 (Points)", fontsize=11.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=15, ha="right", fontsize=10, fontweight="bold")
        ax.axhline(0, color="black", linestyle="-", linewidth=1.2, alpha=0.7)

    # Symmetric y-limits
    all_d = [
        high_df[(high_df["task"] == t[0]) & (high_df["model"] == m)]["delta_judge"].mean()
        for t in TASKS for m in models
        if len(high_df[(high_df["task"] == t[0]) & (high_df["model"] == m)]) > 0
    ]
    max_d = max(max(map(abs, all_d)), 8.0) + 4.0
    for ax in axes:
        ax.set_ylim(-max_d, max_d)

    fig.suptitle(
        f"Mean Score Change (Δ = Round 2 – Round 1) for High-Quality Answers (Human ≥ {int(human_threshold)})\n"
        "(Red = Degradation, Green = Improvement)",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    delta_path = output_dir / "reevaluation_high_human_score_deltas.png"
    fig.savefig(delta_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {delta_path.name}")
    saved_paths.append(delta_path)

    return saved_paths


def print_summary_table(df: pd.DataFrame):
    """Prints a formatted summary comparing Round 1, Round 2, and Human scores."""
    print("\n" + "=" * 95)
    print("       RE-EVALUATION VS. HUMAN GROUND TRUTH EMPIRICAL SUMMARY (FinalJson_RERUN)")
    print("=" * 95)

    for t_key, t_title in TASKS:
        t_data = df[df["task"] == t_key]
        print(f"\n[Task: {t_title}]")
        print(f"{'Model':<16} | {'Human':<7} | {'Judge R1':<9} | {'Judge R2':<9} | {'R2-R1':<7} | {'MAE R1':<7} | {'MAE R2':<7} | {'r(R1,H)':<7} | {'r(R2,H)':<7}")
        print("-" * 95)

        for m in CANONICAL_MODEL_ORDER:
            sub = t_data[t_data["model"] == m]
            if sub.empty:
                continue
            h_m = sub["human_score"].mean()
            r1_m = sub["judge_r1"].mean()
            r2_m = sub["judge_r2"].mean()
            diff = r2_m - r1_m
            mae1 = sub["mae_r1"].mean()
            mae2 = sub["mae_r2"].mean()
            r1_corr = sub["judge_r1"].corr(sub["human_score"])
            r2_corr = sub["judge_r2"].corr(sub["human_score"])

            diff_str = f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}"
            print(
                f"{m:<16} | {h_m:>7.1f} | {r1_m:>9.1f} | {r2_m:>9.1f} | {diff_str:>7} | {mae1:>7.1f} | {mae2:>7.1f} | {r1_corr:>7.2f} | {r2_corr:>7.2f}"
            )

    # High-quality degradation analysis section (Human Score >= 95)
    print("\n" + "=" * 95)
    print("       HIGH-QUALITY ANSWER DEGRADATION ANALYSIS (Human Score >= 95)")
    print("=" * 95)
    high_df = df[df["human_score"] >= 95.0]

    for t_key, t_title in TASKS:
        t_data = high_df[high_df["task"] == t_key]
        print(f"\n[Task: {t_title} (Human >= 95)]")
        print(f"{'Model':<16} | {'N':<4} | {'Judge R1':<9} | {'Judge R2':<9} | {'Delta':<7} | {'Degraded (%)':<14} | {'Same (%)':<10} | {'Improved (%)':<12}")
        print("-" * 95)

        for m in CANONICAL_MODEL_ORDER:
            sub = t_data[t_data["model"] == m]
            if sub.empty:
                continue
            n = len(sub)
            r1_m = sub["judge_r1"].mean()
            r2_m = sub["judge_r2"].mean()
            diff = r2_m - r1_m
            diff_str = f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}"

            deg = (sub["delta_judge"] < -0.01).sum()
            same = (sub["delta_judge"].abs() <= 0.01).sum()
            imp = (sub["delta_judge"] > 0.01).sum()

            deg_pct = f"{deg} ({deg/n*100:.1f}%)"
            same_pct = f"{same} ({same/n*100:.1f}%)"
            imp_pct = f"{imp} ({imp/n*100:.1f}%)"

            print(
                f"{m:<16} | {n:>4d} | {r1_m:>9.1f} | {r2_m:>9.1f} | {diff_str:>7} | {deg_pct:>14} | {same_pct:>10} | {imp_pct:>12}"
            )

    print("\n" + "=" * 95)


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready plots comparing Round 1, Round 2, and Human Ground Truth from FinalJson_RERUN."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Path to results directory (default: results/FinalJson_RERUN).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for generated plots.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=95.0,
        help="Human score threshold for high-quality answer degradation analysis (default: 95.0).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI resolution for exported plots (default: 300).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    print(f"[Info] Loading re-evaluation data from: {args.results_dir.resolve()}")
    df = load_reevaluation_human_data(args.results_dir)
    print(f"[Info] Loaded {len(df)} total evaluated recipe instances across {df['model'].nunique()} models.")

    print("\n--- Generating Correctness Comparison Plots (R1 vs. R2 vs. Human) ---")
    plot_correctness_split(df, args.output_dir, dpi=args.dpi)

    print("\n--- Generating Score Distribution Boxplots ---")
    plot_distributions_split(df, args.output_dir, dpi=args.dpi)

    print("\n--- Generating Human Correlation & Alignment Plots ---")
    plot_human_correlation_split(df, args.output_dir, dpi=args.dpi)

    print("\n--- Generating Discrepancy & Gap Plots (MAE from Human) ---")
    plot_mae_gap(df, args.output_dir, dpi=args.dpi)

    print(f"\n--- Generating High-Quality Answer Degradation Plots (Human >= {args.threshold}) ---")
    plot_high_human_degradation(df, args.output_dir, human_threshold=args.threshold, dpi=args.dpi)

    print_summary_table(df)
    print(f"\n[Success] All plots successfully saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

