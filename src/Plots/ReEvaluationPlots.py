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

from src.metrics import compute_iou_stats

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "FinalJson_RERUN"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "Plots" / "SavedPlots" / "ReEvaluation"

# Target generator models in FinalJson
ALLOWED_GENERATOR_FOLDERS = [
    "gemma3_1b",
    "granite4.1_3b",
    "hf.co_bartowski_Llama-3.2-3B-Instruct-GGUF",
    "ministral-3_3b",
    "qwen3.5_4b",
]

# Canonical display order for generator models
CANONICAL_MODEL_ORDER = [
    "Gemma 3 1B",
    "Granite 4.1 3B",
    "Llama 3.2 3B",
    "Ministral 3 3B",
    "Qwen 3.5 4B",
]

# Display names for generator models
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

# Display names for judge models
JUDGE_DISPLAY_NAMES = {
    "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF": "Llama 3.2 3B",
    "gemma3:1b": "Gemma 3 1B",
    "ministral-3:3b": "Ministral 3 3B",
    "qwen3.5:2b": "Qwen 3.5 2B",
}

# Color palette tokens
COLOR_R1_IOU = "#2471a3"  # Cobalt Blue (IoU Round 1)
COLOR_R2_IOU = "#7fb3d5"  # Soft Sky Blue (IoU Round 2)
COLOR_R1_JUDGE = "#d35400"  # Rust / Burnt Orange (Judge Round 1)
COLOR_R2_JUDGE = "#f39c12"  # Amber / Golden Orange (Judge Round 2)


def setup_style():
    """Configures consistent publication-ready plot style."""
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


def load_reevaluation_data(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    allowed_models: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads re-evaluation data from the target directory and computes IoU and Average Judge metrics."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    if allowed_models is None:
        # Discover all subdirectories containing JSON files, prioritizing known order
        discovered = [d.name for d in results_dir.iterdir() if d.is_dir() and list(d.glob("*.json"))]
        allowed_models = [m for m in ALLOWED_GENERATOR_FOLDERS if m in discovered]
        # Append any other folders discovered
        for d_name in sorted(discovered):
            if d_name not in allowed_models:
                allowed_models.append(d_name)

    qa_records = []
    judge_records = []

    for model_folder in allowed_models:
        folder_path = results_dir / model_folder
        if not folder_path.is_dir():
            continue

        for json_file in sorted(folder_path.glob("*.json")):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            gen_model_raw = data.get("model", model_folder)
            gen_model_name = GENERATOR_DISPLAY_NAMES.get(model_folder, GENERATOR_DISPLAY_NAMES.get(gen_model_raw, gen_model_raw))
            task = data.get("target_type", "")  # "ingredients" or "directions"
            qa_pairs = data.get("qa_pairs", [])

            for qa in qa_pairs:
                recipe = qa.get("recipe_name", "Unknown")
                orig_ans = qa.get("original_answer", "")
                corr_ans = qa.get("round_2", {}).get("corrected_answer", "") or ""
                refs = qa.get("correct_answers", [])

                iou_r1, _, _ = compute_iou_stats(orig_ans, refs)
                iou_r2, _, _ = compute_iou_stats(corr_ans, refs)
                iou_r1_pct = iou_r1 * 100
                iou_r2_pct = iou_r2 * 100

                r1_data = qa.get("round_1", {})
                r2_data = qa.get("round_2", {})
                judge_r1 = float(r1_data.get("average_score", 0.0))
                judge_r2 = float(r2_data.get("average_score", 0.0))

                qa_records.append({
                    "generator_folder": model_folder,
                    "generator_model": gen_model_name,
                    "task": task,
                    "recipe": recipe,
                    "iou_r1": iou_r1_pct,
                    "iou_r2": iou_r2_pct,
                    "delta_iou": iou_r2_pct - iou_r1_pct,
                    "judge_r1": judge_r1,
                    "judge_r2": judge_r2,
                    "delta_judge": judge_r2 - judge_r1,
                })

                # Individual judge records
                r1_judges = r1_data.get("judges", {})
                r2_judges = r2_data.get("judges", {})
                for jname_raw, jinfo1 in r1_judges.items():
                    s1 = jinfo1.get("score")
                    s2 = r2_judges.get(jname_raw, {}).get("score")
                    if s1 is not None and s2 is not None:
                        jname_disp = JUDGE_DISPLAY_NAMES.get(jname_raw, jname_raw)
                        judge_records.append({
                            "generator_folder": model_folder,
                            "generator_model": gen_model_name,
                            "task": task,
                            "recipe": recipe,
                            "judge_raw": jname_raw,
                            "judge_model": jname_disp,
                            "score_r1": float(s1),
                            "score_r2": float(s2),
                            "delta_score": float(s2) - float(s1),
                        })

    df_qa = pd.DataFrame(qa_records)
    df_judge = pd.DataFrame(judge_records)
    return df_qa, df_judge


def _place_pair_labels(ax, bar1, val1: float, fmt1: str, bar2, val2: float, fmt2: str, fontsize: float = 8.5):
    """Places value labels on a pair of adjacent bars, vertically staggering if values are within 3.5 points to avoid horizontal collision."""
    x1 = bar1.get_x() + bar1.get_width() / 2
    x2 = bar2.get_x() + bar2.get_width() / 2

    if abs(val1 - val2) < 3.5:
        if val1 <= val2:
            y1 = val1 + 1.2
            y2 = val2 + 4.2
        else:
            y1 = val1 + 4.2
            y2 = val2 + 1.2
    else:
        y1 = val1 + 1.2
        y2 = val2 + 1.2

    ax.text(x1, y1, fmt1.format(val1), ha="center", va="bottom", fontsize=fontsize, fontweight="bold")
    ax.text(x2, y2, fmt2.format(val2), ha="center", va="bottom", fontsize=fontsize, fontweight="bold")


def plot_reevaluation_task_comparison(df: pd.DataFrame, output_dir: Path):
    """Generates task-level Round 1 vs Round 2 comparisons: standalone Ingredients, standalone Directions, and overview (no deltas)."""
    setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("ingredients", "Ingredients"),
        ("directions", "Directions"),
    ]

    # Standalone plots for each task
    for task_key, task_title in tasks:
        df_task = df[df["task"] == task_key]
        if df_task.empty:
            continue

        r1_judge = df_task["judge_r1"].mean()
        r2_judge = df_task["judge_r2"].mean()

        fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)

        x_pos = [0.0, 1.0]
        bar_width = 0.45
        b1 = ax.bar(
            x_pos[0],
            r1_judge,
            width=bar_width,
            color=COLOR_R1_JUDGE,
            edgecolor="black",
            linewidth=0.8,
            label="Round 1 (Original)",
        )
        b2 = ax.bar(
            x_pos[1],
            r2_judge,
            width=bar_width,
            color=COLOR_R2_JUDGE,
            edgecolor="black",
            linewidth=0.8,
            label="Round 2 (Re-evaluated)",
        )

        ax.text(
            x_pos[0],
            r1_judge + 1.2,
            f"{r1_judge:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            x_pos[1],
            r2_judge + 1.2,
            f"{r2_judge:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(["Round 1\n(Original)", "Round 2\n(Re-evaluated)"], fontsize=11.5, fontweight="bold")
        ax.set_ylabel("Average Judges Score (0–100)", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 108)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(
            f"Answer Re-Evaluation: {task_title}\nRound 1 (Original) vs. Round 2 (Judge Corrected)",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )

        legend_elements = [
            Patch(facecolor=COLOR_R1_JUDGE, edgecolor="black", label="Average Judges Score (Round 1)"),
            Patch(facecolor=COLOR_R2_JUDGE, edgecolor="black", label="Average Judges Score (Round 2)"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=2,
            framealpha=0.95,
            fontsize=10.5,
        )

        plt.tight_layout()
        filepath = output_dir / f"reevaluation_task_{task_key}.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {filepath.name}")

    # Combined overview plot: comparing Ingredients and Directions side by side
    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=300)
    task_x = np.arange(len(tasks)) * 1.2
    bar_width = 0.35

    for i, (task_key, task_title) in enumerate(tasks):
        df_task = df[df["task"] == task_key]
        r1_judge = df_task["judge_r1"].mean()
        r2_judge = df_task["judge_r2"].mean()
        cx = task_x[i]

        b1 = ax.bar(
            cx - bar_width / 2,
            r1_judge,
            width=bar_width,
            color=COLOR_R1_JUDGE,
            edgecolor="black",
            linewidth=0.8,
        )
        b2 = ax.bar(
            cx + bar_width / 2,
            r2_judge,
            width=bar_width,
            color=COLOR_R2_JUDGE,
            edgecolor="black",
            linewidth=0.8,
        )

        _place_pair_labels(ax, b1[0], r1_judge, "{:.1f}", b2[0], r2_judge, "{:.1f}", fontsize=11.0)

    ax.set_xticks(task_x)
    ax.set_xticklabels([t[1] for t in tasks], fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Judges Score (0–100)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlim(task_x[0] - 0.9, task_x[-1] + 0.9)
    ax.set_title(
        "Answer Re-Evaluation: Overall Task Comparison\nRound 1 (Original) vs. Round 2 (Judge Corrected)",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    legend_elements = [
        Patch(facecolor=COLOR_R1_JUDGE, edgecolor="black", label="Average Judges Score (Round 1)"),
        Patch(facecolor=COLOR_R2_JUDGE, edgecolor="black", label="Average Judges Score (Round 2)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        framealpha=0.95,
        fontsize=11,
    )

    plt.tight_layout()
    filepath = output_dir / "reevaluation_task_overview.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {filepath.name}")


def plot_reevaluation_model_comparison(df: pd.DataFrame, output_dir: Path):
    """Generates generator model breakdown plots across all 5 evaluated models (no deltas)."""
    setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("ingredients", "Ingredients"),
        ("directions", "Directions"),
    ]

    # Order models by canonical list
    present_models = [m for m in CANONICAL_MODEL_ORDER if m in df["generator_model"].unique()]
    if not present_models:
        present_models = sorted(df["generator_model"].unique())

    # Standalone plots per task
    for task_key, task_title in tasks:
        df_task = df[df["task"] == task_key]
        if df_task.empty:
            continue

        fig, ax = plt.subplots(figsize=(11.0, 6.0), dpi=300)

        n_models = len(present_models)
        group_spacing = 1.0
        x_centers = np.arange(n_models) * group_spacing
        bar_w = 0.32

        for m_idx, m_name in enumerate(present_models):
            m_sub = df_task[df_task["generator_model"] == m_name]
            r1_j = m_sub["judge_r1"].mean() if not m_sub.empty else 0.0
            r2_j = m_sub["judge_r2"].mean() if not m_sub.empty else 0.0

            cx = x_centers[m_idx]

            b1 = ax.bar(cx - bar_w / 2, r1_j, width=bar_w, color=COLOR_R1_JUDGE, edgecolor="black", linewidth=0.8)
            b2 = ax.bar(cx + bar_w / 2, r2_j, width=bar_w, color=COLOR_R2_JUDGE, edgecolor="black", linewidth=0.8)

            _place_pair_labels(ax, b1[0], r1_j, "{:.1f}", b2[0], r2_j, "{:.1f}", fontsize=9.5)

        ax.set_xticks(x_centers)
        ax.set_xticklabels(present_models, fontsize=11, fontweight="bold")
        ax.set_ylabel("Average Judges Score (0–100)", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 108)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_xlim(x_centers[0] - 0.7, x_centers[-1] + 0.7)
        ax.set_title(
            f"Model Breakdown: {task_title}\nRound 1 vs. Round 2 Comparison",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )

        legend_elements = [
            Patch(facecolor=COLOR_R1_JUDGE, edgecolor="black", label="Average Judges Score (Round 1)"),
            Patch(facecolor=COLOR_R2_JUDGE, edgecolor="black", label="Average Judges Score (Round 2)"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=2,
            framealpha=0.95,
            fontsize=11,
        )

        plt.tight_layout()
        filepath = output_dir / f"reevaluation_model_{task_key}.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {filepath.name}")

    # Overview plot (side-by-side)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300, sharey=True)
    group_spacing = 1.0
    x_centers = np.arange(len(present_models)) * group_spacing
    bar_w = 0.32

    for i, (task_key, task_title) in enumerate(tasks):
        ax = axes[i]
        df_task = df[df["task"] == task_key]

        for m_idx, m_name in enumerate(present_models):
            m_sub = df_task[df_task["generator_model"] == m_name]
            r1_j = m_sub["judge_r1"].mean() if not m_sub.empty else 0.0
            r2_j = m_sub["judge_r2"].mean() if not m_sub.empty else 0.0

            cx = x_centers[m_idx]

            b1 = ax.bar(cx - bar_w / 2, r1_j, width=bar_w, color=COLOR_R1_JUDGE, edgecolor="black", linewidth=0.8)
            b2 = ax.bar(cx + bar_w / 2, r2_j, width=bar_w, color=COLOR_R2_JUDGE, edgecolor="black", linewidth=0.8)

            _place_pair_labels(ax, b1[0], r1_j, "{:.1f}", b2[0], r2_j, "{:.1f}", fontsize=9.0)

        ax.set_xticks(x_centers)
        ax.set_xticklabels(present_models, fontsize=10.5, fontweight="bold")
        if i == 0:
            ax.set_ylabel("Average Judges Score (0–100)", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 108)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_xlim(x_centers[0] - 0.7, x_centers[-1] + 0.7)
        ax.set_title(f"Task: {task_title}", fontsize=13, fontweight="bold")

    legend_elements = [
        Patch(facecolor=COLOR_R1_JUDGE, edgecolor="black", label="Average Judges Score (Round 1)"),
        Patch(facecolor=COLOR_R2_JUDGE, edgecolor="black", label="Average Judges Score (Round 2)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.04),
        framealpha=0.95,
        fontsize=11,
    )
    fig.suptitle(
        "Generator Model Re-Evaluation Breakdown (All Models)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    filepath = output_dir / "reevaluation_model_overview.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {filepath.name}")


def plot_per_judge_reevaluation(df_judge: pd.DataFrame, output_dir: Path):
    """Generates a comparison of how individual judge models scored Round 1 vs Round 2 (no deltas)."""
    setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("ingredients", "Ingredients"),
        ("directions", "Directions"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300, sharey=True)

    judge_order = ["Gemma 3 1B", "Llama 3.2 3B", "Ministral 3 3B", "Qwen 3.5 2B"]

    for i, (task_key, task_title) in enumerate(tasks):
        ax = axes[i]
        df_sub = df_judge[df_judge["task"] == task_key]
        if df_sub.empty:
            continue

        # Compute stats per judge model
        judge_stats = df_sub.groupby("judge_model")[["score_r1", "score_r2"]].mean()
        present_judges = [j for j in judge_order if j in judge_stats.index]
        if not present_judges:
            present_judges = sorted(judge_stats.index)

        x_indices = np.arange(len(present_judges))
        bar_w = 0.35

        r1_vals = [judge_stats.loc[j, "score_r1"] for j in present_judges]
        r2_vals = [judge_stats.loc[j, "score_r2"] for j in present_judges]

        b1 = ax.bar(
            x_indices - bar_w / 2,
            r1_vals,
            width=bar_w,
            color=COLOR_R1_JUDGE,
            edgecolor="black",
            linewidth=0.8,
            label="Round 1 (Original)",
        )
        b2 = ax.bar(
            x_indices + bar_w / 2,
            r2_vals,
            width=bar_w,
            color=COLOR_R2_JUDGE,
            edgecolor="black",
            linewidth=0.8,
            label="Round 2 (Re-evaluated)",
        )

        for bar, val in zip(b1, r1_vals):
            ax.text(
                bar.get_x() + bar_w / 2,
                val + 1.2,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        for bar, val in zip(b2, r2_vals):
            ax.text(
                bar.get_x() + bar_w / 2,
                val + 1.2,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_xticks(x_indices)
        ax.set_xticklabels(present_judges, fontsize=11, fontweight="bold")
        if i == 0:
            ax.set_ylabel("Average Score (0–100)", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.set_title(f"Task: {task_title}", fontsize=13, fontweight="bold")

    legend_elements = [
        Patch(facecolor=COLOR_R1_JUDGE, edgecolor="black", label="Judge Score (Round 1)"),
        Patch(facecolor=COLOR_R2_JUDGE, edgecolor="black", label="Judge Score (Round 2)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.04),
        framealpha=0.95,
        fontsize=11,
    )
    fig.suptitle(
        "Individual Judge Evaluator Scoring: Round 1 vs. Round 2",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    filepath = output_dir / "reevaluation_per_judge_comparison.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {filepath.name}")


def print_summary_table(df: pd.DataFrame, df_judge: pd.DataFrame):
    """Prints a formatted summary table of all empirical metrics across tasks and models."""
    print("\n" + "=" * 80)
    print("                RAG ANSWER RE-EVALUATION EMPIRICAL SUMMARY")
    print("=" * 80)

    # Task level summary
    print("\n[1] Overall Task Performance (Round 1 vs. Round 2):")
    print(f"{'Task':<14} | {'IoU R1 (%)':<11} | {'IoU R2 (%)':<11} | {'Judge R1':<10} | {'Judge R2':<10}")
    print("-" * 65)
    for task in ["ingredients", "directions"]:
        sub = df[df["task"] == task]
        iou1, iou2 = sub["iou_r1"].mean(), sub["iou_r2"].mean()
        j1, j2 = sub["judge_r1"].mean(), sub["judge_r2"].mean()
        print(f"{task.capitalize():<14} | {iou1:>10.2f}% | {iou2:>10.2f}% | {j1:>10.2f} | {j2:>10.2f}")

    # Model level summary
    print("\n[2] Generator Model Breakdown:")
    print(f"{'Generator Model':<18} | {'Task':<12} | {'IoU R1':<9} | {'IoU R2':<9} | {'Judge R1':<9} | {'Judge R2':<9}")
    print("-" * 75)
    ordered_models = [m for m in CANONICAL_MODEL_ORDER if m in df["generator_model"].unique()]
    for model in ordered_models:
        for task in ["ingredients", "directions"]:
            sub = df[(df["generator_model"] == model) & (df["task"] == task)]
            iou1, iou2 = sub["iou_r1"].mean(), sub["iou_r2"].mean()
            j1, j2 = sub["judge_r1"].mean(), sub["judge_r2"].mean()
            print(f"{model:<18} | {task.capitalize():<12} | {iou1:>8.2f}% | {iou2:>8.2f}% | {j1:>9.2f} | {j2:>9.2f}")

    # Individual judge summary
    print("\n[3] Evaluator Judge Scoring (Round 1 vs. Round 2):")
    print(f"{'Judge Evaluator':<18} | {'Task':<12} | {'Score R1':<10} | {'Score R2':<10}")
    print("-" * 55)
    for j_mod in sorted(df_judge["judge_model"].unique()):
        for task in ["ingredients", "directions"]:
            sub = df_judge[(df_judge["judge_model"] == j_mod) & (df_judge["task"] == task)]
            if not sub.empty:
                s1, s2 = sub["score_r1"].mean(), sub["score_r2"].mean()
                print(f"{j_mod:<18} | {task.capitalize():<12} | {s1:>10.2f} | {s2:>10.2f}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication plots for RAG Answer Re-Evaluation analysis across all models."
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Path to results directory containing evaluated json files (default: results/FinalJson_RERUN).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save generated publication plots.",
    )
    parser.add_argument(
        "--compare_human",
        "--compare-human",
        action="store_true",
        dest="compare_human",
        help="Also generate comparison plots against human ground truth (Round 1 vs Round 2 vs Human).",
    )
    args = parser.parse_args()

    print(f"Loading re-evaluation data from: {args.results_dir}")
    df_qa, df_judge = load_reevaluation_data(args.results_dir)
    print(
        f"Loaded {len(df_qa)} QA evaluations and {len(df_judge)} individual judge scores "
        f"across {df_qa['generator_model'].nunique()} models."
    )

    print(f"\nGenerating plots in: {args.output_dir}...")
    plot_reevaluation_task_comparison(df_qa, args.output_dir)
    plot_reevaluation_model_comparison(df_qa, args.output_dir)
    plot_per_judge_reevaluation(df_judge, args.output_dir)

    if args.compare_human:
        from src.Plots.ReEvaluationHumanPlots import (
            load_reevaluation_human_data,
            plot_correctness_split,
            plot_distributions_split,
            plot_human_correlation_split,
            plot_mae_gap,
            plot_high_human_degradation,
            print_summary_table as print_human_summary_table,
        )
        human_out = args.output_dir / "Human_Comparison"
        human_out.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Generating Re-Evaluation vs. Human Plots in: {human_out} ---")
        df_human = load_reevaluation_human_data(args.results_dir)
        plot_correctness_split(df_human, human_out)
        plot_distributions_split(df_human, human_out)
        plot_human_correlation_split(df_human, human_out)
        plot_mae_gap(df_human, human_out)
        plot_high_human_degradation(df_human, human_out)
        print_human_summary_table(df_human)

    print_summary_table(df_qa, df_judge)
    print("All re-evaluation plots generated successfully!")


if __name__ == "__main__":
    main()
