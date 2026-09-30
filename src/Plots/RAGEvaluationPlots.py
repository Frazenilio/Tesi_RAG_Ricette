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
COLOR_IOU = "#2b5c8f"        # Deep Slate Blue (Jaccard Distance)
COLOR_HUMAN = "#27ae60"      # Emerald Green (Human Evaluation)

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

MODEL_COLORS = {
    "Llama-3.2-3B-Instruct-GGUF": "#2980b9",
    "gemma3:1b": "#d35400",
    "granite4.1:3b": "#7f8c8d",
    "ministral-3:3b": "#c0392b",
    "qwen3.5:4b": "#16a085",
}

TASKS = [("ingredients", "Ingredients"), ("directions", "Directions")]


def setup_style():
    """Sets consistent publication-ready plot style."""
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


def load_rag_data(directories: list[Path]) -> pd.DataFrame:
    """Loads all RAG generation results and computes Jaccard Distance and Human scores on [0.0 - 1.0] scale.
    Note: IoU/Jaccard is only computed for 'ingredients'. For 'directions', IoU makes no semantic sense
    so it is kept as NaN.
    """
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

            is_directions = (target_type == "directions")

            for q in qa_pairs:
                orig = q.get("original_answer", "")
                refs = q.get("correct_answers", [])
                
                # Only compute lexical Jaccard on ingredients
                if not is_directions:
                    max_iou, mean_iou, _ = compute_iou_stats(orig, refs)
                else:
                    max_iou, mean_iou = np.nan, np.nan

                raw_human = q.get("human_score")
                human_score = float(raw_human) / 100.0 if raw_human is not None else np.nan

                rows.append({
                    "model_raw": gen_model,
                    "model": gen_model_short,
                    "task": target_type,
                    "recipe_name": q.get("recipe_name", ""),
                    "iou_max": max_iou,        # [0.0, 1.0] scale
                    "iou_mean": mean_iou,      # [0.0, 1.0] scale
                    "human_score": human_score,# [0.0, 1.0] scale
                })

    df = pd.DataFrame(rows)
    return df


def plot_rag_correctness_split(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int = 300,
    human_only: bool = False,
    iou_only: bool = False,
) -> list[Path]:
    """Plot 1 (Split): Generates dedicated, full-size plots comparing Jaccard Distance vs Human Evaluation.
    All scores are on the normalized [0.0 – 1.0] scale.
    Directions only uses Human Evaluation (IoU is omitted as it has no semantic sense for directions).
    """
    saved_paths = []
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    model_display_names = [MODEL_LABELS.get(m, m) for m in models]
    x = np.arange(len(models))

    for t_key, t_title in TASKS:
        t_data = df[df["task"] == t_key]

        # For directions in iou_only mode: skip since directions has no IoU
        if iou_only and t_key == "directions":
            print(f"Skipping IoU-only plot for {t_title}: Jaccard Distance is not applicable to recipe directions.")
            continue

        fig, ax = plt.subplots(figsize=(10, 6.5))

        human_means = [t_data[t_data["model"] == m]["human_score"].mean() for m in models]
        human_sems = [t_data[t_data["model"] == m]["human_score"].sem() for m in models]
        iou_means = [t_data[t_data["model"] == m]["iou_max"].mean() for m in models]
        iou_sems = [t_data[t_data["model"] == m]["iou_max"].sem() for m in models]

        if iou_only:
            width = 0.52
            rects = ax.bar(
                x,
                iou_means,
                width,
                yerr=iou_sems,
                capsize=5,
                label="Jaccard Distance",
                color=COLOR_IOU,
                alpha=0.92,
                edgecolor="white",
                linewidth=1.2,
            )
            for rect, sem in zip(rects, iou_sems):
                h = rect.get_height()
                if not np.isnan(h) and h > 0:
                    y_pos = h + (sem if not np.isnan(sem) else 0) + 0.02
                    ax.annotate(
                        f"{h:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, y_pos),
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        fontweight="bold",
                        color="#2c3e50",
                    )
            ax.set_title(
                f"RAG Answer Correctness: {t_title}\nJaccard Distance Across Models",
                pad=14,
            )
            ax.set_ylabel("Jaccard Distance (0.0 – 1.0 Scale)", fontsize=12, fontweight="bold")
            ax.set_ylim(0, 1.15)
        elif human_only or t_key == "directions":
            width = 0.52
            rects = ax.bar(
                x,
                human_means,
                width,
                yerr=human_sems,
                capsize=5,
                label="Human Evaluation Score",
                color=COLOR_HUMAN,
                alpha=0.92,
                edgecolor="white",
                linewidth=1.2,
            )
            for rect, sem in zip(rects, human_sems):
                h = rect.get_height()
                if not np.isnan(h) and h > 0:
                    y_pos = h + (sem if not np.isnan(sem) else 0) + 0.02
                    ax.annotate(
                        f"{h:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, y_pos),
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        fontweight="bold",
                        color="#2c3e50",
                    )
            subtitle = "Human Evaluation Score Across Models" if t_key != "directions" else "Human Evaluation Score (Directions Quality)"
            ax.set_title(
                f"RAG Answer Correctness: {t_title}\n{subtitle}",
                pad=14,
            )
            ax.set_ylabel("Human Evaluation Score (0.0 – 1.0 Scale)", fontsize=12, fontweight="bold")
            ax.set_ylim(0, 1.15)
        else:
            # Ingredients combined (Jaccard Distance vs Human Evaluation on 0.0 - 1.0 scale)
            width = 0.35
            r_iou = ax.bar(
                x - width / 2,
                iou_means,
                width,
                yerr=iou_sems,
                capsize=4.5,
                label="Jaccard Distance",
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
                capsize=4.5,
                label="Human Evaluation Score",
                color=COLOR_HUMAN,
                alpha=0.92,
                edgecolor="white",
                linewidth=1.2,
            )

            for rects, sems in [(r_iou, iou_sems), (r_human, human_sems)]:
                for rect, sem in zip(rects, sems):
                    h = rect.get_height()
                    if not np.isnan(h) and h > 0:
                        y_pos = h + (sem if not np.isnan(sem) else 0) + 0.02
                        ax.annotate(
                            f"{h:.2f}",
                            xy=(rect.get_x() + rect.get_width() / 2, y_pos),
                            ha="center",
                            va="bottom",
                            fontsize=10.5,
                            fontweight="bold",
                            color="#2c3e50",
                        )

            ax.set_title(
                f"RAG Answer Correctness: {t_title}\nJaccard Distance vs. Human Evaluation Across Models",
                pad=14,
            )
            ax.set_ylabel("Score (0.0 – 1.0 Scale)", fontsize=12, fontweight="bold")
            ax.set_ylim(0, 1.18)
            ax.legend(loc="upper right", frameon=True, fontsize=11)

        ax.set_xticks(x)
        ax.set_xticklabels(model_display_names, rotation=15, ha="right", fontsize=11, fontweight="bold")
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.35, linewidth=1)

        plt.tight_layout()
        out_path = output_dir / f"rag_correctness_{t_key}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
        saved_paths.append(out_path)

    # Combined overview for both tasks
    if iou_only:
        # In iou_only, only Ingredients exists
        ing_data = df[df["task"] == "ingredients"]
        fig, ax = plt.subplots(figsize=(10.5, 6.5))
        iou_means = [ing_data[ing_data["model"] == m]["iou_max"].mean() for m in models]
        iou_sems = [ing_data[ing_data["model"] == m]["iou_max"].sem() for m in models]
        rects = ax.bar(x, iou_means, 0.52, yerr=iou_sems, capsize=5, label="Jaccard Distance", color=COLOR_IOU, alpha=0.92, edgecolor="white", linewidth=1.2)
        for rect, sem in zip(rects, iou_sems):
            h = rect.get_height()
            if not np.isnan(h) and h > 0:
                y_pos = h + (sem if not np.isnan(sem) else 0) + 0.02
                ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, y_pos), ha="center", va="bottom", fontsize=11, fontweight="bold", color="#2c3e50")
        ax.set_title("RAG Answer Correctness: Ingredients\nJaccard Distance Across Models", pad=14)
        ax.set_ylabel("Jaccard Distance (0.0 – 1.0 Scale)", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_display_names, rotation=15, ha="right", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.35, linewidth=1)
        plt.tight_layout()
        combined_path = output_dir / "rag_correctness_by_model_and_task.png"
        fig.savefig(combined_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {combined_path}")
        saved_paths.append(combined_path)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
        # Left: Ingredients (Jaccard Distance vs Human Evaluation)
        ax0 = axes[0]
        ing_data = df[df["task"] == "ingredients"]
        ing_h_means = [ing_data[ing_data["model"] == m]["human_score"].mean() for m in models]
        ing_h_sems = [ing_data[ing_data["model"] == m]["human_score"].sem() for m in models]
        ing_j_means = [ing_data[ing_data["model"] == m]["iou_max"].mean() for m in models]
        ing_j_sems = [ing_data[ing_data["model"] == m]["iou_max"].sem() for m in models]

        r1 = ax0.bar(x - 0.18, ing_j_means, 0.35, yerr=ing_j_sems, capsize=4, label="Jaccard Distance", color=COLOR_IOU, alpha=0.92, edgecolor="white", linewidth=1.2)
        r2 = ax0.bar(x + 0.18, ing_h_means, 0.35, yerr=ing_h_sems, capsize=4, label="Human Evaluation Score", color=COLOR_HUMAN, alpha=0.92, edgecolor="white", linewidth=1.2)
        for rects, sems in [(r1, ing_j_sems), (r2, ing_h_sems)]:
            for rect, sem in zip(rects, sems):
                h = rect.get_height()
                if not np.isnan(h) and h > 0:
                    y_pos = h + (sem if not np.isnan(sem) else 0) + 0.02
                    ax0.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, y_pos), ha="center", va="bottom", fontsize=9, fontweight="bold", color="#2c3e50")
        ax0.set_title("Task: Ingredients (Jaccard Distance vs. Human)", fontsize=13, pad=12)
        ax0.set_ylabel("Score (0.0 – 1.0 Scale)", fontsize=11, fontweight="bold")
        ax0.set_xticks(x)
        ax0.set_xticklabels(model_display_names, rotation=15, ha="right", fontsize=10.5)
        ax0.legend(loc="upper right", frameon=True, fontsize=10)
        ax0.set_ylim(0, 1.18)
        ax0.axhline(1.0, color="gray", linestyle="--", alpha=0.35, linewidth=1)

        # Right: Directions (Human Evaluation Only)
        ax1 = axes[1]
        dir_data = df[df["task"] == "directions"]
        dir_h_means = [dir_data[dir_data["model"] == m]["human_score"].mean() for m in models]
        dir_h_sems = [dir_data[dir_data["model"] == m]["human_score"].sem() for m in models]
        rects_d = ax1.bar(x, dir_h_means, 0.52, yerr=dir_h_sems, capsize=4.5, label="Human Evaluation Score", color=COLOR_HUMAN, alpha=0.92, edgecolor="white", linewidth=1.2)
        for rect, sem in zip(rects_d, dir_h_sems):
            h = rect.get_height()
            if not np.isnan(h) and h > 0:
                y_pos = h + (sem if not np.isnan(sem) else 0) + 0.02
                ax1.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, y_pos), ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#2c3e50")
        ax1.set_title("Task: Directions (Human Evaluation Only)", fontsize=13, pad=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_display_names, rotation=15, ha="right", fontsize=10.5)
        ax1.legend(loc="upper right", frameon=True, fontsize=10)
        ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.35, linewidth=1)

        fig.suptitle("RAG Answer Correctness Across Models\nIngredients (Jaccard Distance vs. Human) & Directions (Human Evaluation)", fontsize=15, fontweight="bold", y=0.99)
        plt.tight_layout()
        combined_path = output_dir / "rag_correctness_by_model_and_task.png"
        fig.savefig(combined_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {combined_path}")
        saved_paths.append(combined_path)

    return saved_paths


def plot_rag_distributions_split(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int = 300,
    human_only: bool = False,
    iou_only: bool = False,
) -> list[Path]:
    """Plot 2 (Split): Generates dedicated, large boxplots showing score distribution for Ingredients and Directions.
    All scores are on the [0.0 – 1.0] scale.
    Directions only shows Human Evaluation distribution.
    """
    saved_paths = []
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    ordered_labels = [MODEL_LABELS.get(m, m) for m in models]

    melted = df.copy()
    melted["model_display"] = melted["model"].map(MODEL_LABELS)

    if iou_only:
        # Only Ingredients has Jaccard Distance
        fig, ax = plt.subplots(figsize=(10.5, 6.5))
        sub = melted[melted["task"] == "ingredients"]

        sns.boxplot(
            data=sub,
            x="model_display",
            y="iou_max",
            order=ordered_labels,
            color=COLOR_IOU,
            ax=ax,
            boxprops=dict(alpha=0.85),
            medianprops=dict(color="#c0392b", linewidth=2.4),
            showmeans=True,
            meanprops=dict(marker="o", markeredgecolor="black", markerfacecolor="white", markersize=7.5),
        )
        ax.set_title("RAG Jaccard Distance Distribution: Ingredients\n(White Circle = Mean, Red Line = Median)", pad=14)
        ax.set_ylabel("Jaccard Distance (0.0 – 1.0 Scale)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Generator LLM", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(ordered_labels)))
        ax.set_xticklabels(ordered_labels, fontsize=11, fontweight="bold")
        ax.set_ylim(-0.05, 1.08)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.35)

        plt.tight_layout()
        out_path = output_dir / "rag_score_distributions_ingredients.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
        saved_paths.append(out_path)

        # Overview
        combined_path = output_dir / "rag_score_distributions_by_model.png"
        fig, ax = plt.subplots(figsize=(12, 6.5))
        sns.boxplot(
            data=sub,
            x="model_display",
            y="iou_max",
            order=ordered_labels,
            color=COLOR_IOU,
            ax=ax,
            boxprops=dict(alpha=0.85),
            medianprops=dict(color="#c0392b", linewidth=2.2),
            showmeans=True,
            meanprops=dict(marker="o", markeredgecolor="black", markerfacecolor="white", markersize=7),
        )
        ax.set_title("RAG Jaccard Distance Distribution & Variance per Model (Ingredients)", fontsize=14, pad=12)
        ax.set_ylabel("Jaccard Distance (0.0 – 1.0 Scale)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Generator LLM", fontsize=11, fontweight="bold")
        ax.set_ylim(-0.05, 1.08)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.35)
        plt.tight_layout()
        fig.savefig(combined_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {combined_path}")
        saved_paths.append(combined_path)

    elif human_only:
        for t_key, t_title in TASKS:
            fig, ax = plt.subplots(figsize=(10.5, 6.5))
            sub = melted[melted["task"] == t_key]

            sns.boxplot(
                data=sub,
                x="model_display",
                y="human_score",
                order=ordered_labels,
                color=COLOR_HUMAN,
                ax=ax,
                boxprops=dict(alpha=0.85),
                medianprops=dict(color="#c0392b", linewidth=2.4),
                showmeans=True,
                meanprops=dict(marker="o", markeredgecolor="black", markerfacecolor="white", markersize=7.5),
            )
            ax.set_title(f"RAG Human Score Distribution: {t_title}\n(White Circle = Mean, Red Line = Median)", pad=14)
            ax.set_ylabel("Human Evaluation Score (0.0 – 1.0 Scale)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Generator LLM", fontsize=12, fontweight="bold")
            ax.set_xticks(range(len(ordered_labels)))
            ax.set_xticklabels(ordered_labels, fontsize=11, fontweight="bold")
            ax.set_ylim(-0.05, 1.08)
            ax.axhline(1.0, color="gray", linestyle="--", alpha=0.35)

            plt.tight_layout()
            out_path = output_dir / f"rag_score_distributions_{t_key}.png"
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_path}")
            saved_paths.append(out_path)

        # Combined stacked
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        for ax_idx, (t_key, t_title) in enumerate(TASKS):
            ax = axes[ax_idx]
            sub = melted[melted["task"] == t_key]
            sns.boxplot(
                data=sub,
                x="model_display",
                y="human_score",
                order=ordered_labels,
                color=COLOR_HUMAN,
                ax=ax,
                boxprops=dict(alpha=0.85),
                medianprops=dict(color="#c0392b", linewidth=2.2),
                showmeans=True,
                meanprops=dict(marker="o", markeredgecolor="black", markerfacecolor="white", markersize=6.5),
            )
            ax.set_title(f"Human Score Distribution: {t_title} (Circle = Mean, Line = Median)", fontsize=13, pad=10)
            ax.set_ylabel("Human Score (0.0 – 1.0)", fontsize=11, fontweight="bold")
            ax.set_xlabel("" if ax_idx == 0 else "LLM Model", fontsize=11, fontweight="bold")
            ax.set_ylim(-0.05, 1.08)
            ax.axhline(1.0, color="gray", linestyle="--", alpha=0.35)
        fig.suptitle("RAG Human Correctness Distribution & Variance per Model\n(Ingredients vs. Directions)", fontsize=15, fontweight="bold", y=0.99)
        plt.tight_layout()
        combined_path = output_dir / "rag_score_distributions_by_model.png"
        fig.savefig(combined_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {combined_path}")
        saved_paths.append(combined_path)

    else:
        # Standard mode:
        # Ingredients has both Jaccard Distance and Human Evaluation
        # Directions has ONLY Human Evaluation
        sub_ing = melted[melted["task"] == "ingredients"]
        melted_iou = sub_ing[["model_display", "task", "iou_max"]].copy()
        melted_iou["Metric"] = "Jaccard Distance"
        melted_iou.rename(columns={"iou_max": "Score"}, inplace=True)

        melted_human_ing = sub_ing[["model_display", "task", "human_score"]].copy()
        melted_human_ing["Metric"] = "Human Evaluation"
        melted_human_ing.rename(columns={"human_score": "Score"}, inplace=True)

        plot_ing = pd.concat([melted_iou, melted_human_ing], ignore_index=True)
        palette = {"Jaccard Distance": COLOR_IOU, "Human Evaluation": COLOR_HUMAN}

        # Ingredients individual plot
        fig, ax = plt.subplots(figsize=(10.5, 6.5))
        sns.boxplot(
            data=plot_ing,
            x="model_display",
            y="Score",
            hue="Metric",
            order=ordered_labels,
            palette=palette,
            ax=ax,
            boxprops=dict(alpha=0.85),
            medianprops=dict(color="#c0392b", linewidth=2.4),
            showmeans=True,
            meanprops=dict(marker="o", markeredgecolor="black", markerfacecolor="white", markersize=7),
        )
        ax.set_title("RAG Score Distribution: Ingredients\n(White Circle = Mean, Red Line = Median)", pad=14)
        ax.set_ylabel("Score (0.0 – 1.0 Scale)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Generator LLM", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(ordered_labels)))
        ax.set_xticklabels(ordered_labels, fontsize=11, fontweight="bold")
        ax.set_ylim(-0.05, 1.08)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.35)
        ax.legend(loc="lower left", frameon=True, fontsize=11)
        plt.tight_layout()
        out_ing = output_dir / "rag_score_distributions_ingredients.png"
        fig.savefig(out_ing, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_ing}")
        saved_paths.append(out_ing)

        # Directions individual plot (Human Only)
        sub_dir = melted[melted["task"] == "directions"]
        fig, ax = plt.subplots(figsize=(10.5, 6.5))
        sns.boxplot(
            data=sub_dir,
            x="model_display",
            y="human_score",
            order=ordered_labels,
            color=COLOR_HUMAN,
            ax=ax,
            boxprops=dict(alpha=0.85),
            medianprops=dict(color="#c0392b", linewidth=2.4),
            showmeans=True,
            meanprops=dict(marker="o", markeredgecolor="black", markerfacecolor="white", markersize=7.5),
        )
        ax.set_title("RAG Human Score Distribution: Directions\n(White Circle = Mean, Red Line = Median)", pad=14)
        ax.set_ylabel("Human Evaluation Score (0.0 – 1.0 Scale)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Generator LLM", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(ordered_labels)))
        ax.set_xticklabels(ordered_labels, fontsize=11, fontweight="bold")
        ax.set_ylim(-0.05, 1.08)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.35)
        plt.tight_layout()
        out_dir = output_dir / "rag_score_distributions_directions.png"
        fig.savefig(out_dir, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_dir}")
        saved_paths.append(out_dir)

        # Combined stacked 2x1
        fig, axes = plt.subplots(2, 1, figsize=(15, 11), sharex=True)
        # Top: Ingredients (Jaccard Distance vs Human)
        ax0 = axes[0]
        sns.boxplot(
            data=plot_ing,
            x="model_display",
            y="Score",
            hue="Metric",
            order=ordered_labels,
            palette=palette,
            ax=ax0,
            boxprops=dict(alpha=0.85),
            medianprops=dict(color="#c0392b", linewidth=2.2),
            showmeans=True,
            meanprops=dict(marker="o", markeredgecolor="black", markerfacecolor="white", markersize=6),
        )
        ax0.set_title("Ingredients Score Distribution: Jaccard Distance vs. Human Evaluation", fontsize=13, pad=10)
        ax0.set_ylabel("Score (0.0 – 1.0)", fontsize=11, fontweight="bold")
        ax0.set_xlabel("")
        ax0.set_ylim(-0.05, 1.08)
        ax0.axhline(1.0, color="gray", linestyle="--", alpha=0.35)
        ax0.legend(loc="lower left", frameon=True, fontsize=10)

        # Bottom: Directions (Human Only)
        ax1 = axes[1]
        sns.boxplot(
            data=sub_dir,
            x="model_display",
            y="human_score",
            order=ordered_labels,
            color=COLOR_HUMAN,
            ax=ax1,
            boxprops=dict(alpha=0.85),
            medianprops=dict(color="#c0392b", linewidth=2.2),
            showmeans=True,
            meanprops=dict(marker="o", markeredgecolor="black", markerfacecolor="white", markersize=6.5),
        )
        ax1.set_title("Directions Score Distribution: Human Evaluation (Circle = Mean, Line = Median)", fontsize=13, pad=10)
        ax1.set_ylabel("Human Score (0.0 – 1.0)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("LLM Model", fontsize=11, fontweight="bold")
        ax1.set_ylim(-0.05, 1.08)
        ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.35)

        fig.suptitle("RAG Correctness Distribution & Variance per Model\nIngredients (Jaccard Distance vs. Human) & Directions (Human Evaluation)", fontsize=15, fontweight="bold", y=0.99)
        plt.tight_layout()
        combined_path = output_dir / "rag_score_distributions_by_model.png"
        fig.savefig(combined_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {combined_path}")
        saved_paths.append(combined_path)

    return saved_paths


def plot_rag_success_rate_split(
    df: pd.DataFrame,
    output_dir: Path,
    human_threshold: float = 0.95,
    iou_threshold: float = 0.95,
    dpi: int = 300,
    human_only: bool = False,
    iou_only: bool = False,
) -> list[Path]:
    """Plot 3 (Split): Generates dedicated plots for High-Quality Percentage for Ingredients and Directions.
    Thresholds are defined on the [0.0 - 1.0] scale (default: 0.95).
    Directions only uses Human threshold.
    """
    saved_paths = []
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    model_display_names = [MODEL_LABELS.get(m, m) for m in models]
    x = np.arange(len(models))

    for t_key, t_title in TASKS:
        if iou_only and t_key == "directions":
            continue

        fig, ax = plt.subplots(figsize=(10, 6.5))
        t_data = df[df["task"] == t_key]

        if iou_only:
            pct_iou_high = [(t_data[t_data["model"] == m]["iou_max"] >= iou_threshold).mean() * 100.0 for m in models]
            width = 0.52
            rects = ax.bar(
                x,
                pct_iou_high,
                width,
                label=f"High Jaccard Distance (≥ {iou_threshold:.2f})",
                color=COLOR_IOU,
                alpha=0.92,
                edgecolor="white",
                linewidth=1.2,
            )
            for rect, val in zip(rects, pct_iou_high):
                h = rect.get_height()
                ax.annotate(
                    f"{val:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, h + 1.5),
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                    color="#2c3e50",
                )
            ax.set_title(
                f"RAG Success Rate: {t_title}\nPercentage of High-Overlap Answers (Jaccard Distance ≥ {iou_threshold:.2f})",
                pad=14,
            )
            ax.set_ylabel(f"% of Answers with Jaccard Distance ≥ {iou_threshold:.2f}", fontsize=12, fontweight="bold")
        elif human_only or t_key == "directions":
            pct_human_high = [(t_data[t_data["model"] == m]["human_score"] >= human_threshold).mean() * 100.0 for m in models]
            width = 0.52
            rects = ax.bar(
                x,
                pct_human_high,
                width,
                label=f"Human Ground Truth Acceptable (≥ {human_threshold:.2f})",
                color=COLOR_HUMAN,
                alpha=0.92,
                edgecolor="white",
                linewidth=1.2,
            )
            for rect, val in zip(rects, pct_human_high):
                h = rect.get_height()
                ax.annotate(
                    f"{val:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, h + 1.5),
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                    color="#2c3e50",
                )
            ax.set_title(
                f"RAG Success Rate: {t_title}\nPercentage of High-Quality Answers (Human Score ≥ {human_threshold:.2f})",
                pad=14,
            )
            ax.set_ylabel(f"% of Answers with Human Score ≥ {human_threshold:.2f}", fontsize=12, fontweight="bold")
        else:
            width = 0.35
            pct_human_high = [(t_data[t_data["model"] == m]["human_score"] >= human_threshold).mean() * 100.0 for m in models]
            pct_iou_high = [(t_data[t_data["model"] == m]["iou_max"] >= iou_threshold).mean() * 100.0 for m in models]
            r1 = ax.bar(
                x - width / 2,
                pct_iou_high,
                width,
                label=f"High Jaccard Distance (≥ {iou_threshold:.2f})",
                color=COLOR_IOU,
                alpha=0.92,
                edgecolor="white",
                linewidth=1.2,
            )
            r2 = ax.bar(
                x + width / 2,
                pct_human_high,
                width,
                label=f"Human Ground Truth Acceptable (≥ {human_threshold:.2f})",
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
                        fontsize=10.5,
                        fontweight="bold",
                        color="#2c3e50",
                    )
            ax.set_title(
                f"RAG Success Rate: {t_title}\nPercentage of High-Quality Answers (Human ≥ {human_threshold:.2f} vs. Jaccard Distance ≥ {iou_threshold:.2f})",
                pad=14,
            )
            ax.set_ylabel("% of Correct / High-Quality Answers", fontsize=12, fontweight="bold")
            ax.legend(loc="upper right", frameon=True, fontsize=11)

        ax.set_xticks(x)
        ax.set_xticklabels(model_display_names, rotation=15, ha="right", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 115)
        ax.axhline(100, color="gray", linestyle="--", alpha=0.3)

        plt.tight_layout()
        out_path = output_dir / f"rag_success_rate_{t_key}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
        saved_paths.append(out_path)

    # Combined overview
    if iou_only:
        t_data = df[df["task"] == "ingredients"]
        fig, ax = plt.subplots(figsize=(10.5, 6.5))
        pct_iou_high = [(t_data[t_data["model"] == m]["iou_max"] >= iou_threshold).mean() * 100.0 for m in models]
        rects = ax.bar(x, pct_iou_high, 0.52, label=f"High Jaccard Distance (≥ {iou_threshold:.2f})", color=COLOR_IOU, alpha=0.92, edgecolor="white", linewidth=1.2)
        for rect, val in zip(rects, pct_iou_high):
            h = rect.get_height()
            ax.annotate(f"{val:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h + 1.5), ha="center", va="bottom", fontsize=11, fontweight="bold", color="#2c3e50")
        ax.set_title(f"RAG Success Rate: Ingredients\nPercentage of High-Overlap Answers (Jaccard Distance ≥ {iou_threshold:.2f})", pad=14)
        ax.set_ylabel(f"% of Answers with Jaccard Distance ≥ {iou_threshold:.2f}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_display_names, rotation=15, ha="right", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 115)
        ax.axhline(100, color="gray", linestyle="--", alpha=0.3)
        plt.tight_layout()
        combined_path = output_dir / "rag_success_rate_by_threshold.png"
        fig.savefig(combined_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {combined_path}")
        saved_paths.append(combined_path)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
        # Left: Ingredients
        ax0 = axes[0]
        ing_data = df[df["task"] == "ingredients"]
        pct_ing_human = [(ing_data[ing_data["model"] == m]["human_score"] >= human_threshold).mean() * 100.0 for m in models]
        pct_ing_iou = [(ing_data[ing_data["model"] == m]["iou_max"] >= iou_threshold).mean() * 100.0 for m in models]
        r1 = ax0.bar(x - 0.18, pct_ing_iou, 0.35, label=f"High Jaccard Distance (≥ {iou_threshold:.2f})", color=COLOR_IOU, alpha=0.92, edgecolor="white", linewidth=1.2)
        r2 = ax0.bar(x + 0.18, pct_ing_human, 0.35, label=f"Human Ground Truth Acceptable (≥ {human_threshold:.2f})", color=COLOR_HUMAN, alpha=0.92, edgecolor="white", linewidth=1.2)
        for rects, vals in [(r1, pct_ing_iou), (r2, pct_ing_human)]:
            for rect, val in zip(rects, vals):
                h = rect.get_height()
                ax0.annotate(f"{val:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h + 1.5), ha="center", va="bottom", fontsize=9, fontweight="bold", color="#2c3e50")
        ax0.set_title("Task: Ingredients", fontsize=13, pad=12)
        ax0.set_ylabel("% of Correct / High-Quality Answers", fontsize=11, fontweight="bold")
        ax0.set_xticks(x)
        ax0.set_xticklabels(model_display_names, rotation=15, ha="right", fontsize=10.5)
        ax0.legend(loc="upper right", frameon=True, fontsize=10)
        ax0.set_ylim(0, 115)
        ax0.axhline(100, color="gray", linestyle="--", alpha=0.3)

        # Right: Directions (Human Only)
        ax1 = axes[1]
        dir_data = df[df["task"] == "directions"]
        pct_dir_human = [(dir_data[dir_data["model"] == m]["human_score"] >= human_threshold).mean() * 100.0 for m in models]
        rects_d = ax1.bar(x, pct_dir_human, 0.52, label=f"Human Ground Truth Acceptable (≥ {human_threshold:.2f})", color=COLOR_HUMAN, alpha=0.92, edgecolor="white", linewidth=1.2)
        for rect, val in zip(rects_d, pct_dir_human):
            h = rect.get_height()
            ax1.annotate(f"{val:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h + 1.5), ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#2c3e50")
        ax1.set_title("Task: Directions (Human Evaluation Only)", fontsize=13, pad=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_display_names, rotation=15, ha="right", fontsize=10.5)
        ax1.legend(loc="upper right", frameon=True, fontsize=10)
        ax1.axhline(100, color="gray", linestyle="--", alpha=0.3)

        fig.suptitle(f"RAG Success Rate: Percentage of High-Quality Answers (Threshold ≥ {human_threshold:.2f})\nIngredients (Jaccard Distance vs. Human) & Directions (Human Evaluation)", fontsize=15, fontweight="bold", y=0.99)
        plt.tight_layout()
        combined_path = output_dir / "rag_success_rate_by_threshold.png"
        fig.savefig(combined_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {combined_path}")
        saved_paths.append(combined_path)

    return saved_paths


def plot_rag_scatter_split(df: pd.DataFrame, output_dir: Path, dpi: int = 300) -> list[Path]:
    """Plot 4 (Split): Generates dedicated scatter plot of Jaccard Distance vs Human Evaluation for Ingredients.
    Omitted for Directions because Jaccard Distance on recipe directions has no semantic sense.
    """
    saved_paths = []
    t_key, t_title = "ingredients", "Ingredients"
    t_data = df[df["task"] == t_key]

    fig, ax = plt.subplots(figsize=(9.5, 7))

    for m in MODEL_ORDER:
        if m not in t_data["model"].unique():
            continue
        sub = t_data[t_data["model"] == m]
        ax.scatter(
            sub["iou_max"],
            sub["human_score"],
            label=MODEL_LABELS.get(m, m),
            color=MODEL_COLORS.get(m, "#333333"),
            alpha=0.70,
            s=64,
            edgecolors="none",
        )

    valid = t_data.dropna(subset=["iou_max", "human_score"])
    if len(valid) > 1:
        z = np.polyfit(valid["iou_max"], valid["human_score"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid["iou_max"].min(), valid["iou_max"].max(), 100)
        y_line = np.clip(p(x_line), 0.0, 1.0)
        corr = valid["iou_max"].corr(valid["human_score"])
        ax.plot(x_line, y_line, color="#2c3e50", linestyle="--", linewidth=2.5, label=f"Trendline (r = {corr:.2f})")

    ax.set_title(f"Correlation: Jaccard Distance vs. Human Evaluation\nTask: {t_title}", pad=14)
    ax.set_xlabel("Max Jaccard Distance (0.0 – 1.0 Scale)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Human Evaluation Score (0.0 – 1.0 Scale)", fontsize=12, fontweight="bold")
    ax.set_xlim(0.15, 1.05)
    ax.set_ylim(-0.05, 1.08)
    ax.legend(loc="upper left", frameon=True, fontsize=11)

    plt.tight_layout()
    out_path = output_dir / f"rag_iou_vs_human_scatter_{t_key}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    saved_paths.append(out_path)

    # Save overview as well (Ingredients is the valid task for Jaccard correlation)
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    for m in MODEL_ORDER:
        if m not in t_data["model"].unique():
            continue
        sub = t_data[t_data["model"] == m]
        ax.scatter(sub["iou_max"], sub["human_score"], label=MODEL_LABELS.get(m, m), color=MODEL_COLORS.get(m, "#333333"), alpha=0.70, s=60, edgecolors="none")
    if len(valid) > 1:
        z = np.polyfit(valid["iou_max"], valid["human_score"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid["iou_max"].min(), valid["iou_max"].max(), 100)
        y_line = np.clip(p(x_line), 0.0, 1.0)
        corr = valid["iou_max"].corr(valid["human_score"])
        ax.plot(x_line, y_line, color="#2c3e50", linestyle="--", linewidth=2.5, label=f"Trendline (r = {corr:.2f})")
    ax.set_title("Correlation: Jaccard Distance vs. Human Quality Score per Recipe\nTask: Ingredients", fontsize=14, pad=12)
    ax.set_xlabel("Max Jaccard Distance (0.0 – 1.0 Scale)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Human Evaluation Score (0.0 – 1.0 Scale)", fontsize=11, fontweight="bold")
    ax.set_xlim(0.15, 1.05)
    ax.set_ylim(-0.05, 1.08)
    ax.legend(loc="upper left", frameon=True, fontsize=10.5)

    plt.tight_layout()
    combined_path = output_dir / "rag_iou_vs_human_scatter.png"
    fig.savefig(combined_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {combined_path}")
    saved_paths.append(combined_path)

    return saved_paths


def plot_rag_iou_violin_split(df: pd.DataFrame, output_dir: Path, dpi: int = 300) -> list[Path]:
    """Plot 4 (Jaccard-Only): Generates violin plots with individual observations for Jaccard Distance on Ingredients.
    Omitted for Directions.
    """
    saved_paths = []
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    ordered_labels = [MODEL_LABELS.get(m, m) for m in models]

    melted = df.copy()
    melted["model_display"] = melted["model"].map(MODEL_LABELS)

    sub = melted[melted["task"] == "ingredients"]

    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    sns.violinplot(
        data=sub,
        x="model_display",
        y="iou_max",
        order=ordered_labels,
        inner=None,
        color=COLOR_IOU,
        alpha=0.35,
        cut=0,
        ax=ax,
    )

    sns.stripplot(
        data=sub,
        x="model_display",
        y="iou_max",
        order=ordered_labels,
        color="#1a365d",
        alpha=0.6,
        size=6,
        jitter=0.22,
        ax=ax,
    )

    means = [sub[sub["model_display"] == lab]["iou_max"].mean() for lab in ordered_labels]
    ax.scatter(
        range(len(ordered_labels)),
        means,
        marker="D",
        color="#c0392b",
        s=60,
        zorder=5,
        label="Mean Score",
    )

    ax.set_title("RAG Jaccard Distance Density & Observations: Ingredients\n(Diamond = Mean, Points = Recipes)", pad=14)
    ax.set_ylabel("Jaccard Distance (0.0 – 1.0 Scale)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Generator LLM", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(ordered_labels)))
    ax.set_xticklabels(ordered_labels, fontsize=11, fontweight="bold")
    ax.set_ylim(-0.05, 1.08)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", frameon=True, fontsize=11)

    plt.tight_layout()
    out_path = output_dir / "rag_iou_scores_violin_ingredients.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    saved_paths.append(out_path)

    # Overview
    combined_path = output_dir / "rag_iou_scores_violin_by_task.png"
    fig.savefig(combined_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {combined_path}")
    saved_paths.append(combined_path)

    return saved_paths


def plot_rag_human_violin_split(df: pd.DataFrame, output_dir: Path, dpi: int = 300) -> list[Path]:
    """Plot 4 (Human-Only): Generates violin plots with individual observations for human scores across tasks."""
    saved_paths = []
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    ordered_labels = [MODEL_LABELS.get(m, m) for m in models]

    melted = df.copy()
    melted["model_display"] = melted["model"].map(MODEL_LABELS)

    for t_key, t_title in TASKS:
        fig, ax = plt.subplots(figsize=(10.5, 6.5))
        sub = melted[melted["task"] == t_key]

        sns.violinplot(
            data=sub,
            x="model_display",
            y="human_score",
            order=ordered_labels,
            inner=None,
            color=COLOR_HUMAN,
            alpha=0.35,
            cut=0,
            ax=ax,
        )

        sns.stripplot(
            data=sub,
            x="model_display",
            y="human_score",
            order=ordered_labels,
            color="#1b4d3e",
            alpha=0.6,
            size=6,
            jitter=0.22,
            ax=ax,
        )

        means = [sub[sub["model_display"] == lab]["human_score"].mean() for lab in ordered_labels]
        ax.scatter(
            range(len(ordered_labels)),
            means,
            marker="D",
            color="#c0392b",
            s=60,
            zorder=5,
            label="Mean Score",
        )

        ax.set_title(f"RAG Human Score Density & Observations: {t_title}\n(Diamond = Mean, Points = Recipes)", pad=14)
        ax.set_ylabel("Human Evaluation Score (0.0 – 1.0 Scale)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Generator LLM", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(ordered_labels)))
        ax.set_xticklabels(ordered_labels, fontsize=11, fontweight="bold")
        ax.set_ylim(-0.05, 1.08)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.35)
        ax.legend(loc="lower left", frameon=True, fontsize=11)

        plt.tight_layout()
        out_path = output_dir / f"rag_human_scores_violin_{t_key}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
        saved_paths.append(out_path)

    # Combined overview (2x1 stacked)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    for ax_idx, (t_key, t_title) in enumerate(TASKS):
        ax = axes[ax_idx]
        sub = melted[melted["task"] == t_key]
        sns.violinplot(
            data=sub,
            x="model_display",
            y="human_score",
            order=ordered_labels,
            inner=None,
            color=COLOR_HUMAN,
            alpha=0.35,
            cut=0,
            ax=ax,
        )
        sns.stripplot(
            data=sub,
            x="model_display",
            y="human_score",
            order=ordered_labels,
            color="#1b4d3e",
            alpha=0.55,
            size=5,
            jitter=0.22,
            ax=ax,
        )
        means = [sub[sub["model_display"] == lab]["human_score"].mean() for lab in ordered_labels]
        ax.scatter(range(len(ordered_labels)), means, marker="D", color="#c0392b", s=50, zorder=5, label="Mean Score")
        ax.set_title(f"Task: {t_title} (Diamond = Mean, Points = Recipes)", fontsize=13, pad=10)
        ax.set_ylabel("Human Score (0.0 – 1.0)", fontsize=11, fontweight="bold")
        ax.set_xlabel("" if ax_idx == 0 else "LLM Model", fontsize=11, fontweight="bold")
        ax.set_ylim(-0.05, 1.08)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.35)
        ax.legend(loc="lower left", frameon=True, fontsize=10)

    fig.suptitle("RAG Human Evaluation Score Density & Individual Recipe Scores\n(Divided by Ingredients and Directions)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    combined_path = output_dir / "rag_human_scores_violin_by_task.png"
    fig.savefig(combined_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {combined_path}")
    saved_paths.append(combined_path)

    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready plots for RAG answer correctness (Jaccard Distance vs Human Evaluation).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for plots")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figures")
    parser.add_argument("--threshold", type=float, default=0.95, help="Correctness threshold on [0.0 - 1.0] scale (default: 0.95)")
    parser.add_argument(
        "--human-only",
        "--only-human",
        action="store_true",
        dest="human_only",
        help="Generate plots using only the Human Evaluation scores (excluding lexical Jaccard metrics)",
    )
    parser.add_argument(
        "--iou-only",
        "--only-iou",
        "--without-human",
        "--no-human",
        action="store_true",
        dest="iou_only",
        help="Generate plots using only the lexical Jaccard Distance scores (excluding Human Evaluation metrics)",
    )
    args = parser.parse_args()

    # Automatically normalize threshold if user passes 95 instead of 0.95
    if args.threshold > 1.0:
        args.threshold /= 100.0

    out_dir = args.output_dir
    if args.human_only and out_dir == DEFAULT_OUTPUT_DIR:
        out_dir = DEFAULT_OUTPUT_DIR / "Human_Only"
    elif args.iou_only and out_dir == DEFAULT_OUTPUT_DIR:
        out_dir = DEFAULT_OUTPUT_DIR / "IoU_Only"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    print(f"Loading RAG evaluation results from {len(DEFAULT_DIRS)} source directories...")
    df = load_rag_data(DEFAULT_DIRS)
    print(f"Loaded {len(df)} total evaluated recipe instances across {df['model'].nunique()} models.")

    if args.human_only:
        print("\n--- Generating Human-Only RAG Correctness Plots ---")
        plot_rag_correctness_split(df, out_dir, args.dpi, human_only=True)
        plot_rag_distributions_split(df, out_dir, args.dpi, human_only=True)
        plot_rag_success_rate_split(
            df,
            out_dir,
            human_threshold=args.threshold,
            iou_threshold=args.threshold,
            dpi=args.dpi,
            human_only=True,
        )
        plot_rag_human_violin_split(df, out_dir, args.dpi)
    elif args.iou_only:
        print("\n--- Generating Jaccard-Only (Without Human Scores) RAG Correctness Plots ---")
        plot_rag_correctness_split(df, out_dir, args.dpi, iou_only=True)
        plot_rag_distributions_split(df, out_dir, args.dpi, iou_only=True)
        plot_rag_success_rate_split(
            df,
            out_dir,
            human_threshold=args.threshold,
            iou_threshold=args.threshold,
            dpi=args.dpi,
            iou_only=True,
        )
        plot_rag_iou_violin_split(df, out_dir, args.dpi)
    else:
        print("\n--- Generating Split & High-Resolution RAG Correctness Plots (Jaccard Distance + Human) ---")
        plot_rag_correctness_split(df, out_dir, args.dpi, human_only=False)
        plot_rag_distributions_split(df, out_dir, args.dpi, human_only=False)
        plot_rag_success_rate_split(
            df,
            out_dir,
            human_threshold=args.threshold,
            iou_threshold=args.threshold,
            dpi=args.dpi,
            human_only=False,
        )
        plot_rag_scatter_split(df, out_dir, args.dpi)

    print("\nAll RAG evaluation plots successfully generated in:", out_dir)


if __name__ == "__main__":
    main()
