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

DEFAULT_JUDGES_CSV = PROJECT_ROOT / "data" / "APICalls" / "API_Max100_judges.csv"
DEFAULT_DATASET_CSV = PROJECT_ROOT / "data" / "judgement_dataset.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "Plots" / "SavedPlots" / "Judges"

# Visual styling tokens
COLOR_IOU = "#2b5c8f"        # Deep Slate Blue
COLOR_HUMAN = "#27ae60"      # Emerald Green
COLOR_AVG_JUDGE = "#34495e"  # Dark Slate Navy (Consensus / Avg Judges)

# Judge Model Palette for API Judges
JUDGE_COLORS = {
    "GEMMA": "#d35400",  # Burnt Orange
    "QWEN": "#16a085",   # Teal
    "LLAMA": "#2980b9",  # Ocean Blue
    "GPT": "#8e44ad",    # Royal Purple
}

JUDGE_LABELS = {
    "GEMMA": "Gemma",
    "QWEN": "Qwen",
    "LLAMA": "Llama",
    "GPT": "GPT",
}

JUDGE_ORDER = ["GEMMA", "QWEN", "LLAMA", "GPT"]

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


def load_and_extract_data(
    judges_csv: Path = DEFAULT_JUDGES_CSV,
    dataset_csv: Path = DEFAULT_DATASET_CSV,
) -> pd.DataFrame:
    """Loads API judge scores from CSV and merges with main dataset to get IoU, Human scores, and generator metadata."""
    judges_csv = Path(judges_csv)
    dataset_csv = Path(dataset_csv)

    if not judges_csv.exists():
        raise FileNotFoundError(f"Judges CSV not found: {judges_csv}")
    if not dataset_csv.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_csv}")

    df_api = pd.read_csv(judges_csv)
    df_main = pd.read_csv(dataset_csv)
    df_main["original_row"] = df_main.index + 2

    # Clean main columns to avoid collisions
    cols_to_keep = [
        "original_row", "Recipe", "Asked", "RAG Model Name",
        "Provided Answer", "Reference Answers", "Human Numeric Score"
    ]
    df_main_clean = df_main[[c for c in cols_to_keep if c in df_main.columns]].copy()

    merged = pd.merge(df_api, df_main_clean, on="original_row", how="inner")

    # Dynamically find judge model columns
    judge_keys = []
    for i in range(1, 10):
        m_col = f"Judge {i} Model Name"
        s_col = f"Judge {i} Numeric Score"
        if m_col in df_api.columns and s_col in df_api.columns:
            m_name = df_api[m_col].dropna().iloc[0]
            judge_keys.append((i, m_name, s_col))

    rows = []
    for _, r in merged.iterrows():
        orig = r.get("Provided Answer", "")
        refs_raw = r.get("Reference Answers", "[]")
        if isinstance(refs_raw, str):
            try:
                refs = json.loads(refs_raw)
            except Exception:
                refs = [refs_raw]
        elif isinstance(refs_raw, list):
            refs = refs_raw
        else:
            refs = []

        max_iou, mean_iou, _ = compute_iou_stats(orig, refs)
        human_score = float(r["Human Numeric Score"]) if pd.notna(r.get("Human Numeric Score")) else np.nan
        rag_model = r.get("RAG Model Name", "unknown")
        rag_model_short = rag_model.split("/")[-1]

        row = {
            "original_row": r["original_row"],
            "target_type": r.get("Asked", "unknown"),
            "gen_model": rag_model,
            "gen_model_short": rag_model_short,
            "recipe_name": r.get("Recipe", ""),
            "human_score": human_score,
            "iou_max": max_iou * 100.0,
            "iou_mean": mean_iou * 100.0,
        }

        judge_scores = []
        for _, m_name, s_col in judge_keys:
            val = float(r[s_col]) if pd.notna(r[s_col]) else np.nan
            row[f"judge_{m_name}"] = val
            if not np.isnan(val):
                judge_scores.append(val)

        row["r1_avg_judge"] = np.mean(judge_scores) if judge_scores else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def plot_overall_task_comparison_split(df: pd.DataFrame, output_dir: Path, dpi: int = 300) -> list[Path]:
    """Plot 1 (Split): Overall benchmark (IoU, Human, Consensus Judges, Individual Judges) for Ingredients and Directions."""
    saved_paths = []
    judge_cols = [c for c in df.columns if c.startswith("judge_")]
    metric_keys = ["iou_max", "human_score", "r1_avg_judge"] + judge_cols
    metric_labels = ["Max IoU (%)", "Human Score", "Consensus Judges"] + [
        JUDGE_LABELS.get(c.replace("judge_", ""), c.replace("judge_", "")) for c in judge_cols
    ]
    colors = [COLOR_IOU, COLOR_HUMAN, COLOR_AVG_JUDGE] + [
        JUDGE_COLORS.get(c.replace("judge_", ""), "#7f8c8d") for c in judge_cols
    ]

    for t_key, t_title in TASKS:
        fig, ax = plt.subplots(figsize=(10.5, 6.5))
        sub = df[df["target_type"] == t_key]

        means = [sub[m].dropna().mean() if len(sub[m].dropna()) > 0 else 0 for m in metric_keys]
        sems = [sub[m].dropna().sem() if len(sub[m].dropna()) > 1 else 0 for m in metric_keys]
        x = np.arange(len(metric_keys))

        rects = ax.bar(
            x,
            means,
            width=0.6,
            yerr=sems,
            capsize=4.5,
            color=colors,
            edgecolor="white",
            linewidth=1.2,
            alpha=0.92,
        )

        for rect, mean, sem in zip(rects, means, sems):
            if mean > 0:
                y_pos = mean + (sem if not np.isnan(sem) else 0) + 1.8
                ax.annotate(
                    f"{mean:.1f}",
                    xy=(rect.get_x() + rect.get_width() / 2, y_pos),
                    ha="center",
                    va="bottom",
                    fontsize=10.5,
                    fontweight="bold",
                    color="#2c3e50",
                )

        ax.set_title(f"Evaluation Benchmark: {t_title}\nIoU Score, Human Ground Truth, and LLM Judges", pad=14)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, rotation=20, ha="right", fontsize=11, fontweight="bold")
        ax.set_ylabel("Score (0 – 100 Scale / % Overlap)", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 115)
        ax.axhline(100, color="gray", linestyle="--", alpha=0.3, linewidth=1)

        plt.tight_layout()
        out_path = output_dir / f"iou_human_judges_{t_key}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
        saved_paths.append(out_path)

    # Combined overview
    fig, ax = plt.subplots(figsize=(13, 7.5))
    tasks_keys = ["ingredients", "directions"]
    task_labels = ["Ingredients", "Directions"]
    n_tasks = len(tasks_keys)
    n_metrics = len(metric_keys)
    x = np.arange(n_tasks)
    width = 0.8 / n_metrics

    for i, (mkey, mlabel, col) in enumerate(zip(metric_keys, metric_labels, colors)):
        means, sems = [], []
        for t in tasks_keys:
            sub = df[df["target_type"] == t][mkey].dropna()
            means.append(sub.mean() if len(sub) > 0 else 0)
            sems.append(sub.sem() if len(sub) > 1 else 0)
        offset = (i - (n_metrics - 1) / 2) * width
        rects = ax.bar(x + offset, means, width, yerr=sems, capsize=4, label=mlabel, color=col, edgecolor="white", linewidth=1.2, alpha=0.92)
        for rect, mean, sem in zip(rects, means, sems):
            if mean > 0:
                y_pos = mean + (sem if not np.isnan(sem) else 0) + 1.5
                ax.annotate(f"{mean:.1f}", xy=(rect.get_x() + rect.get_width() / 2, y_pos), ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#2c3e50")

    ax.set_title("Evaluation Benchmark: IoU Score, Human Score, and LLM Judges\nDivided by Task (Ingredients vs. Directions)", pad=18, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=12, fontweight="bold")
    ax.set_ylabel("Score (0 – 100 Scale / % Overlap)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.axhline(100, color="gray", linestyle="--", alpha=0.3, linewidth=1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=True, facecolor="white", edgecolor="#bdc3c7", fontsize=9.5)
    plt.tight_layout()
    combined_path = output_dir / "iou_human_judges_by_task.png"
    fig.savefig(combined_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {combined_path}")
    saved_paths.append(combined_path)

    return saved_paths


def plot_per_judge_model_breakdown_split(df: pd.DataFrame, output_dir: Path, dpi: int = 300) -> list[Path]:
    """Plot 2 (Split): Dedicated plots comparing each Judge model against Human Ground Truth & IoU for Ingredients and Directions."""
    saved_paths = []
    judge_cols = [c for c in df.columns if c.startswith("judge_")]
    judge_names = [c.replace("judge_", "") for c in judge_cols]
    judge_labels = [JUDGE_LABELS.get(j, j) for j in judge_names]

    x = np.arange(len(judge_cols))
    width = 0.26

    for t_key, t_title in TASKS:
        fig, ax = plt.subplots(figsize=(11, 7))
        t_data = df[df["target_type"] == t_key]

        judge_means, judge_sems = [], []
        human_means, human_sems = [], []
        iou_means, iou_sems = [], []

        for jcol in judge_cols:
            sub = t_data.dropna(subset=[jcol])
            judge_means.append(sub[jcol].mean())
            judge_sems.append(sub[jcol].sem())
            human_means.append(sub["human_score"].mean())
            human_sems.append(sub["human_score"].sem())
            iou_means.append(sub["iou_max"].mean())
            iou_sems.append(sub["iou_max"].sem())

        r_judge = ax.bar(
            x - width,
            judge_means,
            width,
            yerr=judge_sems,
            capsize=4,
            label="Judge Model Score",
            color=COLOR_AVG_JUDGE,
            alpha=0.92,
            edgecolor="white",
            linewidth=1.2,
        )

        r_human = ax.bar(
            x,
            human_means,
            width,
            yerr=human_sems,
            capsize=4,
            label="Human Ground Truth",
            color=COLOR_HUMAN,
            alpha=0.92,
            edgecolor="white",
            linewidth=1.2,
        )

        r_iou = ax.bar(
            x + width,
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

        # Annotations above whiskers
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
                        fontsize=9.5,
                        fontweight="bold",
                        color="#2c3e50",
                    )

        # Delta callouts (Judge - Human)
        for i_j, (jm, hm) in enumerate(zip(judge_means, human_means)):
            delta = jm - hm
            sign = "+" if delta >= 0 else ""
            d_color = "#c0392b" if delta < -5 else ("#e67e22" if delta > 5 else "#27ae60")
            ax.text(
                i_j - width / 2,
                110,
                f"Δ = {sign}{delta:.1f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=d_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9f9", edgecolor=d_color, alpha=0.95),
            )

        ax.set_title(f"LLM Judges vs. Human Ground Truth & IoU: {t_title}\n(Δ = Judge Score − Human Score)", pad=14)
        ax.set_xticks(x)
        ax.set_xticklabels(judge_labels, fontsize=11, fontweight="bold")
        ax.set_ylabel("Score (0 – 100 Scale / % Overlap)", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 120)
        ax.axhline(100, color="gray", linestyle="--", alpha=0.3)
        ax.legend(loc="lower left", fontsize=10.5, frameon=True)

        plt.tight_layout()
        out_path = output_dir / f"judge_vs_human_by_model_{t_key}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
        saved_paths.append(out_path)

    # Retain 2x2 grid as combined overview
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharey=True)
    axes = axes.flatten()
    task_keys_list = ["ingredients", "directions"]
    task_labels_list = ["Ingredients", "Directions"]
    for idx, jcol in enumerate(judge_cols):
        ax = axes[idx]
        j_key = jcol.replace("judge_", "")
        j_label = JUDGE_LABELS.get(j_key, j_key)
        j_color = JUDGE_COLORS.get(j_key, "#2980b9")
        sub_df = df.dropna(subset=[jcol]).copy()
        x_g = np.arange(len(task_keys_list))
        w_g = 0.24
        jm_l, hm_l, im_l = [], [], []
        js_l, hs_l, is_l = [], [], []
        for t in task_keys_list:
            t_data = sub_df[sub_df["target_type"] == t]
            jm_l.append(t_data[jcol].mean())
            js_l.append(t_data[jcol].sem())
            hm_l.append(t_data["human_score"].mean())
            hs_l.append(t_data["human_score"].sem())
            im_l.append(t_data["iou_max"].mean())
            is_l.append(t_data["iou_max"].sem())
        ax.bar(x_g - w_g, jm_l, w_g, yerr=js_l, capsize=4, label=f"Judge ({j_label})", color=j_color, alpha=0.9, edgecolor="white")
        ax.bar(x_g, hm_l, w_g, yerr=hs_l, capsize=4, label="Human Ground Truth", color=COLOR_HUMAN, alpha=0.9, edgecolor="white")
        ax.bar(x_g + w_g, im_l, w_g, yerr=is_l, capsize=4, label="IoU Score (%)", color=COLOR_IOU, alpha=0.9, edgecolor="white")
        for t_i, (jm, hm) in enumerate(zip(jm_l, hm_l)):
            delta = jm - hm
            sign = "+" if delta >= 0 else ""
            d_color = "#c0392b" if delta < -5 else ("#e67e22" if delta > 5 else "#27ae60")
            ax.text(t_i, 108, f"Δ(Judge - Human) = {sign}{delta:.1f}", ha="center", va="center", fontsize=9.5, fontweight="bold", color=d_color, bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9f9", edgecolor=d_color, alpha=0.95))
        ax.set_title(f"Judge: {j_label} (Evaluated n={len(sub_df)})", fontsize=12, pad=12)
        ax.set_xticks(x_g)
        ax.set_xticklabels(task_labels_list, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 120)
        ax.axhline(100, color="gray", linestyle="--", alpha=0.3)
        ax.legend(loc="lower left", fontsize=8.5, frameon=True)
    fig.suptitle("Individual Judge Model Analysis vs. Human Ground Truth & IoU Score\n(Divided by Ingredients and Directions)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    combined_path = output_dir / "judge_vs_human_iou_by_judge_model.png"
    fig.savefig(combined_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {combined_path}")
    saved_paths.append(combined_path)

    return saved_paths


def plot_generator_models_breakdown_split(df: pd.DataFrame, output_dir: Path, dpi: int = 300) -> list[Path]:
    """Plot 3 (Split): Generator models evaluated by IoU, Human, and Consensus Judges for Ingredients and Directions."""
    saved_paths = []
    gen_models = [m for m in MODEL_ORDER if m in df["gen_model_short"].unique()]
    x = np.arange(len(gen_models))
    width = 0.26

    for t_key, t_title in TASKS:
        fig, ax = plt.subplots(figsize=(10.5, 6.5))
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
            capsize=3.5,
            label="IoU Score (%)",
            color=COLOR_IOU,
            alpha=0.92,
            edgecolor="white",
            linewidth=1.2,
        )
        r2 = ax.bar(
            x,
            human_vals,
            width,
            yerr=human_err,
            capsize=3.5,
            label="Human Score",
            color=COLOR_HUMAN,
            alpha=0.92,
            edgecolor="white",
            linewidth=1.2,
        )
        r3 = ax.bar(
            x + width,
            judge_vals,
            width,
            yerr=judge_err,
            capsize=3.5,
            label="Consensus Judges",
            color=COLOR_AVG_JUDGE,
            alpha=0.92,
            edgecolor="white",
            linewidth=1.2,
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
                        fontsize=9.5,
                        fontweight="bold",
                        color="#2c3e50",
                    )

        ax.set_title(f"Generator Models Performance: {t_title}\nIoU Score vs. Human Score vs. Consensus Judges", pad=14)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in gen_models], rotation=15, ha="right", fontsize=11, fontweight="bold")
        ax.set_ylabel("Score (0 – 100)", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 118)
        ax.axhline(100, color="gray", linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=10.5, frameon=True)

        plt.tight_layout()
        out_path = output_dir / f"generator_models_{t_key}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
        saved_paths.append(out_path)

    # Combined overview
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax_idx, (t_key, t_title) in enumerate(TASKS):
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
        r1 = ax.bar(x - width, iou_vals, width, yerr=iou_err, capsize=3, label="IoU Score (%)", color=COLOR_IOU, alpha=0.9, edgecolor="white")
        r2 = ax.bar(x, human_vals, width, yerr=human_err, capsize=3, label="Human Score", color=COLOR_HUMAN, alpha=0.9, edgecolor="white")
        r3 = ax.bar(x + width, judge_vals, width, yerr=judge_err, capsize=3, label="Consensus Judges", color=COLOR_AVG_JUDGE, alpha=0.9, edgecolor="white")
        for rects, errs in [(r1, iou_err), (r2, human_err), (r3, judge_err)]:
            for r, err in zip(rects, errs):
                h = r.get_height()
                if h > 0:
                    y_pos = h + (err if not np.isnan(err) else 0) + 1.8
                    ax.annotate(f"{h:.1f}", xy=(r.get_x() + r.get_width() / 2, y_pos), ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        ax.set_title(f"Task: {t_title}", fontsize=13, pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in gen_models], rotation=15, ha="right", fontsize=10.5)
        ax.set_ylabel("Score (0 – 100)" if ax_idx == 0 else "", fontsize=11)
        ax.set_ylim(0, 118)
        ax.axhline(100, color="gray", linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=9.5, frameon=True)
    fig.suptitle("Performance Across Generator Models: IoU vs. Human vs. Consensus Judges", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    combined_path = output_dir / "generator_models_iou_human_judges.png"
    fig.savefig(combined_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {combined_path}")
    saved_paths.append(combined_path)

    return saved_paths


def plot_correlation_heatmaps_split(df: pd.DataFrame, output_dir: Path, dpi: int = 300) -> list[Path]:
    """Plot 4 (Split): Generates dedicated, large correlation heatmaps for Ingredients and Directions."""
    saved_paths = []
    judge_cols = [c for c in df.columns if c.startswith("judge_")]
    cols_to_corr = ["human_score", "iou_max", "r1_avg_judge"] + judge_cols
    col_labels = ["Human", "IoU Max", "Consensus"] + [
        JUDGE_LABELS.get(c.replace("judge_", ""), c.replace("judge_", "")) for c in judge_cols
    ]

    for t_key, t_title in TASKS:
        fig, ax = plt.subplots(figsize=(8.5, 7))
        sub = df[df["target_type"] == t_key][cols_to_corr].copy()
        sub.columns = col_labels

        corr = sub.corr(method="pearson")
        mask = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 11, "weight": "bold"},
            cmap="vlag",
            vmin=-0.5,
            vmax=1.0,
            center=0.0,
            linewidths=1.0,
            cbar=True,
            cbar_kws={"label": "Pearson Correlation (r)"},
            ax=ax,
        )

        ax.set_title(f"Metric Alignment Matrix: {t_title}\n(Human Ground Truth vs. IoU vs. LLM Judges)", pad=14)
        plt.tight_layout()
        out_path = output_dir / f"correlation_matrix_{t_key}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
        saved_paths.append(out_path)

    # Combined overview
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax_idx, (t_key, t_title) in enumerate(TASKS):
        ax = axes[ax_idx]
        sub = df[df["target_type"] == t_key][cols_to_corr].copy()
        sub.columns = col_labels
        corr = sub.corr(method="pearson")
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="vlag", vmin=-0.5, vmax=1.0, center=0.0, linewidths=0.8, cbar=ax_idx == 1, cbar_kws={"label": "Pearson Correlation (r)"}, ax=ax)
        ax.set_title(f"Metric Alignment Matrix: {t_title}", fontsize=13, pad=12)
    fig.suptitle("Correlation & Alignment: Human Ground Truth vs. IoU vs. LLM Judges", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    combined_path = output_dir / "iou_judges_human_correlation_matrix.png"
    fig.savefig(combined_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {combined_path}")
    saved_paths.append(combined_path)

    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready plots for IoU, Human and LLM Judge scores using API judge results.")
    parser.add_argument("--judges-csv", type=Path, default=DEFAULT_JUDGES_CSV, help="Path to API judges CSV file")
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET_CSV, help="Path to main judgement dataset CSV file")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for plots")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figures")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    print(f"Loading API judge scores from: {args.judges_csv}")
    print(f"Loading ground truth and recipes from: {args.dataset_csv}")
    df = load_and_extract_data(args.judges_csv, args.dataset_csv)
    print(f"Loaded {len(df)} evaluated instances across {df['gen_model_short'].nunique()} generator models and {len([c for c in df.columns if c.startswith('judge_')])} judge models.")

    print("\n--- Generating Split & High-Resolution Judge Comparison Plots ---")
    plot_overall_task_comparison_split(df, args.output_dir, args.dpi)
    plot_per_judge_model_breakdown_split(df, args.output_dir, args.dpi)
    plot_generator_models_breakdown_split(df, args.output_dir, args.dpi)
    plot_correlation_heatmaps_split(df, args.output_dir, args.dpi)

    print("\nAll judge comparison plots successfully generated in:", args.output_dir)


if __name__ == "__main__":
    main()
