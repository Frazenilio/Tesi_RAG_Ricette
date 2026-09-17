import os
import sys
import json
import argparse
from pathlib import Path
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths relative to script location
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "bias_tests"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "Plots" / "SavedPlots" / "Biases"

# Color Palette for Thesis Plots
PALETTE_PRIMARY = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77", "#e7298a", "#66a61e"]
COLOR_LLAMA = "#2b5c8f"      # Slate blue
COLOR_GEMMA = "#d95f02"      # Amber orange
COLOR_NEUTRAL = "#4a7c59"    # Forest green
COLOR_AI = "#c0392b"         # Soft crimson
COLOR_HUMAN = "#2980b9"      # Ocean blue
COLOR_ACCENT = "#8e44ad"     # Violet

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
        "figure.titleweight": "bold"
    })

def find_latest_file(directory: Path, pattern: str) -> Path | None:
    """Finds the latest JSON file matching a glob pattern."""
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None

def load_all_bias_data(results_dir: Path) -> dict:
    """Discovers and loads the latest result JSON files for all bias tests."""
    results_dir = Path(results_dir)
    data = {}

    # 1. Self-Enchantment
    f_se_llama = find_latest_file(results_dir, "self_enchantment_llama_*.json")
    f_se_gemma = find_latest_file(results_dir, "self_enchantment_gemma_*.json")
    if f_se_llama and f_se_llama.exists():
        with open(f_se_llama, "r", encoding="utf-8") as f:
            data["self_enchantment_llama"] = json.load(f)
    if f_se_gemma and f_se_gemma.exists():
        with open(f_se_gemma, "r", encoding="utf-8") as f:
            data["self_enchantment_gemma"] = json.load(f)

    # 2. Compassion-Fade / Attribution Bias
    f_cf_llama = find_latest_file(results_dir, "compassion_fade_llama_*.json")
    f_cf_gemma = find_latest_file(results_dir, "compassion_fade_gemma_*.json")
    if f_cf_llama and f_cf_llama.exists():
        with open(f_cf_llama, "r", encoding="utf-8") as f:
            data["compassion_fade_llama"] = json.load(f)
    if f_cf_gemma and f_cf_gemma.exists():
        with open(f_cf_gemma, "r", encoding="utf-8") as f:
            data["compassion_fade_gemma"] = json.load(f)

    # 3. Length Bias
    f_len = find_latest_file(results_dir, "length_bias_*.json")
    if f_len and f_len.exists():
        with open(f_len, "r", encoding="utf-8") as f:
            data["length_bias"] = json.load(f)

    # 4. Position Bias
    f_pos = find_latest_file(results_dir, "position_bias_*.json")
    if f_pos and f_pos.exists():
        with open(f_pos, "r", encoding="utf-8") as f:
            data["position_bias"] = json.load(f)

    return data


# ==============================================================================
# PLOT 1: Self-Enchantment Bias
# ==============================================================================
def plot_self_enchantment(data_llama: dict | None, data_gemma: dict | None, output_dir: Path) -> Path | None:
    """Plots Self-Enchantment comparison (Self-score vs Other-score) across judges."""
    if not data_llama and not data_gemma:
        print("Warning: No Self-Enchantment data found to plot.")
        return None

    setup_style()
    fig, (ax_bar, ax_box) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.1, 1]})

    bar_records = []
    box_records = []

    configs = [
        ("LLaMA-3.2-3B", data_llama),
        ("Gemma-3-1B", data_gemma)
    ]

    for judge_label, data in configs:
        if not data:
            continue
        summary = data.get("summary", {})
        mean_self = summary.get("mean_self_score", 0.0)
        mean_other = summary.get("mean_other_score", 0.0)
        mean_delta = summary.get("mean_delta", 0.0)

        bar_records.append({"Judge": judge_label, "Target": "Self Answer", "Score": mean_self, "Delta": mean_delta})
        bar_records.append({"Judge": judge_label, "Target": "Other Model Answer", "Score": mean_other, "Delta": mean_delta})

        for r in data.get("results", []):
            s_val = r.get("self_evaluation", {}).get("score", np.nan)
            o_val = r.get("other_evaluation", {}).get("score", np.nan)
            d_val = r.get("delta", np.nan)
            box_records.append({"Judge": judge_label, "Score": s_val, "Type": "Self"})
            box_records.append({"Judge": judge_label, "Score": o_val, "Type": "Other"})
            box_records.append({"Judge": judge_label, "Score": d_val, "Type": "Delta (Self - Other)"})

    df_bar = pd.DataFrame(bar_records)
    df_box = pd.DataFrame(box_records)

    # Panel 1: Grouped Bar Chart of Means
    x = np.arange(len(df_bar["Judge"].unique()))
    width = 0.32

    judges = df_bar["Judge"].unique()
    self_scores = [df_bar[(df_bar["Judge"] == j) & (df_bar["Target"] == "Self Answer")]["Score"].values[0] for j in judges]
    other_scores = [df_bar[(df_bar["Judge"] == j) & (df_bar["Target"] == "Other Model Answer")]["Score"].values[0] for j in judges]
    deltas = [df_bar[(df_bar["Judge"] == j) & (df_bar["Target"] == "Self Answer")]["Delta"].values[0] for j in judges]

    rects1 = ax_bar.bar(x - width/2, self_scores, width, label="Self Answer", color="#2b5c8f", edgecolor="black", linewidth=0.8)
    rects2 = ax_bar.bar(x + width/2, other_scores, width, label="Other Model Answer", color="#7f9db9", edgecolor="black", linewidth=0.8)

    # Annotate bar values and deltas
    for rect in rects1:
        h = rect.get_height()
        ax_bar.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", weight="bold", fontsize=10)
    for rect in rects2:
        h = rect.get_height()
        ax_bar.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", weight="bold", fontsize=10)

    # Annotate delta badge
    for idx, (jx, delta) in enumerate(zip(x, deltas)):
        sign = "+" if delta > 0 else ""
        badge_color = "#27ae60" if delta > 0 else "#c0392b" if delta < 0 else "#7f8c8d"
        ax_bar.text(jx, 95,
                    f"Δ(Self - Other) = {sign}{delta:.2f}",
                    ha="center", va="center", color=badge_color, weight="bold", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec=badge_color, lw=1.2))

    ax_bar.set_ylabel("Mean Judge Score (0 - 100)")
    ax_bar.set_title("Mean Scores: Self vs. Other Generations")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(judges, weight="bold")
    ax_bar.set_ylim(0, 115)
    ax_bar.legend(loc="upper left")

    # Panel 2: Distribution of Per-Question Deltas
    df_delta = df_box[df_box["Type"] == "Delta (Self - Other)"]
    palette_delta = {"LLaMA-3.2-3B": "#2b5c8f", "Gemma-3-1B": "#d95f02"}
    
    sns.boxplot(data=df_delta, x="Judge", y="Score", hue="Judge", ax=ax_box, palette=palette_delta, legend=False, width=0.45, boxprops=dict(alpha=0.85))
    sns.stripplot(data=df_delta, x="Judge", y="Score", ax=ax_box, color="black", alpha=0.5, jitter=0.2, size=5)
    ax_box.axhline(0, color="gray", linestyle="--", linewidth=1.2, label="Zero Bias (Δ = 0)")

    ax_box.set_ylabel("Score Delta (Self - Other)")
    ax_box.set_xlabel("")
    ax_box.set_title("Distribution of Score Deltas (Per Recipe)")
    ax_box.legend(loc="upper right")

    fig.suptitle("Self-Enchantment Bias Analysis across LLM Judges", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()

    out_file = output_dir / "self_enchantment_comparison.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")
    return out_file


# ==============================================================================
# PLOT 2: Compassion-Fade / Attribution Bias
# ==============================================================================
def plot_compassion_fade(data_llama: dict | None, data_gemma: dict | None, output_dir: Path) -> Path | None:
    """Plots Compassion-Fade / Attribution Framing effect (Neutral vs AI vs Human)."""
    if not data_llama and not data_gemma:
        print("Warning: No Compassion-Fade data found to plot.")
        return None

    setup_style()
    fig, (ax_main, ax_delta) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.3, 1]})

    bar_records = []
    configs = [
        ("LLaMA-3.2-3B Judge", data_llama),
        ("Gemma-3-1B Judge", data_gemma)
    ]

    for label, d in configs:
        if not d:
            continue
        s = d.get("summary", {})
        bar_records.append({"Judge": label, "Attribution": "Neutral\n(No Label)", "Score": s.get("mean_neutral_score", 0.0), "Type": "Neutral"})
        bar_records.append({"Judge": label, "Attribution": "AI Agent\n(Framed as AI)", "Score": s.get("mean_ai_score", 0.0), "Type": "AI"})
        bar_records.append({"Judge": label, "Attribution": "Chef Human\n(Framed as Human)", "Score": s.get("mean_human_score", 0.0), "Type": "Human"})

    df_cf = pd.DataFrame(bar_records)
    judges = df_cf["Judge"].unique()

    # Panel 1: Grouped bar chart of the 3 conditions
    x = np.arange(len(judges))
    width = 0.25

    neutral_vals = [df_cf[(df_cf["Judge"] == j) & (df_cf["Type"] == "Neutral")]["Score"].values[0] for j in judges]
    ai_vals = [df_cf[(df_cf["Judge"] == j) & (df_cf["Type"] == "AI")]["Score"].values[0] for j in judges]
    human_vals = [df_cf[(df_cf["Judge"] == j) & (df_cf["Type"] == "Human")]["Score"].values[0] for j in judges]

    rects_n = ax_main.bar(x - width, neutral_vals, width, label="Neutral (Unattributed)", color="#7f8c8d", edgecolor="black", linewidth=0.8)
    rects_ai = ax_main.bar(x, ai_vals, width, label="AI Agent Attributed", color="#c0392b", edgecolor="black", linewidth=0.8)
    rects_h = ax_main.bar(x + width, human_vals, width, label="Human Chef Attributed", color="#2980b9", edgecolor="black", linewidth=0.8)

    for rects in [rects_n, rects_ai, rects_h]:
        for rect in rects:
            h = rect.get_height()
            ax_main.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width()/2, h),
                             xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", weight="bold", fontsize=9.5)

    ax_main.set_ylabel("Mean Judge Score (0 - 100)")
    ax_main.set_title("Evaluation Score by Source Attribution Label")
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(judges, weight="bold")
    ax_main.set_ylim(0, 105)
    ax_main.legend(loc="lower right")

    # Panel 2: Framing Deltas (AI Penalty vs Human Favor relative to Neutral)
    delta_data = []
    for label, d in configs:
        if not d:
            continue
        s = d.get("summary", {})
        delta_data.append({
            "Judge": label.replace(" Judge", ""),
            "Metric": "AI Penalty\n(AI - Neutral)",
            "Delta": s.get("delta_ai_vs_neutral", 0.0)
        })
        delta_data.append({
            "Judge": label.replace(" Judge", ""),
            "Metric": "Human Favor\n(Human - Neutral)",
            "Delta": s.get("delta_human_vs_neutral", 0.0)
        })

    df_delta = pd.DataFrame(delta_data)
    unique_metrics = df_delta["Metric"].unique()
    x2 = np.arange(len(df_delta["Judge"].unique()))
    w2 = 0.35

    delta_ai = [df_delta[(df_delta["Judge"] == j) & (df_delta["Metric"] == "AI Penalty\n(AI - Neutral)")]["Delta"].values[0] for j in df_delta["Judge"].unique()]
    delta_h = [df_delta[(df_delta["Judge"] == j) & (df_delta["Metric"] == "Human Favor\n(Human - Neutral)")]["Delta"].values[0] for j in df_delta["Judge"].unique()]

    r_dai = ax_delta.bar(x2 - w2/2, delta_ai, w2, label="AI vs. Neutral", color="#e74c3c", edgecolor="black", linewidth=0.8)
    r_dh = ax_delta.bar(x2 + w2/2, delta_h, w2, label="Human vs. Neutral", color="#3498db", edgecolor="black", linewidth=0.8)

    ax_delta.axhline(0, color="black", linewidth=1.2)

    for rect in r_dai:
        h = rect.get_height()
        va = "bottom" if h >= 0 else "top"
        y_offset = 4 if h >= 0 else -4
        ax_delta.annotate(f"{h:+.2f}", xy=(rect.get_x() + rect.get_width()/2, h),
                          xytext=(0, y_offset), textcoords="offset points", ha="center", va=va, weight="bold", fontsize=10)
    for rect in r_dh:
        h = rect.get_height()
        va = "bottom" if h >= 0 else "top"
        y_offset = 4 if h >= 0 else -4
        ax_delta.annotate(f"{h:+.2f}", xy=(rect.get_x() + rect.get_width()/2, h),
                          xytext=(0, y_offset), textcoords="offset points", ha="center", va=va, weight="bold", fontsize=10)

    ax_delta.set_ylabel("Score Delta (Points)")
    ax_delta.set_title("Attribution Impact Relative to Neutral")
    ax_delta.set_xticks(x2)
    ax_delta.set_xticklabels(df_delta["Judge"].unique(), weight="bold")
    ax_delta.set_ylim(min(delta_ai + delta_h) - 3, max(delta_ai + delta_h) + 4)
    ax_delta.legend(loc="best")

    fig.suptitle("Compassion-Fade / Attribution Bias in LLM Evaluators", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()

    out_file = output_dir / "compassion_fade_attribution.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")
    return out_file


# ==============================================================================
# PLOT 3: Length / Verbosity Bias Overview
# ==============================================================================
def plot_length_bias(data_length: dict | None, output_dir: Path) -> Path | None:
    """Plots Length/Verbosity Bias: Longest Answer Win Rate and Correlation with Length."""
    if not data_length:
        print("Warning: No Length Bias data found to plot.")
        return None

    setup_style()
    fig, (ax_win, ax_corr) = plt.subplots(1, 2, figsize=(14, 6))

    by_judge = data_length.get("summary_by_judge", {})
    records = []

    # Rename keys for clear presentation
    label_map = {
        "gemma3:1b": "Gemma-3-1B",
        "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF": "LLaMA-3.2-3B",
        "ministral-3:3b": "Ministral-3-3B",
        "qwen3.5:2b": "Qwen-3.5-2B",
        "LLM_Average": "LLM Consensus Avg",
        "Human": "Human Feedback"
    }

    for raw_name, s in by_judge.items():
        clean_name = label_map.get(raw_name, raw_name)
        records.append({
            "Evaluator": clean_name,
            "IsHuman": (clean_name == "Human Feedback"),
            "LongestWinRate": s.get("longest_top_score_percentage", 0.0),
            "ExclWinRate": s.get("longest_exclusive_top_score_percentage", 0.0),
            "PearsonR": s.get("pearson_correlation_chars", 0.0),
            "SpearmanR": s.get("spearman_correlation_chars", 0.0),
            "ScoreDelta": s.get("delta_longest_vs_shortest", 0.0)
        })

    df = pd.DataFrame(records)
    # Order: LLMs first, then LLM Average, then Human
    eval_order = ["Gemma-3-1B", "LLaMA-3.2-3B", "Ministral-3-3B", "Qwen-3.5-2B", "LLM Consensus Avg", "Human Feedback"]
    df["Order"] = df["Evaluator"].map(lambda e: eval_order.index(e) if e in eval_order else 99)
    df = df.sort_values("Order").reset_index(drop=True)

    # Panel 1: Longest Answer Win %
    colors = ["#2b5c8f" if not is_h else "#c0392b" for is_h in df["IsHuman"]]
    bars1 = ax_win.barh(df["Evaluator"], df["LongestWinRate"], color=colors, edgecolor="black", linewidth=0.8, alpha=0.85)

    for bar in bars1:
        w = bar.get_width()
        ax_win.annotate(f"{w:.1f}%", xy=(w, bar.get_y() + bar.get_height()/2),
                        xytext=(5, 0), textcoords="offset points", ha="left", va="center", weight="bold", fontsize=10)

    ax_win.set_xlabel("% of Questions where Longest Answer Won Highest Score")
    ax_win.set_title("Verbosity Preference: Longest Answer Win Rate")
    ax_win.set_xlim(0, 100)
    ax_win.invert_yaxis()

    # Panel 2: Pearson Correlation with Character Length
    corr_colors = ["#27ae60" if r > 0 else "#e74c3c" for r in df["PearsonR"]]
    bars2 = ax_corr.barh(df["Evaluator"], df["PearsonR"], color=corr_colors, edgecolor="black", linewidth=0.8, alpha=0.85)

    ax_corr.axvline(0, color="black", linewidth=1.2)

    for bar in bars2:
        w = bar.get_width()
        ha = "right" if w < 0 else "left"
        x_off = -5 if w < 0 else 5
        ax_corr.annotate(f"{w:+.3f}", xy=(w, bar.get_y() + bar.get_height()/2),
                         xytext=(x_off, 0), textcoords="offset points", ha=ha, va="center", weight="bold", fontsize=10)

    ax_corr.set_xlabel("Pearson Correlation (r) between Score and Answer Length")
    ax_corr.set_title("Correlation: Score vs. Character Length")
    ax_corr.set_xlim(-1.0, 0.2)
    ax_corr.invert_yaxis()

    fig.suptitle("Length (Verbosity) Bias in LLM Judges vs. Human Feedback", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()

    out_file = output_dir / "length_bias_overview.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")
    return out_file


# ==============================================================================
# PLOT 4: Length Bias Breakdown by Task Type (Directions vs. Ingredients)
# ==============================================================================
def plot_length_bias_by_task(data_length: dict | None, output_dir: Path) -> Path | None:
    """Plots Length Bias win rate and delta broken down by Directions vs Ingredients."""
    if not data_length:
        return None

    setup_style()
    fig, (ax_dir, ax_ing) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    by_task = data_length.get("summary_by_task_type", {})
    dirs_dict = by_task.get("directions", {})
    ingr_dict = by_task.get("ingredients", {})

    label_map = {
        "gemma3:1b": "Gemma-3-1B",
        "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF": "LLaMA-3.2-3B",
        "ministral-3:3b": "Ministral-3-3B",
        "qwen3.5:2b": "Qwen-3.5-2B",
        "LLM_Average": "LLM Consensus",
        "Human": "Human Feedback"
    }
    eval_order = ["Gemma-3-1B", "LLaMA-3.2-3B", "Ministral-3-3B", "Qwen-3.5-2B", "LLM Consensus", "Human Feedback"]

    def build_df(source_dict):
        rows = []
        for k, s in source_dict.items():
            name = label_map.get(k, k)
            rows.append({
                "Evaluator": name,
                "WinRate": s.get("longest_top_score_percentage", 0.0),
                "Delta": s.get("delta_longest_vs_shortest", 0.0),
                "IsHuman": (name == "Human Feedback")
            })
        d = pd.DataFrame(rows)
        d["Order"] = d["Evaluator"].map(lambda e: eval_order.index(e) if e in eval_order else 99)
        return d.sort_values("Order").reset_index(drop=True)

    df_dirs = build_df(dirs_dict)
    df_ingr = build_df(ingr_dict)

    # Plot Directions
    colors_d = ["#2b5c8f" if not h else "#c0392b" for h in df_dirs["IsHuman"]]
    bars_d = ax_dir.barh(df_dirs["Evaluator"], df_dirs["WinRate"], color=colors_d, edgecolor="black", linewidth=0.8, alpha=0.85)
    for bar, delta in zip(bars_d, df_dirs["Delta"]):
        w = bar.get_width()
        ax_dir.annotate(f"{w:.1f}%  (Δ {delta:+.1f})", xy=(w, bar.get_y() + bar.get_height()/2),
                        xytext=(5, 0), textcoords="offset points", ha="left", va="center", weight="bold", fontsize=9)
    ax_dir.set_title("Task: Recipe Directions (53 Questions)")
    ax_dir.set_xlabel("Longest Win % (and Δ Longest vs Shortest)")
    ax_dir.set_xlim(0, 115)
    ax_dir.invert_yaxis()

    # Plot Ingredients
    colors_i = ["#2b5c8f" if not h else "#c0392b" for h in df_ingr["IsHuman"]]
    bars_i = ax_ing.barh(df_ingr["Evaluator"], df_ingr["WinRate"], color=colors_i, edgecolor="black", linewidth=0.8, alpha=0.85)
    for bar, delta in zip(bars_i, df_ingr["Delta"]):
        w = bar.get_width()
        ax_ing.annotate(f"{w:.1f}%  (Δ {delta:+.1f})", xy=(w, bar.get_y() + bar.get_height()/2),
                        xytext=(5, 0), textcoords="offset points", ha="left", va="center", weight="bold", fontsize=9)
    ax_ing.set_title("Task: Recipe Ingredients (53 Questions)")
    ax_ing.set_xlabel("Longest Win % (and Δ Longest vs Shortest)")
    ax_ing.set_xlim(0, 115)

    fig.suptitle("Task Sensitivity of Length Bias: Directions vs. Ingredients", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()

    out_file = output_dir / "length_bias_by_task.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")
    return out_file


# ==============================================================================
# PLOT 5: Position Bias Impact
# ==============================================================================
def plot_position_bias(data_position: dict | None, output_dir: Path) -> Path | None:
    """Plots Position Bias: Sequence Similarity between original and shuffled answers."""
    if not data_position:
        print("Warning: No Position Bias data found to plot.")
        return None

    setup_style()
    results = data_position.get("results", [])
    if not results:
        return None

    similarities = []
    chunk0_preferred_orig = 0
    chunk0_preferred_shuf = 0

    for r in results:
        orig = r.get("original_answer", "").strip()
        shuf = r.get("shuffled_answer", "").strip()
        sm = SequenceMatcher(None, orig, shuf).ratio()
        similarities.append(sm)

        orig_refs = r.get("reference_answers", [])
        shuf_refs = r.get("shuffled_reference_answers", [])

        if orig_refs:
            best_orig_idx = max(range(len(orig_refs)), key=lambda i: SequenceMatcher(None, orig, orig_refs[i]).ratio())
            if best_orig_idx == 0:
                chunk0_preferred_orig += 1

        if shuf_refs:
            best_shuf_idx = max(range(len(shuf_refs)), key=lambda i: SequenceMatcher(None, shuf, shuf_refs[i]).ratio())
            if best_shuf_idx == 0:
                chunk0_preferred_shuf += 1

    total_q = len(results)
    mean_sim = np.mean(similarities)
    exact_matches = sum(1 for s in similarities if s >= 0.999)

    fig, (ax_hist, ax_primacy) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.2, 0.8]})

    # Panel 1: Sequence Similarity Distribution
    sns.histplot(similarities, bins=12, kde=True, color="#2b5c8f", ax=ax_hist, edgecolor="black", alpha=0.7)
    ax_hist.axvline(mean_sim, color="#c0392b", linestyle="--", linewidth=2, label=f"Mean Similarity = {mean_sim:.2f}")
    ax_hist.axvline(1.0, color="gray", linestyle=":", linewidth=1.5, label=f"Identical Outputs ({exact_matches}/{total_q})")

    ax_hist.set_xlabel("Sequence Similarity (Original vs. Shuffled Answer)")
    ax_hist.set_ylabel("Count of Queries")
    ax_hist.set_title("Sensitivity of Output Generation to Document Order")
    ax_hist.set_xlim(0.0, 1.05)
    ax_hist.legend(loc="upper right")

    # Panel 2: Primacy Effect (First Chunk Preference)
    p_orig = (chunk0_preferred_orig / total_q) * 100
    p_shuf = (chunk0_preferred_shuf / total_q) * 100

    categories = ["Original Order\n(Rank 1 Chunk)", "Shuffled Order\n(Arbitrary Chunk 0)"]
    pcts = [p_orig, p_shuf]
    bars = ax_primacy.bar(categories, pcts, color=["#2b5c8f", "#e67e22"], edgecolor="black", linewidth=0.8, width=0.5, alpha=0.85)

    for bar in bars:
        h = bar.get_height()
        ax_primacy.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", weight="bold", fontsize=11)

    ax_primacy.set_ylabel("% Answer Aligned with Chunk in Position 0")
    ax_primacy.set_title("Primacy Effect (First Position Bias)")
    ax_primacy.set_ylim(0, 100)

    fig.suptitle("Position Bias & Primacy Sensitivity under Context Shuffling", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()

    out_file = output_dir / "position_bias_impact.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")
    return out_file


# ==============================================================================
# PLOT 6: Executive Summary Dashboard (4 Biases in One Master Figure)
# ==============================================================================
def plot_biases_summary_dashboard(
    data_se_llama: dict | None,
    data_se_gemma: dict | None,
    data_cf_llama: dict | None,
    data_cf_gemma: dict | None,
    data_len: dict | None,
    data_pos: dict | None,
    output_dir: Path
) -> Path | None:
    """Generates a cohesive 2x2 multi-panel master summary dashboard of all 4 bias dimensions."""
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    (ax1, ax2), (ax3, ax4) = axes

    # --- PANEL A: Self-Enchantment Bias ---
    configs_se = [("LLaMA-3.2-3B", data_se_llama), ("Gemma-3-1B", data_se_gemma)]
    se_judges = [c[0] for c in configs_se if c[1]]
    se_self = [c[1]["summary"]["mean_self_score"] for c in configs_se if c[1]]
    se_other = [c[1]["summary"]["mean_other_score"] for c in configs_se if c[1]]
    se_deltas = [c[1]["summary"]["mean_delta"] for c in configs_se if c[1]]

    x = np.arange(len(se_judges))
    w = 0.32
    ax1.bar(x - w/2, se_self, w, label="Self Generation", color="#2b5c8f", edgecolor="black")
    ax1.bar(x + w/2, se_other, w, label="Other Generation", color="#7f9db9", edgecolor="black")

    for i, (jx, d) in enumerate(zip(x, se_deltas)):
        sign = "+" if d > 0 else ""
        ax1.text(jx, max(se_self[i], se_other[i]) + 8, f"Δ = {sign}{d:.2f}",
                 ha="center", weight="bold", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.2", fc="#f8f9fa", ec="gray", lw=1))

    ax1.set_xticks(x)
    ax1.set_xticklabels(se_judges, weight="bold")
    ax1.set_ylabel("Mean Judge Score (0-100)")
    ax1.set_ylim(0, 105)
    ax1.set_title("(A) Self-Enchantment Bias (Self vs. Other)", weight="bold")
    ax1.legend(loc="lower right")

    # --- PANEL B: Attribution / Compassion-Fade Framing ---
    configs_cf = [("LLaMA-3.2-3B", data_cf_llama), ("Gemma-3-1B", data_cf_gemma)]
    cf_judges = [c[0] for c in configs_cf if c[1]]
    cf_neutral = [c[1]["summary"]["mean_neutral_score"] for c in configs_cf if c[1]]
    cf_ai = [c[1]["summary"]["mean_ai_score"] for c in configs_cf if c[1]]
    cf_human = [c[1]["summary"]["mean_human_score"] for c in configs_cf if c[1]]

    x2 = np.arange(len(cf_judges))
    w2 = 0.25
    ax2.bar(x2 - w2, cf_neutral, w2, label="Neutral", color="#7f8c8d", edgecolor="black")
    ax2.bar(x2, cf_ai, w2, label="AI Attributed", color="#c0392b", edgecolor="black")
    ax2.bar(x2 + w2, cf_human, w2, label="Human Attributed", color="#2980b9", edgecolor="black")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(cf_judges, weight="bold")
    ax2.set_ylabel("Mean Judge Score (0-100)")
    ax2.set_ylim(0, 105)
    ax2.set_title("(B) Attribution Framing Bias (AI vs. Human Labels)", weight="bold")
    ax2.legend(loc="lower right")

    # --- PANEL C: Length Bias Win Rate ---
    if data_len:
        by_j = data_len.get("summary_by_judge", {})
        label_map = {
            "gemma3:1b": "Gemma-3-1B",
            "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF": "LLaMA-3.2-3B",
            "ministral-3:3b": "Ministral-3-3B",
            "qwen3.5:2b": "Qwen-3.5-2B",
            "LLM_Average": "LLM Consensus",
            "Human": "Human Feedback"
        }
        eval_order = ["Gemma-3-1B", "LLaMA-3.2-3B", "Ministral-3-3B", "Qwen-3.5-2B", "LLM Consensus", "Human Feedback"]
        rates = []
        names = []
        is_human = []
        for k in eval_order:
            for raw_k, s in by_j.items():
                if label_map.get(raw_k) == k:
                    names.append(k)
                    rates.append(s.get("longest_top_score_percentage", 0.0))
                    is_human.append(k == "Human Feedback")

        bar_c = ["#2b5c8f" if not h else "#c0392b" for h in is_human]
        bars3 = ax3.barh(names, rates, color=bar_c, edgecolor="black", alpha=0.85)
        for b in bars3:
            w_val = b.get_width()
            ax3.annotate(f"{w_val:.1f}%", xy=(w_val, b.get_y() + b.get_height()/2),
                         xytext=(5, 0), textcoords="offset points", ha="left", va="center", weight="bold", fontsize=9.5)
        ax3.set_xlabel("% Longest Answer Wins Highest Score")
        ax3.set_title("(C) Length Bias: Longest Answer Top-Score Rate", weight="bold")
        ax3.set_xlim(0, 105)
        ax3.invert_yaxis()

    # --- PANEL D: Position Bias Primacy Effect ---
    if data_pos:
        results = data_pos.get("results", [])
        similarities = [SequenceMatcher(None, r.get("original_answer", ""), r.get("shuffled_answer", "")).ratio() for r in results]
        mean_sim = np.mean(similarities) if similarities else 0.0

        sns.histplot(similarities, bins=10, kde=True, color="#2b5c8f", ax=ax4, edgecolor="black", alpha=0.7)
        ax4.axvline(mean_sim, color="#c0392b", linestyle="--", linewidth=2, label=f"Mean Similarity = {mean_sim:.2f}")
        ax4.set_xlabel("Lexical Similarity (Original vs. Shuffled Context)")
        ax4.set_ylabel("Count of Recipes")
        ax4.set_title("(D) Position Bias: Output Consistency Under Shuffling", weight="bold")
        ax4.legend(loc="upper right")

    fig.suptitle("RAG Recipe Benchmark: Comprehensive Multi-Bias Evaluation", fontsize=18, weight="bold", y=0.99)
    plt.tight_layout()

    out_file = output_dir / "biases_summary_dashboard.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")
    return out_file


# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready plots for RAG Bias Tests.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Path to folder with bias test JSON outputs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Path to folder where PNG plots will be saved")
    parser.add_argument("--plot", type=str, default="all", choices=["all", "self", "compassion", "length", "position", "dashboard"], help="Specific plot to generate")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    if not results_dir.exists():
        print(f"Error: Results folder '{results_dir}' does not exist.")
        sys.exit(1)

    # Ensure output subfolder 'Biases' exists
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=======================================================")
    print(f"       GENERATING BIAS EVALUATION PLOTS")
    print(f"Results Source : {results_dir}")
    print(f"Output Target  : {output_dir}")
    print(f"=======================================================")

    data = load_all_bias_data(results_dir)

    data_se_llama = data.get("self_enchantment_llama")
    data_se_gemma = data.get("self_enchantment_gemma")
    data_cf_llama = data.get("compassion_fade_llama")
    data_cf_gemma = data.get("compassion_fade_gemma")
    data_len = data.get("length_bias")
    data_pos = data.get("position_bias")

    plot_target = args.plot.lower()

    if plot_target in ("all", "self"):
        print("\n[1/6] Generating Self-Enchantment Plot...")
        plot_self_enchantment(data_se_llama, data_se_gemma, output_dir)

    if plot_target in ("all", "compassion"):
        print("\n[2/6] Generating Compassion-Fade / Attribution Plot...")
        plot_compassion_fade(data_cf_llama, data_cf_gemma, output_dir)

    if plot_target in ("all", "length"):
        print("\n[3/6] Generating Length Bias Overview Plot...")
        plot_length_bias(data_len, output_dir)
        print("\n[4/6] Generating Length Bias by Task Type Plot...")
        plot_length_bias_by_task(data_len, output_dir)

    if plot_target in ("all", "position"):
        print("\n[5/6] Generating Position Bias Plot...")
        plot_position_bias(data_pos, output_dir)

    if plot_target in ("all", "dashboard"):
        print("\n[6/6] Generating Master Summary Dashboard Plot...")
        plot_biases_summary_dashboard(
            data_se_llama, data_se_gemma,
            data_cf_llama, data_cf_gemma,
            data_len, data_pos,
            output_dir
        )

    print(f"\n=======================================================")
    print(f"   ALL BIAS PLOTS GENERATED SUCCESSFULLY!")
    print(f"Saved into: {output_dir}")
    print(f"=======================================================\n")

if __name__ == "__main__":
    main()
