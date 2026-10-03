import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import numpy as np
import os

# Paths (relative to this script)
ROOT_DIR = Path(__file__).parent.parent
INPUT_CSV = ROOT_DIR / "data" / "judgement_dataset.csv"
OUTPUT_PLOT_DIR = ROOT_DIR / "data" / "Boxplots" 

# Visual and Model Mappings
JUDGE_DISPLAY_NAMES = {
    "GEMMA": "Gemma-4-26B-A4B-IT",
    "QWEN": "Qwen-3.6-27B",
    "LLAMA": "Llama-3.1-8B-Instant",
    "GPT": "GPT-OSS-20B",
}

# Cohesive palette matching thesis plots (e.g. JudgeComparisonPlots.py)
PALETTE = {
    "Human Feedback": "#27ae60",       # Emerald Green
    "Gemma-4-26B-A4B-IT": "#d35400",   # Burnt Orange
    "Qwen-3.6-27B": "#16a085",         # Teal
    "Llama-3.1-8B-Instant": "#2980b9", # Ocean Blue
    "GPT-OSS-20B": "#8e44ad",          # Royal Purple
}

# Dictionary mapping API CSV filenames (inside the data/APICalls/ folder) 
# to a tuple containing (Output Plot Filename, Max Score for Scaling).
CSV_FILES = {
    "API_Max100_judges.csv": ("API_Judges_0_100_Boxplot.png", 100),
    "API_Max10_judges.csv": ("API_Judges_0_10_Boxplot.png", 10),
    "API_Max1_judges.csv": ("API_Judges_0_1_Boxplot.png", 1)
}

def main():
    if not INPUT_CSV.exists():
        print(f"Error: Could not find {INPUT_CSV}")
        return

    OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading Human data...")
    # 1. Read Human Scores
    df_human = pd.read_csv(INPUT_CSV)
    df_human["original_row"] = df_human.index + 2
    
    for csv_name, (plot_name, max_score) in CSV_FILES.items():
        api_csv_path = ROOT_DIR / "data" / "APICalls" / csv_name
        if not api_csv_path.exists() and csv_name == "API_Max100_judges.csv":
            fallback = ROOT_DIR / "data" / "APICalls" / "API_judges.csv"
            if fallback.exists():
                api_csv_path = fallback
        
        output_plot_path = OUTPUT_PLOT_DIR / plot_name
        
        if not api_csv_path.exists():
            print(f"Skipping {csv_name}: File not found.")
            continue
            
        print(f"Processing {csv_name}...")
        # 2. Read Judge Scores
        df_api = pd.read_csv(api_csv_path)
        
        # Merge datasets on 'original_row' to ensure exact alignment
        df = pd.merge(df_api, df_human[["original_row", "Human Numeric Score"]], on="original_row", how="inner")
        
        # 3. Restructure data for plotting
        plot_data = []
        
        # Parse Human Feedback (Scaled to Max Score)
        for score in df["Human Numeric Score"]:
            try:
                val = float(score)
                if not pd.isna(val) and val != -1:
                    scaled_val = (val / 100.0) * max_score
                    plot_data.append({"Evaluator": "Human Feedback", "Score": scaled_val})
            except (ValueError, TypeError):
                continue
                
        # Parse each Judge
        score_cols = [col for col in df.columns if col.endswith("Numeric Score") and col != "Human Numeric Score"]
        
        for score_col in score_cols:
            # Extract judge prefix (e.g. "Judge 1")
            judge_prefix = score_col.replace(" Numeric Score", "")
            
            # Get Model Name
            model_name_col = f"{judge_prefix} Model Name"
            model_name = "Unknown"
            if model_name_col in df.columns:
                valid_names = df[model_name_col].dropna()
                if not valid_names.empty:
                    raw_name = str(valid_names.iloc[0]).strip()
                    model_name = JUDGE_DISPLAY_NAMES.get(raw_name.upper(), raw_name)
                    
            # Append scores
            for score in df[score_col]:
                try:
                    val = float(score)
                    if not pd.isna(val) and val != -1:
                        plot_data.append({"Evaluator": model_name, "Score": val})
                except (ValueError, TypeError):
                    continue
                    
        df_plot = pd.DataFrame(plot_data)
        
        if df_plot.empty:
            print(f"No valid scores found to plot for {csv_name}.")
            continue
            
        print(f"Plotting {len(df_plot)} total scores for {csv_name}...")
        
        # Define evaluator order: Human Feedback first, then judges in standard order
        canonical_order = [
            "Human Feedback",
            "Gemma-4-26B-A4B-IT",
            "Qwen-3.6-27B",
            "Llama-3.1-8B-Instant",
            "GPT-OSS-20B"
        ]
        present_evals = df_plot["Evaluator"].unique().tolist()
        evaluators_order = [e for e in canonical_order if e in present_evals] + [e for e in present_evals if e not in canonical_order]
        
        # 4. Plot using seaborn if available, otherwise fallback to pandas/matplotlib
        plt.figure(figsize=(10, 6))
        
        try:
            import seaborn as sns
            sns.set_theme(style="whitegrid")
            sns.boxplot(
                x="Evaluator",
                y="Score",
                hue="Evaluator",
                data=df_plot,
                palette=PALETTE,
                order=evaluators_order,
                legend=False,
                width=0.55
            )
        except ImportError:
            print("Seaborn not found, falling back to Pandas built-in boxplot.")
            # Pandas boxplot is tricky to order, so we convert column to Categorical to force order
            df_plot['Evaluator'] = pd.Categorical(df_plot['Evaluator'], categories=evaluators_order, ordered=True)
            df_plot.boxplot(column="Score", by="Evaluator", grid=True, figsize=(10, 6), ax=plt.gca())
            plt.suptitle("") # Remove default pandas title
        
        plt.title("Score Distribution: Human Feedback vs. LLM Judges", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Evaluator", fontsize=12, fontweight="bold")
        plt.ylabel("Numeric Score", fontsize=12, fontweight="bold")
        plt.xticks(fontsize=10.5)
        
        # Dynamically set y-limits with a little padding
        min_score = df_plot["Score"].min()
        max_score_val = df_plot["Score"].max()
        padding = (max_score_val - min_score) * 0.05 if max_score_val != min_score else 1
        plt.ylim(min_score - padding, max_score_val + padding)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save & Show
        plt.savefig(output_plot_path, dpi=300)
        print(f"Plot successfully saved to {output_plot_path.absolute()}\n")
        
        # We close the figure here so it doesn't overlap for the next CSV in the loop
        plt.close()

def plot_score_variation():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_calls_dir = os.path.join(current_dir, '..', 'data', "APICalls")
    
    # Dictionary mapping filename to its max score for scaling
    CSV_FILES_AND_MAX = {
        "API_Max1_judges.csv": 1,
        "API_Max10_judges.csv": 10,
        "API_Max100_judges.csv": 100
    }
    
    OUTPUT_PLOT_DIR = os.path.join(current_dir, '..', 'data', 'ScaleDifferencePlots')
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
    
    # judge_data[model_name][original_row] = [scaled_score1, scaled_score2, ...]
    judge_data = {}
    
    for filename, max_score in CSV_FILES_AND_MAX.items():
        csv_path = os.path.join(api_calls_dir, filename)
        if not os.path.exists(csv_path):
            print(f"Skipping {filename}: File not found.")
            continue
            
        df = pd.read_csv(csv_path)
        
        if "original_row" not in df.columns:
            df["original_row"] = df.index + 2
            
        for i in range(1, 5):
            model_col = f'Judge {i} Model Name'
            num_col = f'Judge {i} Numeric Score'
            
            if model_col in df.columns and num_col in df.columns:
                for _, row in df.iterrows():
                    model_name = row[model_col]
                    raw_score = row[num_col]
                    row_idx = row["original_row"]
                    
                    if pd.isna(model_name) or raw_score == -1:
                        continue
                        
                    raw_name = str(model_name).strip()
                    model_name = JUDGE_DISPLAY_NAMES.get(raw_name.upper(), raw_name)
                        
                    try:
                        scaled_score = (float(raw_score) / max_score) * 100.0
                    except (ValueError, TypeError):
                        continue
                        
                    if model_name not in judge_data:
                        judge_data[model_name] = {}
                    if row_idx not in judge_data[model_name]:
                        judge_data[model_name][row_idx] = []
                        
                    judge_data[model_name][row_idx].append(scaled_score)
                    
    # Now calculate metrics
    ## https://en.wikipedia.org/wiki/Coefficient_of_variation
    ## Dev std / avg --> lower avg, bigger number compared to dev std
    plot_data_cv = []
    plot_data_sd = []
    plot_data_spread = []
    avg_spreads = {}
    
    for model_name, rows in judge_data.items():
        cvs = []
        spreads = []
        sds = []
        
        for row_idx, scores in rows.items():
            if len(scores) == len(CSV_FILES_AND_MAX): # Only include if judge scored it in ALL files
                mean_score = np.mean(scores)
                std_dev = np.std(scores) 
                
                cv = (std_dev / mean_score) if mean_score > 0 else 0
                spread = np.max(scores) - np.min(scores)
                
                cvs.append(cv)
                spreads.append(spread)
                sds.append(std_dev)
                
                plot_data_cv.append({"Judge": model_name, "CV": cv})
                plot_data_sd.append({"Judge": model_name, "SD": std_dev})
                plot_data_spread.append({"Judge": model_name, "Spread": spread})
                
        if len(cvs) > 0:
            avg_spreads[model_name] = np.mean(spreads)
            
    if not plot_data_cv:
        print("No valid data to plot.")
        return
        
    df_cv = pd.DataFrame(plot_data_cv)
    df_sd = pd.DataFrame(plot_data_sd)
    df_spread = pd.DataFrame(plot_data_spread)
    
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid")
    except ImportError:
        pass
        
    # Plot 1: Boxplot of CVs
    plt.figure(figsize=(10, 6))
    if 'seaborn' in sys.modules:
        sns.boxplot(x="Judge", y="CV", hue="Judge", data=df_cv, palette="Set2", legend=False)
    else:
        df_cv.boxplot(column="CV", by="Judge", grid=True, figsize=(10, 6), ax=plt.gca())
        plt.suptitle("")
        
    plt.title("Score Volatility: Coefficient of Variation (CV) Distribution per Judge", fontsize=14)
    plt.ylabel("Coefficient of Variation (SD / Mean)", fontsize=12)
    plt.xlabel("Judge (Model)", fontsize=12)
    plt.tight_layout()
    cv_plot_path = os.path.join(OUTPUT_PLOT_DIR, "CoefficientOfVariation_Boxplot.png")
    plt.savefig(cv_plot_path, dpi=300)
    plt.close()
    
    # Plot 1.5: Boxplot of SDs
    plt.figure(figsize=(10, 6))
    if 'seaborn' in sys.modules:
        sns.boxplot(x="Judge", y="SD", hue="Judge", data=df_sd, palette="Set2", legend=False)
    else:
        df_sd.boxplot(column="SD", by="Judge", grid=True, figsize=(10, 6), ax=plt.gca())
        plt.suptitle("")
        
    plt.title("Score Volatility: Standard Deviation Distribution per Judge", fontsize=14)
    plt.ylabel("Standard Deviation", fontsize=12)
    plt.xlabel("Judge (Model)", fontsize=12)
    plt.tight_layout()
    sd_plot_path = os.path.join(OUTPUT_PLOT_DIR, "StandardDeviation_Boxplot.png")
    plt.savefig(sd_plot_path, dpi=300)
    plt.close()

    # Plot 1.75: Boxplot of Spreads
    plt.figure(figsize=(10, 6))
    if 'seaborn' in sys.modules:
        sns.boxplot(x="Judge", y="Spread", hue="Judge", data=df_spread, palette="Set2", legend=False)
    else:
        df_spread.boxplot(column="Spread", by="Judge", grid=True, figsize=(10, 6), ax=plt.gca())
        plt.suptitle("")
        
    plt.title("Score Volatility: Spread (Max - Min) Distribution per Judge across scales", fontsize=14)
    plt.ylabel("Spread (%)", fontsize=12)
    plt.xlabel("Judge (Model)", fontsize=12)
    plt.tight_layout()
    spread_box_plot_path = os.path.join(OUTPUT_PLOT_DIR, "Spread_Boxplot.png")
    plt.savefig(spread_box_plot_path, dpi=300)
    plt.close()
    
    # Plot 2: Bar chart of Avg Spread
    plt.figure(figsize=(10, 6))
    judges = list(avg_spreads.keys())
    spread_vals = list(avg_spreads.values())
    
    if 'seaborn' in sys.modules:
        sns.barplot(x=judges, y=spread_vals, hue=judges, palette="Set3", legend=False)
    else:
        plt.bar(judges, spread_vals, color='skyblue')
        
    plt.title("Average Score Spread (Max - Min) per Judge across scales", fontsize=14)
    plt.ylabel("Average Spread (%)", fontsize=12)
    plt.xlabel("Judge (Model)", fontsize=12)
    if spread_vals:
        plt.ylim(0, max(spread_vals) * 1.2) # Give 20% headroom
    plt.tight_layout()
    spread_plot_path = os.path.join(OUTPUT_PLOT_DIR, "AverageSpread_BarChart.png")
    plt.savefig(spread_plot_path, dpi=300)
    plt.close()
    
    print(f"\nPlots successfully saved to:\n- {os.path.abspath(cv_plot_path)}\n- {os.path.abspath(sd_plot_path)}\n- {os.path.abspath(spread_box_plot_path)}\n- {os.path.abspath(spread_plot_path)}")


def plot_coherence_accuracy():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_calls_dir = os.path.join(current_dir, '..', 'data', "APICalls")
    judgement_csv = os.path.join(current_dir, '..', 'data', 'judgement_dataset.csv')
    coherence_csv = os.path.join(api_calls_dir, 'API_Coherence_judges.csv')
    
    OUTPUT_PLOT_DIR = os.path.join(current_dir, '..', 'data', 'ScaleDifferencePlots')
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
    
    if not os.path.exists(coherence_csv) or not os.path.exists(judgement_csv):
        print("Missing required CSV files for coherence accuracy.")
        return
        
    df_coh = pd.read_csv(coherence_csv)
    df_judge = pd.read_csv(judgement_csv)
    df_judge["original_row"] = df_judge.index + 2
    
    # Merge on original_row to bring in the True Human Correctness string
    df_merged = pd.merge(df_coh, df_judge[["original_row", "Human Correctness"]], on="original_row", how="inner")
    df_merged["Human Correctness"] = df_merged["Human Correctness"].str.strip().str.lower().str.replace("midway correct", "midway")
    
    accuracies = {}
    distribution_data = []
    
    for i in range(1, 5):
        model_col = f'Judge {i} Model Name'
        eval_col = f'Judge {i} Evaluation'
        
        if model_col in df_merged.columns and eval_col in df_merged.columns:
            model_names = df_merged[model_col].dropna().unique()
            if len(model_names) == 0: continue
            raw_name = str(model_names[0]).strip()
            model_name = JUDGE_DISPLAY_NAMES.get(raw_name.upper(), raw_name)
            
            valid_rows = df_merged[df_merged[eval_col].notna() & (df_merged[eval_col] != "ERROR")].copy()
            
            if len(valid_rows) == 0:
                continue
                
            # Normalize strings for comparison
            valid_rows[eval_col] = valid_rows[eval_col].str.strip().str.lower()
            valid_rows["Human Correctness"] = valid_rows["Human Correctness"].str.strip().str.lower().str.replace("midway correct", "midway")
            
            matches = (valid_rows[eval_col] == valid_rows["Human Correctness"])
            accuracies[model_name] = matches.mean() * 100.0
            
            # For distribution plot
            counts = valid_rows[eval_col].value_counts().to_dict()
            distribution_data.append({
                "Judge": model_name,
                "Correct": counts.get("correct", 0),
                "Midway": counts.get("midway", 0),
                "Wrong": counts.get("wrong", 0)
            })
            
    if not accuracies:
        print("No valid coherence data to plot.")
        return
        
    try:
        import seaborn as sns
    except ImportError:
        pass
        
    # Plot 1: Accuracy Bar Chart
    plt.figure(figsize=(10, 6))
    judges = list(accuracies.keys())
    acc_vals = list(accuracies.values())
    
    if 'seaborn' in sys.modules:
        sns.barplot(x=judges, y=acc_vals, hue=judges, palette="Set1", legend=False)
    else:
        plt.bar(judges, acc_vals, color='lightgreen')
        
    plt.title("Coherence Evaluation Accuracy per Judge (vs Human)", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xlabel("Judge (Model)", fontsize=12)
    plt.ylim(0, 100)
    
    # Add values on top of bars
    for idx, val in enumerate(acc_vals):
        plt.text(idx, val + 1, f"{val:.1f}%", ha='center', fontsize=10)
        
    plt.tight_layout()
    
    acc_plot_path = os.path.join(OUTPUT_PLOT_DIR, "CoherenceAccuracy_BarChart.png")
    plt.savefig(acc_plot_path, dpi=300)
    plt.close()
    
    # Plot 2: Distribution Stacked Bar Chart
    df_dist = pd.DataFrame(distribution_data).set_index("Judge")
    
    # Add human baseline distribution
    human_labels = df_merged["Human Correctness"].str.strip().str.lower().str.replace("midway correct", "midway")
    human_counts = human_labels.value_counts().to_dict()
    df_dist.loc["HUMAN (Baseline)"] = {
        "Correct": human_counts.get("correct", 0),
        "Midway": human_counts.get("midway", 0),
        "Wrong": human_counts.get("wrong", 0)
    }
    
    colors = ["#4CAF50", "#FFC107", "#F44336"] # Green, Yellow, Red
    df_dist[["Correct", "Midway", "Wrong"]].plot(kind='bar', stacked=True, figsize=(10, 6), color=colors)
    
    plt.title("Distribution of Coherence Labels (Judges vs Human)", fontsize=14)
    plt.ylabel("Number of Recipes", fontsize=12)
    plt.xlabel("Judge / Baseline", fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    dist_plot_path = os.path.join(OUTPUT_PLOT_DIR, "CoherenceDistribution_StackedBar.png")
    plt.savefig(dist_plot_path, dpi=300)
    plt.close()

    # Plot 3: 2x2 Grid of Confusion Matrices
    classes = ['correct', 'midway', 'wrong']
    # Plot 3: Individual Confusion Matrix per Judge and Combined 2x2 Grid
    classes = ['correct', 'midway', 'wrong']
    class_labels = ['Correct', 'Midway', 'Wrong']
    judge_configs = [
        ('Judge 1', 'Gemma-4-26B-A4B-IT', 'Gemma', PALETTE['Gemma-4-26B-A4B-IT']),
        ('Judge 2', 'Qwen-3.6-27B', 'Qwen', PALETTE['Qwen-3.6-27B']),
        ('Judge 3', 'Llama-3.1-8B-Instant', 'Llama', PALETTE['Llama-3.1-8B-Instant']),
        ('Judge 4', 'GPT-OSS-20B', 'GPT', PALETTE['GPT-OSS-20B']),
    ]

    judges_plot_dir = os.path.join(current_dir, 'Plots', 'SavedPlots', 'Judges')
    individual_paths = []

    # Combined 2x2 figure
    fig_comb, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()

    for idx, (prefix, model_name, short_name, col) in enumerate(judge_configs):
        e_col = f'{prefix} Evaluation'
        if e_col not in df_merged.columns:
            continue
        sub = df_merged[['Human Correctness', e_col]].dropna().copy()
        sub[e_col] = sub[e_col].str.strip().str.lower()
        sub = sub[sub[e_col].isin(classes)]

        total = len(sub)
        correct_matches = (sub['Human Correctness'] == sub[e_col]).sum()
        acc = (correct_matches / total) * 100.0 if total > 0 else 0

        cm = pd.crosstab(sub['Human Correctness'], sub[e_col], rownames=['Human True'], colnames=['Judge Pred'])
        cm = cm.reindex(index=classes, columns=classes, fill_value=0)

        # 1. Plot on combined 2x2 axes
        ax = axes[idx]
        if 'seaborn' in sys.modules:
            import seaborn as sns
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                cbar=True,
                vmin=0,
                vmax=30,
                linewidths=1.2,
                linecolor='white',
                annot_kws={'fontsize': 12, 'fontweight': 'bold'},
                xticklabels=class_labels,
                yticklabels=class_labels,
                ax=ax
            )
        else:
            cax = ax.matshow(cm, cmap='Blues')
            fig_comb.colorbar(cax, ax=ax)
            for r in range(len(classes)):
                for c in range(len(classes)):
                    ax.text(c, r, str(cm.values[r, c]), ha='center', va='center', fontsize=12, fontweight='bold')

        ax.set_title(f'{model_name}', fontsize=13, fontweight='bold', pad=12, color=col)
        ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Human Label', fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', labelsize=10.5)
        ax.tick_params(axis='y', labelsize=10.5, rotation=0)

        # 2. Individual Dedicated Figure for this Judge
        fig_single, ax_single = plt.subplots(figsize=(6.5, 5.5))
        if 'seaborn' in sys.modules:
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                cbar=True,
                vmin=0,
                vmax=30,
                linewidths=1.2,
                linecolor='white',
                annot_kws={'fontsize': 13, 'fontweight': 'bold'},
                xticklabels=class_labels,
                yticklabels=class_labels,
                ax=ax_single
            )
        else:
            cax = ax_single.matshow(cm, cmap='Blues')
            fig_single.colorbar(cax, ax=ax_single)
            for r in range(len(classes)):
                for c in range(len(classes)):
                    ax_single.text(c, r, str(cm.values[r, c]), ha='center', va='center', fontsize=13, fontweight='bold')

        ax_single.set_title(f'Confusion Matrix: {model_name}', fontsize=13.5, fontweight='bold', pad=12, color=col)
        ax_single.set_xlabel('Predicted Label', fontsize=11.5, fontweight='bold')
        ax_single.set_ylabel('True Human Label', fontsize=11.5, fontweight='bold')
        ax_single.tick_params(axis='x', labelsize=11)
        ax_single.tick_params(axis='y', labelsize=11, rotation=0)
        plt.tight_layout()

        single_out = os.path.join(OUTPUT_PLOT_DIR, f"Coherence_Confusion_Matrix_{short_name}.png")
        fig_single.savefig(single_out, dpi=300, bbox_inches='tight')
        if os.path.exists(judges_plot_dir):
            fig_single.savefig(os.path.join(judges_plot_dir, f"Coherence_Confusion_Matrix_{short_name}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_single)
        individual_paths.append(single_out)

    fig_comb.suptitle('Coherence Classification: Confusion Matrices per LLM Judge', fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()

    cm_plot_path = os.path.join(OUTPUT_PLOT_DIR, "Coherence_Confusion_Matrices.png")
    fig_comb.savefig(cm_plot_path, dpi=300, bbox_inches='tight')

    if os.path.exists(judges_plot_dir):
        fig_comb.savefig(os.path.join(judges_plot_dir, "Coherence_Confusion_Matrices.png"), dpi=300, bbox_inches='tight')

    plt.close(fig_comb)

    ind_str = "\n- ".join([os.path.abspath(p) for p in individual_paths])
    print(f"\nPlots successfully saved to:\n- {os.path.abspath(acc_plot_path)}\n- {os.path.abspath(dist_plot_path)}\n- {os.path.abspath(cm_plot_path)}\n- {ind_str}")


def plot_histograms():
    """Generates faceted histograms for all three scales (0-100, 0-10, 0-1)."""
    if not INPUT_CSV.exists():
        print(f"Error: Could not find {INPUT_CSV}")
        return

    OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df_human = pd.read_csv(INPUT_CSV)
    df_human["original_row"] = df_human.index + 2

    scales_config = [
        ("API_Max100_judges.csv", "API_Judges_0_100_Histogram.png", 100, np.linspace(0, 100, 11), np.arange(0, 101, 10), "{:.0f}"),
        ("API_Max10_judges.csv", "API_Judges_0_10_Histogram.png", 10, np.linspace(0, 10, 11), np.arange(0, 11, 1), "{:.0f}"),
        ("API_Max1_judges.csv", "API_Judges_0_1_Histogram.png", 1, np.linspace(0, 1, 11), np.round(np.arange(0, 1.05, 0.1), 1), "{:.1f}"),
    ]

    evaluator_info = [
        ("Human Feedback", "Human", PALETTE["Human Feedback"]),
        ("Gemma-4-26B-A4B-IT", "Judge 1", PALETTE["Gemma-4-26B-A4B-IT"]),
        ("Qwen-3.6-27B", "Judge 2", PALETTE["Qwen-3.6-27B"]),
        ("Llama-3.1-8B-Instant", "Judge 3", PALETTE["Llama-3.1-8B-Instant"]),
        ("GPT-OSS-20B", "Judge 4", PALETTE["GPT-OSS-20B"]),
    ]

    for csv_name, plot_name, max_score, bins, xticks, tick_fmt in scales_config:
        csv_path = ROOT_DIR / "data" / "APICalls" / csv_name
        if not csv_path.exists() and csv_name == "API_Max100_judges.csv":
            fallback = ROOT_DIR / "data" / "APICalls" / "API_judges.csv"
            if fallback.exists():
                csv_path = fallback

        if not csv_path.exists():
            print(f"Skipping histogram for {csv_name}: File not found.")
            continue

        df_api = pd.read_csv(csv_path)
        merged = pd.merge(df_api, df_human[["original_row", "Human Numeric Score"]], on="original_row")

        fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharey=True)

        for ax, (disp_name, prefix, col) in zip(axes, evaluator_info):
            if prefix == "Human":
                raw_scores = merged["Human Numeric Score"].astype(float)
                scores = (raw_scores / 100.0) * max_score
            else:
                col_name = f"{prefix} Numeric Score"
                scores = merged[col_name].astype(float)

            # Plot histogram counts within expected range
            counts, edges, patches = ax.hist(
                scores,
                bins=bins,
                range=(0, max_score),
                color=col,
                alpha=0.85,
                edgecolor="white",
                linewidth=1.2,
                rwidth=0.9,
            )

            # Annotate counts on top of bars
            bin_width = bins[1] - bins[0]
            for count, edge in zip(counts, edges):
                if count > 0:
                    ax.text(
                        edge + bin_width / 2,
                        count + 1.2,
                        f"{int(count)}",
                        ha="center",
                        va="bottom",
                        fontsize=9.5,
                        fontweight="bold",
                        color="#2c3e50",
                    )


            ax.set_ylabel("Recipes", fontsize=10.5, fontweight="bold")
            ax.set_title(disp_name, fontsize=12, fontweight="bold", loc="left", pad=4, color="#2c3e50")
            ax.grid(axis="y", linestyle=":", alpha=0.6)
            ax.set_ylim(0, 82)
            ax.set_xlim(-bin_width * 0.4, max_score + bin_width * 0.4)

            # Ensure numbers are shown on EVERY x-axis
            ax.set_xticks(xticks)
            ax.set_xticklabels([tick_fmt.format(x) for x in xticks], fontsize=10.5)
            ax.tick_params(labelbottom=True)

        axes[-1].set_xlabel(f"Numeric Score (0 – {max_score})", fontsize=12, fontweight="bold")

        plt.suptitle("Score Distribution: Human vs LLM judges", fontsize=14, fontweight="bold", y=0.995)
        plt.tight_layout()
        out_path = OUTPUT_PLOT_DIR / plot_name
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Histogram successfully saved to {out_path.absolute()}")


if __name__ == "__main__":
    main()
    plot_histograms()
    plot_score_variation()
    plot_coherence_accuracy()
