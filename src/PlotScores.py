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

# Dictionary mapping API CSV filenames (inside the data/APICalls/ folder) 
# to a tuple containing (Output Plot Filename, Max Score for Scaling).
CSV_FILES = {
    "API_judges.csv": ("API_Judges_0_100_Boxplot.png", 100),
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
                    model_name = str(valid_names.iloc[0])
                    
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
        
        # Define evaluator order: Human Feedback first, then the rest
        unique_evals = df_plot["Evaluator"].unique().tolist()
        if "Human Feedback" in unique_evals:
            unique_evals.remove("Human Feedback")
        evaluators_order = ["Human Feedback"] + unique_evals
        
        # 4. Plot using seaborn if available, otherwise fallback to pandas/matplotlib
        plt.figure(figsize=(10, 6))
        
        try:
            import seaborn as sns
            sns.set_theme(style="whitegrid")
            sns.boxplot(x="Evaluator", y="Score", hue="Evaluator", data=df_plot, palette="Set2", order=evaluators_order, legend=False)
        except ImportError:
            print("Seaborn not found, falling back to Pandas built-in boxplot.")
            # Pandas boxplot is tricky to order, so we convert column to Categorical to force order
            df_plot['Evaluator'] = pd.Categorical(df_plot['Evaluator'], categories=evaluators_order, ordered=True)
            df_plot.boxplot(column="Score", by="Evaluator", grid=True, figsize=(10, 6), ax=plt.gca())
            plt.suptitle("") # Remove default pandas title
        
        plt.title(f"Score Distribution: Human Feedback vs. LLM Judges ({csv_name})", fontsize=14)
        plt.xlabel("Evaluator", fontsize=12)
        plt.ylabel("Numeric Score", fontsize=12)
        
        # Dynamically set y-limits with a little padding
        min_score = df_plot["Score"].min()
        max_score = df_plot["Score"].max()
        padding = (max_score - min_score) * 0.05 if max_score != min_score else 1
        plt.ylim(min_score - padding, max_score + padding)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save & Show
        plt.savefig(output_plot_path, dpi=300)
        print(f"Plot successfully saved to {output_plot_path.absolute()}\n")
        
        # We close the figure here so it doesn't overlap for the next CSV in the loop
        # We don't call plt.show() here to prevent the script from pausing mid-loop
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
    
    accuracies = {}
    distribution_data = []
    
    for i in range(1, 5):
        model_col = f'Judge {i} Model Name'
        eval_col = f'Judge {i} Evaluation'
        
        if model_col in df_merged.columns and eval_col in df_merged.columns:
            model_names = df_merged[model_col].dropna().unique()
            if len(model_names) == 0: continue
            model_name = model_names[0]
            
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
    
    print(f"\nPlots successfully saved to:\n- {os.path.abspath(acc_plot_path)}\n- {os.path.abspath(dist_plot_path)}")

if __name__ == "__main__":
    main()
    plot_score_variation()
    plot_coherence_accuracy()
