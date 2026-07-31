import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

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

if __name__ == "__main__":
    main()
