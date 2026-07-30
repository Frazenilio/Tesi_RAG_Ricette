import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Paths (relative to this script)
ROOT_DIR = Path(__file__).parent.parent
INPUT_CSV = ROOT_DIR / "data" / "judgement_dataset.csv"
API_CSV = ROOT_DIR / "data" / "API_judges.csv"
OUTPUT_PLOT = ROOT_DIR / "data" / "boxplot_scores.png"

def main():
    if not INPUT_CSV.exists() or not API_CSV.exists():
        print(f"Error: Could not find {INPUT_CSV} or {API_CSV}")
        return

    print("Reading data...")
    # 1. Read Human Scores
    df_human = pd.read_csv(INPUT_CSV)
    df_human["original_row"] = df_human.index + 2
    
    # 2. Read Judge Scores
    df_api = pd.read_csv(API_CSV)
    
    # Merge datasets on 'original_row' to ensure exact alignment
    df = pd.merge(df_api, df_human[["original_row", "Human Numeric Score"]], on="original_row", how="inner")
    
    # 3. Restructure data for plotting
    plot_data = []
    
    # Parse Human Feedback
    for score in df["Human Numeric Score"]:
        try:
            val = float(score)
            if not pd.isna(val) and val != -1:
                plot_data.append({"Evaluator": "Human Feedback", "Score": val})
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
        print("No valid scores found to plot.")
        return
        
    print(f"Plotting {len(df_plot)} total scores...")
    
    # 4. Plot using seaborn if available, otherwise fallback to pandas/matplotlib
    plt.figure(figsize=(10, 6))
    
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        sns.boxplot(x="Evaluator", y="Score", data=df_plot, palette="Set2")
    except ImportError:
        print("Seaborn not found, falling back to Pandas built-in boxplot.")
        df_plot.boxplot(column="Score", by="Evaluator", grid=True, figsize=(10, 6), ax=plt.gca())
        plt.suptitle("") # Remove default pandas title
    
    plt.title("Score Distribution: Human Feedback vs. LLM Judges", fontsize=14)
    plt.xlabel("Evaluator", fontsize=12)
    plt.ylabel("Numeric Score (0-100)", fontsize=12)
    plt.ylim(-5, 105)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save & Show
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Plot successfully saved to {OUTPUT_PLOT.absolute()}")
    plt.show()

if __name__ == "__main__":
    main()
