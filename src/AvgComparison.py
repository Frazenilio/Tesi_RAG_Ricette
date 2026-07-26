import pandas as pd
import numpy as np
import os

def main():
    # Construct the path to the CSV file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '..', 'data', 'API_judges.csv')
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # We will collect the deltas for each judge name (model name)
    judge_stats = {}
    
    for i in range(1, 5):
        model_col = f'Judge {i} Model Name'
        delta_col = f'Judge {i} delta score'
        
        if model_col in df.columns and delta_col in df.columns:
            for idx, row in df.iterrows():
                model_name = row[model_col]
                delta_val = row[delta_col]
                
                if pd.isna(model_name):
                    continue
                    
                # Convert delta to numeric
                try:
                    delta_val = float(delta_val)
                except:
                    delta_val = np.nan
                
                if not pd.isna(delta_val):
                    if model_name not in judge_stats:
                        judge_stats[model_name] = []
                    judge_stats[model_name].append(delta_val)
                    
    print("Average delta score per judge:")
    for model_name, deltas in judge_stats.items():
        avg_delta = np.mean(deltas)
        print(f"{model_name}: {avg_delta:.4f} (based on {len(deltas)} values)")

if __name__ == "__main__":
    main()