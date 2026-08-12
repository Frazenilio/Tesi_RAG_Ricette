import pandas as pd
import numpy as np
import os
import sys

try:
    import pingouin as pg
except ImportError:
    pg = None

try:
    import krippendorff
except ImportError:
    krippendorff = None

def human_difference():
    # Construct the path to the CSV file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '..', 'data', "APICalls", 'API_Max1_judges.csv')
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # We will collect the deltas for each judge name (model name)
    judge_stats = {}
    
    for i in range(1, 5):
        model_col = f'Judge {i} Model Name'
        delta_col = f'Judge {i} delta score'
        num_col = f'Judge {i} Numeric Score'
        
        if model_col in df.columns and delta_col in df.columns:
            for idx, row in df.iterrows():
                model_name = row[model_col]
                delta_val = row[delta_col]
                
                if row[num_col] == -1:
                    print("Skipping value")
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
                    
    print("\n--- Average delta score per judge ---")
    for model_name, deltas in judge_stats.items():
        avg_delta = np.mean(deltas)
        print(f"{model_name}: {avg_delta:.4f} (based on {len(deltas)} values)")

def score_variation():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_calls_dir = os.path.join(current_dir, '..', 'data', "APICalls")
    
    # Dictionary mapping filename to its max score for scaling
    CSV_FILES_AND_MAX = {
        "API_Max1_judges.csv": 1,
        "API_Max10_judges.csv": 10,
        "API_Max100_judges.csv": 100
    }
    
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
                    
    print("\n--- Score Variation Metrics ---")
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
                
        if len(cvs) > 0:
            avg_cv = np.mean(cvs)
            avg_spread = np.mean(spreads)
            avg_sd = np.mean(sds)
            print(f"{model_name}: Avg CV = {avg_cv:.4f} | Avg SD = {avg_sd:.2f} | Avg Spread = {avg_spread:.2f}% (based on {len(cvs)} shared rows)")
        else:
            print(f"{model_name}: No common rows found across all files.")

    print("\n--- Inter-Rater Reliability (Numeric) ---")
    if pg is not None:
        for filename, max_score in CSV_FILES_AND_MAX.items():
            csv_path = os.path.join(api_calls_dir, filename)
            if not os.path.exists(csv_path): continue
            
            df_file = pd.read_csv(csv_path)
            if "original_row" not in df_file.columns:
                df_file["original_row"] = df_file.index + 2
                
            icc_data = []
            for i in range(1, 5):
                model_col = f'Judge {i} Model Name'
                num_col = f'Judge {i} Numeric Score'
                
                if model_col in df_file.columns and num_col in df_file.columns:
                    for _, row in df_file.iterrows():
                        model_name = row[model_col]
                        raw_score = row[num_col]
                        row_idx = row["original_row"]
                        if pd.isna(model_name) or raw_score == -1:
                            continue
                        icc_data.append({"item": row_idx, "rater": model_name, "score": raw_score})
                        
            if icc_data:
                df_icc = pd.DataFrame(icc_data)
                try:
                    icc = pg.intraclass_corr(data=df_icc, targets="item", raters="rater", ratings="score")
                    icc_31 = icc[icc["Type"].isin(["ICC3", "ICC(C,1)"])]
                    if not icc_31.empty:
                        icc_val = icc_31["ICC"].values[0]
                        ci = icc_31["CI95"].values[0] if "CI95" in icc_31.columns else ""
                        print(f"{filename} ICC(3,1): {icc_val:.4f} (95% CI: {ci})")
                except Exception as e:
                    print(f"Could not compute ICC for {filename}: {e}")
    else:
        print("pingouin library not installed. Cannot compute ICC(3,1).")

def coherence_accuracy():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_calls_dir = os.path.join(current_dir, '..', 'data', "APICalls")
    judgement_csv = os.path.join(current_dir, '..', 'data', 'judgement_dataset.csv')
    coherence_csv = os.path.join(api_calls_dir, 'API_Coherence_judges.csv')
    
    if not os.path.exists(coherence_csv) or not os.path.exists(judgement_csv):
        print("Missing required CSV files for coherence accuracy.")
        return
        
    df_coh = pd.read_csv(coherence_csv)
    df_judge = pd.read_csv(judgement_csv)
    df_judge["original_row"] = df_judge.index + 2
    
    # Merge on original_row to bring in the True Human Correctness string
    df_merged = pd.merge(df_coh, df_judge[["original_row", "Human Correctness"]], on="original_row", how="inner")
    
    print("\n--- Coherence Accuracy Metrics ---")
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
            accuracy = matches.mean() * 100.0
            
            print(f"{model_name} Accuracy: {accuracy:.2f}% (over {len(valid_rows)} valid judgments)")

    print("\n--- Krippendorff's Alpha (Coherence Labels) ---")
    if krippendorff is not None:
        label_map = {"correct": 2, "midway": 1, "wrong": 0}
        items = sorted(df_merged["original_row"].unique())
        judge_names = []
        for i in range(1, 5):
            col = f'Judge {i} Model Name'
            if col in df_merged.columns:
                names = df_merged[col].dropna().unique()
                if len(names) > 0:
                    judge_names.append((names[0], f'Judge {i} Evaluation'))
                    
        reliability_data = []
        for judge_name, eval_col in judge_names:
            judge_scores = []
            for item in items:
                row_data = df_merged[df_merged["original_row"] == item]
                if len(row_data) > 0:
                    val = row_data.iloc[0][eval_col]
                    if pd.isna(val) or val == "ERROR":
                        judge_scores.append(np.nan)
                    else:
                        norm_val = str(val).strip().lower()
                        judge_scores.append(label_map.get(norm_val, np.nan))
                else:
                    judge_scores.append(np.nan)
            reliability_data.append(judge_scores)
            
        try:
            alpha = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement="nominal")
            print(f"Alpha (4 LLM Judges): {alpha:.4f}")
        except Exception as e:
            print(f"Could not compute Krippendorff's alpha: {e}")
    else:
        print("krippendorff library not installed. Cannot compute alpha.")

if __name__ == "__main__":
    human_difference()
    score_variation()
    coherence_accuracy()