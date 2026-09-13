import pandas as pd
import numpy as np
import os
import sys
from collections import Counter
import random

try:
    import pingouin as pg
except ImportError:
    pg = None

try:
    import krippendorff
except ImportError:
    krippendorff = None

try:
    import statsmodels.stats.inter_rater as irr
except ImportError:
    irr = None

def human_difference():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_calls_dir = os.path.join(current_dir, '..', 'data', "APICalls")
    
    CSV_FILES_AND_MAX = {
        "API_Max1_judges.csv": 1.0,
        "API_Max10_judges.csv": 10.0,
        "API_Max100_judges.csv": 100.0
    }
    
    scale_results = {}
    
    for filename, max_score in CSV_FILES_AND_MAX.items():
        csv_path = os.path.join(api_calls_dir, filename)
        if not os.path.exists(csv_path):
            print(f"Skipping {filename}: File not found.")
            continue
            
        df = pd.read_csv(csv_path)
        judge_stats = {}
        
        for i in range(1, 5):
            model_col = f'Judge {i} Model Name'
            delta_col = f'Judge {i} delta score'
            num_col = f'Judge {i} Numeric Score'
            
            if model_col in df.columns and delta_col in df.columns:
                for idx, row in df.iterrows():
                    model_name = row[model_col]
                    raw_score = row[num_col]
                    delta_val = row[delta_col]
                    
                    if pd.isna(model_name) or raw_score == -1:
                        continue
                        
                    model_name = str(model_name).upper().strip()
                    try:
                        delta_num = float(delta_val)
                    except (ValueError, TypeError):
                        continue
                        
                    if not pd.isna(delta_num):
                        if model_name not in judge_stats:
                            judge_stats[model_name] = {
                                "raw": [],
                                "norm": [],
                                "abs_raw": [],
                                "abs_norm": []
                            }
                        judge_stats[model_name]["raw"].append(delta_num)
                        judge_stats[model_name]["norm"].append(delta_num / max_score)
                        judge_stats[model_name]["abs_raw"].append(abs(delta_num))
                        judge_stats[model_name]["abs_norm"].append(abs(delta_num) / max_score)
                        
        scale_results[filename] = (max_score, judge_stats)
        
        print(f"\n--- Delta Scores Breakdown ({filename} | Scale 0-{int(max_score)}) ---")
        for model_name in sorted(judge_stats.keys()):
            stats = judge_stats[model_name]
            raw_signed = np.mean(stats["raw"])
            norm_signed = np.mean(stats["norm"])
            raw_abs = np.mean(stats["abs_raw"])
            norm_abs = np.mean(stats["abs_norm"])
            n = len(stats["raw"])
            print(f"  {model_name:<6}: "
                  f"Raw Delta = {raw_signed:+.3f} (Abs: {raw_abs:.3f}) | "
                  f"Norm Delta (0-1) = {norm_signed:+.4f} ({norm_signed*100:+.2f}%) | "
                  f"Abs Norm = {norm_abs:.4f} ({norm_abs*100:.2f}%) [n={n}]")

    # Summary table across all scales
    all_models = sorted(list({m for _, (_, stats) in scale_results.items() for m in stats.keys()}))
    
    print("\n" + "="*84)
    print("--- Summary: Normalized Signed Delta (LLM - Human) Across Scales ---")
    print(f"{'Judge':<8} | {'Max 1 (0-1)':<20} | {'Max 10 (0-10)':<20} | {'Max 100 (0-100)':<20}")
    print("-" * 84)
    for model in all_models:
        row_str = f"{model:<8} | "
        for fname in ["API_Max1_judges.csv", "API_Max10_judges.csv", "API_Max100_judges.csv"]:
            if fname in scale_results and model in scale_results[fname][1]:
                norm_d = np.mean(scale_results[fname][1][model]["norm"])
                row_str += f"{norm_d:+.4f} ({norm_d*100:+.2f}%)     | "
            else:
                row_str += f"{'N/A':<20} | "
        print(row_str.rstrip(" |"))

    print("\n--- Summary: Normalized Absolute Delta |LLM - Human| Across Scales ---")
    print(f"{'Judge':<8} | {'Max 1 (0-1)':<20} | {'Max 10 (0-10)':<20} | {'Max 100 (0-100)':<20}")
    print("-" * 84)
    for model in all_models:
        row_str = f"{model:<8} | "
        for fname in ["API_Max1_judges.csv", "API_Max10_judges.csv", "API_Max100_judges.csv"]:
            if fname in scale_results and model in scale_results[fname][1]:
                abs_norm_d = np.mean(scale_results[fname][1][model]["abs_norm"])
                row_str += f"{abs_norm_d:.4f} ({abs_norm_d*100:.2f}%)      | "
            else:
                row_str += f"{'N/A':<20} | "
        print(row_str.rstrip(" |"))
    print("="*84)

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
                        
                        if krippendorff is not None:
                            try:
                                pivot_df = df_icc.pivot(index='rater', columns='item', values='score')
                                alpha_llm = krippendorff.alpha(reliability_data=pivot_df.values, level_of_measurement='interval')
                                print(f"{filename} Krippendorff Alpha (Interval): {alpha_llm:.4f}")
                            except Exception as e:
                                print(f"Could not compute Krippendorff Alpha for {filename}: {e}")
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

def human_reliability():
    print("\n--- Human Annotator Reliability ---")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xlsx_path = os.path.join(current_dir, '..', 'Le tediose ricette di Francesco (Risposte).xlsx')
    
    if not os.path.exists(xlsx_path):
        print(f"File not found: {xlsx_path}")
        return
        
    try:
        df = pd.read_excel(xlsx_path)
    except Exception as e:
        print(f"Could not read excel file: {e}")
        return
        
    icc_data = []
    reliability_data = []
    reliability_data_labels = []

    for rater_idx, row in df.iterrows():
        rater_scores = []
        rater_labels = []
        scores = row[1:].values
        for item_idx, score in enumerate(scores):
            if pd.isna(score) or score == -1 or str(score).strip() == "":
                val = np.nan
                val_label = np.nan
            else:
                try:
                    val = float(score)
                    if val >= 95:
                        val_label = 2
                    elif val <= 30:
                        val_label = 0
                    else:
                        val_label = 1
                except ValueError:
                    val = np.nan
                    val_label = np.nan
                
            icc_data.append({
                'item': item_idx,
                'rater': rater_idx,
                'score': val
            })
            rater_scores.append(val)
            rater_labels.append(val_label)
        reliability_data.append(rater_scores)
        reliability_data_labels.append(rater_labels)

    if pg is not None:
        df_icc = pd.DataFrame(icc_data).dropna()
        if not df_icc.empty:
            try:
                icc = pg.intraclass_corr(data=df_icc, targets='item', raters='rater', ratings='score')
                icc_31 = icc[icc['Type'].isin(['ICC3', 'ICC(C,1)'])]
                if not icc_31.empty:
                    icc_val = icc_31['ICC'].values[0]
                    ci = icc_31['CI95%'].values[0] if 'CI95%' in icc_31.columns else (icc_31['CI95'].values[0] if 'CI95' in icc_31.columns else '')
                    print(f"ICC(3,1): {icc_val:.4f} (95% CI: {ci})")
                else:
                    print("Could not find ICC3 type in pingouin output.")
            except Exception as e:
                print(f"Error computing ICC: {e}")
        else:
            print("Not enough valid data for ICC.")
    else:
        print("pingouin library not installed. Cannot compute ICC(3,1).")

    if krippendorff is not None:
        try:
            alpha_nominal = krippendorff.alpha(reliability_data=reliability_data_labels, level_of_measurement='nominal')
            print(f"Krippendorff Alpha (Coherence Labels): {alpha_nominal:.4f}")
        except Exception as e:
            print(f"Error computing Krippendorff Alpha: {e}")
    else:
        print("krippendorff library not installed. Cannot compute alpha.")

    if irr is not None:
        try:
            data_t = np.array(reliability_data_labels).T
            if np.isnan(data_t).any():
                print("Warning: Missing values detected. Fleiss' Kappa requires complete data.")
            else:
                table, categories = irr.aggregate_raters(data_t)
                kappa = irr.fleiss_kappa(table)
                print(f"Fleiss' Kappa (Coherence Labels): {kappa:.4f}")
        except Exception as e:
            print(f"Error computing Fleiss' Kappa: {e}")

def aggregate_comparison():
    print("\n--- Aggregated Human vs LLM Comparison ---")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_calls_dir = os.path.join(current_dir, '..', 'data', "APICalls")
    
    xlsx_path = os.path.join(current_dir, '..', 'Le tediose ricette di Francesco (Risposte).xlsx')
    llm_numeric_csv = os.path.join(api_calls_dir, 'API_Max100_judges.csv')
    llm_label_csv = os.path.join(api_calls_dir, 'API_Coherence_judges.csv')
    
    if not all(os.path.exists(p) for p in [xlsx_path, llm_numeric_csv, llm_label_csv]):
        print("Missing required files for aggregated comparison.")
        return
        
    try:
        df_human = pd.read_excel(xlsx_path)
        df_llm_num = pd.read_csv(llm_numeric_csv)
        df_llm_lab = pd.read_csv(llm_label_csv)
    except Exception as e:
        print(f"Could not read files: {e}")
        return

    if "original_row" not in df_llm_num.columns:
        df_llm_num["original_row"] = df_llm_num.index + 2
    if "original_row" not in df_llm_lab.columns:
        df_llm_lab["original_row"] = df_llm_lab.index + 2
    
    human_agg = {}
    for item_idx in range(len(df_human.columns) - 1):
        human_agg[item_idx] = {'scores': [], 'labels': []}
        
    for rater_idx, row in df_human.iterrows():
        scores = row[1:].values
        for item_idx, score in enumerate(scores):
            if pd.isna(score) or score == -1 or str(score).strip() == "":
                continue
            try:
                val = float(score)
                human_agg[item_idx]['scores'].append(val)
                if val >= 95:
                    human_agg[item_idx]['labels'].append(2)
                elif val <= 30:
                    human_agg[item_idx]['labels'].append(0)
                else:
                    human_agg[item_idx]['labels'].append(1)
            except ValueError:
                pass
                
    final_human_agg = {}
    for item_idx, data in human_agg.items():
        if data['scores']:
            avg_score = np.mean(data['scores'])
            majority_label = Counter(data['labels']).most_common(1)[0][0]
            final_human_agg[item_idx] = {'score': avg_score, 'label': majority_label}

    CSV_FILES_AND_MAX = {
        "API_Max1_judges.csv": 1,
        "API_Max10_judges.csv": 10,
        "API_Max100_judges.csv": 100
    }
    
    for file_name, max_score in CSV_FILES_AND_MAX.items():
        csv_path = os.path.join(api_calls_dir, file_name)
        if not os.path.exists(csv_path):
            continue
            
        df_llm_num = pd.read_csv(csv_path)
        llm_num_map = {}
        for item_idx, row in df_llm_num.iterrows():
            valid_scores = []
            for i in range(1, 5):
                model_col = f'Judge {i} Model Name'
                num_col = f'Judge {i} Numeric Score'
                if model_col in df_llm_num.columns and num_col in df_llm_num.columns:
                    model_name = str(row[model_col]).lower()
                    if "llama" in model_name:
                        continue
                    val = row[num_col]
                    if not pd.isna(val) and val != -1:
                        valid_scores.append(float(val))
            if valid_scores:
                llm_num_map[item_idx] = np.mean(valid_scores)
                
        shared_items = set(final_human_agg.keys()).intersection(set(llm_num_map.keys()))
        if not shared_items:
            continue
            
        deltas = []
        cvs = []
        sds = []
        icc_data = []
        rel_num_data = [[], []]
        
        for item_idx in sorted(list(shared_items)):
            h_score_100 = final_human_agg[item_idx]['score']
            h_score_scaled = (h_score_100 / 100.0) * max_score
            l_score = llm_num_map[item_idx]
            
            delta = abs(h_score_scaled - l_score) / max_score
            deltas.append(delta)
            
            mean_score = np.mean([h_score_scaled, l_score])
            std_dev = np.std([h_score_scaled, l_score])
            
            cv = (std_dev / mean_score) if mean_score > 0 else 0
            cvs.append(cv)
            sds.append(std_dev / max_score)
            
            icc_data.append({'item': item_idx, 'rater': 'Human', 'score': h_score_scaled})
            icc_data.append({'item': item_idx, 'rater': 'LLM', 'score': l_score})
            rel_num_data[0].append(h_score_scaled)
            rel_num_data[1].append(l_score)
            
        print(f"\n--- Numeric Comparison ({file_name}) ---")
        print(f"Based on {len(shared_items)} shared items.")
        print(f"Avg Delta Score (Scaled 0-1): {np.mean(deltas):.4f}")
        print(f"Avg Standard Deviation (Scaled 0-1): {np.mean(sds):.4f}")
        print(f"Avg Coefficient of Variation (CV): {np.mean(cvs):.4f}")
        
        if pg is not None:
            df_icc = pd.DataFrame(icc_data)
            try:
                icc = pg.intraclass_corr(data=df_icc, targets='item', raters='rater', ratings='score')
                icc_31 = icc[icc['Type'].isin(['ICC3', 'ICC(C,1)'])]
                if not icc_31.empty:
                    icc_val = icc_31['ICC'].values[0]
                    ci = icc_31['CI95%'].values[0] if 'CI95%' in icc_31.columns else (icc_31['CI95'].values[0] if 'CI95' in icc_31.columns else '')
                    print(f"Numeric ICC(3,1) (Human Avg vs LLM Avg): {icc_val:.4f} (95% CI: {ci})")
            except Exception as e:
                print(f"Error computing ICC: {e}")
                
        if krippendorff is not None:
            try:
                alpha_num = krippendorff.alpha(reliability_data=rel_num_data, level_of_measurement='interval')
                print(f"Numeric Krippendorff Alpha (Human Avg vs LLM Avg): {alpha_num:.4f}")
            except Exception as e:
                pass

    if os.path.exists(llm_label_csv):
        df_llm_lab = pd.read_csv(llm_label_csv)
        llm_lab_map = {}
        llm_lab_map_4 = {}
        label_map = {"correct": 2, "midway": 1, "wrong": 0}
        for item_idx, row in df_llm_lab.iterrows():
            labels = []
            labels_4 = []
            for i in range(1, 5):
                model_col = f'Judge {i} Model Name'
                eval_col = f'Judge {i} Evaluation'
                if model_col in df_llm_lab.columns and eval_col in df_llm_lab.columns:
                    model_name = str(row[model_col]).lower()
                    eval_str = str(row[eval_col]).strip().lower()
                    
                    if eval_str in label_map:
                        lbl = label_map[eval_str]
                        labels_4.append(lbl)
                        if "llama" not in model_name:
                            labels.append(lbl)
                            
            if labels:
                majority_label = Counter(labels).most_common(1)[0][0]
                llm_lab_map[item_idx] = majority_label
                
            if labels_4:
                c4 = Counter(labels_4)
                max_count = max(c4.values())
                candidates = [k for k, v in c4.items() if v == max_count]
                llm_lab_map_4[item_idx] = random.choice(candidates)
                
        shared_items_lab = set(final_human_agg.keys()).intersection(set(llm_lab_map.keys()))
        if shared_items_lab:
            matches = 0
            matches_4 = 0
            rel_lab_data = [[], []]
            
            for item_idx in sorted(list(shared_items_lab)):
                h_lab = final_human_agg[item_idx]['label']
                l_lab = llm_lab_map[item_idx]
                l_lab_4 = llm_lab_map_4[item_idx]
                
                if h_lab == l_lab:
                    matches += 1
                if h_lab == l_lab_4:
                    matches_4 += 1
                    
                rel_lab_data[0].append(h_lab)
                rel_lab_data[1].append(l_lab)
                
            accuracy = matches / len(shared_items_lab)
            accuracy_4 = matches_4 / len(shared_items_lab)
            
            print(f"\n--- Labels Comparison (API_Coherence_judges.csv) ---")
            print(f"Based on {len(shared_items_lab)} shared items.")
            print(f"Accuracy (Human Maj == LLM Maj): {accuracy:.2%}")
            print(f"Accuracy (4 judges, tie-break rng): {accuracy_4:.2%}")
            
            if krippendorff is not None:
                try:
                    alpha_lab = krippendorff.alpha(reliability_data=rel_lab_data, level_of_measurement='nominal')
                    print(f"Labels Krippendorff Alpha (Human Maj vs LLM Maj): {alpha_lab:.4f}")
                except:
                    pass
                    
            if irr is not None:
                try:
                    data_t = np.array(rel_lab_data).T
                    if not np.isnan(data_t).any():
                        table, categories = irr.aggregate_raters(data_t)
                        kappa = irr.fleiss_kappa(table)
                        print(f"Labels Fleiss' Kappa (Human Maj vs LLM Maj): {kappa:.4f}")
                except:
                    pass

def combined_reliability():
    print("\n--- Combined Annotator Reliability (3 Humans + 4 LLMs) ---")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_calls_dir = os.path.join(current_dir, '..', 'data', "APICalls")
    xlsx_path = os.path.join(current_dir, '..', 'Le tediose ricette di Francesco (Risposte).xlsx')
    llm_label_csv = os.path.join(api_calls_dir, 'API_Coherence_judges.csv')
    
    try:
        df_human = pd.read_excel(xlsx_path)
    except Exception as e:
        print(f"Could not read excel file: {e}")
        return
        
    human_num_data = [] 
    human_lab_data = [] 
    for rater_idx, row in df_human.iterrows():
        rater_scores = []
        rater_labels = []
        scores = row[1:].values
        for item_idx, score in enumerate(scores):
            if pd.isna(score) or score == -1 or str(score).strip() == "":
                rater_scores.append(np.nan)
                rater_labels.append(np.nan)
            else:
                try:
                    val = float(score)
                    rater_scores.append(val)
                    if val >= 95:
                        rater_labels.append(2)
                    elif val <= 30:
                        rater_labels.append(0)
                    else:
                        rater_labels.append(1)
                except ValueError:
                    rater_scores.append(np.nan)
                    rater_labels.append(np.nan)
        human_num_data.append(rater_scores)
        human_lab_data.append(rater_labels)

    CSV_FILES_AND_MAX = {
        "API_Max1_judges.csv": 1,
        "API_Max10_judges.csv": 10,
        "API_Max100_judges.csv": 100
    }
    
    for file_name, max_score in CSV_FILES_AND_MAX.items():
        csv_path = os.path.join(api_calls_dir, file_name)
        if not os.path.exists(csv_path):
            continue
            
        df_llm_num = pd.read_csv(csv_path)
        
        num_items = len(human_num_data[0])
        llm_num_data = [[np.nan]*num_items for _ in range(4)]
        
        for item_idx, row in df_llm_num.iterrows():
            if item_idx >= num_items:
                break
            for i in range(1, 5):
                num_col = f'Judge {i} Numeric Score'
                if num_col in df_llm_num.columns:
                    val = row[num_col]
                    if not pd.isna(val) and val != -1:
                        llm_num_data[i-1][item_idx] = float(val)
                        
        scaled_human_num_data = [[(s / 100.0 * max_score) if not pd.isna(s) else np.nan for s in rater] for rater in human_num_data]
        combined_num_data = scaled_human_num_data + llm_num_data
        
        icc_data = []
        for r_idx, rater_scores in enumerate(combined_num_data):
            rater_name = f"Human_{r_idx+1}" if r_idx < 3 else f"LLM_{r_idx-2}"
            for i_idx, score in enumerate(rater_scores):
                if not pd.isna(score):
                    icc_data.append({'item': i_idx, 'rater': rater_name, 'score': score})
                    
        print(f"\n--- Combined Numeric Comparison ({file_name}) ---")
        
        if pg is not None:
            df_icc = pd.DataFrame(icc_data)
            try:
                icc = pg.intraclass_corr(data=df_icc, targets='item', raters='rater', ratings='score')
                icc_31 = icc[icc['Type'].isin(['ICC3', 'ICC(C,1)'])]
                if not icc_31.empty:
                    icc_val = icc_31['ICC'].values[0]
                    ci = icc_31['CI95%'].values[0] if 'CI95%' in icc_31.columns else (icc_31['CI95'].values[0] if 'CI95' in icc_31.columns else '')
                    print(f"Numeric ICC(3,1) (7 Raters): {icc_val:.4f} (95% CI: {ci})")
            except Exception as e:
                print(f"Error computing ICC: {e}")
                
        if krippendorff is not None:
            try:
                alpha_num = krippendorff.alpha(reliability_data=combined_num_data, level_of_measurement='interval')
                print(f"Numeric Krippendorff Alpha (7 Raters): {alpha_num:.4f}")
            except Exception as e:
                pass

    if os.path.exists(llm_label_csv):
        df_llm_lab = pd.read_csv(llm_label_csv)
        label_map = {"correct": 2, "midway": 1, "wrong": 0}
        
        num_items = len(human_lab_data[0])
        llm_lab_data = [[np.nan]*num_items for _ in range(4)]
        
        for item_idx, row in df_llm_lab.iterrows():
            if item_idx >= num_items:
                break
            for i in range(1, 5):
                eval_col = f'Judge {i} Evaluation'
                if eval_col in df_llm_lab.columns:
                    eval_str = str(row[eval_col]).strip().lower()
                    if eval_str in label_map:
                        llm_lab_data[i-1][item_idx] = label_map[eval_str]
                        
        combined_lab_data = human_lab_data + llm_lab_data
        
        print(f"\n--- Combined Labels Comparison (API_Coherence_judges.csv) ---")
        if krippendorff is not None:
            try:
                alpha_lab = krippendorff.alpha(reliability_data=combined_lab_data, level_of_measurement='nominal')
                print(f"Labels Krippendorff Alpha (7 Raters): {alpha_lab:.4f}")
            except Exception as e:
                pass
                
        if irr is not None:
            try:
                data_t = np.array(combined_lab_data).T
                data_t_clean = data_t[~np.isnan(data_t).any(axis=1)]
                if len(data_t_clean) > 0:
                    table, categories = irr.aggregate_raters(data_t_clean)
                    kappa = irr.fleiss_kappa(table)
                    print(f"Labels Fleiss' Kappa (7 Raters) [on {len(data_t_clean)} complete items]: {kappa:.4f}")
                else:
                    print("No complete items found for Fleiss' Kappa across all 7 raters.")
            except Exception as e:
                print(f"Error computing Fleiss Kappa: {e}")

def singular_judge_comparison():
    print("\n--- Singular Judge vs Human Avg Comparison ---")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_calls_dir = os.path.join(current_dir, '..', 'data', "APICalls")
    xlsx_path = os.path.join(current_dir, '..', 'Le tediose ricette di Francesco (Risposte).xlsx')
    
    try:
        df_human = pd.read_excel(xlsx_path)
    except Exception as e:
        print(f"Could not read excel file: {e}")
        return
        
    human_agg = {}
    for rater_idx, row in df_human.iterrows():
        scores = row[1:].values
        for item_idx, score in enumerate(scores):
            if item_idx not in human_agg:
                human_agg[item_idx] = {'scores': [], 'labels': []}
            if pd.isna(score) or score == -1 or str(score).strip() == "":
                continue
            try:
                val = float(score)
                human_agg[item_idx]['scores'].append(val)
                if val >= 95:
                    human_agg[item_idx]['labels'].append(2)
                elif val <= 30:
                    human_agg[item_idx]['labels'].append(0)
                else:
                    human_agg[item_idx]['labels'].append(1)
            except ValueError:
                pass
                
    final_human_agg = {}
    for item_idx, data in human_agg.items():
        if data['scores']:
            final_human_agg[item_idx] = {
                'score': np.mean(data['scores']),
                'label': Counter(data['labels']).most_common(1)[0][0]
            }

    CSV_FILES_AND_MAX = {
        "API_Max1_judges.csv": 1,
        "API_Max10_judges.csv": 10,
        "API_Max100_judges.csv": 100
    }
    
    for file_name, max_score in CSV_FILES_AND_MAX.items():
        csv_path = os.path.join(api_calls_dir, file_name)
        if not os.path.exists(csv_path):
            continue
            
        df_llm_num = pd.read_csv(csv_path)
        
        judge_scores = {}
        
        for item_idx, row in df_llm_num.iterrows():
            for i in range(1, 5):
                model_col = f'Judge {i} Model Name'
                num_col = f'Judge {i} Numeric Score'
                if model_col in df_llm_num.columns and num_col in df_llm_num.columns:
                    model_name = str(row[model_col]).upper().strip()
                    if pd.isna(row[model_col]) or model_name == "NAN":
                        continue
                    val = row[num_col]
                    if not pd.isna(val) and val != -1:
                        if model_name not in judge_scores:
                            judge_scores[model_name] = {}
                        judge_scores[model_name][item_idx] = float(val)
                        
        print(f"\n--- Singular Judge Comparison ({file_name}) ---")
        for model_name, scores_map in judge_scores.items():
            shared_items = set(final_human_agg.keys()).intersection(set(scores_map.keys()))
            if not shared_items:
                continue
                
            deltas = []
            cvs = []
            sds = []
            spreads = []
            
            for item_idx in sorted(list(shared_items)):
                h_score_100 = final_human_agg[item_idx]['score']
                h_score_scaled = (h_score_100 / 100.0) * max_score
                l_score = scores_map[item_idx]
                
                delta = abs(h_score_scaled - l_score) / max_score
                deltas.append(delta)
                
                mean_score = np.mean([h_score_scaled, l_score])
                std_dev = np.std([h_score_scaled, l_score])
                spread = (max(h_score_scaled, l_score) - min(h_score_scaled, l_score)) / max_score
                
                cv = (std_dev / mean_score) if mean_score > 0 else 0
                cvs.append(cv)
                sds.append(std_dev / max_score)
                spreads.append(spread)
                
            avg_delta = np.mean(deltas)
            avg_cv = np.mean(cvs)
            avg_sd = np.mean(sds)
            avg_spread = np.mean(spreads) * 100 
            
            print(f"{model_name}: Avg Delta (Scaled 0-1) = {avg_delta:.4f} | Avg CV = {avg_cv:.4f} | Avg SD (Scaled 0-1) = {avg_sd:.4f} | Avg Spread = {avg_spread:.2f}% (based on {len(shared_items)} shared items)")

    llm_label_csv = os.path.join(api_calls_dir, 'API_Coherence_judges.csv')
    if os.path.exists(llm_label_csv):
        df_llm_lab = pd.read_csv(llm_label_csv)
        judge_labels = {}
        label_map = {"correct": 2, "midway": 1, "wrong": 0}
        
        for item_idx, row in df_llm_lab.iterrows():
            for i in range(1, 5):
                model_col = f'Judge {i} Model Name'
                eval_col = f'Judge {i} Evaluation'
                if model_col in df_llm_lab.columns and eval_col in df_llm_lab.columns:
                    model_name = str(row[model_col]).upper().strip()
                    if pd.isna(row[model_col]) or model_name == "NAN":
                        continue
                    eval_str = str(row[eval_col]).strip().lower()
                    if eval_str in label_map:
                        if model_name not in judge_labels:
                            judge_labels[model_name] = {}
                        judge_labels[model_name][item_idx] = label_map[eval_str]
                        
        print(f"\n--- Singular Judge Labels Comparison (API_Coherence_judges.csv) ---")
        for model_name, labels_map in judge_labels.items():
            shared_items = set(final_human_agg.keys()).intersection(set(labels_map.keys()))
            if not shared_items:
                continue
                
            matches = 0
            for item_idx in shared_items:
                h_lab = final_human_agg[item_idx]['label']
                l_lab = labels_map[item_idx]
                if h_lab == l_lab:
                    matches += 1
                    
            accuracy = matches / len(shared_items)
            print(f"{model_name} Accuracy: {accuracy:.2%} (based on {len(shared_items)} shared items)")

if __name__ == "__main__":
    human_difference()
    score_variation()
    coherence_accuracy()
    human_reliability()
    aggregate_comparison()
    combined_reliability()
    singular_judge_comparison()