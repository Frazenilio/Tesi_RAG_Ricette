import pandas as pd
import os
import json
import ast
from datetime import datetime

def create_labeling_sheet():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_calls_dir = os.path.join(current_dir, '..', 'data', "APICalls")
    data_dir = os.path.join(current_dir, '..', 'data')
    
    # 1. Get the 90 rows that were evaluated by the API judges
    max100_csv = os.path.join(api_calls_dir, 'API_Max100_judges.csv')
    if not os.path.exists(max100_csv):
        print(f"Error: Could not find {max100_csv}")
        return
        
    df_api = pd.read_csv(max100_csv)
    # Ensure original_row is present
    if "original_row" not in df_api.columns:
        df_api["original_row"] = df_api.index + 2
        
    sampled_rows = df_api["original_row"].unique()
    print(f"Found {len(sampled_rows)} sampled rows used by the API judges.")
    
    # 2. Get the actual text content from the main dataset
    judgement_csv = os.path.join(data_dir, 'judgement_dataset.csv')
    if not os.path.exists(judgement_csv):
        print(f"Error: Could not find {judgement_csv}")
        return
        
    df_main = pd.read_csv(judgement_csv)
    df_main["original_row"] = df_main.index + 2
    
    # Filter only the sampled rows
    df_labeling = df_main[df_main["original_row"].isin(sampled_rows)].copy()
    
    # 3. Create the simplified format for human labelers
    # We only keep what they NEED to read to evaluate
    output_df = pd.DataFrame()
    output_df["Row ID"] = df_labeling["original_row"]
    output_df["Question"] = df_labeling["Question"]
    output_df["Provided Answer (LLM)"] = df_labeling["Provided Answer"]
    output_df["Reference Answers (Correct)"] = df_labeling["Reference Answers"]
    
    # Add empty columns for the labelers to fill out
    output_df["YOUR SCORE (0-100)"] = -1
    output_df["YOUR EXPLANATION/NOTES (Optional)"] = ""
    
    # Sort by Row ID just to keep it organized
    output_df = output_df.sort_values(by="Row ID")
    
    # 4. Save to CSV and Excel (Excel is usually much easier for humans to read multi-line text)
    output_csv = os.path.join(data_dir, 'Human_Labeling_Task.csv')
    output_df.to_csv(output_csv, index=False)
    print(f"Successfully generated labeling sheet at: {output_csv}")
    
    try:
        output_excel = os.path.join(data_dir, 'Human_Labeling_Task.xlsx')
        output_df.to_excel(output_excel, index=False)
        print(f"Also generated an Excel version (easier to read) at: {output_excel}")
    except ImportError:
        print("\nTip: Run 'pip install openpyxl' if you want it to also generate a highly-readable Excel (.xlsx) file!")

    # 5. Generate JSON format
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "qa_pairs": []
    }
    
    for _, row in df_labeling.iterrows():
        try:
            correct_answers = ast.literal_eval(row["Reference Answers"])
        except (ValueError, SyntaxError):
            correct_answers = row["Reference Answers"]
            
        qa_pair = {
            "qa_index": int(row["original_row"]),
            "query": row["Question"],
            "correct_answers": correct_answers,
            "original_answer": row["Provided Answer"],
            "human_score": -1,
            "human_explanation": ""
        }
        json_data["qa_pairs"].append(qa_pair)
        
    output_json = os.path.join(data_dir, 'Human_Labeling_Task.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"Successfully generated JSON labeling sheet at: {output_json}")

if __name__ == "__main__":
    create_labeling_sheet()
