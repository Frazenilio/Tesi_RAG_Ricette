import csv
import random
import re
from pathlib import Path
import sys
import time
import pandas as pd

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.APIRequests.APIModels import Model
from src.APIRequests.APICall import APICall
from src.APIRequests.OpenRouterAPIRequest import OpenRouterAPIRequest
from src.APIRequests.GroqAPIRequest import GroqAPI
from src.JudgementDatasetBuilding import CORRECT_STR, WRONG_STR, MIDWAY_STR
from src.prompts import PROMPT_LLM_JUDGE_SHORT as PROMPT_LLM_JUDGE
from src.prompts import PROMPT_LLM_JUDGE_COHERENCE

# --- Configuration ---
INPUT_CSV: Path = Path("data/judgement_dataset.csv")
# OUTPUT_CSV: Path = Path("data/API_judges.csv")
REFERENCE_CSV: Path = Path("data/APICalls/API_Max100_judges.csv")
OUTPUT_CSV: Path = Path("data/APICalls/API_Max1_judges.csv")
# JUDGEMENT_CSV: Path = Path("data/APICalls/API_Explanations.csv")
JUDGEMENT_CSV: Path = Path("data/APICalls/API_Explanations_Max1.csv")

COHERENCE_OUT_CSV: Path = Path("data/APICalls/API_Coherence_judges.csv")
COHERENCE_EXP_CSV: Path = Path("data/APICalls/API_Explanations_Coherence.csv")

SAMPLE_SIZE: int = 30  # per category
MAX_RETRIES: int = 3
RETRY_DELAY: float = 5.0
COOLDOWN_JUDGE: float = 5.0

MAX_SCORE: int = 1
MIN_SCORE: int = 0
DEFAULT_SCORE: int = 0.8

# --- Judges to invoke (fill manually) ---
# Each entry is an APICall instance.
judges: dict[Model, APICall] = {
    Model.GEMMA : OpenRouterAPIRequest(Model.GEMMA),
    Model.QWEN : GroqAPI(Model.QWEN),
    Model.LLAMA : GroqAPI(Model.LLAMA),
    Model.GPT : GroqAPI(Model.GPT)
}

def parse_response(raw_response: str) -> tuple[float, str]:
    """Extracts Decision and Explanation from the raw API response."""
    # Find Decision
    score = -1.0
    # This allows things like "**Decision:**", "Decision :", or "Decision: The score is 100"
    score_match = re.search(r"(?i)Decision[^\d]*(\d+(?:\.\d+)?)", raw_response)
    if score_match:
        try:
            score = float(score_match.group(1))
        except ValueError:
            pass
            
    # Find Explanation
    explanation = raw_response
    exp_match = re.search(r"(?i)Explanation[^\w]*(.*)", raw_response, re.DOTALL)
    if exp_match:
        explanation = exp_match.group(1).strip()
        
    return score, explanation

def parse_response_coherence(raw_response: str) -> tuple[str, str]:
    """Extracts Classification and Explanation from the raw API response for Coherence."""
    classification = "ERROR"
    # Find Decision: [Correct, Midway, Wrong]
    class_match = re.search(r"(?i)Decision[^\w]*(Correct|Midway|Wrong)", raw_response)
    if class_match:
        classification = class_match.group(1).title()  # Normalize capitalization
        
    explanation = raw_response
    exp_match = re.search(r"(?i)Explanation[^\w]*(.*)", raw_response, re.DOTALL)
    if exp_match:
        explanation = exp_match.group(1).strip()
        
    return classification, explanation

def getPrompts(original_prompt: str) -> tuple[str, str]:
    prompt_parts = original_prompt.split("-------- CONTEXT START --------")
    system_prompt = prompt_parts[0].strip()
    question_template = "-------- CONTEXT START --------\n" + prompt_parts[1].strip()
    return system_prompt, question_template

def interrogateJudge(judge: APICall, system_prompt: str, formatted_question: str) -> tuple[float, str]:
    for attempt in range(MAX_RETRIES):
        try:
            raw_response = judge.call(question=formatted_question, system=system_prompt)
            score, explanation = parse_response(raw_response)
            print(f"Done (Score: {score})")
            return score, explanation
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"\nAPI Error on attempt {attempt+1}/{MAX_RETRIES}: {e}. Retrying in {RETRY_DELAY}s...", end="", flush=True)
                time.sleep(RETRY_DELAY)
            else:
                print(f"\nFailed again: {e}")
                return -1.0, f"API_ERROR: {e}"
    return -1.0, "Unknown Error"

def interrogateJudge_coherence(judge: APICall, system_prompt: str, formatted_question: str) -> tuple[str, str]:
    for attempt in range(MAX_RETRIES):
        try:
            raw_response = judge.call(question=formatted_question, system=system_prompt)
            classification, explanation = parse_response_coherence(raw_response)
            print(f"Done (Decision: {classification})")
            return classification, explanation
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"\nAPI Error on attempt {attempt+1}/{MAX_RETRIES}: {e}. Retrying in {RETRY_DELAY}s...", end="", flush=True)
                time.sleep(RETRY_DELAY)
            else:
                print(f"\nFailed again: {e}")
                return "ERROR", f"API_ERROR: {e}"
    return "ERROR", "Unknown Error"

def calculate_delta(human_score: int, model_score: float, max_score: int = MAX_SCORE) -> float:
    """Calculates the delta by scaling the 0-100 human score to the current max_score."""
    scaled_human = (human_score / 100.0) * max_score
    return round(model_score - scaled_human, 2)

def saveJudge(df_scores: pd.DataFrame, df_exps: pd.DataFrame, score: float, explanation: str, human_score: int, idx_row: int, idx_exp: int, idx_judge: int) -> None:
    score_col = f"Judge {idx_judge} Numeric Score"
    exp_col = f"Judge {idx_judge} Explanation"
    delta_col = f"Judge {idx_judge} delta score"
    
    df_scores.loc[idx_row, score_col] = score
    if score != -1:
        df_scores.loc[idx_row, delta_col] = calculate_delta(human_score, score)
    else:
        df_scores.loc[idx_row, delta_col] = pd.NA
        
    df_exps.loc[idx_exp, exp_col] = explanation

def saveJudge_coherence(df_scores: pd.DataFrame, df_exps: pd.DataFrame, classification: str, explanation: str, human_score: int, idx_row: int, idx_exp: int, idx_judge: int) -> None:
    class_col = f"Judge {idx_judge} Evaluation"
    exp_col = f"Judge {idx_judge} Explanation"
    
    df_scores.loc[idx_row, class_col] = classification
    df_exps.loc[idx_exp, exp_col] = explanation

def fill_csv(csv_path: Path, rows: list = None) -> None:
    if rows is None:
        rows = []
    if not INPUT_CSV.exists():
        print(f"Error: Input CSV {INPUT_CSV} not found.")
        return
        
    print(f"Reading {INPUT_CSV}...")
    
    df_input = pd.read_csv(INPUT_CSV)
    df_input["_original_row"] = df_input.index + 2  # Header is row 1
    
    if rows:
        # Use specific rows provided by the user
        sampled_df = df_input[df_input["_original_row"].isin(rows)]
        print(f"Using {len(sampled_df)} specifically requested rows.")
    else:
        # Sample rows randomly
        sampled_dfs = []
        for cat in [CORRECT_STR, WRONG_STR, MIDWAY_STR]:
            cat_df = df_input[df_input["Human Correctness"] == cat]
            sample_count = min(SAMPLE_SIZE, len(cat_df))
            if sample_count > 0:
                sampled_dfs.append(cat_df.sample(n=sample_count))
                print(f"Sampled {sample_count} rows for category '{cat}'")
            else:
                print(f"Warning: 0 rows found for category '{cat}'")
        sampled_df = pd.concat(sampled_dfs) if sampled_dfs else pd.DataFrame()
                
    total_sampled = len(sampled_df)
    print(f"Total sampled rows: {total_sampled}")
    
    if total_sampled == 0:
        print("Nothing to process.")
        return
        
    # Split prompt into system and question template
    system_prompt_raw, question_template = getPrompts(PROMPT_LLM_JUDGE)
    
    system_prompt = system_prompt_raw.format(
        min_score=MIN_SCORE,
        max_score=MAX_SCORE,
        default_score=DEFAULT_SCORE
    )

    # Build dynamic headers for output CSV
    out_headers = ["original_row"]
    exp_headers = ["original_row"]
    for i in range(1, len(judges) + 1):
        out_headers.extend([
            f"Judge {i} Model Name", 
            f"Judge {i} Numeric Score", 
            f"Judge {i} delta score"
        ])
        exp_headers.extend([
            f"Judge {i} Explanation"
        ])
        
    # Create or open output CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    exp_file_exists = JUDGEMENT_CSV.exists()
    
    for i, (_, row) in enumerate(sampled_df.iterrows(), start=1):
        orig_row_idx = int(row["_original_row"])
        human_corr = row.get("Human Correctness", "unknown")
        
        try:
            human_score = int(row.get("Human Numeric Score", 0))
        except (ValueError, TypeError):
            human_score = 0
            
        out_df = pd.DataFrame(columns=out_headers, index=[0])
        out_df.loc[0, "original_row"] = orig_row_idx
        
        exp_df = pd.DataFrame(columns=exp_headers, index=[0])
        exp_df.loc[0, "original_row"] = orig_row_idx
        
        q = row.get("Question", "")
        ans = row.get("Provided Answer", "")
        refs = row.get("Reference Answers", "[]")
        
        if pd.isna(q): q = ""
        if pd.isna(ans): ans = ""
        if pd.isna(refs): refs = "[]"
        
        formatted_question = question_template.format(
            prompted_query=q,
            llm_rag_answer=ans,
            correct_answers=refs,
            min_score = MIN_SCORE,
            max_score = MAX_SCORE,
            default_score = DEFAULT_SCORE
        )
        
        # Invoke judges
        for j_idx, judge in enumerate(judges.values(), start=1):
            model_name = judge.model.value
            print(f"[{i}/{total_sampled}] Row {orig_row_idx} ({human_corr}) — Judge {j_idx}/{len(judges)} ({model_name})... ", end="", flush=True)
            
            out_df.loc[0, f"Judge {j_idx} Model Name"] = model_name
            
            score, explanation = interrogateJudge(judge, system_prompt, formatted_question)
            saveJudge(out_df, exp_df, score, explanation, human_score, 0, 0, j_idx)
            
            time.sleep(COOLDOWN_JUDGE)
            
        # Write immediately
        out_df.to_csv(csv_path, mode='a', header=not file_exists, index=False)
        exp_df.to_csv(JUDGEMENT_CSV, mode='a', header=not exp_file_exists, index=False)
        file_exists = True
        exp_file_exists = True
            
    print(f"\nFinished processing. Results saved to {csv_path} and {JUDGEMENT_CSV}")

def fill_csv_coherence(csv_path: Path, exp_csv_path: Path, rows: list = None) -> None:
    if rows is None:
        rows = []
    if not INPUT_CSV.exists():
        print(f"Error: Input CSV {INPUT_CSV} not found.")
        return
        
    print(f"Reading {INPUT_CSV}...")
    df_input = pd.read_csv(INPUT_CSV)
    df_input["_original_row"] = df_input.index + 2
    
    if rows:
        sampled_df = df_input[df_input["_original_row"].isin(rows)]
        print(f"Using {len(sampled_df)} specifically requested rows.")
    else:
        sampled_dfs = []
        for cat in [CORRECT_STR, WRONG_STR, MIDWAY_STR]:
            cat_df = df_input[df_input["Human Correctness"] == cat]
            sample_count = min(SAMPLE_SIZE, len(cat_df))
            if sample_count > 0:
                sampled_dfs.append(cat_df.sample(n=sample_count))
                print(f"Sampled {sample_count} rows for category '{cat}'")
            else:
                print(f"Warning: 0 rows found for category '{cat}'")
        sampled_df = pd.concat(sampled_dfs) if sampled_dfs else pd.DataFrame()
                
    total_sampled = len(sampled_df)
    print(f"Total sampled rows: {total_sampled}")
    
    if total_sampled == 0:
        return
        
    system_prompt, question_template = getPrompts(PROMPT_LLM_JUDGE_COHERENCE)
    
    out_headers = ["original_row", "Human Numeric Score"]
    exp_headers = ["original_row"]
    for i in range(1, len(judges) + 1):
        out_headers.extend([f"Judge {i} Model Name", f"Judge {i} Evaluation"])
        exp_headers.extend([f"Judge {i} Explanation"])
        
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exp_csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    exp_file_exists = exp_csv_path.exists()
    
    for i, (_, row) in enumerate(sampled_df.iterrows(), start=1):
        orig_row_idx = int(row["_original_row"])
        human_corr = row.get("Human Correctness", "unknown")
        
        try:
            human_score = int(row.get("Human Numeric Score", 0))
        except (ValueError, TypeError):
            human_score = 0
            
        out_df = pd.DataFrame(columns=out_headers, index=[0])
        out_df.loc[0, "original_row"] = orig_row_idx
        out_df.loc[0, "Human Numeric Score"] = human_score
        
        exp_df = pd.DataFrame(columns=exp_headers, index=[0])
        exp_df.loc[0, "original_row"] = orig_row_idx
        
        q = row.get("Question", "") if not pd.isna(row.get("Question")) else ""
        ans = row.get("Provided Answer", "") if not pd.isna(row.get("Provided Answer")) else ""
        refs = row.get("Reference Answers", "[]") if not pd.isna(row.get("Reference Answers")) else "[]"
        
        formatted_question = question_template.format(
            prompted_query=q,
            llm_rag_answer=ans,
            correct_answers=refs
        )
        
        for j_idx, judge in enumerate(judges.values(), start=1):
            model_name = judge.model.value
            print(f"[{i}/{total_sampled}] Row {orig_row_idx} ({human_corr}) — Judge {j_idx}/{len(judges)} ({model_name})... ", end="", flush=True)
            
            out_df.loc[0, f"Judge {j_idx} Model Name"] = model_name
            
            classification, explanation = interrogateJudge_coherence(judge, system_prompt, formatted_question)
            saveJudge_coherence(out_df, exp_df, classification, explanation, human_score, 0, 0, j_idx)
            
            time.sleep(COOLDOWN_JUDGE)
            
        out_df.to_csv(csv_path, mode='a', header=not file_exists, index=False)
        exp_df.to_csv(exp_csv_path, mode='a', header=not exp_file_exists, index=False)
        file_exists = True
        exp_file_exists = True
            
    print(f"\nFinished processing. Results saved to {csv_path} and {exp_csv_path}")

def fix_missing(csv_missing: Path, csv_exp: Path = JUDGEMENT_CSV) -> None:
    df = pd.read_csv(csv_missing)
    df_exp = pd.read_csv(csv_exp) if csv_exp.exists() else pd.DataFrame()
    
    # We need the original questions/answers to re-prompt
    df_input = pd.read_csv(INPUT_CSV)
    df_input["_original_row"] = df_input.index + 2
    
    system_prompt_raw, question_template = getPrompts(PROMPT_LLM_JUDGE)
    system_prompt = system_prompt_raw.format(
        min_score=MIN_SCORE,
        max_score=MAX_SCORE,
        default_score=DEFAULT_SCORE
    )
    
    score_cols = [col for col in df.columns if col.endswith("Numeric Score")]
    mask = (df[score_cols] == -1).any(axis=1)
    rows_to_redo = df[mask].index.tolist()
    
    print(f"Found {len(rows_to_redo)} rows with errors that need re-evaluation.")
    
    for idx in rows_to_redo:
        orig_row_idx = int(df.loc[idx, "original_row"])
        
        orig_data_matches = df_input[df_input["_original_row"] == orig_row_idx]
        if orig_data_matches.empty:
            continue
        orig_data = orig_data_matches.iloc[0]
            
        try:
            human_score = int(orig_data.get("Human Numeric Score", 0))
        except (ValueError, TypeError):
            human_score = 0
            
        q = orig_data.get("Question", "")
        ans = orig_data.get("Provided Answer", "")
        refs = orig_data.get("Reference Answers", "[]")
        
        if pd.isna(q): q = ""
        if pd.isna(ans): ans = ""
        if pd.isna(refs): refs = "[]"
        
        formatted_question = question_template.format(
            prompted_query=q,
            llm_rag_answer=ans,
            correct_answers=refs,
            min_score = MIN_SCORE,
            max_score = MAX_SCORE,
            default_score = DEFAULT_SCORE
        )
        
        for j_idx, judge in enumerate(judges.values(), start=1):
            score_col = f"Judge {j_idx} Numeric Score"
            
            # Only re-evaluate if THIS specific judge failed on this row
            if df.loc[idx, score_col] == -1:
                model_name = judge.model.value
                print(f"Re-evaluating Row {orig_row_idx} — Judge {j_idx} ({model_name})... ", end="", flush=True)
                
                score, explanation = interrogateJudge(judge, system_prompt, formatted_question)
                
                exp_idx_matches = df_exp[df_exp["original_row"] == orig_row_idx].index if not df_exp.empty and "original_row" in df_exp.columns else []
                exp_idx = exp_idx_matches[0] if len(exp_idx_matches) > 0 else idx
                
                if exp_idx not in df_exp.index:
                    df_exp.loc[exp_idx, "original_row"] = orig_row_idx
                
                saveJudge(df, df_exp, score, explanation, human_score, idx, exp_idx, j_idx)
                
                time.sleep(COOLDOWN_JUDGE)
                
    # Save the updated DataFrame back to the CSV in-place
    df.to_csv(csv_missing, index=False)
    if not df_exp.empty:
        df_exp.to_csv(csv_exp, index=False)
    print(f"Saved fixed rows back to {csv_missing} and {csv_exp}")

def fix_missing_coherence(csv_missing: Path, csv_exp: Path) -> None:
    df = pd.read_csv(csv_missing)
    df_exp = pd.read_csv(csv_exp) if csv_exp.exists() else pd.DataFrame()
    
    df_input = pd.read_csv(INPUT_CSV)
    df_input["_original_row"] = df_input.index + 2
    
    system_prompt, question_template = getPrompts(PROMPT_LLM_JUDGE_COHERENCE)
    
    # Identify rows where ANY judge failed with "ERROR"
    class_cols = [col for col in df.columns if col.endswith(" Evaluation")]
    mask = (df[class_cols] == "ERROR").any(axis=1)
    rows_to_redo = df[mask].index.tolist()
    
    print(f"Found {len(rows_to_redo)} rows with errors that need re-evaluation.")
    
    for idx in rows_to_redo:
        orig_row_idx = int(df.loc[idx, "original_row"])
        
        orig_data_matches = df_input[df_input["_original_row"] == orig_row_idx]
        if orig_data_matches.empty:
            continue
        orig_data = orig_data_matches.iloc[0]
            
        try:
            human_score = int(orig_data.get("Human Numeric Score", 0))
        except (ValueError, TypeError):
            human_score = 0
            
        q = orig_data.get("Question", "") if not pd.isna(orig_data.get("Question")) else ""
        ans = orig_data.get("Provided Answer", "") if not pd.isna(orig_data.get("Provided Answer")) else ""
        refs = orig_data.get("Reference Answers", "[]") if not pd.isna(orig_data.get("Reference Answers")) else "[]"
        
        formatted_question = question_template.format(
            prompted_query=q,
            llm_rag_answer=ans,
            correct_answers=refs
        )
        
        for j_idx, judge in enumerate(judges.values(), start=1):
            class_col = f"Judge {j_idx} Evaluation"
            
            if df.loc[idx, class_col] == "ERROR":
                model_name = judge.model.value
                print(f"Re-evaluating Row {orig_row_idx} — Judge {j_idx} ({model_name})... ", end="", flush=True)
                
                classification, explanation = interrogateJudge_coherence(judge, system_prompt, formatted_question)
                
                exp_idx_matches = df_exp[df_exp["original_row"] == orig_row_idx].index if not df_exp.empty and "original_row" in df_exp.columns else []
                exp_idx = exp_idx_matches[0] if len(exp_idx_matches) > 0 else idx
                
                if exp_idx not in df_exp.index:
                    df_exp.loc[exp_idx, "original_row"] = orig_row_idx
                
                saveJudge_coherence(df, df_exp, classification, explanation, human_score, idx, exp_idx, j_idx)
                
                time.sleep(COOLDOWN_JUDGE)
                
    df.to_csv(csv_missing, index=False)
    if not df_exp.empty:
        df_exp.to_csv(csv_exp, index=False)
    print(f"Saved fixed rows back to {csv_missing} and {csv_exp}")

def main():
    # Read the exact rows evaluated in the previous run
    df_old = pd.read_csv(REFERENCE_CSV)
    old_rows = df_old["original_row"].tolist()
    
    # Pass them to fill_csv to ensure a 1-to-1 comparison
    ## For creating the csv anew (cfr.: new score scale)
    # fill_csv(OUTPUT_CSV, rows=old_rows)
    ## This one is for fixing missing rows (cfr.: token finished on previous run)
    # fix_missing(OUTPUT_CSV)
    
    ## Uncomment to run the full Coherence evaluation
    # fill_csv_coherence(COHERENCE_OUT_CSV, COHERENCE_EXP_CSV, rows=old_rows)
    
    ## Uncomment to fix missing coherence rows
    fix_missing_coherence(COHERENCE_OUT_CSV, COHERENCE_EXP_CSV)

if __name__ == "__main__":
    main()
