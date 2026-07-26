import csv
import random
import re
from pathlib import Path
import sys
import time

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.APIRequests.APIModels import Model
from src.APIRequests.APICall import APICall
from src.APIRequests.OpenRouterAPIRequest import OpenRouterAPIRequest
from src.APIRequests.GroqAPIRequest import GroqAPI
from src.JudgementDatasetBuilding import CORRECT_STR, WRONG_STR, MIDWAY_STR
from src.prompts import PROMPT_LLM_JUDGE

# --- Configuration ---
INPUT_CSV: Path = Path("data/judgement_dataset.csv")
OUTPUT_CSV: Path = Path("data/API_judges.csv")
SAMPLE_SIZE: int = 30  # per category
MAX_RETRIES: int = 3
RETRY_DELAY: float = 5.0
COOLDOWN_JUDGE: float = 5.0

# --- Judges to invoke (fill manually) ---
# Each entry is an APICall instance.
judges: list[APICall] = [
    OpenRouterAPIRequest(Model.GEMMA),
    GroqAPI(Model.QWEN),
    GroqAPI(Model.LLAMA),
    GroqAPI(Model.GPT)
]

def parse_response(raw_response: str) -> tuple[int, str]:
    """Extracts Decision and Explanation from the raw API response."""
    # Find Decision
    score = -1
    score_match = re.search(r"Decision:\s*(\d+)", raw_response)
    if score_match:
        try:
            score = int(score_match.group(1))
        except ValueError:
            pass
            
    # Find Explanation
    explanation = raw_response
    exp_match = re.search(r"Explanation:\s*(.*)", raw_response, re.DOTALL)
    if exp_match:
        explanation = exp_match.group(1).strip()
        
    return score, explanation

def main():
    if not INPUT_CSV.exists():
        print(f"Error: Input CSV {INPUT_CSV} not found.")
        return
        
    print(f"Reading {INPUT_CSV}...")
    
    # Read rows and track original row number (1-based index including header)
    # Header is row 1, data starts at row 2
    all_rows = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):
            row["_original_row"] = i
            all_rows.append(row)
            
    # Group by Human Correctness
    buckets = {
        CORRECT_STR: [],
        WRONG_STR: [],
        MIDWAY_STR: []
    }
    
    for row in all_rows:
        human_corr = row.get("Human Correctness", "")
        if human_corr in buckets:
            buckets[human_corr].append(row)
            
    # Sample rows
    sampled_rows = []
    for cat, items in buckets.items():
        sample_count = min(SAMPLE_SIZE, len(items))
        if sample_count > 0:
            sampled_rows.extend(random.sample(items, sample_count))
            print(f"Sampled {sample_count} rows for category '{cat}'")
        else:
            print(f"Warning: 0 rows found for category '{cat}'")
            
    total_sampled = len(sampled_rows)
    print(f"Total sampled rows: {total_sampled}")
    
    if total_sampled == 0:
        print("Nothing to process.")
        return
        
    # Split prompt into system and question template
    prompt_parts = PROMPT_LLM_JUDGE.split("-------- CONTEXT START --------")
    system_prompt = prompt_parts[0].strip()
    question_template = "-------- CONTEXT START --------\n" + prompt_parts[1].strip()
    
    # Build dynamic headers for output CSV
    out_headers = ["original_row"]
    for i in range(1, len(judges) + 1):
        out_headers.extend([
            f"Judge {i} Model Name", 
            f"Judge {i} Numeric Score", 
            f"Judge {i} Explanation",
            f"Judge {i} delta score"
        ])
        
    # Create or open output CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = OUTPUT_CSV.exists()
    
    # Open in append mode so we can resume/save as we go
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=out_headers)
        if not file_exists:
            writer.writeheader()
            
        for i, row in enumerate(sampled_rows, start=1):
            orig_row_idx = row["_original_row"]
            human_corr = row.get("Human Correctness", "unknown")
            try:
                human_score = int(row.get("Human Numeric Score", 0))
            except ValueError:
                human_score = 0
            
            # Prepare out_row
            out_row = {"original_row": orig_row_idx}
            
            # Prepare question
            q = row.get("Question", "")
            ans = row.get("Provided Answer", "")
            refs = row.get("Reference Answers", "[]")
            
            formatted_question = question_template.format(
                prompted_query=q,
                llm_rag_answer=ans,
                correct_answers=refs
            )
            
            # Invoke judges
            for j_idx, judge in enumerate(judges, start=1):
                model_name = judge.model.value
                print(f"[{i}/{total_sampled}] Row {orig_row_idx} ({human_corr}) — Judge {j_idx}/{len(judges)} ({model_name})... ", end="", flush=True)
                
                out_row[f"Judge {j_idx} Model Name"] = model_name
                
                for attempt in range(MAX_RETRIES):
                    try:
                        raw_response = judge.call(question=formatted_question, system=system_prompt)
                        score, explanation = parse_response(raw_response)
                        
                        out_row[f"Judge {j_idx} Numeric Score"] = score
                        out_row[f"Judge {j_idx} Explanation"] = explanation
                        if score != -1:
                            out_row[f"Judge {j_idx} delta score"] = score - human_score
                        else:
                            out_row[f"Judge {j_idx} delta score"] = ""
                        print(f"Done (Score: {score})")
                        ## Now wait before the next judge to prevent exaggerating with calls
                        time.sleep(COOLDOWN_JUDGE)
                        break  # Break out of retry loop on success
                    except Exception as e:
                        if attempt < MAX_RETRIES - 1:
                            print(f"\nAPI Error on attempt {attempt+1}/{MAX_RETRIES}: {e}. Retrying in {RETRY_DELAY}s...", end="", flush=True)
                            time.sleep(RETRY_DELAY)
                        else:
                            print(f"\nFailed after {MAX_RETRIES} attempts: {e}")
                            out_row[f"Judge {j_idx} Numeric Score"] = -1
                            out_row[f"Judge {j_idx} Explanation"] = f"API_ERROR: {e}"
                            out_row[f"Judge {j_idx} delta score"] = ""
                    
            # Write immediately
            writer.writerow(out_row)
            out_f.flush()
            
    print(f"\nFinished processing. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
