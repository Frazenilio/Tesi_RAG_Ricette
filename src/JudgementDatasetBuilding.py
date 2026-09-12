import json
import csv
import os
from pathlib import Path

# Add project root to path if running directly to import metrics
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.metrics import compute_iou_stats

# --- Configuration ---
ROOT_DIR = Path(__file__).parent.parent
TARGET_FOLDERS = [
    ROOT_DIR / "results" / "2026-07-15T11-32-35_START_DS",
    ROOT_DIR / "results" / "2026-07-17T10-32-24",
]
DROP_EXISTING_DATASET = True
OUTPUT_CSV = ROOT_DIR / "data" / "judgement_dataset.csv"

# --- Constants ---
CORRECT_STR = "correct"
WRONG_STR = "wrong"
MIDWAY_STR = "midway correct"

IOU_THRESH_HIGH = 0.95
IOU_THRESH_LOW = 0.3

HUMAN_THRESH_HIGH = 95
HUMAN_THRESH_LOW = 30

def evaluate_iou_correctness(original_answer: str, correct_answers: list[str]) -> str:
    """Evaluate correctness based on IoU score with reference answers."""
    max_iou, _, _ = compute_iou_stats(original_answer, correct_answers)
    if max_iou >= IOU_THRESH_HIGH:
        return CORRECT_STR
    elif max_iou <= IOU_THRESH_LOW:
        return WRONG_STR
    else:
        return MIDWAY_STR

def evaluate_human_correctness(score: int) -> str:
    if score >= HUMAN_THRESH_HIGH:
        return CORRECT_STR
    elif score <= HUMAN_THRESH_LOW:
        return WRONG_STR
    else:
        return MIDWAY_STR

def build_dataset():
    if DROP_EXISTING_DATASET and OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()
        print(f"Dropped existing dataset at {OUTPUT_CSV}")

    # Make sure output directory exists
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    file_exists = OUTPUT_CSV.exists()
    
    headers = [
        "Question", ## From question we can retrieve the recipe name and direction/ingredients but we save it anyway
        "Recipe",
        "Asked",
        "Reference Answers", 
        "RAG Model Name", 
        "Provided Answer", 
        "IoU Correctness", "Human Correctness",
        "Judge 1 Model Name", "Judge 1 Numeric Score", "Judge 1 Explanation",
        "Judge 2 Model Name", "Judge 2 Numeric Score", "Judge 2 Explanation",
        "Judge 3 Model Name", "Judge 3 Numeric Score", "Judge 3 Explanation",
        "Human Numeric Score", 
        "Human Explanation"
    ]

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        
        if not file_exists:
            writer.writeheader()
            
        rows_processed = 0

        for folder in TARGET_FOLDERS:
            for root, _, files in sorted(os.walk(folder)):
                for file in sorted(files):
                    if not file.endswith(".json"):
                        continue
                    
                    filepath = os.path.join(root, file)
                
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        
                    # Basic validation
                    if "qa_pairs" not in data or "model" not in data:
                        continue
                        
                    rag_model_name = data["model"]
                    
                    for pair in data["qa_pairs"]:
                        row = {}
                        row["Question"] = pair.get("query", "")
                        row["Recipe"] = pair.get("recipe_name", "")
                        row["Asked"] = "directions" if "directions" in pair.get("query", "").lower() else "ingredients"
                        
                        # Serialize the list of reference answers to JSON string for CSV safely
                        correct_answers = pair.get("correct_answers", [])
                        row["Reference Answers"] = json.dumps(correct_answers)
                        
                        row["RAG Model Name"] = rag_model_name
                        
                        original_answer = pair.get("original_answer", "")
                        row["Provided Answer"] = original_answer
                        
                        row["IoU Correctness"] = evaluate_iou_correctness(original_answer, correct_answers)
                        row["Human Correctness"] = evaluate_human_correctness(pair.get("human_score", 0))
                        
                        row["Human Numeric Score"] = pair.get("human_score", 0)
                        row["Human Explanation"] = pair.get("human_explanation", "")
                        
                        # Extract judges
                        round_1 = pair.get("round_1", {})
                        judges_dict = round_1.get("judges", {})
                        
                        # Sort judge model names to ensure deterministic assignment
                        judge_names = sorted(list(judges_dict.keys()))
                        
                        for i in range(1, 4):
                            judge_model_key = f"Judge {i} Model Name"
                            judge_score_key = f"Judge {i} Numeric Score"
                            judge_exp_key = f"Judge {i} Explanation"
                            
                            if i <= len(judge_names):
                                j_name = judge_names[i-1]
                                j_data = judges_dict[j_name]
                                row[judge_model_key] = j_name
                                row[judge_score_key] = j_data.get("score", "")
                                row[judge_exp_key] = j_data.get("explanation", "")
                            else:
                                row[judge_model_key] = ""
                                row[judge_score_key] = ""
                                row[judge_exp_key] = ""
                                
                        writer.writerow(row)
                        rows_processed += 1
                        
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
                    
    print(f"Successfully processed {rows_processed} QA pairs.")
    print(f"Dataset saved to: {OUTPUT_CSV}")
    
def print_dataset_stats():
    # Track the number of each correctness category by reading the final dataset
    iou_final_counts = { CORRECT_STR: 0, WRONG_STR: 0, MIDWAY_STR: 0 }
    human_final_counts = { CORRECT_STR: 0, WRONG_STR: 0, MIDWAY_STR: 0 }
    
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                iou_corr = row.get("IoU Correctness", "")
                human_corr = row.get("Human Correctness", "")
                
                if iou_corr in iou_final_counts:
                    iou_final_counts[iou_corr] += 1
                if human_corr in human_final_counts:
                    human_final_counts[human_corr] += 1
                    
    print("\n--- Correctness Breakdown (Total Dataset) ---")
    print(f"{'Category':<16} | {'IoU':<6} | {'Human':<6}")
    print("-" * 34)
    for category in [CORRECT_STR, WRONG_STR, MIDWAY_STR]:
        iou_c = iou_final_counts.get(category, 0)
        hum_c = human_final_counts.get(category, 0)
        print(f"{category.capitalize():<16} | {iou_c:<6} | {hum_c:<6}")

def main():
    build_dataset()
    print_dataset_stats()

if __name__ == "__main__":
    main()
