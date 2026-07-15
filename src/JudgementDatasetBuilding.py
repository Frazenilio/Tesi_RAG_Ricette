import json
import csv
import os
from pathlib import Path

# Add project root to path if running directly to import metrics
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.metrics import compute_iou_stats

# --- Configuration ---
TARGET_FOLDER = Path(r"C:\Users\Franc\Documents\UniBS\Tesi\Tesi_RAG_Ricette\results\2026-07-15T11-32-35")
DROP_EXISTING_DATASET = True
OUTPUT_CSV = Path(r"C:\Users\Franc\Documents\UniBS\Tesi\Tesi_RAG_Ricette\data\judgement_dataset.csv")

def evaluate_correctness(original_answer: str, correct_answers: list[str]) -> str:
    """Evaluate correctness based on IoU score with reference answers."""
    max_iou, _, _ = compute_iou_stats(original_answer, correct_answers)
    if max_iou >= 0.95:
        return "correct"
    elif max_iou <= 0.3:
        return "wrong"
    else:
        return "midway correct"

def main():
    if DROP_EXISTING_DATASET and OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()
        print(f"Dropped existing dataset at {OUTPUT_CSV}")

    # Make sure output directory exists
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    file_exists = OUTPUT_CSV.exists()
    
    headers = [
        "Question", 
        "Reference Answers", 
        "RAG Model Name", 
        "Provided Answer", 
        "Correctness",
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

        for root, _, files in os.walk(TARGET_FOLDER):
            for file in files:
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
                        
                        # Serialize the list of reference answers to JSON string for CSV safely
                        ## TODO Aggiungivalutazione sulla base del punteggio umano
                        correct_answers = pair.get("correct_answers", [])
                        row["Reference Answers"] = json.dumps(correct_answers)
                        
                        row["RAG Model Name"] = rag_model_name
                        
                        original_answer = pair.get("original_answer", "")
                        row["Provided Answer"] = original_answer
                        
                        row["Correctness"] = evaluate_correctness(original_answer, correct_answers)
                        
                        row["Human Numeric Score"] = pair.get("human_score", "")
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
    
    # Track the number of each correctness category by reading the final dataset
    final_counts = {
        "correct": 0,
        "wrong": 0,
        "midway correct": 0
    }
    
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                correctness = row.get("Correctness", "")
                if correctness in final_counts:
                    final_counts[correctness] += 1
                    
    print("\n--- Correctness Breakdown (Total Dataset) ---")
    for category, count in final_counts.items():
        print(f"{category.capitalize()}: {count}")

if __name__ == "__main__":
    main()
