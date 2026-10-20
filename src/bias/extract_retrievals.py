import os
import json
import argparse
from pathlib import Path
import pandas as pd

# Default project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_LLAMA_JSON = PROJECT_ROOT / "results" / "2026-06-29T16-55-34" / "hf.co_bartowski_Llama-3.2-3B-Instruct-GGUF" / "rag" / "Doppio" / "directions" / "results_2026-06-29T16-55-34.json"
DEFAULT_GEMMA_JSON = PROJECT_ROOT / "results" / "2026-06-29T16-55-34" / "gemma3_1b" / "rag" / "Doppio" / "directions" / "results_2026-06-29T16-55-34.json"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "retrieval_base_test.csv"

def format_reference_answers(ref_list: list[str]) -> str:
    """Formats a list of reference answers into a numbered string block."""
    lines = []
    for idx, ans in enumerate(ref_list, 1):
        clean_ans = str(ans).strip()
        lines.append(f"Reference {idx}: {clean_ans}")
    return "\n".join(lines)

def extract_base_retrievals(
    llama_json_path: Path = DEFAULT_LLAMA_JSON,
    gemma_json_path: Path = DEFAULT_GEMMA_JSON,
    output_csv_path: Path = DEFAULT_OUTPUT_CSV,
    min_refs: int = 4,
    max_refs: int = 10,
    target_count: int = 30
) -> pd.DataFrame:
    """
    Extracts base retrieval examples from Llama 3.2 3B and Gemma 3 1B RAG results.
    Filters recipes by reference answer count (min_refs <= count <= max_refs).
    """
    if not llama_json_path.exists():
        raise FileNotFoundError(f"Llama results file not found at: {llama_json_path}")
    
    print(f"Loading Llama results from: {llama_json_path}")
    with open(llama_json_path, "r", encoding="utf-8") as f:
        llama_data = json.load(f)

    gemma_data = None
    if gemma_json_path and gemma_json_path.exists():
        print(f"Loading Gemma results from: {gemma_json_path}")
        with open(gemma_json_path, "r", encoding="utf-8") as f:
            gemma_data = json.load(f)
    else:
        print("Note: Gemma results file not found or not specified. Skipping Gemma responses.")

    # Index Gemma responses by recipe_name for quick lookup
    gemma_lookup = {}
    if gemma_data and "results" in gemma_data:
        for size_str, size_val in gemma_data["results"].items():
            for qa in size_val.get("qa_pairs", []):
                r_name = qa.get("recipe_name", "").strip().lower()
                if r_name:
                    gemma_lookup[r_name] = qa.get("response", "").strip()

    llama_results = llama_data.get("results", {})
    
    # Collect eligible sizes
    eligible_sizes = []
    for k in llama_results.keys():
        try:
            k_int = int(k)
            if min_refs <= k_int <= max_refs:
                eligible_sizes.append(k_int)
        except ValueError:
            continue
    eligible_sizes.sort()

    print(f"Eligible reference answer sizes in range [{min_refs}, {max_refs}]: {eligible_sizes}")

    records = []
    for size in eligible_sizes:
        qa_pairs = llama_results[str(size)].get("qa_pairs", [])
        for qa in qa_pairs:
            recipe_name = qa.get("recipe_name", "").strip()
            recipe_id = qa.get("recipe_id", "")
            query = qa.get("query", "")
            
            # Extract references
            correct_dirs = qa.get("correct_directions", []) or []
            correct_ings = qa.get("correct_ingredients", []) or []
            ref_list = correct_dirs if correct_dirs else correct_ings
            num_refs = len(ref_list)

            # Strict verification of constraint
            if not (min_refs <= num_refs <= max_refs):
                continue

            ref_formatted = format_reference_answers(ref_list)
            ref_json = json.dumps(ref_list, ensure_ascii=False)

            llama_resp = qa.get("response", "").strip()
            gemma_resp = gemma_lookup.get(recipe_name.lower(), "")

            # Baseline judge scores from original run (if available)
            judges_info = qa.get("judges", {}).get("response", {})
            baseline_gemma = judges_info.get("gemma3:1b", {}).get("score", None)
            baseline_ministral = judges_info.get("ministral-3:3b", {}).get("score", None)
            baseline_qwen = judges_info.get("qwen3.5:2b", {}).get("score", None)
            baseline_avg = judges_info.get("average_score", None)

            records.append({
                "recipe_name": recipe_name,
                "query": query,
                "target_type": "directions" if "directions" in query.lower() else "ingredients",
                "num_references": num_refs,
                "reference_answers_formatted": ref_formatted,
                "reference_answers_json": ref_json,
                "llama_rag_response": llama_resp,
                "gemma_rag_response": gemma_resp,
                "baseline_gemma_score": baseline_gemma,
                "baseline_ministral_score": baseline_ministral,
                "baseline_qwen_score": baseline_qwen,
                "baseline_avg_score": baseline_avg
            })

            if target_count and len(records) >= target_count:
                break
        if target_count and len(records) >= target_count:
            break

    df = pd.DataFrame(records)
    
    # Ensure output directory exists
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    
    print(f"\nSuccessfully extracted {len(df)} records to: {output_csv_path}")
    print("\n--- Summary Breakdown by Number of Reference Answers ---")
    counts = df["num_references"].value_counts().sort_index()
    for n_ref, count in counts.items():
        print(f"  {n_ref} References: {count} recipes")
    print(f"Total: {len(df)} recipes (Target: {target_count})")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Extract base RAG retrievals for bias testing.")
    parser.add_argument("--llama_json", type=Path, default=DEFAULT_LLAMA_JSON, help="Path to Llama results JSON")
    parser.add_argument("--gemma_json", type=Path, default=DEFAULT_GEMMA_JSON, help="Path to Gemma results JSON")
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output CSV path")
    parser.add_argument("--min_refs", type=int, default=4, help="Minimum reference answers (strict > 3)")
    parser.add_argument("--max_refs", type=int, default=10, help="Maximum reference answers")
    parser.add_argument("--target_count", type=int, default=30, help="Target number of recipes to extract")
    args = parser.parse_args()

    extract_base_retrievals(
        llama_json_path=args.llama_json,
        gemma_json_path=args.gemma_json,
        output_csv_path=args.output_csv,
        min_refs=args.min_refs,
        max_refs=args.max_refs,
        target_count=args.target_count
    )

if __name__ == "__main__":
    main()
