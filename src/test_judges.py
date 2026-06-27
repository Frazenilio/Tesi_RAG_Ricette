import json
import re
import sys
from pathlib import Path

# Add project root to path if running directly
sys.path.append(str(Path(__file__).parent.parent))

from src.config import load_config
from src.ollama_utils import check_ollama_server
from src.generation import create_model_runtime, query_llm
from src.prompts import PROMPT_LLM_JUDGE

def main():
    print("Loading config...")
    cfg = load_config()

    if not cfg.judges:
        print("Error: No judges configured in config.yaml.")
        sys.exit(1)

    print("Checking Ollama server...")
    ollama_ok = check_ollama_server(raise_on_failure=False)
    if not ollama_ok:
        print("Warning: Ollama server not reachable. Ensure it's running if you use Ollama models.")

    # Initialize the judge models
    judge_runtimes = []
    print("\nInitializing judges...")
    for judge_spec in cfg.judges:
        runtime = create_model_runtime(
            judge_spec,
            device=cfg.device,
            timeout=cfg.llm_timeout,
            ollama_available=ollama_ok,
            delete_after_run=cfg.delete_after_run,
        )
        judge_runtimes.append(runtime)

    # Find JSON results
    results_dir = Path("results")
    if not results_dir.exists():
        print("No results directory found.")
        sys.exit(1)

    # We will look for JSON files in the results directory
    json_files = list(results_dir.rglob("*.json"))
    if not json_files:
        print("No JSON files found in results directory.")
        sys.exit(0)

    # Filter files if filter_recipes is set
    filtered_json_files = []
    for jf in json_files:
        if not cfg.filter_recipes:
            filtered_json_files.append(jf)
            continue
            
        # Check if any filtered recipe is in the file path
        # Results paths look like: results/<timestamp>/<model>/<experiment>/<recipe>/...
        if any(recipe in str(jf) for recipe in cfg.filter_recipes):
            filtered_json_files.append(jf)

    print(f"\nFound {len(filtered_json_files)} JSON files to evaluate.")

    try:
        for json_path in filtered_json_files:
            print(f"\nProcessing {json_path}...")
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
                continue

            results = data.get("results", {})
            updated = False

            for size, size_data in results.items():
                qa_pairs = size_data.get("qa_pairs", [])
                for qa in qa_pairs:
                    prompted_query = qa.get("query", "")
                    
                    # Combine correct answers
                    correct_ing = qa.get("correct_ingredients", [])
                    correct_dir = qa.get("correct_directions", [])
                    correct_answers = "\n".join(correct_ing + correct_dir)

                    if not correct_answers:
                        continue

                    # Determine which responses exist
                    responses_to_evaluate = {}
                    for key in ["response", "response_rag", "response_llm"]:
                        if key in qa and qa[key]:
                            responses_to_evaluate[key] = qa[key]

                    if "judges" not in qa:
                        qa["judges"] = {}

                    for resp_key, llm_rag_answer in responses_to_evaluate.items():
                        if resp_key not in qa["judges"]:
                            qa["judges"][resp_key] = {}

                        judges_dict = qa["judges"][resp_key]
                        
                        scores = []
                        all_judges_ran = True

                        for runtime in judge_runtimes:
                            model_name = runtime.spec.visible_label
                            
                            # Skip if this judge has already evaluated this response
                            if model_name in judges_dict and "score" in judges_dict[model_name]:
                                scores.append(judges_dict[model_name]["score"])
                                continue
                            
                            # Format prompt
                            user_message = PROMPT_LLM_JUDGE.format(
                                prompted_query=prompted_query,
                                llm_rag_answer=llm_rag_answer,
                                correct_answers=correct_answers
                            )
                            system_prompt = "You are an impartial judge evaluating an answer."

                            print(f"  -> Asking {model_name} to evaluate {resp_key}...")
                            eval_text = query_llm(
                                system_prompt=system_prompt,
                                user_message=user_message,
                                model_runtime=runtime,
                                num_ctx=cfg.llm_num_ctx,
                                num_predict=cfg.llm_num_predict,
                                think=cfg.llm_think,
                                temperature=0.0, # Use greedy decoding for evaluation
                                timeout=cfg.llm_timeout,
                                retries=cfg.llm_retries,
                                keep_alive=cfg.llm_keep_alive,
                            )

                            # Parse Decision and Explanation
                            score = 0
                            explanation = eval_text
                            
                            # Regex to find Decision: [number]
                            match = re.search(r"(?i)Decision:\s*(\d+)", eval_text)
                            if match:
                                score = int(match.group(1))
                                # Optional: extract explanation
                                exp_match = re.search(r"(?i)Explanation:\s*(.*)", eval_text, re.DOTALL)
                                if exp_match:
                                    explanation = exp_match.group(1).strip()
                            else:
                                print(f"    Warning: Could not parse score from {model_name}. Using 0.")

                            judges_dict[model_name] = {
                                "score": score,
                                "explanation": explanation
                            }
                            scores.append(score)
                            updated = True

                        if scores:
                            avg_score = sum(scores) / len(scores)
                            judges_dict["average_score"] = avg_score

            if updated:
                print(f"Saving updated evaluations to {json_path}")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)

    finally:
        for runtime in judge_runtimes:
            runtime.close()

if __name__ == "__main__":
    main()
