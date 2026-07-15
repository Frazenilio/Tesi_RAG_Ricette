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

def normalize_model_name(name: str) -> str:
    if not name:
        return ""
    name = name.lower().strip()
    name = name.removeprefix("hf.co/")
    if name.endswith(":latest"):
        name = name.removesuffix(":latest")
    # Extract the base model name (e.g. from a path/repo structure)
    name = name.split("/")[-1]
    return name

def parse_judge_output(text: str) -> tuple[int, str]:
    # Normalize markdown bold markers and white space to make parsing robust
    cleaned = text.replace("**", "").replace("__", "").replace("`", "").strip()
    
    # Try Pattern 1: Standard structured format "Decision: [score]" or "Score: [score]"
    match = re.search(r"(?i)(?:Decision|Score)\s*:?\s*(\d+)", cleaned)
    if match:
        score = int(match.group(1))
        # Find explanation in rest of text, stopping if it hits another Decision/Score block
        exp_match = re.search(r"(?i)Explanation\s*:?\s*(.*?)(?=\n(?:Decision|Score)\s*:|$)", cleaned, re.DOTALL)
        explanation = exp_match.group(1).strip() if exp_match else text
        return min(max(score, 0), 100), explanation

    # Try Pattern 2: Any matching word followed by is/was/rating [score]
    match = re.search(r"(?i)(?:Decision|Score)\s+(?:is|was|rating|score)\s*(\d+)", cleaned)
    if match:
        score = int(match.group(1))
        return min(max(score, 0), 100), text

    # Try Pattern 3: Look for any line containing "decision" (case-insensitive) and extract the first number on that line
    for line in cleaned.split("\n"):
        if "decision" in line.lower() or "score" in line.lower():
            num_match = re.search(r"(\d+)", line)
            if num_match:
                score = int(num_match.group(1))
                return min(max(score, 0), 100), text

    # Try Pattern 4: Fallback to searching for any number between 0 and 100 in the entire response text
    numbers = re.findall(r"\b\d+\b", cleaned)
    if numbers:
        for num_str in numbers:
            val = int(num_str)
            if 0 <= val <= 100:
                return val, text

    return 0, text

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

    print(f"\nFound {len(json_files)} JSON files. Filtering and evaluating...")

    try:
        for json_path in json_files:
            # Skip path-based elements to speed up
            if "rag_vs_llm" in json_path.parts:
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
                continue

            # 1. Skip rag_vs_llm experiment
            if data.get("experiment") == "rag_vs_llm":
                continue

            # 2. Skip strategies not currently active in config (e.g. old names like 'combined')
            strategy = data.get("strategy")
            if strategy not in cfg.strategies:
                continue

            # 3. Skip recipe files if filter_recipes is set and recipe doesn't match
            recipe_in_json = data.get("recipe_name")
            if cfg.filter_recipes and recipe_in_json and recipe_in_json not in cfg.filter_recipes:
                continue

            print(f"\nProcessing {json_path}...")
            results = data.get("results", {})
            updated = False

            for size, size_data in results.items():
                qa_pairs = size_data.get("qa_pairs", [])
                for qa in qa_pairs:
                    recipe_name = qa.get("recipe_name", "")
                    
                    # Recipe filter for global JSON results files
                    if cfg.filter_recipes and recipe_name not in cfg.filter_recipes:
                        continue

                    prompted_query = qa.get("query", "")
                    
                    # Combine correct answers with numbered labels for the judge
                    correct_ing = qa.get("correct_ingredients", []) or []
                    correct_dir = qa.get("correct_directions", []) or []
                    correct_list = correct_ing + correct_dir
                    correct_answers = ""
                    for idx, ans in enumerate(correct_list, 1):
                        correct_answers += f"Reference {idx}: {ans}\n"

                    if not correct_answers:
                        continue

                    # Determine which responses exist
                    responses_to_evaluate = {}
                    for key in ["response", "response_rag", "response_llm"]:
                        if key in qa and qa[key]:
                            responses_to_evaluate[key] = qa[key]

                    if "judges" not in qa:
                        qa["judges"] = {}

                    generator_model = data.get("model", "")

                    for resp_key, llm_rag_answer in responses_to_evaluate.items():
                        if resp_key not in qa["judges"]:
                            qa["judges"][resp_key] = {}

                        judges_dict = qa["judges"][resp_key]
                        
                        # Filter judge runtimes to exclude the generator model itself
                        allowed_runtimes = []
                        for runtime in judge_runtimes:
                            model_name = runtime.spec.visible_label
                            if normalize_model_name(model_name) != normalize_model_name(generator_model):
                                allowed_runtimes.append(runtime)
                        
                        # Select exactly the first 3 eligible judges
                        selected_runtimes = allowed_runtimes[:3]
                        selected_names = {r.spec.visible_label for r in selected_runtimes}
                        
                        # Clean up any existing judge evaluations not in the selected 3
                        for judge_key in list(judges_dict.keys()):
                            if judge_key != "average_score" and judge_key not in selected_names:
                                judges_dict.pop(judge_key)
                                updated = True
                        
                        scores = []
                        all_judges_ran = True

                        for runtime in selected_runtimes:
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
                            score, explanation = parse_judge_output(eval_text)
                            if score == 0 and "0" not in eval_text:
                                print(f"    Warning: Could not parse non-zero score from {model_name}. Using 0. Raw text: {eval_text[:100]}...")

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
