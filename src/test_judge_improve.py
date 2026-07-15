import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path if running directly
sys.path.append(str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import load_config, Config
from src.data import load_data, build_grouped_by_size, build_grouped_by_size_controlled
from src.experiments import prepare_generation_cases
from src.generation import create_model_runtime, query_llm
from src.ollama_utils import check_ollama_server
from src.prompts import PROMPT_LLM_JUDGE, PROMPT_LIST_CORRECTOR, SYSTEM_PROMPT_CORRECTOR, PROMPT_ANSWER_CORRECTOR
from src.test_judges import normalize_model_name, parse_judge_output


second_round: bool = False

def parse_corrector_output(text: str) -> tuple[str, str]:
    """
    Parse the output of the corrector LLM to extract the Corrected Answer
    and the Correction List.
    """
    cleaned = text.strip()
    
    # Use regex to find Corrected Answer: and Correction List:
    # re.DOTALL ensures . matches newlines, re.IGNORECASE makes it case insensitive
    match = re.search(r"Corrected Answer\s*:?\s*(.*?)Correction List\s*:?\s*(.*)", cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    if match:
        corrected_answer = match.group(1).strip()
        correction_list = match.group(2).strip()
        return corrected_answer, correction_list
    
    # If the regex fails, it might just be the answer, or formatted unexpectedly.
    # Fallback: assume the whole text is the corrected answer
    return cleaned, "Parse error: couldn't separate answer and correction list."

def select_judges(judge_runtimes: list, generator_model: str, num_judges: int = 3) -> list:
    """
    Select up to `num_judges` judge runtimes, excluding the generator model itself.
    """
    allowed_runtimes = []
    for runtime in judge_runtimes:
        model_name = runtime.spec.visible_label
        if normalize_model_name(model_name) != normalize_model_name(generator_model):
            allowed_runtimes.append(runtime)
            if len(allowed_runtimes) == num_judges:
                break
    return allowed_runtimes

def run_judge_round(answer: str, query: str, correct_answers_str: str, judges: list, cfg: Config) -> tuple[dict, float]:
    """
    Run a round of evaluation using the provided judges on a single answer.
    """
    judges_dict = {}
    scores = []
    
    for runtime in judges:
        model_name = runtime.spec.visible_label
        
        user_message = PROMPT_LLM_JUDGE.format(
            prompted_query=query,
            llm_rag_answer=answer,
            correct_answers=correct_answers_str
        )
        system_prompt = "You are an impartial judge evaluating an answer."
        
        print(f"      -> Asking judge {model_name}...")
        eval_text = query_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            model_runtime=runtime,
            num_ctx=cfg.llm_num_ctx,
            num_predict=cfg.llm_num_predict,
            think=cfg.llm_think,
            temperature=0.0, # Greedy decoding for evaluation
            timeout=cfg.llm_timeout,
            retries=cfg.llm_retries,
            keep_alive=cfg.llm_keep_alive,
        )
        
        score, explanation = parse_judge_output(eval_text)
        if score == 0 and "0" not in eval_text:
            print(f"        Warning: Could not parse non-zero score from {model_name}. Using 0. Raw text: {eval_text[:100]}...")
            
        judges_dict[model_name] = {
            "score": score,
            "explanation": explanation
        }
        scores.append(score)
        
    avg_score = sum(scores) / len(scores) if scores else 0.0
    return judges_dict, avg_score

def provide_judgement(original_answer: str, judgement: str, current_cl: str, \
    generator_runtime, cfg) -> str:
    ## if there is a list use it. If it's an empty string then use "Empty." as by prompt
    list: str = current_cl if current_cl else "Empty."
    corrector_user_message = PROMPT_LIST_CORRECTOR.format(
                original_answer=original_answer,
                judgements=judgement,
                correction_list=list
            )

    ## TODO add the parsing method based on the PROMPT_LIST_CORRECTOR. The output is a str
    ## but the format is JSON-like (see the prompt to understand) 
    raw_response: str = query_llm(
        system_prompt=SYSTEM_PROMPT_CORRECTOR,
        user_message=corrector_user_message,
        model_runtime=generator_runtime,
        num_ctx=cfg.llm_num_ctx,
        num_predict=cfg.llm_num_predict,
        think=cfg.llm_think,
        temperature=cfg.llm_temperature, # Same as RAG generation
        timeout=cfg.llm_timeout,
        retries=cfg.llm_retries,
        keep_alive=cfg.llm_keep_alive,
    )

    # Parse the output based on PROMPT_LIST_CORRECTOR
    match = re.search(r"<correction_list>(.*?)<[/\\]correction_list>", raw_response, flags=re.IGNORECASE | re.DOTALL)
    if match:
        ans = match.group(1).strip()
    else:
        ans = raw_response.strip()
        
    # Clean up markdown code blocks if the model wrapped it anyway
    ans = re.sub(r"^```(?:json)?\n?", "", ans, flags=re.IGNORECASE)
    ans = re.sub(r"\n?```$", "", ans, flags=re.IGNORECASE)
    return ans.strip()

def correct_answer(original_answer: str, correction_list: str, \
        generator_runtime, cfg) -> str:
    corrector_message: str = PROMPT_ANSWER_CORRECTOR.format(
        original_answer=original_answer,
        correction_list=correction_list
    )

    raw_response: str = query_llm(
        system_prompt=SYSTEM_PROMPT_CORRECTOR,
        user_message=corrector_message,
        model_runtime=generator_runtime,
        num_ctx=cfg.llm_num_ctx,
        num_predict=cfg.llm_num_predict,
        think=cfg.llm_think,
        temperature=cfg.llm_temperature, # Same as RAG generation
        timeout=cfg.llm_timeout,
        retries=cfg.llm_retries,
        keep_alive=cfg.llm_keep_alive,
    )

    # Parse the output based on PROMPT_ANSWER_CORRECTOR
    # Support <correct_answer> as well since some models hallucinate the exact tag name
    match = re.search(r"<(?:corrected_answer|correct_answer)>(.*?)<[/\\](?:corrected_answer|correct_answer)>", raw_response, flags=re.IGNORECASE | re.DOTALL)
    if match:
        ans = match.group(1).strip()
    else:
        ans = raw_response.strip()
        
    # Clean up markdown code blocks if the model wrapped it anyway
    ans = re.sub(r"^```(?:markdown|text)?\n?", "", ans, flags=re.IGNORECASE)
    ans = re.sub(r"\n?```$", "", ans, flags=re.IGNORECASE)
    return ans.strip()
    
    

def main():
    parser = argparse.ArgumentParser(description="2-Round RAG -> Judge -> Corrector -> Judge Pipeline")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("config.yaml"),
        metavar="PATH",
        help="Path to a YAML config file (default: config.yaml at project root)",
    )
    parser.add_argument(
        "--use-oracle",
        action="store_true",
        help="Use the oracle database (bypass FAISS and use ground truth chunks as context)",
    )
    args = parser.parse_args()

    print("Loading config...")
    cfg = load_config(args.config)
    
    if not cfg.judges:
        print("Error: No judges configured in config.yaml.")
        sys.exit(1)

    print("Checking Ollama backend...")
    ollama_available = check_ollama_server()
    if not ollama_available:
        print("Warning: Ollama server not reachable. Ensure it's running if you use Ollama models.")

    # Initialize the judge models
    judge_runtimes = []
    print("\nInitializing judges...")
    for judge_spec in cfg.judges:
        runtime = create_model_runtime(
            judge_spec,
            device=cfg.device,
            timeout=cfg.llm_timeout,
            ollama_available=ollama_available,
            delete_after_run=cfg.delete_after_run,
        )
        judge_runtimes.append(runtime)

    try:
        encoding_model = SentenceTransformer(cfg.encoding_model, device=cfg.device)

        print("\nLoading data...")
        df = load_data(
            cfg.base_path,
            cfg.recipes_csv,
            cfg.ingredients_csv,
            cfg.directions_csv,
            cfg.filter_recipes,
        )
        
        # Hardcode strategy as per requirements
        strategy = "Singolo-Distinti"
        
        if args.use_oracle:
            print(f"\nUsing ORACLE mode. FAISS indexing will be bypassed.")
            
        print(f"\nBuilding grouped data for strategy '{strategy}'...")
        if cfg.controlled_dataset:
            groups = build_grouped_by_size_controlled(
                df, encoding_model, cfg.device, strategy, cfg.n_per_size
            )
        else:
            groups = build_grouped_by_size(
                df, encoding_model, cfg.device, strategy
            )

        timestamp = datetime.now().isoformat(timespec="seconds")
        ts_slug = timestamp.replace(":", "-")
        results_dir = Path("results") / ts_slug
        results_dir.mkdir(parents=True, exist_ok=True)

        for model_spec in cfg.language_models:
            print(f"\n--- Preparing generator model: {model_spec.visible_label} ---")
            with create_model_runtime(
                model_spec,
                device=cfg.device,
                timeout=cfg.llm_timeout,
                ollama_available=ollama_available,
                delete_after_run=cfg.delete_after_run,
            ) as generator_runtime:
                
                model_dir = results_dir / generator_runtime.slug
                model_dir.mkdir(parents=True, exist_ok=True)
                
                # Select the 3 eligible judges (excluding the generator model)
                selected_judges = select_judges(judge_runtimes, generator_runtime.results_label)
                judge_names = [j.spec.visible_label for j in selected_judges]
                print(f"  Selected judges for {generator_runtime.results_label}: {judge_names}")

                for target_type in ["ingredients", "directions"]:
                    print(f"\n  Running target: {target_type.upper()}")
                    
                    # Run RAG generation
                    generated_cases = prepare_generation_cases(
                        groups,
                        encoding_model,
                        cfg,
                        generator_runtime,
                        target_type,
                        include_rag=True,
                        include_llm=False, # We only need RAG
                        use_oracle=args.use_oracle,
                        trace_path=None,   # Don't save trace for simplicity
                    )
                    
                    correct_key = "correct_ingredients" if target_type == "ingredients" else "correct_directions"
                    qa_pairs = []

                    for size, cases in generated_cases.items():
                        print(f"    Evaluating size {size}...")
                        for case in cases:
                            # 1. Format correct answers
                            correct_list = getattr(case, "correct_chunks", [])
                            correct_answers_str = ""
                            for idx, ans in enumerate(correct_list, 1):
                                correct_answers_str += f"Reference {idx}: {ans}\n"
                            
                            original_answer = case.response_rag
                            
                            # Skip if something is missing
                            if not correct_answers_str or original_answer is None:
                                continue

                            qa_entry = {
                                "qa_index": case.qa_index,
                                "recipe_name": case.recipe_name,
                                "query": case.query,
                                "correct_answers": [f"Reference {idx}: {ans}" for idx, ans in enumerate(correct_list, 1)],
                                "original_answer": original_answer,
                                "human_score": 50,
                                "human_explanation": "",
                            }
                            
                            print(f"      [Round 1] Judging {case.recipe_name}...")
                            # 2. Round 1: Judges evaluate original answer
                            r1_judges_dict, r1_avg = run_judge_round(
                                original_answer, 
                                case.query, 
                                correct_answers_str, 
                                selected_judges, 
                                cfg
                            )
                            
                            qa_entry["round_1"] = {
                                "judges": r1_judges_dict,
                                "average_score": r1_avg
                            }

                            if second_round:
                                print(f"      [Correction] {generator_runtime.results_label} is creating the CL")
                                # Build judgements string for the corrector
                                # judgements_str = ""
                                correction_list = ""
                                for judge_name, judge_data in r1_judges_dict.items():
                                    ## Score removed for simplicity since it wasn't used
                                    # judgements_str += f"Judgement: {judge_data['explanation']}\n"
                                    correction_list = provide_judgement(original_answer, judge_data['explanation'], \
                                        correction_list, generator_runtime=generator_runtime, cfg=cfg)
                                
                                ## 4. Now ask the LLM to fix the answer
                                print(f"      [Correction] {generator_runtime.results_label} is correcting the answer")

                                corrected_answer = correct_answer(original_answer, correction_list, generator_runtime=generator_runtime, cfg=cfg)


                                # 5. Round 2: Judges evaluate corrected answer
                                print(f"      [Round 2] Judging corrected answer for {case.recipe_name}...")
                                r2_judges_dict, r2_avg = run_judge_round(
                                    corrected_answer, 
                                    case.query, 
                                    correct_answers_str, 
                                    selected_judges, 
                                    cfg
                                )
                                
                                qa_entry["round_2"] = {
                                    "corrected_answer": corrected_answer,
                                    "correction_list": correction_list,
                                    "judges": r2_judges_dict,
                                    "average_score": r2_avg,
                                }
                            
                            qa_pairs.append(qa_entry)
                            
                    # Save JSON for this model + target_type
                    payload = {
                        "model": generator_runtime.results_label,
                        "strategy": strategy,
                        "target_type": target_type,
                        "timestamp": ts_slug,
                        "qa_pairs": qa_pairs
                    }
                    
                    out_path = model_dir / f"judge_improve_{target_type}_{ts_slug}.json"
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2)
                    print(f"  Saved JSON to {out_path}")

    finally:
        for runtime in judge_runtimes:
            runtime.close()

if __name__ == "__main__":
    main()
