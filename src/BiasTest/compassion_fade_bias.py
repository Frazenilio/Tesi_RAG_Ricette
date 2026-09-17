import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from ollama import Client

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import load_config
try:
    from src.prompts import PROMPT_LLM_JUDGE_SHORT as PROMPT_LLM_JUDGE
except ImportError:
    try:
        from src.prompts import PROMPT_LLM_JUDGE
    except ImportError:
        PROMPT_LLM_JUDGE = (
            "You are an impartial judge evaluating an answer.\n"
            "Score from {min_score} to {max_score} based on logical match to Reference Answers.\n"
            "Question: {prompted_query}\n"
            "Provided Answer: {llm_rag_answer}\n"
            "Reference Answers: {correct_answers}\n"
            "Decision: [Score]\nExplanation: [Explanation]"
        )
from src.test_judges import parse_judge_output

DEFAULT_INPUT_CSV = PROJECT_ROOT / "data" / "retrieval_base_test.csv"
DEFAULT_SAVE_FOLDER = PROJECT_ROOT / "results" / "bias_tests"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"

# Model identifiers
LLAMA_MODEL_NAME = "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF"
GEMMA_MODEL_NAME = "gemma3:1b"

MIN_SCORE = 0
MAX_SCORE = 100
DEFAULT_SCORE = 50

# Attribution Framing Labels
ATTRIBUTION_HUMAN = "Chef Expert Human"
ATTRIBUTION_AI = "Autonomous AI Agent"
ATTRIBUTION_NEUTRAL = "Culinary Assistant"

TAG_AI = ATTRIBUTION_AI
TAG_HUMAN = ATTRIBUTION_HUMAN

def unload_model(client: Client | None, model_name: str) -> None:
    """Unloads a model from Ollama memory by setting keep_alive=0 to prevent OOM."""
    if client is None:
        return
    try:
        client.generate(model=model_name, keep_alive=0)
        print(f"Unloaded '{model_name}' from Ollama memory.")
    except Exception as e:
        print(f"Note: could not unload '{model_name}': {e}")

def evaluate_with_attribution(
    client: Client | None,
    judge_model: str,
    query: str,
    provided_answer: str,
    correct_answers_formatted: str,
    cfg,
    dry_run: bool = False,
    mock_score: float = 85.0
) -> tuple[float, str]:
    """Evaluates a single answer using the specified judge model via Ollama with automatic retries."""
    if dry_run or client is None:
        return mock_score, "[DRY RUN - Judge evaluation skipped. Prompt formatted successfully.]"

    user_message = PROMPT_LLM_JUDGE.format(
        min_score=MIN_SCORE,
        max_score=MAX_SCORE,
        default_score=DEFAULT_SCORE,
        prompted_query=query,
        llm_rag_answer=provided_answer,
        correct_answers=correct_answers_formatted
    )
    system_prompt = "You are an impartial judge evaluating an answer."

    max_retries = getattr(cfg, "llm_retries", 3)
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat(
                model=judge_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                stream=False,
                options={
                    "num_ctx": cfg.llm_num_ctx,
                    "num_predict": cfg.llm_num_predict,
                    "temperature": 0.0,  # Greedy decoding for consistency
                },
                keep_alive=cfg.llm_keep_alive,
            )
            eval_text = response.get("message", {}).get("content", "").strip()
            score, explanation = parse_judge_output(eval_text)
            return float(score), explanation
        except Exception as e:
            if attempt < max_retries:
                print(f"\n  [Retry {attempt}/{max_retries}] Error querying {judge_model}: {e}. Waiting 5s before retrying...")
                try:
                    unload_model(client, judge_model)
                except Exception:
                    pass
                time.sleep(5)
            else:
                raise e

evaluate_single_answer = evaluate_with_attribution

def ensure_model_available(client: Client, model_name: str) -> None:
    """Checks if model is available on Ollama server, pulling it if missing."""
    try:
        resp = client.list()
        local_models = getattr(resp, "models", []) if hasattr(resp, "models") else (resp.get("models", []) if isinstance(resp, dict) else [])
        model_names = []
        for m in local_models:
            name = getattr(m, "model", None) or getattr(m, "name", None)
            if not name and isinstance(m, dict):
                name = m.get("model") or m.get("name")
            if name:
                model_names.append(name)

        if model_name in model_names or f"{model_name}:latest" in model_names:
            return

        for name in model_names:
            if name.startswith(model_name) or model_name.startswith(name):
                return

        print(f"Model '{model_name}' not found in Ollama. Pulling now (please wait a moment)...")
        client.pull(model_name)
        print(f"Successfully pulled '{model_name}'!")
    except Exception as e:
        print(f"Note on checking/pulling '{model_name}': {e}")


def run_compassion_fade_for_judge(
    judge_key: str,  # "llama" or "gemma"
    judge_model: str,
    target_response: str = "llama",  # "llama" or "gemma"
    input_csv: Path = DEFAULT_INPUT_CSV,
    save_folder: Path = DEFAULT_SAVE_FOLDER,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    dry_run: bool = False
) -> Path:
    """Runs the compassion-fade / attribution bias test for a given judge and target response set."""
    df = pd.read_csv(input_csv)
    cfg = load_config()

    print(f"\n=======================================================")
    print(f"Running Compassion-Fade Test for Judge: {judge_key.upper()} ({judge_model})")
    print(f"Evaluating target response set: {target_response.upper()}")
    print(f"Ollama host: {ollama_host} | Dry run: {dry_run}")
    print(f"=======================================================")

    client = None
    if not dry_run:
        print(f"Connecting to Ollama at {ollama_host}...")
        client = Client(host=ollama_host, timeout=cfg.llm_timeout)
        try:
            client.list()
            print("Successfully connected to Ollama server.")
        except Exception as e:
            raise ConnectionError(f"Could not connect to Ollama at {ollama_host}: {e}")
        ensure_model_available(client, judge_model)

    results = []
    neutral_scores = []
    ai_scores = []
    human_scores = []

    target_col = f"{target_response}_rag_response"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {input_csv}. Available: {df.columns.tolist()}")

    total_recipes = len(df)
    for idx, row in df.iterrows():
        recipe_name = row["recipe_name"]
        query = row["query"]
        raw_answer = str(row[target_col]).strip()
        correct_answers_formatted = str(row["reference_answers_formatted"]).strip()

        print(f"[{idx+1:2d}/{total_recipes:2d}] Evaluating: {recipe_name}...")

        # 1. Neutral condition
        answer_neutral = raw_answer
        neutral_score, neutral_exp = evaluate_single_answer(
            client=client,
            judge_model=judge_model,
            query=query,
            provided_answer=answer_neutral,
            correct_answers_formatted=correct_answers_formatted,
            cfg=cfg,
            dry_run=dry_run,
            mock_score=85.0
        )

        # 2. AI-Attributed condition
        answer_ai = f"{TAG_AI}\n{raw_answer}"
        ai_score, ai_exp = evaluate_single_answer(
            client=client,
            judge_model=judge_model,
            query=query,
            provided_answer=answer_ai,
            correct_answers_formatted=correct_answers_formatted,
            cfg=cfg,
            dry_run=dry_run,
            mock_score=80.0
        )

        # 3. Human-Attributed condition
        answer_human = f"{TAG_HUMAN}\n{raw_answer}"
        human_score, human_exp = evaluate_single_answer(
            client=client,
            judge_model=judge_model,
            query=query,
            provided_answer=answer_human,
            correct_answers_formatted=correct_answers_formatted,
            cfg=cfg,
            dry_run=dry_run,
            mock_score=90.0
        )

        delta_ai_vs_neutral = round(ai_score - neutral_score, 2)
        delta_human_vs_neutral = round(human_score - neutral_score, 2)
        delta_ai_vs_human = round(ai_score - human_score, 2)

        neutral_scores.append(neutral_score)
        ai_scores.append(ai_score)
        human_scores.append(human_score)

        results.append({
            "recipe_name": recipe_name,
            "query": query,
            "neutral_evaluation": {
                "score": neutral_score,
                "explanation": neutral_exp
            },
            "ai_attributed_evaluation": {
                "attribution_tag": TAG_AI,
                "score": ai_score,
                "explanation": ai_exp
            },
            "human_attributed_evaluation": {
                "attribution_tag": TAG_HUMAN,
                "score": human_score,
                "explanation": human_exp
            },
            "delta_ai_vs_neutral": delta_ai_vs_neutral,
            "delta_human_vs_neutral": delta_human_vs_neutral,
            "delta_ai_vs_human": delta_ai_vs_human
        })

    mean_neutral = round(float(np.mean(neutral_scores)), 2)
    mean_ai = round(float(np.mean(ai_scores)), 2)
    mean_human = round(float(np.mean(human_scores)), 2)
    delta_ai_vs_neutral = round(mean_ai - mean_neutral, 2)
    delta_human_vs_neutral = round(mean_human - mean_neutral, 2)
    delta_ai_vs_human = round(mean_ai - mean_human, 2)

    timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")
    output_data = {
        "bias_test": "compassion-fade",
        "judge_model": judge_model,
        "evaluated_answers_source": target_response,
        "backend": "ollama",
        "timestamp": timestamp,
        "summary": {
            "mean_neutral_score": mean_neutral,
            "mean_ai_score": mean_ai,
            "mean_human_score": mean_human,
            "delta_ai_vs_neutral": delta_ai_vs_neutral,
            "delta_human_vs_neutral": delta_human_vs_neutral,
            "delta_ai_vs_human": delta_ai_vs_human
        },
        "results": results
    }

    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    out_file = save_folder / f"compassion_fade_{judge_key}_{target_response}_{timestamp}.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nCompleted Compassion-Fade evaluation for Judge: {judge_key.upper()} on {target_response.upper()} answers.")
    print(f"Summary:")
    print(f"  Mean Neutral Score:       {mean_neutral}")
    print(f"  Mean AI-Attributed Score: {mean_ai}")
    print(f"  Mean Human Score:         {mean_human}")
    print(f"  Delta (AI - Neutral):     {delta_ai_vs_neutral:+0.2f}")
    print(f"  Delta (Human - Neutral):  {delta_human_vs_neutral:+0.2f}")
    print(f"  Delta (AI - Human):       {delta_ai_vs_human:+0.2f}")
    print(f"Saved results to: {out_file}")

    return out_file

def run_compassion_fade_bias(
    judge: str = "both",
    target_response: str = "llama",
    save_folder: Path = DEFAULT_SAVE_FOLDER,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    dry_run: bool = False
) -> list[Path]:
    """Orchestrates the compassion-fade test across requested judge models and response sets."""
    cfg = load_config()
    llama_model = LLAMA_MODEL_NAME
    gemma_model = GEMMA_MODEL_NAME

    if cfg.judges:
        for j in cfg.judges:
            if "llama" in j.ollama_model.lower():
                llama_model = j.ollama_model
            elif "gemma" in j.ollama_model.lower():
                gemma_model = j.ollama_model

    judges_to_run = []
    judge_clean = judge.strip().lower()

    if judge_clean in ("llama", "both"):
        judges_to_run.append(("llama", llama_model))
    if judge_clean in ("gemma", "both"):
        judges_to_run.append(("gemma", gemma_model))

    if not judges_to_run:
        raise ValueError(f"Invalid judge '{judge}'. Must be 'llama', 'gemma', or 'both'.")

    targets_to_run = []
    target_clean = target_response.strip().lower()
    if target_clean in ("llama", "both"):
        targets_to_run.append("llama")
    if target_clean in ("gemma", "both"):
        targets_to_run.append("gemma")

    if not targets_to_run:
        raise ValueError(f"Invalid target-response '{target_response}'. Must be 'llama', 'gemma', or 'both'.")

    output_files = []
    for j_key, j_model in judges_to_run:
        for t_target in targets_to_run:
            out_file = run_compassion_fade_for_judge(
                judge_key=j_key,
                judge_model=j_model,
                target_response=t_target,
                save_folder=save_folder,
                ollama_host=ollama_host,
                dry_run=dry_run
            )
            output_files.append(out_file)
        if not dry_run:
            unload_model(Client(host=ollama_host, timeout=cfg.llm_timeout), j_model)

    return output_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compassion-Fade / Attribution Bias Test")
    parser.add_argument("--judge", type=str, default="both", choices=["llama", "gemma", "both"], help="Judge model to test")
    parser.add_argument("--target-response", type=str, default="llama", choices=["llama", "gemma", "both"], help="Response set to evaluate")
    parser.add_argument("--save-file", type=Path, default=DEFAULT_SAVE_FOLDER, help="Folder to save output JSON")
    parser.add_argument("--ollama_host", type=str, default=DEFAULT_OLLAMA_HOST, help="Ollama server host URL")
    parser.add_argument("--dry-run", action="store_true", help="Run without sending API calls")

    args = parser.parse_args()
    run_compassion_fade_bias(
        judge=args.judge,
        target_response=args.target_response,
        save_folder=args.save_file,
        ollama_host=args.ollama_host,
        dry_run=args.dry_run
    )
