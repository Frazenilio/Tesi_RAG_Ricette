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
DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# Model identifiers for Self-Enchantment
LLAMA_MODEL_NAME = "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF"
GEMMA_MODEL_NAME = "gemma3:1b"

MIN_SCORE = 0
MAX_SCORE = 100
DEFAULT_SCORE = 50

def unload_model(client: Client | None, model_name: str) -> None:
    """Unloads a model from Ollama memory by setting keep_alive=0 to prevent OOM."""
    if client is None:
        return
    try:
        client.generate(model=model_name, keep_alive=0)
        print(f"Unloaded '{model_name}' from Ollama memory.")
    except Exception as e:
        print(f"Note: could not unload '{model_name}': {e}")

def evaluate_single_answer(
    client: Client | None,
    judge_model: str,
    query: str,
    provided_answer: str,
    correct_answers_formatted: str,
    cfg,
    dry_run: bool = False
) -> tuple[float, str]:
    """Evaluates a single answer using the specified judge model via Ollama with automatic retries."""
    if dry_run or client is None:
        return 85.0, "[DRY RUN - Judge evaluation skipped. Prompt formatted successfully.]"

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
                    "temperature": 0.0,  # Greedy decoding for consistent evaluation
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


def run_self_enchantment_bias_for_judge(
    judge_key: str,  # "llama" or "gemma"
    judge_model: str,
    input_csv: Path = DEFAULT_INPUT_CSV,
    save_folder: Path = DEFAULT_SAVE_FOLDER,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    dry_run: bool = False
) -> Path:
    """Runs Self-Enchantment bias test for a single judge against its own generations vs the other model."""
    df = pd.read_csv(input_csv)
    cfg = load_config()

    print(f"\n=======================================================")
    print(f"Running Self-Enchantment Test for Judge: {judge_key.upper()} ({judge_model})")
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
    self_scores = []
    other_scores = []
    deltas = []

    for idx, row in df.iterrows():
        recipe_name = str(row["recipe_name"]).strip()
        query = str(row["query"]).strip()
        llama_ans = str(row.get("llama_rag_response", "")).strip()
        gemma_ans = str(row.get("gemma_rag_response", "")).strip()
        ref_formatted = str(row.get("reference_answers_formatted", "")).strip()

        # Identify which answer is Self and which is Other
        if judge_key.lower() == "llama":
            self_gen = "llama"
            self_ans = llama_ans
            other_gen = "gemma"
            other_ans = gemma_ans
        else:
            self_gen = "gemma"
            self_ans = gemma_ans
            other_gen = "llama"
            other_ans = llama_ans

        if not dry_run:
            print(f"[{idx+1:2d}/{len(df)}] Evaluating {recipe_name}...")
            print(f"  -> Judging SELF ({self_gen})...")
        self_score, self_exp = evaluate_single_answer(
            client=client,
            judge_model=judge_model,
            query=query,
            provided_answer=self_ans,
            correct_answers_formatted=ref_formatted,
            cfg=cfg,
            dry_run=dry_run
        )

        if not dry_run:
            print(f"  -> Judging OTHER ({other_gen})...")
        other_score, other_exp = evaluate_single_answer(
            client=client,
            judge_model=judge_model,
            query=query,
            provided_answer=other_ans,
            correct_answers_formatted=ref_formatted,
            cfg=cfg,
            dry_run=dry_run
        )

        delta = round(self_score - other_score, 2)
        self_scores.append(self_score)
        other_scores.append(other_score)
        deltas.append(delta)

        results.append({
            "recipe_name": recipe_name,
            "query": query,
            "self_evaluation": {
                "generator": self_gen,
                "answer": self_ans,
                "score": self_score,
                "explanation": self_exp
            },
            "other_evaluation": {
                "generator": other_gen,
                "answer": other_ans,
                "score": other_score,
                "explanation": other_exp
            },
            "delta": delta
        })

    # Summary statistics
    mean_self = round(float(np.mean(self_scores)), 2)
    mean_other = round(float(np.mean(other_scores)), 2)
    mean_delta = round(float(np.mean(deltas)), 2)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    judge_slug = judge_key.lower()

    output_data = {
        "bias_test": "self-enchantment",
        "judge_model": judge_model,
        "backend": "ollama",
        "timestamp": timestamp,
        "summary": {
            "mean_self_score": mean_self,
            "mean_other_score": mean_other,
            "mean_delta": mean_delta
        },
        "results": results
    }

    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    out_file = save_folder / f"self_enchantment_{judge_slug}_{timestamp}.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"\nCompleted Self-Enchantment test for {judge_key.upper()}!")
    print(f"  Mean Self Score:  {mean_self}")
    print(f"  Mean Other Score: {mean_other}")
    print(f"  Mean Delta:       {mean_delta}")
    print(f"Saved results to: {out_file}")
    return out_file

def run_self_enchantment_bias(
    judge: str = "both",
    input_csv: Path = DEFAULT_INPUT_CSV,
    save_folder: Path = DEFAULT_SAVE_FOLDER,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    dry_run: bool = False
) -> list[Path]:
    """Runs Self-Enchantment bias test for LLaMA, Gemma, or both."""
    judge = judge.lower().strip()
    generated_files = []

    cfg = load_config()
    llama_model = LLAMA_MODEL_NAME
    gemma_model = GEMMA_MODEL_NAME

    # Check if config has custom model names
    if cfg.judges:
        for j in cfg.judges:
            if "llama" in j.ollama_model.lower():
                llama_model = j.ollama_model
            elif "gemma" in j.ollama_model.lower():
                gemma_model = j.ollama_model

    if judge in ("llama", "both"):
        f_llama = run_self_enchantment_bias_for_judge(
            judge_key="llama",
            judge_model=llama_model,
            input_csv=input_csv,
            save_folder=save_folder,
            ollama_host=ollama_host,
            dry_run=dry_run
        )
        generated_files.append(f_llama)
        if not dry_run:
            unload_model(Client(host=ollama_host, timeout=cfg.llm_timeout), llama_model)

    if judge in ("gemma", "both"):
        f_gemma = run_self_enchantment_bias_for_judge(
            judge_key="gemma",
            judge_model=gemma_model,
            input_csv=input_csv,
            save_folder=save_folder,
            ollama_host=ollama_host,
            dry_run=dry_run
        )
        generated_files.append(f_gemma)
        if not dry_run:
            unload_model(Client(host=ollama_host, timeout=cfg.llm_timeout), gemma_model)

    return generated_files

def main():
    parser = argparse.ArgumentParser(description="Run Self-Enchantment Bias experiment.")
    parser.add_argument("--judge", type=str, default="both", choices=["llama", "gemma", "both"], help="Judge model to test")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Input base CSV")
    parser.add_argument("--save-file", type=Path, default=DEFAULT_SAVE_FOLDER, help="Folder where to save the result JSON")
    parser.add_argument("--ollama-host", type=str, default=DEFAULT_OLLAMA_HOST, help="Ollama server host URL")
    parser.add_argument("--dry-run", action="store_true", help="Preview output without querying the judge")
    args = parser.parse_args()

    run_self_enchantment_bias(
        judge=args.judge,
        input_csv=args.input_csv,
        save_folder=args.save_file,
        ollama_host=args.ollama_host,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
