import os
import sys
import json
import random
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
from ollama import Client

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import load_config
from src.prompts import SYSTEM_PROMPT_DIRECTIONS_RAG, SYSTEM_PROMPT_RAG

DEFAULT_INPUT_CSV = PROJECT_ROOT / "data" / "retrieval_base_test.csv"
DEFAULT_SAVE_FOLDER = PROJECT_ROOT / "results" / "bias_tests"
DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

def deterministic_shuffle(items: list[str], seed: int, recipe_idx: int) -> tuple[list[tuple[int, str]], list[str]]:
    """
    Shuffles items deterministically while ensuring items change positions.
    Returns:
      - indexed_shuffled: list of (original_1_indexed_pos, item_text)
      - raw_shuffled: list of item_text (clean for RAG prompt)
    """
    n = len(items)
    indexed = list(enumerate(items, 1))  # [(1, text1), (2, text2), ...]
    
    rng = random.Random(seed + recipe_idx * 1000)
    
    # Generate a permutation that moves at least the first item (or as many as possible)
    shuffled = indexed.copy()
    for _ in range(10):
        rng.shuffle(shuffled)
        # Check if first item moved or order changed
        if shuffled[0][0] != 1 or n <= 1:
            break
            
    raw_shuffled = [text for _, text in shuffled]
    return shuffled, raw_shuffled

def unload_model(client: Client | None, model_name: str) -> None:
    """Unloads a model from Ollama memory by setting keep_alive=0 to prevent OOM."""
    if client is None:
        return
    try:
        client.generate(model=model_name, keep_alive=0)
        print(f"Unloaded '{model_name}' from Ollama memory.")
    except Exception as e:
        print(f"Note: could not unload '{model_name}': {e}")

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


def run_position_bias(
    input_csv: Path = DEFAULT_INPUT_CSV,
    save_folder: Path = DEFAULT_SAVE_FOLDER,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    model_override: str | None = None,
    dry_run: bool = False,
    seed: int = 42
) -> Path:
    """
    Executes the Position Bias test by shuffling RAG context and querying the model.
    Saves results to a JSON file in save_folder.
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"Input base dataset not found at: {input_csv}. Run extraction first.")

    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} recipes from {input_csv}")

    cfg = load_config()
    
    # Determine model to use (from config.yaml or override)
    if model_override:
        model_name = model_override
    elif cfg.language_models:
        model_name = cfg.language_models[0].ollama_model
    else:
        model_name = "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF"

    print(f"Using model: {model_name}")
    print(f"Ollama host: {ollama_host}")
    print(f"Dry run: {dry_run}")
    print(f"Shuffle seed: {seed}")

    client = None
    if not dry_run:
        print(f"Initializing Ollama client for host: {ollama_host}...")
        client = Client(host=ollama_host, timeout=cfg.llm_timeout)
        # Verify connection
        try:
            client.list()
            print("Successfully connected to Ollama server.")
        except Exception as e:
            raise ConnectionError(
                f"Could not connect to Ollama at {ollama_host}: {e}\n"
                "If running locally, ensure 'ollama serve' is active.\n"
                "If running on Google Colab, pass the tunnel URL via --ollama_host <URL>."
            )
        ensure_model_available(client, model_name)

    results = []
    
    for idx, row in df.iterrows():
        recipe_name = str(row["recipe_name"]).strip()
        query = str(row["query"]).strip()
        original_ans = str(row.get("llama_rag_response", "")).strip()

        # Parse reference answers from JSON
        raw_refs = json.loads(row["reference_answers_json"])
        
        # 1. Format original reference answers: ["1. ...", "2. ..."]
        original_reference_answers = [f"{i}. {text.strip()}" for i, text in enumerate(raw_refs, 1)]

        # 2. Shuffle context deterministically
        shuffled_indexed, raw_shuffled_chunks = deterministic_shuffle(raw_refs, seed=seed, recipe_idx=idx)

        # 3. Format shuffled reference answers for human inspection: ["1. (old_idx) ...", ...]
        shuffled_reference_answers = [
            f"{new_pos}. ({old_pos}) {text.strip()}"
            for new_pos, (old_pos, text) in enumerate(shuffled_indexed, 1)
        ]

        # 4. Construct RAG prompt with RAW shuffled context (NO indexes in prompt)
        is_directions = "directions" in query.lower()
        sys_template = SYSTEM_PROMPT_DIRECTIONS_RAG if is_directions else SYSTEM_PROMPT_RAG
        context_str = "\n\n".join(raw_shuffled_chunks)
        system_prompt = sys_template.format(context=context_str)

        # 5. Query the model or preview in dry-run mode
        if dry_run:
            shuffled_answer = "[DRY RUN - Model query skipped. Context shuffled successfully.]"
        else:
            print(f"[{idx+1:2d}/{len(df)}] Querying RAG with shuffled context for: {recipe_name}...")
            response = client.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                stream=False,
                options={
                    "num_ctx": cfg.llm_num_ctx,
                    "num_predict": cfg.llm_num_predict,
                    "temperature": cfg.llm_temperature,
                },
                keep_alive=cfg.llm_keep_alive,
            )
            shuffled_answer = response.get("message", {}).get("content", "").strip()

        results.append({
            "recipe_name": recipe_name,
            "query": query,
            "reference_answers": original_reference_answers,
            "original_answer": original_ans,
            "shuffled_reference_answers": shuffled_reference_answers,
            "shuffled_answer": shuffled_answer
        })

    # Prepare final output structure
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    model_slug = model_name.replace("/", "_").replace(":", "_")
    output_data = {
        "model": model_name,
        "backend": "ollama",
        "experiment": "rag",
        "bias_test": "position",
        "timestamp": timestamp,
        "config": {
            "temperature": cfg.llm_temperature,
            "num_ctx": cfg.llm_num_ctx,
            "num_predict": cfg.llm_num_predict,
            "shuffle_seed": seed
        },
        "results": results
    }

    # Ensure save directory exists
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    out_file = save_folder / f"position_bias_{model_slug}_{timestamp}.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"\nPosition Bias test complete!")
    print(f"Saved {len(results)} queries to: {out_file}")

    if not dry_run and client is not None:
        unload_model(client, model_name)

    return out_file

def main():
    parser = argparse.ArgumentParser(description="Run Position Bias experiment via RAG context shuffling.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Input CSV path")
    parser.add_argument("--save-file", type=Path, default=DEFAULT_SAVE_FOLDER, help="Folder where to save the result JSON")
    parser.add_argument("--ollama-host", type=str, default=DEFAULT_OLLAMA_HOST, help="Ollama server host URL")
    parser.add_argument("--model", type=str, default=None, help="Override model name from config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Format and export JSON without invoking Ollama")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    run_position_bias(
        input_csv=args.input_csv,
        save_folder=args.save_file,
        ollama_host=args.ollama_host,
        model_override=args.model,
        dry_run=args.dry_run,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
