import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import requests

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.BiasTest.extract_base_retrievals import (
    extract_base_retrievals,
    DEFAULT_OUTPUT_CSV,
)
from src.BiasTest.position_bias import run_position_bias, DEFAULT_SAVE_FOLDER
from src.BiasTest.self_enchantment_bias import run_self_enchantment_bias

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

SUPPORTED_BIASES = {
    "position": "Tests RAG generation under shuffled context order (Manual inspection output).",
    "self-enchantment": "Tests if judge scores its own generated answers higher than other models.",
    "length": "Planned - Manual length variations.",
    "compassion-fade": "Planned - Evaluation of AI vs Human attribution tags."
}

def check_ollama(host: str = DEFAULT_OLLAMA_HOST) -> bool:
    """Checks if the Ollama server (local or Colab tunnel) is reachable."""
    print(f"\n--- Checking Ollama Server at: {host} ---")
    url = host.rstrip("/") + "/api/tags"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            models_info = resp.json().get("models", [])
            model_names = [m.get("name") for m in models_info]
            print(f"Status: Connected successfully!")
            print(f"Available models ({len(model_names)}): {', '.join(model_names) if model_names else 'None found'}")
            return True
        else:
            print(f"Status: Server returned status code {resp.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Status: Could not connect to Ollama server ({e})")
        print("Note: If running in Google Colab, provide the tunnel URL via --ollama_host <URL>.")
        return False

def show_dataset_info(csv_path: Path = DEFAULT_OUTPUT_CSV):
    """Displays information and sample rows from the base retrieval dataset."""
    print(f"\n--- Base Retrieval Dataset Information: {csv_path} ---")
    if not csv_path.exists():
        print(f"File not found: {csv_path}. Run with --extract to generate it.")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df)}")
    print(f"Number of Reference Answers Range: [{df['num_references'].min()} - {df['num_references'].max()}]")
    print("\nBreakdown by Reference Count:")
    counts = df["num_references"].value_counts().sort_index()
    for n_ref, c in counts.items():
        print(f"  {n_ref} References: {c} recipes")
    
    print("\nRecipes list:")
    for idx, row in df.iterrows():
        print(f"  {idx+1:2d}. {row['recipe_name']:<30} (Refs: {row['num_references']:2d})")

def main():
    parser = argparse.ArgumentParser(
        description="Main runner for the Recipe RAG Bias Testing suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check dataset info
  python src/BiasTest/MainBias.py --info

  # Re-extract base 30-recipe dataset
  python src/BiasTest/MainBias.py --extract

  # Run Position Bias (Dry Run to test file generation without Ollama)
  python src/BiasTest/MainBias.py --run position --dry-run --save-file results/bias_tests

  # Run Self-Enchantment Bias (Dry Run or Live)
  python src/BiasTest/MainBias.py --run self-enchantment --judge both --dry-run --save-file results/bias_tests
  python src/BiasTest/MainBias.py --run self-enchantment --judge llama --save-file results/bias_tests
        """
    )
    parser.add_argument("--run", type=str, default=None, help="Name of the bias to test (e.g. 'position', 'self-enchantment')")
    parser.add_argument("--judge", type=str, default="both", choices=["llama", "gemma", "both"], help="Judge model for Self-Enchantment test")
    parser.add_argument("--save-file", type=Path, default=DEFAULT_SAVE_FOLDER, help="Folder path where to save the result JSON")
    parser.add_argument("--ollama_host", type=str, default=DEFAULT_OLLAMA_HOST, help="Ollama server host URL")
    parser.add_argument("--model", type=str, default=None, help="Override LLM model name specified in config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Generate output JSON without querying the LLM")
    parser.add_argument("--seed", type=int, default=42, help="Seed for shuffling context")
    parser.add_argument("--extract", action="store_true", help="Extract base retrieval dataset to CSV")
    parser.add_argument("--info", action="store_true", help="Show summary of the extracted base dataset")
    parser.add_argument("--check-ollama", action="store_true", help="Check connectivity to Ollama server")

    args = parser.parse_args()

    # If --run is specified
    if args.run:
        bias_name = args.run.strip().lower().replace("_", "-")
        if bias_name == "position":
            print(f"\n--- Starting Bias Test: Position Bias (Context Shuffling) ---")
            run_position_bias(
                save_folder=args.save_file,
                ollama_host=args.ollama_host,
                model_override=args.model,
                dry_run=args.dry_run,
                seed=args.seed
            )
        elif bias_name in ("self-enchantment", "self"):
            print(f"\n--- Starting Bias Test: Self-Enchantment Bias ---")
            run_self_enchantment_bias(
                judge=args.judge,
                save_folder=args.save_file,
                ollama_host=args.ollama_host,
                dry_run=args.dry_run
            )
        else:
            print(f"\nError: Unknown bias '{args.run}'.")
            print("Supported biases:")
            for b_name, b_desc in SUPPORTED_BIASES.items():
                print(f"  - {b_name:<18}: {b_desc}")
            sys.exit(1)
        return

    if args.extract:
        print("Running base retrieval extraction...")
        extract_base_retrievals()
        return

    if args.check_ollama:
        check_ollama(args.ollama_host)
        return

    # Default / fallback to --info
    show_dataset_info()

if __name__ == "__main__":
    main()
