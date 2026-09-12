import pandas as pd
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
INPUT_CSV = ROOT_DIR / "data" / "judgement_dataset.csv"
OUTPUT_CSV = ROOT_DIR / "data" / "human_judgement_dataset.csv"

COLUMNS_TO_KEEP = [
    "Question",
    "Recipe",
    "Asked",
    "Reference Answers",
    "RAG Model Name",
    "Provided Answer",
    "IoU Correctness",
    "Human Correctness",
    "Human Numeric Score",
    "Human Explanation"
]

def extract_human_judgements(input_path: Path = INPUT_CSV, output_path: Path = OUTPUT_CSV) -> pd.DataFrame:
    """Extract human judgments and QA data, omitting LLM judge scores, models, and explanations."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    df = pd.read_csv(input_path)
    df_human = df[COLUMNS_TO_KEEP].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_human.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Extracted {len(df_human)} rows with {len(COLUMNS_TO_KEEP)} columns.")
    print(f"Saved to: {output_path}")
    return df_human

if __name__ == "__main__":
    extract_human_judgements()
