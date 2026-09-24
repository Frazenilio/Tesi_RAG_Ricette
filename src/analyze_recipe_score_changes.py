import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFAULT_FINAL_JSON_DIR = (
    PROJECT_ROOT / "results" / "FinalJson_RERUN"
    if (PROJECT_ROOT / "results" / "FinalJson_RERUN").exists()
    else PROJECT_ROOT / "results" / "FinalJson"
)

MODEL_DISPLAY_NAMES = {
    "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF": "Llama 3.2 3B",
    "hf.co_bartowski_Llama-3.2-3B-Instruct-GGUF": "Llama 3.2 3B",
    "gemma3:1b": "Gemma 3 1B",
    "gemma3_1b": "Gemma 3 1B",
    "granite4.1:3b": "Granite 4.1 3B",
    "granite4.1_3b": "Granite 4.1 3B",
    "ministral-3:3b": "Ministral 3 3B",
    "ministral-3_3b": "Ministral 3 3B",
    "qwen3.5:4b": "Qwen 3.5 4B",
    "qwen3.5_4b": "Qwen 3.5 4B",
}


def find_json_files(input_dir: Path) -> list[Path]:
    """Finds all JSON files recursively in the given directory."""
    if not input_dir.exists():
        return []
    return sorted(input_dir.glob("**/*.json"))


def parse_qa_entries(file_path: Path) -> dict[str, Any]:
    """Loads a JSON result file and extracts QA pairs with their scores and deltas."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_raw = data.get("model", file_path.parent.name)
    model_name = MODEL_DISPLAY_NAMES.get(model_raw, model_raw)
    target_type = data.get("target_type", "unknown").lower()
    qa_pairs = data.get("qa_pairs", [])

    entries = []
    for qa in qa_pairs:
        recipe_name = qa.get("recipe_name", "Unknown")
        human_score = float(qa.get("human_score", 0.0))
        r1_raw = qa.get("round_1", {}).get("average_score")
        r2_raw = qa.get("round_2", {}).get("average_score")
        r1_score = float(r1_raw) if r1_raw is not None else 0.0
        r2_score = float(r2_raw) if r2_raw is not None else 0.0
        delta = round(r2_score - r1_score, 2)
        if abs(delta) < 1e-6:
            delta = 0.0

        entries.append({
            "qa_index": qa.get("qa_index"),
            "recipe_name": recipe_name,
            "human_score": human_score,
            "r1_score": round(r1_score, 1),
            "r2_score": round(r2_score, 1),
            "delta": delta,
        })

    return {
        "file_path": file_path,
        "model_raw": model_raw,
        "model_name": model_name,
        "target_type": target_type,
        "entries": entries,
    }


def filter_top_recipes(
    entries: list[dict],
    threshold_high: float = 95.0,
    threshold_low: float = 90.0,
    top_n: int = 5,
) -> dict[str, list[dict]]:
    """Partitions entries into the 4 target categories and ranks top N."""
    high_group = [e for e in entries if e["human_score"] >= threshold_high]
    low_group = [e for e in entries if e["human_score"] < threshold_low]

    # Category 1: Human >= 95, Top Score Increases (delta > 0, highest delta first)
    high_inc = sorted([e for e in high_group if e["delta"] > 0], key=lambda x: x["delta"], reverse=True)[:top_n]

    # Category 2: Human >= 95, Top Score Decreases (delta < 0, largest drop first)
    high_dec = sorted([e for e in high_group if e["delta"] < 0], key=lambda x: x["delta"])[:top_n]

    # Category 3: Human < 90, Top Score Increases (delta > 0, highest delta first)
    low_inc = sorted([e for e in low_group if e["delta"] > 0], key=lambda x: x["delta"], reverse=True)[:top_n]

    # Category 4: Human < 90, Top Score Decreases (delta < 0, largest drop first)
    low_dec = sorted([e for e in low_group if e["delta"] < 0], key=lambda x: x["delta"])[:top_n]

    return {
        "high_inc": high_inc,
        "high_dec": high_dec,
        "low_inc": low_inc,
        "low_dec": low_dec,
    }


def print_detailed_report(
    results: list[dict],
    threshold_high: float = 95.0,
    threshold_low: float = 90.0,
    top_n: int = 5,
    names_only: bool = False,
):
    """Prints a structured, readable report to the console."""
    categories_meta = [
        ("high_inc", f"Top {top_n} Score INCREASES (Human Score >= {threshold_high:.0f})", "+"),
        ("high_dec", f"Top {top_n} Score DECREASES (Human Score >= {threshold_high:.0f})", "-"),
        ("low_inc", f"Top {top_n} Score INCREASES (Human Score < {threshold_low:.0f})", "+"),
        ("low_dec", f"Top {top_n} Score DECREASES (Human Score < {threshold_low:.0f})", "-"),
    ]

    for item in results:
        model_name = item["model_name"]
        task = item["target_type"].capitalize()
        filtered = item["filtered"]

        print("\n" + "=" * 80)
        print(f" MODEL: {model_name:<20} | TASK: {task:<15} ({item['file_path'].name})")
        print("=" * 80)

        for cat_key, cat_title, sign in categories_meta:
            recipes = filtered.get(cat_key, [])
            print(f"\n  [*] {cat_title}:")

            if not recipes:
                print("      (None found)")
                continue

            if names_only:
                names = [r["recipe_name"] for r in recipes]
                print(f"      {', '.join(names)}")
            else:
                for rank, r in enumerate(recipes, 1):
                    sign_str = f"+{r['delta']:.1f}" if r["delta"] > 0 else f"{r['delta']:.1f}"
                    print(
                        f"      {rank}. {r['recipe_name']:<30} "
                        f"Delta: {sign_str:>6} | "
                        f"R1: {r['r1_score']:>5.1f} -> R2: {r['r2_score']:>5.1f} | "
                        f"Human: {r['human_score']:>3.0f}"
                    )


def export_markdown_report(
    results: list[dict],
    output_path: Path,
    threshold_high: float = 95.0,
    threshold_low: float = 90.0,
    top_n: int = 5,
):
    """Exports the full analysis into a clean markdown document."""
    categories_meta = [
        ("high_inc", f"Human Score ≥ {threshold_high:.0f} — Top Score INCREASES"),
        ("high_dec", f"Human Score ≥ {threshold_high:.0f} — Top Score DECREASES"),
        ("low_inc", f"Human Score < {threshold_low:.0f} — Top Score INCREASES"),
        ("low_dec", f"Human Score < {threshold_low:.0f} — Top Score DECREASES"),
    ]

    lines = [
        "# Top Recipe Score Changes: Round 1 vs Round 2",
        "",
        f"- **Human Score (High)**: ≥ {threshold_high:.0f}",
        f"- **Human Score (Low)**: < {threshold_low:.0f}",
        f"- **Top N per category**: {top_n}",
        "",
    ]

    for item in results:
        model_name = item["model_name"]
        task = item["target_type"].capitalize()
        filtered = item["filtered"]

        lines.append(f"## {model_name} — {task}")
        lines.append("")

        for cat_key, cat_title in categories_meta:
            recipes = filtered.get(cat_key, [])
            lines.append(f"### {cat_title}")
            lines.append("")

            if not recipes:
                lines.append("_None found._\n")
                continue

            lines.append("| # | Recipe Name | Human Score | Round 1 Score | Round 2 Score | Delta (R2 - R1) |")
            lines.append("|---|:---|:---:|:---:|:---:|:---:|")
            for rank, r in enumerate(recipes, 1):
                sign_str = f"+{r['delta']:.1f}" if r["delta"] > 0 else f"{r['delta']:.1f}"
                lines.append(
                    f"| {rank} | **{r['recipe_name']}** | {r['human_score']:.0f} | {r['r1_score']:.1f} | {r['r2_score']:.1f} | `{sign_str}` |"
                )
            lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[Exported] Markdown report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Top 5 recipes with largest score increases/decreases grouped by Human Score."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_FINAL_JSON_DIR,
        help="Path to results directory containing FinalJson files (default: results/FinalJson_RERUN).",
    )
    parser.add_argument(
        "--threshold_high",
        type=float,
        default=95.0,
        help="Threshold for high human score (default: 95.0).",
    )
    parser.add_argument(
        "--threshold_low",
        type=float,
        default=90.0,
        help="Threshold for low human score (default: 90.0).",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="Number of recipes to show in each category (default: 5).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=None,
        help="Filter to specific generator models (e.g. qwen, gemma, llama).",
    )
    parser.add_argument(
        "--target_type",
        choices=["ingredients", "directions"],
        default=None,
        help="Filter to ingredients or directions.",
    )
    parser.add_argument(
        "--names_only",
        action="store_true",
        help="Output only the recipe names list.",
    )
    parser.add_argument(
        "--export_markdown",
        type=Path,
        default=None,
        help="Optional path to save report as a Markdown file.",
    )
    args = parser.parse_args()

    print(f"\n[Info] Analyzing results from: {args.input_dir.resolve()}")

    json_files = find_json_files(args.input_dir)
    if not json_files:
        print(f"Error: No JSON files found in {args.input_dir}")
        return

    parsed_items = []
    for f in json_files:
        parsed = parse_qa_entries(f)
        if args.models:
            if not any(m.lower() in parsed["model_raw"].lower() for m in args.models):
                continue
        if args.target_type and parsed["target_type"] != args.target_type.lower():
            continue

        parsed["filtered"] = filter_top_recipes(
            parsed["entries"],
            threshold_high=args.threshold_high,
            threshold_low=args.threshold_low,
            top_n=args.top_n,
        )
        parsed_items.append(parsed)

    # Sort parsed items by model name and target type
    parsed_items.sort(key=lambda x: (x["model_name"], x["target_type"]))

    print_detailed_report(
        parsed_items,
        threshold_high=args.threshold_high,
        threshold_low=args.threshold_low,
        top_n=args.top_n,
        names_only=args.names_only,
    )

    if args.export_markdown:
        export_markdown_report(
            parsed_items,
            args.export_markdown,
            threshold_high=args.threshold_high,
            threshold_low=args.threshold_low,
            top_n=args.top_n,
        )


if __name__ == "__main__":
    main()
