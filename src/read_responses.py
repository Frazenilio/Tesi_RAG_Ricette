import json
import argparse
import sys
from pathlib import Path

# Make sure to use a raw string (r"path\to\file") or forward slashes (path/to/file) to avoid escape sequence issues on Windows.
JSON_PATH = r"results\2026-06-21T11-48-57\qwen3.5_0.8b\rag\Baked Macaroni and Cheese\Singolo-Aggregati\directions\results_2026-06-21T11-48-57.json"

def format_split_list(text, label):
    lines = []
    lines.append(f"#### {label}")
    if not text:
        lines.append("* *(empty)*")
        return lines
    # Split by semicolon
    parts = [p.strip() for p in text.split(";") if p.strip()]
    for part in parts:
        cleaned_part = part
        for prefix in ["The ingredients of", "The directions of"]:
            if cleaned_part.lower().startswith(prefix.lower()):
                colon_idx = cleaned_part.find(":")
                if colon_idx != -1:
                    cleaned_part = cleaned_part[colon_idx + 1:].strip()
        lines.append(f"* {cleaned_part}")
    return lines

def main():
    target_path_str = JSON_PATH.strip() if JSON_PATH else ""
    if not target_path_str:
        parser = argparse.ArgumentParser(description="Beautifully format and save recipe JSON output responses to Markdown.")
        parser.add_argument("json_path", type=str, nargs="?", help="Path to the JSON results file.")
        args = parser.parse_args()
        target_path_str = args.json_path

    if not target_path_str:
        print("Error: No JSON path specified. Please set JSON_PATH at the top of the script or pass it as an argument.", file=sys.stderr)
        sys.exit(1)

    json_path = Path(target_path_str)
    if not json_path.exists():
        print(f"Error: File not found at {json_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}", file=sys.stderr)
        sys.exit(1)

    md_lines = []
    md_lines.append(f"# Recipe Generation Results Analysis")
    md_lines.append(f"* **Source File**: `{json_path.name}`")
    md_lines.append(f"* **Model**: `{data.get('model', 'N/A')}`")
    md_lines.append(f"* **Experiment**: `{data.get('experiment', 'N/A')}`")
    md_lines.append(f"* **Strategy**: `{data.get('strategy', 'N/A')}`")
    md_lines.append(f"* **Target Type**: `{data.get('target_type', 'N/A')}`")
    md_lines.append("")
    md_lines.append("---")

    results = data.get("results", {})
    for size, size_data in results.items():
        md_lines.append(f"\n## Variant Size / Key: {size}")
        qa_pairs = size_data.get("qa_pairs", [])
        for qa in qa_pairs:
            md_lines.append(f"\n### Recipe: {qa.get('recipe_name', 'N/A')}")
            md_lines.append(f"**Query**: {qa.get('query', 'N/A')}")
            md_lines.append("")
            
            # Ground truth
            correct_ing = qa.get("correct_ingredients")
            correct_dir = qa.get("correct_directions")
            gt_list = correct_ing or correct_dir
            if gt_list:
                md_lines.append("#### Ground Truth Reference")
                for step in gt_list:
                    sub_steps = [s.strip() for s in step.split(";") if s.strip()]
                    for sub in sub_steps:
                        cleaned_sub = sub
                        for prefix in ["The ingredients of", "The directions of"]:
                            if cleaned_sub.lower().startswith(prefix.lower()):
                                colon_idx = cleaned_sub.find(":")
                                if colon_idx != -1:
                                    cleaned_sub = cleaned_sub[colon_idx + 1:].strip()
                        md_lines.append(f"* {cleaned_sub}")
                md_lines.append("")

            # Responses
            response = qa.get("response")
            response_rag = qa.get("response_rag")
            response_llm = qa.get("response_llm")

            if response:
                md_lines.extend(format_split_list(response, "Response"))
                md_lines.append("")
            if response_rag:
                md_lines.extend(format_split_list(response_rag, "RAG Response"))
                md_lines.append("")
            if response_llm:
                md_lines.extend(format_split_list(response_llm, "LLM-only Response"))
                md_lines.append("")
            
            md_lines.append("---")

    # Output path in same folder as target JSON file
    out_md_path = json_path.with_suffix(".md")
    try:
        with open(out_md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")
        print(f"\n[Success] Formatted results successfully written to markdown file:")
        print(f"-> {out_md_path.absolute()}")
    except Exception as e:
        print(f"Error writing Markdown file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
