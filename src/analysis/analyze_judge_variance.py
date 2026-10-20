import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "FinalJson_RERUN"

# Canonical display names
GENERATOR_DISPLAY_NAMES = {
    "gemma3_1b": "Gemma 3 1B",
    "gemma3:1b": "Gemma 3 1B",
    "granite4.1_3b": "Granite 4.1 3B",
    "granite4.1:3b": "Granite 4.1 3B",
    "hf.co_bartowski_Llama-3.2-3B-Instruct-GGUF": "Llama 3.2 3B",
    "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF": "Llama 3.2 3B",
    "ministral-3_3b": "Ministral 3 3B",
    "ministral-3:3b": "Ministral 3 3B",
    "qwen3.5_4b": "Qwen 3.5 4B",
    "qwen3.5:4b": "Qwen 3.5 4B",
}

JUDGE_DISPLAY_NAMES = {
    "gemma3_1b": "Gemma 3 1B",
    "gemma3:1b": "Gemma 3 1B",
    "granite4.1_3b": "Granite 4.1 3B",
    "granite4.1:3b": "Granite 4.1 3B",
    "hf.co_bartowski_Llama-3.2-3B-Instruct-GGUF": "Llama 3.2 3B",
    "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF": "Llama 3.2 3B",
    "ministral-3_3b": "Ministral 3 3B",
    "ministral-3:3b": "Ministral 3 3B",
    "qwen3.5_2b": "Qwen 3.5 2B",
    "qwen3.5:2b": "Qwen 3.5 2B",
}


def analyze_unchanged_answers(results_dir: Path) -> dict:
    """Scans all JSON evaluation files and extracts statistics on identical answers
    between Round 1 and Round 2, analyzing whether judges produced differing scores.
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    json_files = sorted(results_dir.glob("*/*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {results_dir}")

    total_evaluated = 0
    total_unchanged = 0
    reused_count = 0
    reevaluated_count = 0

    # Model and judge counters
    model_stats = defaultdict(lambda: {
        "total_qa": 0,
        "unchanged": 0,
        "reused": 0,
        "reevaluated": 0,
        "score_varied": 0,
    })

    judge_variance_counts = defaultdict(int)
    model_judge_variance = defaultdict(lambda: defaultdict(int))
    varied_cases = []

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Warning] Failed to read {jf}: {e}", file=sys.stderr)
            continue

        raw_model = data.get("model", jf.parent.name)
        model_name = GENERATOR_DISPLAY_NAMES.get(raw_model, raw_model)
        target_type = data.get("target_type", "")
        qa_pairs = data.get("qa_pairs", [])

        for qa in qa_pairs:
            total_evaluated += 1
            model_stats[model_name]["total_qa"] += 1

            r1_ans = (qa.get("original_answer") or "").strip()
            r2_info = qa.get("round_2") or {}
            r2_ans = (r2_info.get("corrected_answer") or "").strip()

            if r1_ans == r2_ans:
                total_unchanged += 1
                model_stats[model_name]["unchanged"] += 1

                r1_judges = (qa.get("round_1") or {}).get("judges") or {}
                r2_judges = r2_info.get("judges") or {}

                # Determine if re-evaluated or copied directly
                # (When copied, explanations and scores match 100%)
                is_copied = True
                for j in r1_judges:
                    if j not in r2_judges:
                        is_copied = False
                        break
                    if r1_judges[j].get("explanation") != r2_judges[j].get("explanation"):
                        is_copied = False
                        break
                    if r1_judges[j].get("score") != r2_judges[j].get("score"):
                        is_copied = False
                        break

                if is_copied:
                    reused_count += 1
                    model_stats[model_name]["reused"] += 1
                else:
                    reevaluated_count += 1
                    model_stats[model_name]["reevaluated"] += 1

                # Check if any judge score differed
                diffs = []
                all_judges = sorted(set(r1_judges.keys()) | set(r2_judges.keys()))
                for j in all_judges:
                    s1 = r1_judges.get(j, {}).get("score")
                    s2 = r2_judges.get(j, {}).get("score")
                    if s1 is not None and s2 is not None and s1 != s2:
                        j_display = JUDGE_DISPLAY_NAMES.get(j, j)
                        diffs.append({
                            "judge_raw": j,
                            "judge": j_display,
                            "r1_score": s1,
                            "r2_score": s2,
                            "delta": s2 - s1,
                            "r1_explanation": r1_judges.get(j, {}).get("explanation", "").strip(),
                            "r2_explanation": r2_judges.get(j, {}).get("explanation", "").strip(),
                        })
                        judge_variance_counts[j_display] += 1
                        model_judge_variance[model_name][j_display] += 1

                if diffs:
                    model_stats[model_name]["score_varied"] += 1
                    varied_cases.append({
                        "recipe_name": qa.get("recipe_name", ""),
                        "corrector_model": model_name,
                        "task": target_type,
                        "human_score": qa.get("human_score"),
                        "human_explanation": qa.get("human_explanation", ""),
                        "r1_avg": (qa.get("round_1") or {}).get("average_score"),
                        "r2_avg": r2_info.get("average_score"),
                        "answer_text": r1_ans,
                        "changed_judges": diffs,
                    })

    return {
        "total_evaluated": total_evaluated,
        "total_unchanged": total_unchanged,
        "reused_count": reused_count,
        "reevaluated_count": reevaluated_count,
        "model_stats": model_stats,
        "judge_variance_counts": judge_variance_counts,
        "model_judge_variance": model_judge_variance,
        "varied_cases": varied_cases,
    }


def print_report(results: dict, verbose: bool = False):
    """Prints a structured summary report."""
    print("=" * 95)
    print(" ANALYSIS OF IDENTICAL ANSWERS WITH JUDGE SCORE VARIANCE (Round 1 vs. Round 2)")
    print("=" * 95)
    print(f"Total evaluated QA instances:             {results['total_evaluated']}")
    print(f"Total instances with UNCHANGED answer:    {results['total_unchanged']} ({results['total_unchanged'] / results['total_evaluated'] * 100:.1f}%)")
    print(f"  -> Reused R1 Judge Evaluation (copied): {results['reused_count']}")
    print(f"  -> Actively Re-Evaluated by Judges:     {results['reevaluated_count']}")
    print(f"  -> Judge Score Varied on Identical Ans: {len(results['varied_cases'])}")
    print("-" * 95)

    print("\n--- [1] BREAKDOWN BY CORRECTOR MODEL ---")
    print(f"{'Corrector Model':<22} | {'Total QA':<8} | {'Unchanged':<10} | {'Reused':<8} | {'Re-eval':<8} | {'Score Varied':<14} | {'% Varied'}")
    print("-" * 95)
    for model, s in sorted(results["model_stats"].items()):
        pct_str = f"{s['score_varied'] / s['reevaluated'] * 100:.1f}%" if s["reevaluated"] > 0 else "0.0%"
        print(
            f"{model:<22} | {s['total_qa']:<8} | {s['unchanged']:<10} | {s['reused']:<8} | {s['reevaluated']:<8} | {s['score_varied']:<14} | {pct_str}"
        )

    print("\n--- [2] BREAKDOWN BY JUDGE (Number of Score Shifts on Identical Text) ---")
    print(f"{'Judge Model':<25} | {'Score Shifts Count'}")
    print("-" * 45)
    for judge, cnt in sorted(results["judge_variance_counts"].items(), key=lambda x: -x[1]):
        print(f"{judge:<25} | {cnt}")

    print("\n--- [3] DETAILED CASES WHERE JUDGES CHANGED SCORE (Total: {}) ---".format(len(results["varied_cases"])))
    for idx, c in enumerate(results["varied_cases"], 1):
        print(f"\n[{idx}] Recipe: {c['recipe_name']} | Task: {c['task'].capitalize()} | Corrector: {c['corrector_model']}")
        print(f"    Human Score: {c['human_score']} | Judge Avg: {c['r1_avg']:.1f} -> {c['r2_avg']:.1f} (Delta: {c['r2_avg'] - c['r1_avg']:+.1f})")
        
        for d in c["changed_judges"]:
            print(f"    * Judge: {d['judge']} -> Score: {d['r1_score']} -> {d['r2_score']} (Diff: {d['delta']:+})")
            if verbose:
                print(f"      [R1 Exp] {d['r1_explanation'][:140]}...")
                print(f"      [R2 Exp] {d['r2_explanation'][:140]}...")

        if verbose:
            preview = c['answer_text'][:120] + ("..." if len(c['answer_text']) > 120 else "")
            print(f"    Answer: \"{preview}\"")

    print("\n" + "=" * 95)


def export_csv(varied_cases: list, output_csv: Path):
    """Exports all variance cases to a clean CSV file."""
    import csv

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "recipe_name",
            "task",
            "corrector_model",
            "human_score",
            "round_1_avg_judge",
            "round_2_avg_judge",
            "judge_name",
            "r1_judge_score",
            "r2_judge_score",
            "judge_score_delta",
            "r1_judge_explanation",
            "r2_judge_explanation",
        ])

        for c in varied_cases:
            for d in c["changed_judges"]:
                writer.writerow([
                    c["recipe_name"],
                    c["task"],
                    c["corrector_model"],
                    c["human_score"],
                    f"{c['r1_avg']:.2f}",
                    f"{c['r2_avg']:.2f}",
                    d["judge"],
                    d["r1_score"],
                    d["r2_score"],
                    d["delta"],
                    d["r1_explanation"],
                    d["r2_explanation"],
                ])

    print(f"\n[Saved] Detailed variance cases exported to CSV: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and analyze cases where the answer was identical between Round 1 and Round 2 but judge scores varied."
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory containing FinalJson evaluation files (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print judge explanation snippets and answer previews for each case.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Optional path to export identified variance cases as CSV.",
    )

    args = parser.parse_args()

    results = analyze_unchanged_answers(args.results_dir)
    print_report(results, verbose=args.verbose)

    if args.output_csv:
        export_csv(results["varied_cases"], args.output_csv)


if __name__ == "__main__":
    main()
