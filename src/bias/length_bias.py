import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

DEFAULT_INPUT_CSV = PROJECT_ROOT / "data" / "judgement_dataset.csv"
DEFAULT_SAVE_FOLDER = PROJECT_ROOT / "results" / "bias_tests"


def extract_evaluations_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts individual (question, recipe, asked, rag_model, answer, judge, score)
    records from the judgement dataset.
    """
    records = []
    for _, row in df.iterrows():
        p_ans = str(row["Provided Answer"]) if pd.notna(row["Provided Answer"]) else ""
        char_len = len(p_ans)
        word_len = len(p_ans.split())
        question = str(row["Question"])
        recipe = str(row["Recipe"])
        asked = str(row["Asked"])
        rag_model = str(row["RAG Model Name"])

        # Extract 3 LLM judges per row
        row_llm_scores = []
        for j_i in [1, 2, 3]:
            j_model = row.get(f"Judge {j_i} Model Name")
            j_score = row.get(f"Judge {j_i} Numeric Score")
            if pd.notna(j_model) and pd.notna(j_score):
                j_model_str = str(j_model).strip()
                j_score_flt = float(j_score)
                row_llm_scores.append(j_score_flt)
                records.append({
                    "question": question,
                    "recipe": recipe,
                    "asked": asked,
                    "rag_model": rag_model,
                    "provided_answer": p_ans,
                    "char_length": char_len,
                    "word_length": word_len,
                    "judge": j_model_str,
                    "score": j_score_flt,
                    "is_human": False
                })

        # LLM Average across judges for this row
        if row_llm_scores:
            records.append({
                "question": question,
                "recipe": recipe,
                "asked": asked,
                "rag_model": rag_model,
                "provided_answer": p_ans,
                "char_length": char_len,
                "word_length": word_len,
                "judge": "LLM_Average",
                "score": float(np.mean(row_llm_scores)),
                "is_human": False
            })

        # Human evaluation
        h_score = row.get("Human Numeric Score")
        if pd.notna(h_score):
            records.append({
                "question": question,
                "recipe": recipe,
                "asked": asked,
                "rag_model": rag_model,
                "provided_answer": p_ans,
                "char_length": char_len,
                "word_length": word_len,
                "judge": "Human",
                "score": float(h_score),
                "is_human": True
            })

    return pd.DataFrame(records)


def compute_judge_metrics(sub_df: pd.DataFrame) -> dict:
    """Computes length bias summary metrics for a given subset of evaluation records."""
    if sub_df.empty:
        return {}

    # Overall correlation across all evaluated answers
    pearson_chars = sub_df["char_length"].corr(sub_df["score"], method="pearson")
    spearman_chars = sub_df["char_length"].corr(sub_df["score"], method="spearman")
    pearson_words = sub_df["word_length"].corr(sub_df["score"], method="pearson")
    spearman_words = sub_df["word_length"].corr(sub_df["score"], method="spearman")

    longest_top_count = 0
    longest_exclusive_count = 0
    longest_scores = []
    shortest_scores = []
    questions_count = 0

    for _, q_group in sub_df.groupby("question"):
        if len(q_group) < 2:
            continue
        questions_count += 1

        # Max and min length candidate within this question
        max_char = q_group["char_length"].max()
        min_char = q_group["char_length"].min()

        longest_rows = q_group[q_group["char_length"] == max_char]
        shortest_rows = q_group[q_group["char_length"] == min_char]

        max_score = q_group["score"].max()
        highest_scoring_rows = q_group[q_group["score"] == max_score]

        # Did longest get top score?
        longest_has_top = (longest_rows["score"] == max_score).any()
        if longest_has_top:
            longest_top_count += 1
            if len(highest_scoring_rows) == 1:
                longest_exclusive_count += 1

        longest_scores.append(float(longest_rows["score"].mean()))
        shortest_scores.append(float(shortest_rows["score"].mean()))

    top_rate = (longest_top_count / questions_count) if questions_count > 0 else 0.0
    exclusive_rate = (longest_exclusive_count / questions_count) if questions_count > 0 else 0.0
    mean_longest = float(np.mean(longest_scores)) if longest_scores else 0.0
    mean_shortest = float(np.mean(shortest_scores)) if shortest_scores else 0.0
    mean_all = float(sub_df["score"].mean()) if not sub_df.empty else 0.0
    delta_longest_shortest = mean_longest - mean_shortest

    return {
        "evaluations_count": len(sub_df),
        "questions_count": questions_count,
        "longest_top_score_count": longest_top_count,
        "longest_top_score_rate": round(top_rate, 4),
        "longest_top_score_percentage": round(top_rate * 100, 2),
        "longest_exclusive_top_score_count": longest_exclusive_count,
        "longest_exclusive_top_score_rate": round(exclusive_rate, 4),
        "longest_exclusive_top_score_percentage": round(exclusive_rate * 100, 2),
        "mean_score_longest": round(mean_longest, 2),
        "mean_score_shortest": round(mean_shortest, 2),
        "mean_score_all": round(mean_all, 2),
        "delta_longest_vs_shortest": round(delta_longest_shortest, 2),
        "pearson_correlation_chars": round(float(pearson_chars), 4) if pd.notna(pearson_chars) else 0.0,
        "spearman_correlation_chars": round(float(spearman_chars), 4) if pd.notna(spearman_chars) else 0.0,
        "pearson_correlation_words": round(float(pearson_words), 4) if pd.notna(pearson_words) else 0.0,
        "spearman_correlation_words": round(float(spearman_words), 4) if pd.notna(spearman_words) else 0.0
    }


def analyze_length_bias(df: pd.DataFrame) -> dict:
    """Performs the full length / verbosity bias analysis on the dataset."""
    eval_df = extract_evaluations_dataframe(df)

    # 1. Summary by Judge across all questions
    judges = eval_df["judge"].unique().tolist()
    # Order nicely: LLMs first, then LLM_Average, then Human
    judge_order = sorted([j for j in judges if j not in ("LLM_Average", "Human")])
    if "LLM_Average" in judges:
        judge_order.append("LLM_Average")
    if "Human" in judges:
        judge_order.append("Human")

    summary_by_judge = {}
    for j in judge_order:
        j_df = eval_df[eval_df["judge"] == j]
        summary_by_judge[j] = compute_judge_metrics(j_df)

    # 2. Summary by Task Type (directions vs ingredients)
    summary_by_task = {}
    for asked in sorted(eval_df["asked"].unique()):
        summary_by_task[asked] = {}
        for j in judge_order:
            j_asked_df = eval_df[(eval_df["judge"] == j) & (eval_df["asked"] == asked)]
            summary_by_task[asked][j] = compute_judge_metrics(j_asked_df)

    # 3. Question-level details
    questions_data = []
    # Group original df by Question
    for question, q_group in df.groupby("Question"):
        recipe = str(q_group["Recipe"].iloc[0])
        asked = str(q_group["Asked"].iloc[0])

        candidates = []
        max_char = -1
        longest_model = ""
        min_char = 999999
        shortest_model = ""

        for _, row in q_group.iterrows():
            rag_model = str(row["RAG Model Name"])
            p_ans = str(row["Provided Answer"]) if pd.notna(row["Provided Answer"]) else ""
            char_len = len(p_ans)
            word_len = len(p_ans.split())

            if char_len > max_char:
                max_char = char_len
                longest_model = rag_model
            if char_len < min_char:
                min_char = char_len
                shortest_model = rag_model

            candidates.append({
                "rag_model": rag_model,
                "char_length": char_len,
                "word_length": word_len
            })

        for c in candidates:
            c["is_longest"] = (c["char_length"] == max_char)
            c["is_shortest"] = (c["char_length"] == min_char)

        # Detailed outcomes per judge for this question
        q_eval_df = eval_df[eval_df["question"] == question]
        judge_outcomes = {}
        for j in judge_order:
            j_q = q_eval_df[q_eval_df["judge"] == j]
            if j_q.empty or len(j_q) < 2:
                continue

            j_max_char = j_q["char_length"].max()
            j_min_char = j_q["char_length"].min()
            j_longest_row = j_q[j_q["char_length"] == j_max_char]
            j_shortest_row = j_q[j_q["char_length"] == j_min_char]

            j_max_score = float(j_q["score"].max())
            j_top_rows = j_q[j_q["score"] == j_max_score]
            j_longest_score = float(j_longest_row["score"].iloc[0])
            j_shortest_score = float(j_shortest_row["score"].iloc[0])

            longest_won = bool((j_longest_row["score"] == j_max_score).any())
            longest_exclusive = bool(longest_won and len(j_top_rows) == 1)

            judge_outcomes[j] = {
                "longest_model": j_longest_row["rag_model"].iloc[0],
                "longest_char_length": int(j_max_char),
                "shortest_model": j_shortest_row["rag_model"].iloc[0],
                "shortest_char_length": int(j_min_char),
                "highest_score": j_max_score,
                "highest_scoring_models": j_top_rows["rag_model"].tolist(),
                "longest_score": j_longest_score,
                "shortest_score": j_shortest_score,
                "longest_got_highest_score": longest_won,
                "longest_got_exclusive_highest_score": longest_exclusive,
                "delta_longest_vs_shortest": round(j_longest_score - j_shortest_score, 2)
            }

        questions_data.append({
            "question": question,
            "recipe": recipe,
            "asked": asked,
            "longest_model": longest_model,
            "shortest_model": shortest_model,
            "candidates": candidates,
            "judge_outcomes": judge_outcomes
        })

    return {
        "summary_by_judge": summary_by_judge,
        "summary_by_task_type": summary_by_task,
        "questions": questions_data
    }


def print_report(results: dict):
    """Prints a clean, formatted report of the length bias analysis."""
    print("\n" + "=" * 80)
    print("                LENGTH (VERBOSITY) BIAS ANALYSIS REPORT")
    print("=" * 80)

    summary_by_judge = results.get("summary_by_judge", {})
    header = f"{'Judge / Evaluator':<30} | {'Questions':<9} | {'Longest Win%':<12} | {'Excl Win%':<10} | {'Score Delta':<11} | {'Pearson r':<9}"
    print(header)
    print("-" * 80)

    for judge, m in summary_by_judge.items():
        q_cnt = m.get("questions_count", 0)
        win_pct = f"{m.get('longest_top_score_percentage', 0.0):.1f}%"
        excl_pct = f"{m.get('longest_exclusive_top_score_percentage', 0.0):.1f}%"
        delta = f"{m.get('delta_longest_vs_shortest', 0.0):+0.2f}"
        p_r = f"{m.get('pearson_correlation_chars', 0.0):+0.3f}"
        print(f"{judge:<30} | {q_cnt:<9} | {win_pct:<12} | {excl_pct:<10} | {delta:<11} | {p_r:<9}")

    print("=" * 80)

    # Breakdown by task type
    print("\n--- Breakdown by Task Type: Directions vs Ingredients ---")
    summary_by_task = results.get("summary_by_task_type", {})
    for task, j_dict in summary_by_task.items():
        print(f"\nTask Type: [{task.upper()}]")
        print(f"{'Judge / Evaluator':<30} | {'Longest Win%':<12} | {'Excl Win%':<10} | {'Score Delta':<11} | {'Pearson r':<9}")
        print("-" * 75)
        for judge, m in j_dict.items():
            win_pct = f"{m.get('longest_top_score_percentage', 0.0):.1f}%"
            excl_pct = f"{m.get('longest_exclusive_top_score_percentage', 0.0):.1f}%"
            delta = f"{m.get('delta_longest_vs_shortest', 0.0):+0.2f}"
            p_r = f"{m.get('pearson_correlation_chars', 0.0):+0.3f}"
            print(f"{judge:<30} | {win_pct:<12} | {excl_pct:<10} | {delta:<11} | {p_r:<9}")
    print("=" * 80 + "\n")


def run_length_bias(
    input_csv: Path = DEFAULT_INPUT_CSV,
    save_folder: Path = DEFAULT_SAVE_FOLDER
) -> Path:
    """Runs the Length Bias analysis and exports the structured JSON."""
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input dataset not found at: {input_csv}")

    print(f"Loading dataset for Length Bias analysis from: {input_csv}...")
    df = pd.read_csv(input_csv)

    results_data = analyze_length_bias(df)

    timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")
    output_data = {
        "bias_test": "length",
        "dataset_source": str(input_csv),
        "timestamp": timestamp,
        "total_rows_analyzed": len(df),
        "total_questions_analyzed": len(results_data["questions"]),
        "summary_by_judge": results_data["summary_by_judge"],
        "summary_by_task_type": results_data["summary_by_task_type"],
        "questions": results_data["questions"]
    }

    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    out_file = save_folder / f"length_bias_{timestamp}.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print_report(results_data)
    print(f"Length Bias results saved to: {out_file}\n")
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Length (Verbosity) Bias in LLM Judges")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Path to judgement dataset CSV")
    parser.add_argument("--save-file", type=Path, default=DEFAULT_SAVE_FOLDER, help="Folder to save output JSON")

    args = parser.parse_args()
    run_length_bias(input_csv=args.input_csv, save_folder=args.save_file)
