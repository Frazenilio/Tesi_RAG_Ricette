import argparse
import copy
import json
import re
import sys
import time
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

from src.config import load_config, Config, _build_inline_model_spec
from src.generation import create_model_runtime, query_llm
from src.ollama_utils import check_ollama_server
from src.prompts import (
    PROMPT_LLM_JUDGE,
    PROMPT_LIST_CORRECTOR,
    SYSTEM_PROMPT_CORRECTOR,
    PROMPT_ANSWER_CORRECTOR,
)
from src.test_judges import parse_judge_output

DEFAULT_INPUT_DIRS = [
    PROJECT_ROOT / "results" / "2026-07-15T11-32-35_START_DS",
    PROJECT_ROOT / "results" / "2026-07-17T10-32-24",
]


def provide_judgement(
    original_answer: str,
    judgement: str,
    current_cl: str,
    generator_runtime,
    cfg: Config,
) -> str:
    """Iteratively builds the correction list by incorporating judge critique."""
    cl_str = current_cl if current_cl else "Empty."
    corrector_user_message = PROMPT_LIST_CORRECTOR.format(
        original_answer=original_answer,
        judgements=judgement,
        correction_list=cl_str,
    )

    raw_response: str = query_llm(
        system_prompt=SYSTEM_PROMPT_CORRECTOR,
        user_message=corrector_user_message,
        model_runtime=generator_runtime,
        num_ctx=cfg.llm_num_ctx,
        num_predict=cfg.llm_num_predict,
        think=cfg.llm_think,
        temperature=cfg.llm_temperature,
        timeout=cfg.llm_timeout,
        retries=cfg.llm_retries,
        keep_alive=cfg.llm_keep_alive,
    )

    match = re.search(
        r"<correction_list>(.*?)<[/\\]correction_list>",
        raw_response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        ans = match.group(1).strip()
    else:
        ans = raw_response.strip()

    ans = re.sub(r"^```(?:json)?\n?", "", ans, flags=re.IGNORECASE)
    ans = re.sub(r"\n?```$", "", ans, flags=re.IGNORECASE)
    return ans.strip()


def correct_answer(
    original_answer: str,
    correction_list: str,
    generator_runtime,
    cfg: Config,
) -> str:
    """Produces the corrected answer given the original answer and the correction list."""
    corrector_message: str = PROMPT_ANSWER_CORRECTOR.format(
        original_answer=original_answer,
        correction_list=correction_list,
    )

    raw_response: str = query_llm(
        system_prompt=SYSTEM_PROMPT_CORRECTOR,
        user_message=corrector_message,
        model_runtime=generator_runtime,
        num_ctx=cfg.llm_num_ctx,
        num_predict=cfg.llm_num_predict,
        think=cfg.llm_think,
        temperature=cfg.llm_temperature,
        timeout=cfg.llm_timeout,
        retries=cfg.llm_retries,
        keep_alive=cfg.llm_keep_alive,
    )

    match = re.search(
        r"<(?:corrected_answer|correct_answer)>(.*?)<[/\\](?:corrected_answer|correct_answer)>",
        raw_response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        ans = match.group(1).strip()
    else:
        ans = raw_response.strip()

    ans = re.sub(r"^```(?:markdown|text)?\n?", "", ans, flags=re.IGNORECASE)
    ans = re.sub(r"\n?```$", "", ans, flags=re.IGNORECASE)
    return ans.strip()


def run_judge_round(
    answer: str,
    query: str,
    correct_answers_str: str,
    judges: list,
    cfg: Config,
) -> tuple[dict, float]:
    """Runs a round of evaluation using the specified judges on a single answer."""
    judges_dict = {}
    scores = []

    for runtime in judges:
        model_name = runtime.spec.visible_label

        min_score = getattr(cfg, "min_score", 0)
        max_score = getattr(cfg, "max_score", 100)
        default_score = getattr(cfg, "default_score", 50)

        user_message = PROMPT_LLM_JUDGE.format(
            min_score=min_score,
            max_score=max_score,
            default_score=default_score,
            prompted_query=query,
            llm_rag_answer=answer,
            correct_answers=correct_answers_str,
        )
        system_prompt = "You are an impartial judge evaluating an answer."

        print(f"        -> Asking judge {model_name}...")
        eval_text = query_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            model_runtime=runtime,
            num_ctx=cfg.llm_num_ctx,
            num_predict=cfg.llm_num_predict,
            think=cfg.llm_think,
            temperature=0.0,  # Greedy decoding for evaluation
            timeout=cfg.llm_timeout,
            retries=cfg.llm_retries,
            keep_alive=cfg.llm_keep_alive,
        )

        score, explanation = parse_judge_output(eval_text)
        if score == 0 and "0" not in eval_text:
            print(f"          Warning: Could not parse non-zero score from {model_name}. Using 0.")

        judges_dict[model_name] = {
            "score": score,
            "explanation": explanation,
        }
        scores.append(score)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    return judges_dict, avg_score


def determine_output_path(
    input_file: Path,
    input_dir: Path,
    custom_output_dir: Path | None,
    output_suffix: str,
) -> Path:
    """Computes the target destination path ensuring the original file is preserved intact."""
    rel_path = input_file.relative_to(input_dir)
    if custom_output_dir is not None:
        target_dir = custom_output_dir / input_dir.name
    else:
        target_dir = input_dir.parent / f"{input_dir.name}{output_suffix}"
    return target_dir / rel_path


def find_json_files(input_dirs: list[Path]) -> list[tuple[Path, Path]]:
    """Discovers all JSON files in the given input directories, returning (file_path, root_input_dir)."""
    results = []
    for d in input_dirs:
        d_path = Path(d)
        if not d_path.exists():
            print(f"[Warning] Input directory does not exist: {d_path}")
            continue
        for json_file in sorted(d_path.glob("**/*.json")):
            results.append((json_file, d_path))
    return results


def process_file(
    input_file: Path,
    output_file: Path,
    cfg: Config,
    *,
    num_judges: int | None = None,
    limit: int | None = None,
    filter_recipe: str | None = None,
    resume: bool = True,
    dry_run: bool = False,
    ollama_available: bool = False,
):
    """Processes a single JSON file, performs Round 2 re-evaluation, and saves a copy."""
    print(f"\n{'='*75}")
    print(f"Processing: {input_file}")
    print(f"Output to : {output_file}")
    print(f"{'='*75}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    gen_model_name = data.get("model")
    task = data.get("target_type")
    qa_pairs = data.get("qa_pairs", [])

    if not qa_pairs:
        print("  [Warning] No qa_pairs found in file. Skipping.")
        return

    # Check judges present in Round 1
    sample_r1_judges = qa_pairs[0].get("round_1", {}).get("judges", {})
    all_judge_names = list(sample_r1_judges.keys())

    if num_judges is not None and num_judges > 0:
        selected_judge_names = all_judge_names[:num_judges]
    else:
        selected_judge_names = all_judge_names

    print(f"  Generator Model : {gen_model_name}")
    print(f"  Target Type     : {task}")
    print(f"  Total QA Pairs  : {len(qa_pairs)}")
    print(f"  Original Judges : {all_judge_names}")
    print(f"  Selected Judges : {selected_judge_names} (N={len(selected_judge_names)})")

    # Filter qa_pairs if requested
    filtered_pairs = []
    for q in qa_pairs:
        if filter_recipe and filter_recipe.lower() not in q.get("recipe_name", "").lower():
            continue
        filtered_pairs.append(q)

    if limit is not None and limit > 0:
        filtered_pairs = filtered_pairs[:limit]

    print(f"  Active QA Pairs to process: {len(filtered_pairs)}")

    if dry_run:
        print("  [Dry-Run] Validation passed. No modifications made.")
        return

    if not ollama_available:
        raise RuntimeError("Ollama server is not available. Please start Ollama before running re-evaluation.")

    # Create output directory structure
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # If resuming, load existing output if present
    output_data = copy.deepcopy(data)
    if resume and output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as of:
                existing_output = json.load(of)
            existing_qa_by_idx = {
                (q.get("qa_index"), q.get("recipe_name")): q
                for q in existing_output.get("qa_pairs", [])
            }
            # Merge existing round_2 if present
            for q in output_data.get("qa_pairs", []):
                key = (q.get("qa_index"), q.get("recipe_name"))
                if key in existing_qa_by_idx and "round_2" in existing_qa_by_idx[key]:
                    q["round_2"] = existing_qa_by_idx[key]["round_2"]
            print(f"  [Resume] Loaded previous progress from {output_file.name}")
        except Exception as e:
            print(f"  [Resume Warning] Could not parse existing output file ({e}), starting fresh.")

    # Helper map for in-memory tracking
    qa_map = {
        (q.get("qa_index"), q.get("recipe_name")): q
        for q in output_data["qa_pairs"]
    }

    # Step 1: Initialize Generator Runtime and compute corrections (CL and CA)
    print(f"\n  [Phase 1] Initializing generator model: {gen_model_name}...")
    gen_spec = _build_inline_model_spec(gen_model_name)
    with create_model_runtime(
        gen_spec,
        device=cfg.device,
        timeout=cfg.llm_timeout,
        ollama_available=ollama_available,
        delete_after_run=False,
    ) as generator_runtime:

        for idx, qa in enumerate(filtered_pairs, 1):
            key = (qa.get("qa_index"), qa.get("recipe_name"))
            target_qa = qa_map[key]

            # If already has round_2 with corrected_answer and correction_list, we can skip phase 1
            if resume and "round_2" in target_qa and target_qa["round_2"].get("corrected_answer"):
                print(f"    ({idx}/{len(filtered_pairs)}) [Skipping Phase 1] {qa.get('recipe_name')} already corrected.")
                continue

            orig_ans = qa.get("original_answer", "")
            r1_judges = qa.get("round_1", {}).get("judges", {})

            print(f"    ({idx}/{len(filtered_pairs)}) Generating correction for: {qa.get('recipe_name')}...")

            # Sequentially build correction list from the selected judges
            correction_list = ""
            for j_name in selected_judge_names:
                if j_name in r1_judges:
                    explanation = r1_judges[j_name].get("explanation", "")
                    correction_list = provide_judgement(
                        orig_ans,
                        explanation,
                        correction_list,
                        generator_runtime,
                        cfg,
                    )

            # Produce corrected answer
            corrected_ans = correct_answer(
                orig_ans,
                correction_list,
                generator_runtime,
                cfg,
            )

            # Initialize round_2 container
            if "round_2" not in target_qa:
                target_qa["round_2"] = {}
            target_qa["round_2"]["corrected_answer"] = corrected_ans
            target_qa["round_2"]["correction_list"] = correction_list

            # Incremental save
            with open(output_file, "w", encoding="utf-8") as f_out:
                json.dump(output_data, f_out, indent=2)

    # Step 2: Initialize Judge Runtimes and evaluate Round 2
    print(f"\n  [Phase 2] Initializing {len(selected_judge_names)} judge models...")
    judge_runtimes = []
    for j_name in selected_judge_names:
        j_spec = _build_inline_model_spec(j_name)
        rt = create_model_runtime(
            j_spec,
            device=cfg.device,
            timeout=cfg.llm_timeout,
            ollama_available=ollama_available,
            delete_after_run=False,
        )
        judge_runtimes.append(rt)

    try:
        for idx, qa in enumerate(filtered_pairs, 1):
            key = (qa.get("qa_index"), qa.get("recipe_name"))
            target_qa = qa_map[key]

            # Check if judging already complete
            if resume and "round_2" in target_qa and target_qa["round_2"].get("judges"):
                existing_r2_judges = target_qa["round_2"]["judges"]
                if all(j in existing_r2_judges for j in selected_judge_names):
                    print(f"    ({idx}/{len(filtered_pairs)}) [Skipping Phase 2] {qa.get('recipe_name')} already judged.")
                    continue

            corrected_ans = target_qa["round_2"].get("corrected_answer", "")
            query = qa.get("query", "")
            correct_answers_list = qa.get("correct_answers", [])
            correct_answers_str = "\n".join(correct_answers_list) + "\n"

            print(f"    ({idx}/{len(filtered_pairs)}) [Round 2] Judging: {qa.get('recipe_name')}...")
            r2_judges_dict, r2_avg = run_judge_round(
                corrected_ans,
                query,
                correct_answers_str,
                judge_runtimes,
                cfg,
            )

            target_qa["round_2"]["judges"] = r2_judges_dict
            target_qa["round_2"]["average_score"] = r2_avg

            # Incremental checkpoint save
            with open(output_file, "w", encoding="utf-8") as f_out:
                json.dump(output_data, f_out, indent=2)

    finally:
        for rt in judge_runtimes:
            rt.close()

    print(f"  [Completed] Saved updated copy to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Round-2 Answer Re-Evaluation on existing Round-1 JSON results, preserving original files intact."
    )
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        type=Path,
        default=DEFAULT_INPUT_DIRS,
        help="Input directories containing Round-1 JSON result files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Root output directory. If omitted, directories with '--output_suffix' are created alongside inputs.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_ROUND2",
        help="Suffix appended to input directory names for copies (default: '_ROUND2').",
    )
    parser.add_argument(
        "-n",
        "--num_judges",
        type=int,
        default=None,
        help="Number of judges to use (e.g. 1 or 2). If less than 3, picks the first N judges in order from Round 1. Default: all (3).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of QA pairs per file (useful for testing).",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default=None,
        help="Filter to a single recipe name substring (e.g. 'Carbonara').",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=None,
        help="Filter to specific generator model names.",
    )
    parser.add_argument(
        "--target_type",
        choices=["ingredients", "directions"],
        default=None,
        help="Filter to ingredients or directions.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resuming from existing output files and recompute from scratch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan files and display execution plan without invoking LLMs or modifying files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("       RAG ANSWER RE-EVALUATION RUNNER (ROUND 2 ON EXISTING JSONS)")
    print("=" * 80)

    cfg = load_config(args.config)
    try:
        ollama_available = check_ollama_server()
    except Exception:
        ollama_available = False

    if not ollama_available:
        if args.dry_run:
            print("[Note] Ollama server is currently offline. Proceeding in dry-run mode.")
        else:
            print("[Warning] Ollama server is currently offline. Please start Ollama before executing.")

    all_files = find_json_files(args.input_dirs)
    if not all_files:
        print("No JSON files found in the specified input directories.")
        return

    print(f"Discovered {len(all_files)} JSON result files across {len(args.input_dirs)} input directories.\n")

    files_to_process = []
    for input_file, root_dir in all_files:
        with open(input_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        gen_model = meta.get("model", "")
        task = meta.get("target_type", "")

        if args.models:
            if not any(m.lower() in gen_model.lower() for m in args.models):
                continue
        if args.target_type and task != args.target_type:
            continue

        output_file = determine_output_path(
            input_file,
            root_dir,
            args.output_dir,
            args.output_suffix,
        )
        files_to_process.append((input_file, output_file))

    print(f"Files matched after filters: {len(files_to_process)}")
    for in_f, out_f in files_to_process:
        print(f"  • {in_f.parent.name}/{in_f.name} -> {out_f.parent.name}/{out_f.name}")

    if args.dry_run:
        print("\n--- Running Dry-Run File Inspection ---")
        for in_f, out_f in files_to_process:
            process_file(
                in_f,
                out_f,
                cfg,
                num_judges=args.num_judges,
                limit=args.limit,
                filter_recipe=args.recipe,
                resume=not args.no_resume,
                dry_run=True,
                ollama_available=ollama_available,
            )
        print("\n[Dry-Run Complete] All configurations and file paths are valid.")
        return

    # Normal execution
    for in_f, out_f in files_to_process:
        process_file(
            in_f,
            out_f,
            cfg,
            num_judges=args.num_judges,
            limit=args.limit,
            filter_recipe=args.recipe,
            resume=not args.no_resume,
            dry_run=False,
            ollama_available=ollama_available,
        )

    print("\nAll files successfully processed and saved!")


if __name__ == "__main__":
    main()
