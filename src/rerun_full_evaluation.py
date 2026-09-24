import argparse
import copy
import json
import re
import shutil
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

DEFAULT_FINAL_JSON_DIR = PROJECT_ROOT / "results" / "FinalJson"


def provide_judgement(
    original_answer: str,
    judgement: str,
    current_cl: str,
    generator_runtime,
    cfg: Config,
    temperature: float = 0.0,
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
        temperature=temperature,  # 0.0 for greedy deterministic correction list
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
    temperature: float = 0.0,
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
        temperature=temperature,  # 0.0 for greedy deterministic corrected answer
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


def _is_r1_done(qa: dict, judges: list[str]) -> bool:
    """Checks if Round 1 judge evaluation is complete for this QA pair."""
    if qa.get("_r1_complete") is True:
        return True
    r1 = qa.get("round_1")
    if not isinstance(r1, dict):
        return False
    j_dict = r1.get("judges")
    if not isinstance(j_dict, dict) or not j_dict:
        return False
    if judges:
        return all(j in j_dict and isinstance(j_dict[j], dict) and j_dict[j].get("score") is not None for j in judges)
    return True


def _is_gen_done(qa: dict) -> bool:
    """Checks if Round 2 answer correction has been generated for this QA pair."""
    if qa.get("_gen_complete") is True:
        return True
    r2 = qa.get("round_2")
    if not isinstance(r2, dict):
        return False
    ans = r2.get("corrected_answer")
    return bool(ans and isinstance(ans, str) and ans.strip())


def _is_r2_done(qa: dict, judges: list[str]) -> bool:
    """Checks if Round 2 judge evaluation is complete for this QA pair."""
    if qa.get("_r2_complete") is True:
        return True
    r2 = qa.get("round_2")
    if not isinstance(r2, dict):
        return False
    j_dict = r2.get("judges")
    if not isinstance(j_dict, dict) or not j_dict:
        return False
    if judges:
        return all(j in j_dict and isinstance(j_dict[j], dict) and j_dict[j].get("score") is not None for j in judges)
    return True


def process_file_full_rerun(
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
    corrector_temperature: float = 0.0,
):
    """
    Reruns both Round 1 and Round 2 completely from scratch:
      1. Evaluates original_answer with the 3 judges (Round 1).
      2. Uses the fresh Round 1 feedback to generate correction_list and corrected_answer with the generator model.
      3. Evaluates corrected_answer with the 3 judges (Round 2).
    """
    print(f"\n{'='*80}")
    print(f"File   : {input_file}")
    print(f"Output : {output_file}")
    print(f"{'='*80}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    gen_model_name = data.get("model")
    task = data.get("target_type")
    qa_pairs = data.get("qa_pairs", [])

    if not qa_pairs:
        print("  [Warning] No qa_pairs found in file. Skipping.")
        return

    # Discover judges from original Round 1 configuration
    sample_r1_judges = qa_pairs[0].get("round_1", {}).get("judges", {})
    all_judge_names = list(sample_r1_judges.keys())

    if num_judges is not None and num_judges > 0:
        selected_judge_names = all_judge_names[:num_judges]
    else:
        selected_judge_names = all_judge_names

    print(f"  Generator Model : {gen_model_name}")
    print(f"  Target Type     : {task}")
    print(f"  Total QA Pairs  : {len(qa_pairs)}")
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

    if not dry_run and not ollama_available:
        raise RuntimeError("Ollama server is not available. Please start Ollama before executing.")

    # Create backup copy (.bak) if modifying in-place
    if output_file.resolve() == input_file.resolve() and not dry_run:
        bak_file = input_file.with_suffix(input_file.suffix + ".bak")
        if not bak_file.exists():
            shutil.copy2(input_file, bak_file)
            print(f"  [Backup] Created backup copy: {bak_file.name}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Initialize working data
    output_data = copy.deepcopy(data)

    # If resuming from existing output file, load previous progress
    if resume and output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as of:
                existing_output = json.load(of)
            existing_qa_list = existing_output.get("qa_pairs", [])
            existing_by_idx_name = {
                (q.get("qa_index"), q.get("recipe_name")): q
                for q in existing_qa_list
            }
            existing_by_name = {
                q.get("recipe_name"): q
                for q in existing_qa_list
                if q.get("recipe_name")
            }

            for i, q in enumerate(output_data.get("qa_pairs", [])):
                key = (q.get("qa_index"), q.get("recipe_name"))
                ex = None
                if key in existing_by_idx_name:
                    ex = existing_by_idx_name[key]
                elif q.get("recipe_name") in existing_by_name:
                    ex = existing_by_name[q.get("recipe_name")]
                elif i < len(existing_qa_list):
                    cand = existing_qa_list[i]
                    if cand.get("recipe_name") == q.get("recipe_name"):
                        ex = cand

                if ex:
                    if _is_r1_done(ex, selected_judge_names):
                        q["round_1"] = copy.deepcopy(ex.get("round_1"))
                        q["_r1_complete"] = True

                        if _is_gen_done(ex):
                            if "round_2" not in q:
                                q["round_2"] = {}
                            q["round_2"]["corrected_answer"] = ex["round_2"].get("corrected_answer", "")
                            q["round_2"]["correction_list"] = ex["round_2"].get("correction_list", "")
                            q["_gen_complete"] = True

                            if _is_r2_done(ex, selected_judge_names):
                                q["round_2"]["judges"] = copy.deepcopy(ex["round_2"].get("judges", {}))
                                q["round_2"]["average_score"] = ex["round_2"].get("average_score", 0.0)
                                q["_r2_complete"] = True

            r1_done_cnt = sum(1 for q in output_data.get("qa_pairs", []) if q.get("_r1_complete"))
            gen_done_cnt = sum(1 for q in output_data.get("qa_pairs", []) if q.get("_gen_complete"))
            r2_done_cnt = sum(1 for q in output_data.get("qa_pairs", []) if q.get("_r2_complete"))
            print(f"  [Resume] Loaded previous progress from {output_file.name}:")
            print(f"           • Round 1 complete: {r1_done_cnt}/{len(output_data['qa_pairs'])}")
            print(f"           • Correction complete: {gen_done_cnt}/{len(output_data['qa_pairs'])}")
            print(f"           • Round 2 complete: {r2_done_cnt}/{len(output_data['qa_pairs'])}")
        except Exception as e:
            print(f"  [Resume Warning] Could not parse existing output file ({e}), starting fresh.")

    qa_map = {
        (q.get("qa_index"), q.get("recipe_name")): q
        for q in output_data["qa_pairs"]
    }

    # Pre-calculate workload across all steps
    pairs_needing_r1 = []
    pairs_needing_gen = []
    pairs_needing_r2 = []
    for qa in filtered_pairs:
        key = (qa.get("qa_index"), qa.get("recipe_name"))
        target_qa = qa_map[key]
        if not (resume and target_qa.get("_r1_complete", False)):
            pairs_needing_r1.append(qa)
        if not (resume and target_qa.get("_gen_complete", False)):
            pairs_needing_gen.append(qa)
        if not (resume and target_qa.get("_r2_complete", False)):
            pairs_needing_r2.append(qa)

    if dry_run:
        print(f"\n  [Dry-Run Plan]")
        print(f"    • Step 1 (Round 1 Judging)      : {len(pairs_needing_r1)}/{len(filtered_pairs)} QA pairs to run")
        print(f"    • Step 2 (Answer Correction)    : {len(pairs_needing_gen)}/{len(filtered_pairs)} QA pairs to run")
        print(f"    • Step 3 (Round 2 Judging)      : {len(pairs_needing_r2)}/{len(filtered_pairs)} QA pairs to run")
        print(f"  [Dry-Run] All checks passed. No modifications made.")
        return

    if pairs_needing_r1:
        print(f"\n  [Step 1/3] Running Round 1 Judging for {len(pairs_needing_r1)} QA pairs...")
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
            for idx, qa in enumerate(pairs_needing_r1, 1):
                key = (qa.get("qa_index"), qa.get("recipe_name"))
                target_qa = qa_map[key]
                recipe_name = qa.get("recipe_name", f"Index {key[0]}")
                query = qa.get("query", "")
                orig_ans = target_qa.get("original_answer", "")
                correct_answers_list = qa.get("correct_answers", [])
                correct_answers_str = "\n".join(correct_answers_list) + "\n"

                print(f"    ({idx}/{len(pairs_needing_r1)}) [Round 1] Judging: {recipe_name}...")
                r1_judges_dict, r1_avg = run_judge_round(
                    orig_ans,
                    query,
                    correct_answers_str,
                    judge_runtimes,
                    cfg,
                )

                target_qa["round_1"] = {
                    "judges": r1_judges_dict,
                    "average_score": r1_avg,
                }
                target_qa["_r1_complete"] = True
                target_qa["_gen_complete"] = False
                target_qa["_r2_complete"] = False

                # Checkpoint save
                with open(output_file, "w", encoding="utf-8") as f_out:
                    json.dump(output_data, f_out, indent=2)

        finally:
            for rt in judge_runtimes:
                rt.close()
    else:
        print(f"\n  [Step 1/3] Round 1 Judging already complete for all {len(filtered_pairs)} QA pairs.")

    # =========================================================================
    # Step 2: Answer Correction Generation (Generator LLM produces Round 2 answer)
    # =========================================================================
    pairs_needing_gen = []
    for qa in filtered_pairs:
        key = (qa.get("qa_index"), qa.get("recipe_name"))
        target_qa = qa_map[key]
        gen_done = resume and target_qa.get("_gen_complete", False)
        if not gen_done:
            pairs_needing_gen.append(qa)

    if pairs_needing_gen:
        print(f"\n  [Step 2/3] Generating Answer Corrections with {gen_model_name} for {len(pairs_needing_gen)} QA pairs...")
        gen_spec = _build_inline_model_spec(gen_model_name)
        with create_model_runtime(
            gen_spec,
            device=cfg.device,
            timeout=cfg.llm_timeout,
            ollama_available=ollama_available,
            delete_after_run=False,
        ) as generator_runtime:

            for idx, qa in enumerate(pairs_needing_gen, 1):
                key = (qa.get("qa_index"), qa.get("recipe_name"))
                target_qa = qa_map[key]
                recipe_name = qa.get("recipe_name", f"Index {key[0]}")
                orig_ans = target_qa.get("original_answer", "")
                r1_judges = target_qa.get("round_1", {}).get("judges", {})

                print(f"    ({idx}/{len(pairs_needing_gen)}) [Correcting] {recipe_name}...")

                # Sequentially incorporate feedback from each judge into correction_list
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
                            temperature=corrector_temperature,
                        )

                # Produce corrected answer
                corrected_ans = correct_answer(
                    orig_ans,
                    correction_list,
                    generator_runtime,
                    cfg,
                    temperature=corrector_temperature,
                )

                if "round_2" not in target_qa:
                    target_qa["round_2"] = {}
                target_qa["round_2"]["corrected_answer"] = corrected_ans
                target_qa["round_2"]["correction_list"] = correction_list
                target_qa["_gen_complete"] = True
                target_qa["_r2_complete"] = False

                # Checkpoint save
                with open(output_file, "w", encoding="utf-8") as f_out:
                    json.dump(output_data, f_out, indent=2)
    else:
        print(f"\n  [Step 2/3] Answer Correction already complete for all {len(filtered_pairs)} QA pairs.")

    # =========================================================================
    # Step 3: Round 2 Evaluation (Evaluate corrected_answer with 3 judges)
    # =========================================================================
    pairs_needing_r2 = []
    for qa in filtered_pairs:
        key = (qa.get("qa_index"), qa.get("recipe_name"))
        target_qa = qa_map[key]
        r2_done = resume and target_qa.get("_r2_complete", False)
        if not r2_done:
            pairs_needing_r2.append(qa)

    if pairs_needing_r2:
        print(f"\n  [Step 3/3] Running Round 2 Judging for {len(pairs_needing_r2)} QA pairs...")
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
            for idx, qa in enumerate(pairs_needing_r2, 1):
                key = (qa.get("qa_index"), qa.get("recipe_name"))
                target_qa = qa_map[key]
                recipe_name = qa.get("recipe_name", f"Index {key[0]}")
                query = qa.get("query", "")
                orig_ans = target_qa.get("original_answer", "")
                corrected_ans = target_qa.get("round_2", {}).get("corrected_answer", "")
                correct_answers_list = qa.get("correct_answers", [])
                correct_answers_str = "\n".join(correct_answers_list) + "\n"

                # Optimization: if the corrected answer is identical to the original answer,
                # reuse Round 1 judge evaluation directly to guarantee 100% score consistency.
                if orig_ans.strip() == corrected_ans.strip():
                    print(f"    ({idx}/{len(pairs_needing_r2)}) [Round 2] Reusing Round 1 evaluation (identical answer) for: {recipe_name}")
                    r1_judges = target_qa.get("round_1", {}).get("judges", {})
                    r1_avg = target_qa.get("round_1", {}).get("average_score", 0.0)
                    target_qa["round_2"]["judges"] = copy.deepcopy(r1_judges)
                    target_qa["round_2"]["average_score"] = r1_avg
                else:
                    print(f"    ({idx}/{len(pairs_needing_r2)}) [Round 2] Judging: {recipe_name}...")
                    r2_judges_dict, r2_avg = run_judge_round(
                        corrected_ans,
                        query,
                        correct_answers_str,
                        judge_runtimes,
                        cfg,
                    )
                    target_qa["round_2"]["judges"] = r2_judges_dict
                    target_qa["round_2"]["average_score"] = r2_avg

                target_qa["_r2_complete"] = True

                # Checkpoint save
                with open(output_file, "w", encoding="utf-8") as f_out:
                    json.dump(output_data, f_out, indent=2)

        finally:
            for rt in judge_runtimes:
                rt.close()
    else:
        print(f"\n  [Step 3/3] Round 2 Judging already complete for all {len(filtered_pairs)} QA pairs.")

    # Clean up internal tracking markers before final save
    for qa in output_data.get("qa_pairs", []):
        qa.pop("_r1_complete", None)
        qa.pop("_gen_complete", None)
        qa.pop("_r2_complete", None)

    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(output_data, f_out, indent=2)

    print(f"  [Completed] Successfully reran and saved: {output_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Rerun full evaluation from scratch: Round 1 Judging -> Answer Correction -> Round 2 Judging."
    )
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        type=Path,
        default=[DEFAULT_FINAL_JSON_DIR],
        help="Input directories containing JSON files (default: results/FinalJson).",
    )
    parser.add_argument(
        "--in_place",
        action="store_true",
        help="Modify the input JSON files in place (creates a .bak backup copy first).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Custom output directory. If omitted and --in_place is not set, copies are saved with --output_suffix.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_RERUN",
        help="Suffix appended to input directory name if output_dir is not specified (default: '_RERUN').",
    )
    parser.add_argument(
        "-n",
        "--num_judges",
        type=int,
        default=None,
        help="Number of judges to use (default: all 3).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of QA pairs per file (useful for testing in Colab).",
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
        help="Filter to specific generator model names (e.g. 'qwen3.5:4b').",
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
        help="Disable resuming from existing progress and start all steps from zero.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Display plan and validation without calling any LLMs or changing files.",
    )
    parser.add_argument(
        "--num_predict",
        type=int,
        default=512,
        help="Max tokens for judge output (default: 512, recommended to prevent truncation).",
    )
    parser.add_argument(
        "--corrector_temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the answer corrector (default: 0.0 for greedy deterministic generation).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("       COMPLETE END-TO-END RAG RE-EVALUATION RUNNER (ROUNDS 1 & 2)")
    print("=" * 80)

    cfg = load_config(args.config)
    if args.num_predict is not None:
        cfg.llm_num_predict = args.num_predict

    try:
        ollama_available = check_ollama_server()
    except Exception:
        ollama_available = False

    if not ollama_available:
        if args.dry_run:
            print("[Note] Ollama server is offline. Proceeding in dry-run mode.")
        else:
            print("[Warning] Ollama server is offline. Start Ollama before executing.")

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

        if args.in_place:
            output_file = input_file
        elif args.output_dir is not None:
            rel = input_file.relative_to(root_dir)
            output_file = args.output_dir / root_dir.name / rel
        else:
            rel = input_file.relative_to(root_dir)
            target_dir = root_dir.parent / f"{root_dir.name}{args.output_suffix}"
            output_file = target_dir / rel

        files_to_process.append((input_file, output_file))

    print(f"Files matched after filters: {len(files_to_process)}")
    for in_f, out_f in files_to_process:
        print(f"  • {in_f.parent.name}/{in_f.name} -> {out_f.parent.name}/{out_f.name}")

    for in_f, out_f in files_to_process:
        process_file_full_rerun(
            in_f,
            out_f,
            cfg,
            num_judges=args.num_judges,
            limit=args.limit,
            filter_recipe=args.recipe,
            resume=not args.no_resume,
            dry_run=args.dry_run,
            ollama_available=ollama_available,
            corrector_temperature=args.corrector_temperature,
        )

    if args.dry_run:
        print("\n[Dry-Run Complete] All configurations and file paths are valid.")
    else:
        print("\nAll files successfully evaluated and saved!")


if __name__ == "__main__":
    main()
