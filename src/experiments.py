import json
from dataclasses import dataclass
from itertools import count
from pathlib import Path
import numpy as np

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import Config
from .generation import query_llm
from .metrics import compute_iou_stats, sentence_iou
from .prompts import (
    SYSTEM_PROMPT_PLAIN,
    SYSTEM_PROMPT_RAG,
    SYSTEM_PROMPT_DIRECTIONS_PLAIN,
    SYSTEM_PROMPT_DIRECTIONS_RAG,
)
from .retrieval import build_faiss_index, retrieve, retrieve_oracle


@dataclass(frozen=True)
class GeneratedCase:
    qa_index: int
    size: int
    recipe_id: str
    recipe_name: str
    query: str
    correct_chunks: list[str]
    context_indices: list[int]
    context_chunks: list[str]
    context: str
    response_rag: str | None
    response_llm: str | None


def prepare_generation_cases(
    grouped_by_size: list,
    encoding_model: SentenceTransformer,
    cfg: Config,
    model_runtime,
    target_type: str,
    *,
    include_rag: bool,
    include_llm: bool,
    trace_path: Path | None = None,
) -> dict[int, list[GeneratedCase]]:
    if not include_rag and not include_llm:
        raise ValueError("At least one response path must be enabled.")

    if target_type == "ingredients":
        sys_prompt_rag = SYSTEM_PROMPT_RAG
        sys_prompt_plain = SYSTEM_PROMPT_PLAIN
    elif target_type == "directions":
        sys_prompt_rag = SYSTEM_PROMPT_DIRECTIONS_RAG
        sys_prompt_plain = SYSTEM_PROMPT_DIRECTIONS_PLAIN
    else:
        raise ValueError(f"Invalid target_type: {target_type}")

    cases_by_size: dict[int, list[GeneratedCase]] = {}
    call_counter = count(1)
    trace_entries = [] if trace_path is not None else None

    for group in tqdm(grouped_by_size, desc=f"Generation Cases ({target_type})"):
        size = group.size
        strategy = group.strategy
        questions = group.questions_ingredients if target_type == "ingredients" else group.questions_directions

        # Build FAISS index/indices for this group if we need to do RAG
        index_ing = None
        index_dir = None
        if include_rag:
            if strategy == "Doppio":
                if len(group.global_chunks_ingredients) > 0:
                    nlist = min(cfg.nlist, len(group.global_chunks_ingredients))
                    embeddings_f32 = np.array(group.embeddings_ingredients).astype(np.float32)
                    index_ing = build_faiss_index(embeddings_f32, nlist=max(1, nlist), nprobe=cfg.nprobe)
                if len(group.global_chunks_directions) > 0:
                    nlist = min(cfg.nlist, len(group.global_chunks_directions))
                    embeddings_f32 = np.array(group.embeddings_directions).astype(np.float32)
                    index_dir = build_faiss_index(embeddings_f32, nlist=max(1, nlist), nprobe=cfg.nprobe)
            else:
                if len(group.global_chunks) > 0:
                    nlist = min(cfg.nlist, len(group.global_chunks))
                    embeddings_f32 = np.array(group.embeddings).astype(np.float32)
                    index_ing = build_faiss_index(embeddings_f32, nlist=max(1, nlist), nprobe=cfg.nprobe)
                    index_dir = index_ing

        size_cases: list[GeneratedCase] = []
        for qa_index, (
            recipe_name,
            recipe_code_id,
            query_text,
            correct_indices,
            correct_ground_truth_chunks,
        ) in enumerate(tqdm(questions, desc=f"Size {size}", leave=False)):
            
            # Select target database chunks
            if strategy == "Doppio":
                current_chunks = group.global_chunks_ingredients if target_type == "ingredients" else group.global_chunks_directions
                current_index = index_ing if target_type == "ingredients" else index_dir
            else:
                current_chunks = group.global_chunks
                current_index = index_ing

            correct_chunks = correct_ground_truth_chunks
            
            response_rag = None
            response_llm = None

            if include_rag:
                if current_index is not None:
                    # Use FAISS to retrieve the context, setting k = len(correct_indices)
                    retrieved_ids, retrieved_chunks, context = retrieve(
                        query=query_text,
                        k=len(correct_indices),
                        index=current_index,
                        global_chunks=current_chunks,
                        correct_indices=correct_indices,
                        encoding_model=encoding_model,
                        device=cfg.device
                    )
                    context_indices = [int(i) for i in retrieved_ids]
                    context_chunks = [current_chunks[i] for i in context_indices]
                else:
                    context_indices = []
                    context_chunks = []
                    context = ""

                response_rag = query_llm(
                    sys_prompt_rag.format(context=context),
                    query_text,
                    model_runtime,
                    call_index=next(call_counter),
                    num_ctx=cfg.llm_num_ctx,
                    num_predict=cfg.llm_num_predict,
                    think=cfg.llm_think,
                    temperature=cfg.llm_temperature,
                    timeout=cfg.llm_timeout,
                    retries=cfg.llm_retries,
                    keep_alive=cfg.llm_keep_alive,
                )
            else:
                context_indices = [int(index_value) for index_value in correct_indices]
                context_chunks = [
                    current_chunks[index_value] for index_value in context_indices
                ]
                context = ""

            if include_llm:
                response_llm = query_llm(
                    sys_prompt_plain,
                    query_text,
                    model_runtime,
                    call_index=next(call_counter),
                    num_ctx=cfg.llm_num_ctx,
                    num_predict=cfg.llm_num_predict,
                    think=cfg.llm_think,
                    temperature=cfg.llm_temperature,
                    timeout=cfg.llm_timeout,
                    retries=cfg.llm_retries,
                    keep_alive=cfg.llm_keep_alive,
                )

            size_cases.append(
                GeneratedCase(
                    qa_index=qa_index,
                    size=int(size),
                    recipe_id=recipe_code_id,
                    recipe_name=recipe_name,
                    query=query_text,
                    correct_chunks=correct_chunks,
                    context_indices=context_indices,
                    context_chunks=context_chunks,
                    context=context,
                    response_rag=response_rag,
                    response_llm=response_llm,
                )
            )

            if trace_entries is not None:
                trace_entries.append(
                    {
                        "qa_index": qa_index,
                        "size": int(size),
                        "recipe_id": recipe_code_id,
                        "recipe_name": recipe_name,
                        "query": query_text,
                        "retrieved_indices": context_indices,
                        "retrieved_chunks": context_chunks,
                        "retrieved_context": context,
                        "context_source": f"faiss_vector_index_{strategy}_{target_type}" if include_rag else f"oracle_dataset_chunks_{strategy}_{target_type}",
                    }
                )

        cases_by_size[int(size)] = size_cases

    if trace_entries is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trace_path, "w") as f:
            json.dump(
                {
                    "trace_version": 2,
                    "model": getattr(
                        model_runtime, "results_label", model_runtime.visible_label
                    ),
                    "experiment": f"rag_{target_type}",
                    "context_source": f"faiss_vector_index_{target_type}",
                    "reference_chunks_source": f"dataset_correct_indices_{target_type}",
                    "entries": trace_entries,
                },
                f,
                indent=2,
            )
    return cases_by_size


def _load_generation_cases(
    grouped_by_size: list,
    encoding_model: SentenceTransformer,
    cfg: Config,
    model_runtime,
    target_type: str,
    *,
    include_rag: bool,
    include_llm: bool,
    trace_path: Path | None = None,
    generated_cases: dict[int, list[GeneratedCase]] | None = None,
) -> dict[int, list[GeneratedCase]]:
    if generated_cases is not None:
        return generated_cases
    return prepare_generation_cases(
        grouped_by_size,
        encoding_model,
        cfg,
        model_runtime,
        target_type,
        include_rag=include_rag,
        include_llm=include_llm,
        trace_path=trace_path,
    )


def _require_response(case: GeneratedCase, field_name: str) -> str:
    response = getattr(case, field_name)
    if response is None:
        raise ValueError(f"Missing {field_name} for query '{case.query}'.")
    return response


def run_test_rag(
    grouped_by_size: list,
    encoding_model: SentenceTransformer,
    cfg: Config,
    model_runtime,
    target_type: str,
    trace_path: Path | None = None,
    generated_cases: dict[int, list[GeneratedCase]] | None = None,
) -> dict:
    """Test 1: Query WITH context."""
    print(
        f"\n=== Test 1: RAG (with context) | Target: {target_type} | Model: {model_runtime.visible_label} ==="
    )
    results = {}
    cases_by_size = _load_generation_cases(
        grouped_by_size,
        encoding_model,
        cfg,
        model_runtime,
        target_type,
        include_rag=True,
        include_llm=False,
        trace_path=trace_path,
        generated_cases=generated_cases,
    )
    correct_key = "correct_ingredients" if target_type == "ingredients" else "correct_directions"
    for group in tqdm(grouped_by_size, desc="RAG Tests"):
        size = group.size
        maxs, means, stddevs = [], [], []
        qa_pairs = []
        for case in tqdm(cases_by_size[int(size)], desc=f"Size {size}", leave=False):
            response = _require_response(case, "response_rag")
            mx, mn, sd = compute_iou_stats(response, case.correct_chunks)
            maxs.append(mx)
            means.append(mn)
            stddevs.append(sd)
            qa_pairs.append(
                {
                    "qa_index": case.qa_index,
                    "recipe_id": case.recipe_id,
                    "recipe_name": case.recipe_name,
                    "query": case.query,
                    correct_key: case.correct_chunks,
                    "response": response,
                }
            )

        results[size] = {
            "maxs": maxs,
            "means": means,
            "stddevs": stddevs,
            "pct_perfect": len([v for v in maxs if v == 1.0]) / len(maxs)
            if maxs
            else 0.0,
            "pct_near": len([v for v in maxs if v >= 0.95]) / len(maxs)
            if maxs
            else 0.0,
            "qa_pairs": qa_pairs,
        }
    return results


def run_test_llm_only(
    grouped_by_size: list,
    encoding_model: SentenceTransformer,
    cfg: Config,
    model_runtime,
    target_type: str,
    generated_cases: dict[int, list[GeneratedCase]] | None = None,
) -> dict:
    """Test 2: Direct query WITHOUT context."""
    print(
        f"\n=== Test 2: LLM only (no context) | Target: {target_type} | Model: {model_runtime.visible_label} ==="
    )
    results = {}
    cases_by_size = _load_generation_cases(
        grouped_by_size,
        encoding_model,
        cfg,
        model_runtime,
        target_type,
        include_rag=False,
        include_llm=True,
        generated_cases=generated_cases,
    )
    correct_key = "correct_ingredients" if target_type == "ingredients" else "correct_directions"
    for group in tqdm(grouped_by_size, desc="LLM-only Tests"):
        size = group.size
        maxs, means, stddevs = [], [], []

        qa_pairs = []
        for case in tqdm(cases_by_size[int(size)], desc=f"Size {size}", leave=False):
            response = _require_response(case, "response_llm")
            mx, mn, sd = compute_iou_stats(response, case.correct_chunks)
            maxs.append(mx)
            means.append(mn)
            stddevs.append(sd)
            qa_pairs.append(
                {
                    "recipe_id": case.recipe_id,
                    "recipe_name": case.recipe_name,
                    "query": case.query,
                    correct_key: case.correct_chunks,
                    "response": response,
                }
            )

        results[size] = {
            "maxs": maxs,
            "means": means,
            "stddevs": stddevs,
            "pct_perfect": len([v for v in maxs if v == 1.0]) / len(maxs)
            if maxs
            else 0.0,
            "pct_near": len([v for v in maxs if v >= 0.95]) / len(maxs)
            if maxs
            else 0.0,
            "qa_pairs": qa_pairs,
        }
    return results


def run_test_rag_vs_llm(
    grouped_by_size: list,
    encoding_model: SentenceTransformer,
    cfg: Config,
    model_runtime,
    target_type: str,
    generated_cases: dict[int, list[GeneratedCase]] | None = None,
) -> dict:
    """Test 3: Compare RAG response vs LLM-only response (divergence)."""
    print(
        f"\n=== Test 3: RAG vs LLM divergence | Target: {target_type} | Model: {model_runtime.visible_label} ==="
    )
    results = {}
    cases_by_size = _load_generation_cases(
        grouped_by_size,
        encoding_model,
        cfg,
        model_runtime,
        target_type,
        include_rag=True,
        include_llm=True,
        generated_cases=generated_cases,
    )
    correct_key = "correct_ingredients" if target_type == "ingredients" else "correct_directions"
    for group in tqdm(grouped_by_size, desc="RAG vs LLM Tests"):
        size = group.size
        scores = []

        qa_pairs = []
        for case in tqdm(cases_by_size[int(size)], desc=f"Size {size}", leave=False):
            response_rag = _require_response(case, "response_rag")
            response_llm = _require_response(case, "response_llm")
            score = sentence_iou(response_rag, response_llm)
            scores.append(score)
            qa_pairs.append(
                {
                    "recipe_id": case.recipe_id,
                    "recipe_name": case.recipe_name,
                    "query": case.query,
                    correct_key: case.correct_chunks,
                    "response_rag": response_rag,
                    "response_llm": response_llm,
                }
            )

        results[size] = {
            "scores": scores,
            "pct_perfect": len([v for v in scores if v == 1.0]) / len(scores)
            if scores
            else 0.0,
            "pct_near": len([v for v in scores if v >= 0.95]) / len(scores)
            if scores
            else 0.0,
            "qa_pairs": qa_pairs,
        }
    return results
