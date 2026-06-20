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
from .plot import plot_divergence_results, plot_results
from .prompts import SYSTEM_PROMPT_PLAIN, SYSTEM_PROMPT_RAG
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
    *,
    include_rag: bool,
    include_llm: bool,
    trace_path: Path | None = None,
) -> dict[int, list[GeneratedCase]]:
    if not include_rag and not include_llm:
        raise ValueError("At least one response path must be enabled.")

    cases_by_size: dict[int, list[GeneratedCase]] = {}
    call_counter = count(1)
    trace_entries = [] if trace_path is not None else None

    for size, global_chunks, embedded_chunks, questions in tqdm(
        grouped_by_size, desc="Generation Cases"
    ):
        # Build FAISS index for this group if we need to do RAG
        index = None
        if include_rag and len(global_chunks) > 0:
            nlist = min(cfg.nlist, len(global_chunks))
            if nlist < 1:
                nlist = 1
            embedded_chunks_f32 = np.array(embedded_chunks).astype(np.float32)
            index = build_faiss_index(embedded_chunks_f32, nlist=nlist, nprobe=cfg.nprobe)

        size_cases: list[GeneratedCase] = []
        for qa_index, (
            recipe_name,
            recipe_code_id,
            query_text,
            correct_indices,
        ) in enumerate(tqdm(questions, desc=f"Size {size}", leave=False)):
            correct_chunks, _ = retrieve_oracle(global_chunks, correct_indices)
            
            response_rag = None
            response_llm = None

            if include_rag:
                if index is not None:
                    # Use FAISS to retrieve the context, setting k = len(correct_indices)
                    retrieved_ids, retrieved_chunks, context = retrieve(
                        query=query_text,
                        k=len(correct_indices),
                        index=index,
                        global_chunks=global_chunks,
                        correct_indices=correct_indices,
                        encoding_model=encoding_model,
                        device=cfg.device
                    )
                    context_indices = [int(i) for i in retrieved_ids]
                    context_chunks = [global_chunks[i] for i in context_indices]
                else:
                    context_indices = []
                    context_chunks = []
                    context = ""

                response_rag = query_llm(
                    SYSTEM_PROMPT_RAG.format(context=context),
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
                    global_chunks[index_value] for index_value in context_indices
                ]
                context = ""

            if include_llm:
                response_llm = query_llm(
                    SYSTEM_PROMPT_PLAIN,
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
                        "context_source": "faiss_vector_index" if include_rag else "oracle_dataset_chunks",
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
                    "experiment": "rag",
                    "context_source": "faiss_vector_index",
                    "reference_chunks_source": "dataset_correct_indices",
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
    plot_dir: Path | None = None,
    trace_path: Path | None = None,
    generated_cases: dict[int, list[GeneratedCase]] | None = None,
) -> dict:
    """Test 1: Query WITH oracle-built context."""
    print(
        f"\n=== Test 1: RAG (with context) | Model: {model_runtime.visible_label} ==="
    )
    results = {}
    cases_by_size = _load_generation_cases(
        grouped_by_size,
        encoding_model,
        cfg,
        model_runtime,
        include_rag=True,
        include_llm=False,
        trace_path=trace_path,
        generated_cases=generated_cases,
    )
    for size, _global_chunks, _embedded_chunks, _questions in tqdm(
        grouped_by_size, desc="RAG Tests"
    ):
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
                    "correct_ingredients": case.correct_chunks,
                    "response": response,
                }
            )

        save_path = plot_dir / f"size_{size}.png" if plot_dir is not None else None
        plot_results(
            maxs,
            means,
            stddevs,
            title=f"Test 1 — RAG | varianti={size}",
            save_path=save_path,
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
    plot_dir: Path | None = None,
    generated_cases: dict[int, list[GeneratedCase]] | None = None,
) -> dict:
    """Test 2: Direct query WITHOUT context."""
    print(
        f"\n=== Test 2: LLM only (no context) | Model: {model_runtime.visible_label} ==="
    )
    results = {}
    cases_by_size = _load_generation_cases(
        grouped_by_size,
        encoding_model,
        cfg,
        model_runtime,
        include_rag=False,
        include_llm=True,
        generated_cases=generated_cases,
    )
    for size, _global_chunks, _embedded_chunks, _questions in tqdm(
        grouped_by_size, desc="LLM-only Tests"
    ):
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
                    "correct_ingredients": case.correct_chunks,
                    "response": response,
                }
            )

        save_path = plot_dir / f"size_{size}.png" if plot_dir is not None else None
        plot_results(
            maxs,
            means,
            stddevs,
            title=f"Test 2 — LLM only | varianti={size}",
            save_path=save_path,
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
    plot_dir: Path | None = None,
    generated_cases: dict[int, list[GeneratedCase]] | None = None,
) -> dict:
    """Test 3: Compare RAG response vs LLM-only response (divergence)."""
    print(
        f"\n=== Test 3: RAG vs LLM divergence | Model: {model_runtime.visible_label} ==="
    )
    results = {}
    cases_by_size = _load_generation_cases(
        grouped_by_size,
        encoding_model,
        cfg,
        model_runtime,
        include_rag=True,
        include_llm=True,
        generated_cases=generated_cases,
    )
    for size, _global_chunks, _embedded_chunks, _questions in tqdm(
        grouped_by_size, desc="RAG vs LLM Tests"
    ):
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
                    "correct_ingredients": case.correct_chunks,
                    "response_rag": response_rag,
                    "response_llm": response_llm,
                }
            )

        save_path = plot_dir / f"size_{size}.png" if plot_dir is not None else None
        plot_divergence_results(
            scores,
            title=f"Test 3 — RAG vs LLM | varianti={size}",
            save_path=save_path,
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
