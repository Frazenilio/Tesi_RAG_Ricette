import argparse
import dataclasses
import json
import pickle
from collections import Counter
from datetime import datetime
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.config import load_config
from src.data import build_grouped_by_size, build_grouped_by_size_controlled, load_data
from src.experiments import (
    prepare_generation_cases,
    run_test_llm_only,
    run_test_rag,
    run_test_rag_vs_llm,
)
from src.generation import create_model_runtime
from src.ollama_utils import check_ollama_server

RESULTS_DIR = Path("results")


def ensure_unique_result_labels(model_specs) -> None:
    duplicates = sorted(
        label
        for label, count in Counter(spec.visible_label for spec in model_specs).items()
        if count > 1
    )
    if duplicates:
        labels = ", ".join(duplicates)
        raise ValueError(
            "Removing variant labels would make result names collide for: "
            f"{labels}. Ensure each configured model has a unique visible name."
        )


def build_public_model_spec(model_spec) -> dict:
    return {
        "display_name": model_spec.visible_label,
        "backend_order": list(model_spec.backend_order),
        "ollama_model": model_spec.ollama_model,
    }


def build_public_config(cfg) -> dict:
    config = dataclasses.asdict(cfg)
    config["language_models"] = [
        build_public_model_spec(model_spec) for model_spec in cfg.language_models
    ]
    config.pop("language_model_types", None)
    return config


def run_selected_experiments(
    groups, encoding_model, cfg, model_runtime, model_dir, ts_slug
):
    runners = {
        "rag": run_test_rag,
        "llm_only": run_test_llm_only,
        "rag_vs_llm": run_test_rag_vs_llm,
    }
    model_results = {}
    plot_dirs = {}
    for experiment in cfg.experiments:
        exp_dir = model_dir / experiment
        exp_dir.mkdir(exist_ok=True)
        plot_dir = exp_dir / f"plots_{ts_slug}"
        plot_dir.mkdir(exist_ok=True)
        plot_dirs[experiment] = plot_dir

    retrieval_trace_file = None
    retrieval_trace_path = None
    if "rag" in cfg.experiments:
        retrieval_trace_file = f"retrieval_trace_{ts_slug}.json"
        retrieval_trace_path = model_dir / "rag" / retrieval_trace_file

    generated_cases = prepare_generation_cases(
        groups,
        cfg,
        model_runtime,
        include_rag="rag" in cfg.experiments or "rag_vs_llm" in cfg.experiments,
        include_llm="llm_only" in cfg.experiments or "rag_vs_llm" in cfg.experiments,
        trace_path=retrieval_trace_path,
    )

    methodology = {
        "rag_context_source": "oracle_dataset_chunks",
        "reference_chunks_source": "dataset_correct_indices",
        "single_generation_pass": True,
        "reuses_responses_across_selected_experiments": len(cfg.experiments) > 1,
    }

    for experiment in cfg.experiments:
        exp_dir = model_dir / experiment
        if experiment == "rag":
            model_results[experiment] = runners[experiment](
                groups,
                encoding_model,
                cfg,
                model_runtime,
                plot_dir=plot_dirs[experiment],
                trace_path=retrieval_trace_path,
                generated_cases=generated_cases,
            )
        else:
            model_results[experiment] = runners[experiment](
                groups,
                encoding_model,
                cfg,
                model_runtime,
                plot_dir=plot_dirs[experiment],
                generated_cases=generated_cases,
            )

        exp_results = model_results[experiment]
        payload = {
            "model": model_runtime.results_label,
            "backend": model_runtime.backend,
            "experiment": experiment,
            "timestamp": ts_slug,
            "model_spec": build_public_model_spec(model_runtime.spec),
            "config": build_public_config(cfg),
            "generation_methodology": methodology,
            "results": exp_results,
        }
        if experiment == "rag" and retrieval_trace_file is not None:
            payload["retrieval_trace_file"] = retrieval_trace_file
        exp_out_path = exp_dir / f"results_{ts_slug}.json"
        with open(exp_out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Experiment '{experiment}' results saved to {exp_out_path}")

    return model_results


def main():
    parser = argparse.ArgumentParser(description="RAG recipe experiment runner")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("config.yaml"),
        metavar="PATH",
        help="Path to a YAML config file (default: config.yaml at project root)",
    )
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Persist grouped_by_size to results/data.pkl for faster re-runs",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_unique_result_labels(cfg.language_models)

    print("Checking Ollama backend...")
    ollama_available = check_ollama_server()

    encoding_model = SentenceTransformer(cfg.encoding_model, device=cfg.device)

    df = load_data(cfg.base_path, cfg.recipes_csv, cfg.ingredients_csv)
    if cfg.controlled_dataset:
        groups = build_grouped_by_size_controlled(
            df, encoding_model, cfg.device, cfg.n_per_size
        )
    else:
        groups = build_grouped_by_size(df, encoding_model, cfg.device)

    timestamp = datetime.now().isoformat(timespec="seconds")
    ts_slug = timestamp.replace(":", "-")
    RESULTS_DIR.mkdir(exist_ok=True)

    if args.save_data:
        data_path = RESULTS_DIR / "data.pkl"
        with open(data_path, "wb") as f:
            pickle.dump(groups, f)
        print(f"Data saved to {data_path}")

    results = {
        "timestamp": timestamp,
        "config": build_public_config(cfg),
        "models": {},
    }

    for model_spec in cfg.language_models:
        print(f"\n--- Preparing model: {model_spec.visible_label} ---")
        with create_model_runtime(
            model_spec,
            device=cfg.device,
            timeout=cfg.llm_timeout,
            ollama_available=ollama_available,
        ) as model_runtime:
            model_dir = RESULTS_DIR / model_runtime.slug
            model_dir.mkdir(exist_ok=True)
            model_results = run_selected_experiments(
                groups, encoding_model, cfg, model_runtime, model_dir, ts_slug
            )
            results["models"][model_runtime.results_label] = {
                "backend": model_runtime.backend,
                "results": model_results,
            }

    out_path = RESULTS_DIR / f"summary_{ts_slug}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nGlobal summary saved to {out_path}")

    del encoding_model


if __name__ == "__main__":
    main()
