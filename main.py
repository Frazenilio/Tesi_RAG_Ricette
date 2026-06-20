import argparse
import dataclasses
import json
import pickle
from collections import Counter
from datetime import datetime
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.config import load_config
from src.data import (
    ExperimentGroup,
    build_grouped_by_size,
    build_grouped_by_size_controlled,
    load_data,
)
from src.experiments import (
    prepare_generation_cases,
    run_test_llm_only,
    run_test_rag,
    run_test_rag_vs_llm,
)
from src.plot import plot_divergence_results, plot_results
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

    # Ensure experiment subdirectories exist first
    for experiment in cfg.experiments:
        (model_dir / experiment).mkdir(parents=True, exist_ok=True)

    retrieval_trace_file = None
    retrieval_trace_path = None
    if "rag" in cfg.experiments:
        retrieval_trace_file = f"retrieval_trace_{ts_slug}.json"
        retrieval_trace_path = model_dir / "rag" / retrieval_trace_file

    generated_cases = prepare_generation_cases(
        groups,
        encoding_model,
        cfg,
        model_runtime,
        include_rag="rag" in cfg.experiments or "rag_vs_llm" in cfg.experiments,
        include_llm="llm_only" in cfg.experiments or "rag_vs_llm" in cfg.experiments,
        trace_path=retrieval_trace_path,
    )

    # Get the unique recipe names across all generated cases
    recipe_names = sorted(list({case.recipe_name for cases in generated_cases.values() for case in cases}))

    methodology = {
        "rag_context_source": "oracle_dataset_chunks",
        "reference_chunks_source": "dataset_correct_indices",
        "single_generation_pass": True,
        "reuses_responses_across_selected_experiments": len(cfg.experiments) > 1,
    }

    # Decide whether to divide by recipe based on whether specific recipes are filtered in the config
    divide_by_recipe = bool(cfg.filter_recipes)

    for experiment in cfg.experiments:
        exp_dir = model_dir / experiment
        model_results[experiment] = {}

        if not divide_by_recipe:
            # LEGACY / BEFORE: run globally on all recipes combined
            plot_dir = exp_dir / f"plots_{ts_slug}"
            plot_dir.mkdir(parents=True, exist_ok=True)

            if experiment == "rag":
                exp_results = runners[experiment](
                    groups,
                    encoding_model,
                    cfg,
                    model_runtime,
                    plot_dir=plot_dir,
                    trace_path=retrieval_trace_path,
                    generated_cases=generated_cases,
                )
            else:
                exp_results = runners[experiment](
                    groups,
                    encoding_model,
                    cfg,
                    model_runtime,
                    plot_dir=plot_dir,
                    generated_cases=generated_cases,
                )

            model_results[experiment] = exp_results

        else:
            # OPTION A: Divide by recipe
            recipe_names = sorted(list({case.recipe_name for cases in generated_cases.values() for case in cases}))
            aggregated_exp_results = {}

            for recipe_name in recipe_names:
                # Create a safe directory name for the recipe
                recipe_name_safe = recipe_name
                for c in '<>:"/\\|?*':
                    recipe_name_safe = recipe_name_safe.replace(c, "_")
                recipe_dir = exp_dir / recipe_name_safe
                recipe_dir.mkdir(parents=True, exist_ok=True)

                # Filter generated cases for this recipe
                recipe_gen_cases = {}
                for size, cases in generated_cases.items():
                    filtered_cases = [c for c in cases if c.recipe_name == recipe_name]
                    if filtered_cases:
                        recipe_gen_cases[size] = filtered_cases

                if not recipe_gen_cases:
                    continue

                # Filter groups for this recipe
                recipe_groups = []
                for g in groups:
                    filtered_qs = [q for q in g.questions if q[0] == recipe_name]
                    if filtered_qs:
                        recipe_groups.append(
                            ExperimentGroup(
                                size=g.size,
                                chunks=g.chunks,
                                embeddings=g.embeddings,
                                questions=filtered_qs,
                            )
                        )

                if experiment == "rag":
                    recipe_results = runners[experiment](
                        recipe_groups,
                        encoding_model,
                        cfg,
                        model_runtime,
                        plot_dir=recipe_dir,
                        trace_path=None,
                        generated_cases=recipe_gen_cases,
                    )
                else:
                    recipe_results = runners[experiment](
                        recipe_groups,
                        encoding_model,
                        cfg,
                        model_runtime,
                        plot_dir=recipe_dir,
                        generated_cases=recipe_gen_cases,
                    )

                # Save recipe-specific results json
                recipe_payload = {
                    "model": model_runtime.results_label,
                    "backend": model_runtime.backend,
                    "experiment": experiment,
                    "timestamp": ts_slug,
                    "recipe_name": recipe_name,
                    "model_spec": build_public_model_spec(model_runtime.spec),
                    "config": build_public_config(cfg),
                    "generation_methodology": methodology,
                    "results": recipe_results,
                }
                recipe_out_path = recipe_dir / f"results_{ts_slug}.json"
                with open(recipe_out_path, "w") as f:
                    json.dump(recipe_payload, f, indent=2)

                # Aggregate for the global experiment-level output
                for size, metrics in recipe_results.items():
                    if size not in aggregated_exp_results:
                        aggregated_exp_results[size] = {}
                    
                    for key, val in metrics.items():
                        if isinstance(val, list):
                            if key not in aggregated_exp_results[size]:
                                aggregated_exp_results[size][key] = []
                            aggregated_exp_results[size][key].extend(val)

            # Recompute percentages for the aggregated results at experiment level
            for size, data in aggregated_exp_results.items():
                if "maxs" in data:
                    maxs = data["maxs"]
                    data["pct_perfect"] = len([v for v in maxs if v == 1.0]) / len(maxs) if maxs else 0.0
                    data["pct_near"] = len([v for v in maxs if v >= 0.95]) / len(maxs) if maxs else 0.0
                if "scores" in data:
                    scores = data["scores"]
                    data["pct_perfect"] = len([v for v in scores if v == 1.0]) / len(scores) if scores else 0.0
                    data["pct_near"] = len([v for v in scores if v >= 0.95]) / len(scores) if scores else 0.0

            # Generate overall/aggregated plots (as done in the original version)
            plot_dir = exp_dir / f"plots_{ts_slug}"
            plot_dir.mkdir(parents=True, exist_ok=True)
            for size, data in aggregated_exp_results.items():
                save_path = plot_dir / f"size_{size}.png"
                if experiment == "rag":
                    plot_results(
                        data["maxs"],
                        data["means"],
                        data["stddevs"],
                        title=f"Test 1 — RAG | varianti={size}",
                        save_path=save_path,
                    )
                elif experiment == "llm_only":
                    plot_results(
                        data["maxs"],
                        data["means"],
                        data["stddevs"],
                        title=f"Test 2 — LLM only | varianti={size}",
                        save_path=save_path,
                    )
                elif experiment == "rag_vs_llm":
                    plot_divergence_results(
                        data["scores"],
                        title=f"Test 3 — RAG vs LLM | varianti={size}",
                        save_path=save_path,
                    )

            model_results[experiment] = aggregated_exp_results

        # Save global experiment results file (representing all recipes aggregated)
        payload = {
            "model": model_runtime.results_label,
            "backend": model_runtime.backend,
            "experiment": experiment,
            "timestamp": ts_slug,
            "model_spec": build_public_model_spec(model_runtime.spec),
            "config": build_public_config(cfg),
            "generation_methodology": methodology,
            "results": aggregated_exp_results,
        }
        if experiment == "rag" and retrieval_trace_file is not None:
            payload["retrieval_trace_file"] = retrieval_trace_file
        exp_out_path = exp_dir / f"results_{ts_slug}.json"
        with open(exp_out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Experiment '{experiment}' aggregated results saved to {exp_out_path}")

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

    df = load_data(
        cfg.base_path, cfg.recipes_csv, cfg.ingredients_csv, cfg.filter_recipes
    )
    if cfg.controlled_dataset:
        groups = build_grouped_by_size_controlled(
            df, encoding_model, cfg.device, cfg.n_per_size
        )
    else:
        groups = build_grouped_by_size(df, encoding_model, cfg.device)

    timestamp = datetime.now().isoformat(timespec="seconds")
    ts_slug = timestamp.replace(":", "-")
    run_results_dir = RESULTS_DIR / ts_slug
    run_results_dir.mkdir(parents=True, exist_ok=True)

    if args.save_data:
        data_path = run_results_dir / "data.pkl"
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
            delete_after_run=cfg.delete_after_run,
        ) as model_runtime:
            model_dir = run_results_dir / model_runtime.slug
            model_dir.mkdir(parents=True, exist_ok=True)
            model_results = run_selected_experiments(
                groups, encoding_model, cfg, model_runtime, model_dir, ts_slug
            )
            results["models"][model_runtime.results_label] = {
                "backend": model_runtime.backend,
                "results": model_results,
            }

    out_path = run_results_dir / f"summary_{ts_slug}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nGlobal summary saved to {out_path}")

    del encoding_model


if __name__ == "__main__":
    main()
