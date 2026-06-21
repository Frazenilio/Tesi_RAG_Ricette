import argparse
import dataclasses
import json
import pickle
import numpy as np
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
from src.plot import plot_divergence_results, plot_results, plot_strategy_comparison
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
    groups_by_strategy, encoding_model, cfg, model_runtime, model_dir, ts_slug
):
    runners = {
        "rag": run_test_rag,
        "llm_only": run_test_llm_only,
        "rag_vs_llm": run_test_rag_vs_llm,
    }
    model_results = {}

    # Initialize nested results structure
    for strategy in cfg.strategies:
        model_results[strategy] = {}
        for experiment in cfg.experiments:
            model_results[strategy][experiment] = {}

    divide_by_recipe = bool(cfg.filter_recipes)

    # Accumulate metrics for comparison plots at the end:
    # structure: target_type -> experiment -> strategy -> {size -> avg_iou}
    comparison_data = {
        "ingredients": {exp: {strat: {} for strat in cfg.strategies} for exp in cfg.experiments},
        "directions": {exp: {strat: {} for strat in cfg.strategies} for exp in cfg.experiments}
    }

    # Accumulate metrics per recipe for comparison:
    # structure: recipe_name -> target_type -> experiment -> strategy -> {size -> avg_iou}
    recipe_comparison_data = {}
    recipe_to_size = {}

    for strategy in cfg.strategies:
        groups = groups_by_strategy[strategy]

        for target_type in ["ingredients", "directions"]:
            print(f"\n==========================================")
            print(f"Running strategy: {strategy.upper()} | Target: {target_type.upper()}")
            print(f"==========================================")

            retrieval_trace_file = None
            retrieval_trace_path = None
            if "rag" in cfg.experiments:
                retrieval_trace_file = f"retrieval_trace_{strategy}_{target_type}_{ts_slug}.json"
                trace_dir = model_dir / "rag" / strategy / target_type
                trace_dir.mkdir(parents=True, exist_ok=True)
                retrieval_trace_path = trace_dir / retrieval_trace_file

            generated_cases = prepare_generation_cases(
                groups,
                encoding_model,
                cfg,
                model_runtime,
                target_type,
                include_rag="rag" in cfg.experiments or "rag_vs_llm" in cfg.experiments,
                include_llm="llm_only" in cfg.experiments or "rag_vs_llm" in cfg.experiments,
                trace_path=retrieval_trace_path,
            )

            methodology = {
                "rag_context_source": "faiss_vector_index",
                "database_strategy": strategy,
                "reference_chunks_source": f"dataset_correct_indices_{target_type}",
                "single_generation_pass": True,
                "reuses_responses_across_selected_experiments": len(cfg.experiments) > 1,
            }

            for experiment in cfg.experiments:
                exp_dir = model_dir / experiment

                if not divide_by_recipe:
                    target_dir = exp_dir / strategy / target_type
                    target_dir.mkdir(parents=True, exist_ok=True)

                    if experiment == "rag":
                        exp_results = runners[experiment](
                            groups,
                            encoding_model,
                            cfg,
                            model_runtime,
                            target_type,
                            plot_dir=target_dir,
                            trace_path=retrieval_trace_path,
                            generated_cases=generated_cases,
                        )
                    else:
                        exp_results = runners[experiment](
                            groups,
                            encoding_model,
                            cfg,
                            model_runtime,
                            target_type,
                            plot_dir=target_dir,
                            generated_cases=generated_cases,
                        )

                    # Save global strategy+target results
                    payload = {
                        "model": model_runtime.results_label,
                        "backend": model_runtime.backend,
                        "experiment": experiment,
                        "strategy": strategy,
                        "target_type": target_type,
                        "timestamp": ts_slug,
                        "model_spec": build_public_model_spec(model_runtime.spec),
                        "config": build_public_config(cfg),
                        "generation_methodology": methodology,
                        "results": exp_results,
                    }
                    if experiment == "rag" and retrieval_trace_file is not None:
                        payload["retrieval_trace_file"] = f"{strategy}/{target_type}/{retrieval_trace_file}"
                    exp_out_path = target_dir / f"results_{ts_slug}.json"
                    with open(exp_out_path, "w") as f:
                        json.dump(payload, f, indent=2)
                    print(f"Strategy '{strategy}' target '{target_type}' results for '{experiment}' saved to {exp_out_path}")

                    model_results[strategy][experiment][target_type] = exp_results

                    # Accumulate for global comparison plot
                    for size, size_data in exp_results.items():
                        if "maxs" in size_data and len(size_data["maxs"]) > 0:
                            avg_iou = np.mean(size_data["maxs"])
                        elif "scores" in size_data and len(size_data["scores"]) > 0:
                            avg_iou = np.mean(size_data["scores"])
                        else:
                            avg_iou = 0.0
                        comparison_data[target_type][experiment][strategy][size] = avg_iou

                else:
                    # Option A: Divide by recipe
                    recipe_names = sorted(list({case.recipe_name for cases in generated_cases.values() for case in cases}))
                    aggregated_exp_results = {}

                    for recipe_name in recipe_names:
                        recipe_name_safe = recipe_name
                        for c in '<>:"/\\|?*':
                            recipe_name_safe = recipe_name_safe.replace(c, "_")

                        recipe_target_dir = exp_dir / recipe_name_safe / strategy / target_type
                        recipe_target_dir.mkdir(parents=True, exist_ok=True)

                        recipe_gen_cases = {}
                        for size, cases in generated_cases.items():
                            filtered_cases = [c for c in cases if c.recipe_name == recipe_name]
                            if filtered_cases:
                                recipe_gen_cases[size] = filtered_cases

                        if not recipe_gen_cases:
                            continue

                        recipe_groups = []
                        for g in groups:
                            filtered_qs_ing = [q for q in g.questions_ingredients if q[0] == recipe_name]
                            filtered_qs_dir = [q for q in g.questions_directions if q[0] == recipe_name]
                            if filtered_qs_ing or filtered_qs_dir:
                                recipe_groups.append(
                                    ExperimentGroup(
                                        size=g.size,
                                        strategy=g.strategy,
                                        global_chunks=g.global_chunks,
                                        embeddings=g.embeddings,
                                        questions_ingredients=filtered_qs_ing,
                                        questions_directions=filtered_qs_dir,
                                        global_chunks_ingredients=g.global_chunks_ingredients,
                                        embeddings_ingredients=g.embeddings_ingredients,
                                        global_chunks_directions=g.global_chunks_directions,
                                        embeddings_directions=g.embeddings_directions,
                                    )
                                )

                        if experiment == "rag":
                            recipe_results = runners[experiment](
                                recipe_groups,
                                encoding_model,
                                cfg,
                                model_runtime,
                                target_type,
                                plot_dir=recipe_target_dir,
                                trace_path=None,
                                generated_cases=recipe_gen_cases,
                            )
                        else:
                            recipe_results = runners[experiment](
                                recipe_groups,
                                encoding_model,
                                cfg,
                                model_runtime,
                                target_type,
                                plot_dir=recipe_target_dir,
                                generated_cases=recipe_gen_cases,
                            )

                        # Save recipe target results
                        recipe_payload = {
                            "model": model_runtime.results_label,
                            "backend": model_runtime.backend,
                            "experiment": experiment,
                            "strategy": strategy,
                            "target_type": target_type,
                            "timestamp": ts_slug,
                            "recipe_name": recipe_name,
                            "model_spec": build_public_model_spec(model_runtime.spec),
                            "config": build_public_config(cfg),
                            "generation_methodology": methodology,
                            "results": recipe_results,
                        }
                        recipe_out_path = recipe_target_dir / f"results_{ts_slug}.json"
                        with open(recipe_out_path, "w") as f:
                            json.dump(recipe_payload, f, indent=2)

                        # Accumulate for recipe comparison graph
                        if recipe_name not in recipe_comparison_data:
                            recipe_comparison_data[recipe_name] = {
                                "ingredients": {e: {s: {} for s in cfg.strategies} for e in cfg.experiments},
                                "directions": {e: {s: {} for s in cfg.strategies} for e in cfg.experiments}
                            }
                        for size, size_res in recipe_results.items():
                            if "maxs" in size_res and len(size_res["maxs"]) > 0:
                                avg_iou = np.mean(size_res["maxs"])
                            elif "scores" in size_res and len(size_res["scores"]) > 0:
                                avg_iou = np.mean(size_res["scores"])
                            else:
                                avg_iou = 0.0
                            recipe_comparison_data[recipe_name][target_type][experiment][strategy][size] = avg_iou
                            recipe_to_size[recipe_name] = size
                            comparison_data[target_type][experiment][strategy][recipe_name] = avg_iou

                        # Aggregate for the global experiment-level output
                        for size, metrics in recipe_results.items():
                            if size not in aggregated_exp_results:
                                aggregated_exp_results[size] = {}
                            
                            for key, val in metrics.items():
                                if isinstance(val, list):
                                    if key not in aggregated_exp_results[size]:
                                        aggregated_exp_results[size][key] = []
                                    aggregated_exp_results[size][key].extend(val)

                    # Recompute percentages for aggregated experiment results
                    for size, data in aggregated_exp_results.items():
                        if "maxs" in data:
                            maxs = data["maxs"]
                            data["pct_perfect"] = len([v for v in maxs if v == 1.0]) / len(maxs) if maxs else 0.0
                            data["pct_near"] = len([v for v in maxs if v >= 0.95]) / len(maxs) if maxs else 0.0
                        if "scores" in data:
                            scores = data["scores"]
                            data["pct_perfect"] = len([v for v in scores if v == 1.0]) / len(scores) if scores else 0.0
                            data["pct_near"] = len([v for v in scores if v >= 0.95]) / len(scores) if scores else 0.0

                    # Save global target plots under the target folder
                    global_target_dir = exp_dir / strategy / target_type
                    global_target_dir.mkdir(parents=True, exist_ok=True)
                    plot_dir = global_target_dir / f"plots_{ts_slug}"
                    plot_dir.mkdir(parents=True, exist_ok=True)

                    for size, data in aggregated_exp_results.items():
                        save_path = plot_dir / f"size_{size}.png"
                        if experiment == "rag":
                            plot_results(
                                data["maxs"],
                                data["means"],
                                data["stddevs"],
                                title=f"Test 1 — RAG ({target_type}) | Strategy: {strategy} | size={size}",
                                save_path=save_path,
                            )
                        elif experiment == "llm_only":
                            plot_results(
                                data["maxs"],
                                data["means"],
                                data["stddevs"],
                                title=f"Test 2 — LLM only ({target_type}) | Strategy: {strategy} | size={size}",
                                save_path=save_path,
                            )
                        elif experiment == "rag_vs_llm":
                            plot_divergence_results(
                                data["scores"],
                                title=f"Test 3 — RAG vs LLM ({target_type}) | Strategy: {strategy} | size={size}",
                                save_path=save_path,
                            )

                    # Save global experiment results file for this strategy + target
                    payload = {
                        "model": model_runtime.results_label,
                        "backend": model_runtime.backend,
                        "experiment": experiment,
                        "strategy": strategy,
                        "target_type": target_type,
                        "timestamp": ts_slug,
                        "model_spec": build_public_model_spec(model_runtime.spec),
                        "config": build_public_config(cfg),
                        "generation_methodology": methodology,
                        "results": aggregated_exp_results,
                    }
                    if experiment == "rag" and retrieval_trace_file is not None:
                        payload["retrieval_trace_file"] = f"{strategy}/{target_type}/{retrieval_trace_file}"
                    exp_out_path = exp_dir / strategy / f"results_{ts_slug}.json"
                    with open(exp_out_path, "w") as f:
                        json.dump(payload, f, indent=2)
                    print(f"Experiment '{experiment}' aggregated results for '{target_type}' saved to {exp_out_path}")

                    model_results[strategy][experiment][target_type] = aggregated_exp_results

                    if not divide_by_recipe:
                        # Accumulate for global comparison plot
                        for size, size_data in aggregated_exp_results.items():
                            if "maxs" in size_data and len(size_data["maxs"]) > 0:
                                avg_iou = np.mean(size_data["maxs"])
                            elif "scores" in size_data and len(size_data["scores"]) > 0:
                                avg_iou = np.mean(size_data["scores"])
                            else:
                                avg_iou = 0.0
                            comparison_data[target_type][experiment][strategy][size] = avg_iou

    # Generate comparative plots at the end
    print("\nGenerating strategy comparison plots...")
    size_to_recipe_labels = {}
    for strategy in cfg.strategies:
        for group in groups_by_strategy[strategy]:
            recipe_names = sorted(list({q[0] for q in group.questions_ingredients}))
            if recipe_names:
                if len(recipe_names) > 3:
                    label = ", ".join(recipe_names[:3]) + "..."
                else:
                    label = ", ".join(recipe_names)
                size_to_recipe_labels[group.size] = label

    for target_type in ["ingredients", "directions"]:
        for experiment in cfg.experiments:
            # 1. Global comparison plot
            strat_metrics = comparison_data[target_type][experiment]
            # Ensure we actually have data for some strategies
            if any(strat_metrics.get(s, {}) for s in cfg.strategies):
                title = f"Database Strategy Comparison — {experiment.upper()} ({target_type})"
                save_path = model_dir / experiment / f"comparison_{target_type}.png"
                if divide_by_recipe:
                    plot_strategy_comparison(strat_metrics, title=title, save_path=save_path, recipe_to_size=recipe_to_size)
                else:
                    plot_strategy_comparison(strat_metrics, title=title, save_path=save_path, x_labels=size_to_recipe_labels)

            # 2. Recipe-specific comparison plots (if divided by recipe)
            if divide_by_recipe:
                for recipe_name, recipe_data in recipe_comparison_data.items():
                    recipe_name_safe = recipe_name
                    for c in '<>:"/\\|?*':
                        recipe_name_safe = recipe_name_safe.replace(c, "_")
                    
                    recipe_strat_metrics = recipe_data[target_type][experiment]
                    if any(recipe_strat_metrics.get(s, {}) for s in cfg.strategies):
                        title = f"Database Strategy Comparison ({recipe_name}) — {experiment.upper()} ({target_type})"
                        save_path = model_dir / experiment / recipe_name_safe / f"comparison_{target_type}.png"
                        
                        recipe_sizes = set()
                        for strat_data in recipe_strat_metrics.values():
                            recipe_sizes.update(strat_data.keys())
                        recipe_x_labels = {size: recipe_name for size in recipe_sizes}
                        
                        plot_strategy_comparison(recipe_strat_metrics, title=title, save_path=save_path, x_labels=recipe_x_labels)

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
        cfg.base_path,
        cfg.recipes_csv,
        cfg.ingredients_csv,
        cfg.directions_csv,
        cfg.filter_recipes,
    )
    groups_by_strategy = {}
    for strategy in cfg.strategies:
        if cfg.controlled_dataset:
            groups_by_strategy[strategy] = build_grouped_by_size_controlled(
                df, encoding_model, cfg.device, strategy, cfg.n_per_size
            )
        else:
            groups_by_strategy[strategy] = build_grouped_by_size(
                df, encoding_model, cfg.device, strategy
            )

    timestamp = datetime.now().isoformat(timespec="seconds")
    ts_slug = timestamp.replace(":", "-")
    run_results_dir = RESULTS_DIR / ts_slug
    run_results_dir.mkdir(parents=True, exist_ok=True)

    if args.save_data:
        data_path = run_results_dir / "data.pkl"
        with open(data_path, "wb") as f:
            pickle.dump(groups_by_strategy, f)
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
                groups_by_strategy, encoding_model, cfg, model_runtime, model_dir, ts_slug
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
