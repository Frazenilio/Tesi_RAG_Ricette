# Restoration Guide for Plots

To restore the plot functionality that was moved to `DISCARDED/plots/`, follow these exact steps:

## 1. Restore the `plot.py` File

Move `plot.py` back to its original location:
```bash
mv DISCARDED/plots/plot.py src/plot.py
```

## 2. Restore `src/experiments.py`

Add back the plot import and the `plot_dir` argument to each of the three runner functions.

### Imports
In `src/experiments.py`, after the `compute_iou_stats, sentence_iou` import, add:
```python
from .plot import plot_divergence_results, plot_results
```

### Function Signatures
In the `run_test_rag` function signature (around line 258), add back:
```python
    plot_dir: Path | None = None,
```
Do the same for `run_test_llm_only` (around line 324) and `run_test_rag_vs_llm` (around line 394).

### Function Bodies
Inside `run_test_rag`, after the `qa_pairs.append(...)` block and right before `results[size] = {`, add:
```python
        save_path = plot_dir / f"size_{size}.png" if plot_dir is not None else None
        plot_results(
            maxs,
            means,
            stddevs,
            title=f"Test 1 — RAG ({target_type}) | varianti={size}",
            save_path=save_path,
        )
```

Inside `run_test_llm_only`, right before `results[size] = {`, add:
```python
        save_path = plot_dir / f"size_{size}.png" if plot_dir is not None else None
        plot_results(
            maxs,
            means,
            stddevs,
            title=f"Test 2 — LLM only ({target_type}) | varianti={size}",
            save_path=save_path,
        )
```

Inside `run_test_rag_vs_llm`, right before `results[size] = {`, add:
```python
        save_path = plot_dir / f"size_{size}.png" if plot_dir is not None else None
        plot_divergence_results(
            scores,
            title=f"Test 3 — RAG vs LLM ({target_type}) | varianti={size}",
            save_path=save_path,
        )
```

## 3. Restore `main.py`

### Imports
Add the plot imports back around line 24:
```python
from src.plot import plot_divergence_results, plot_results, plot_strategy_comparison
```

### Setup Comparison Accumulators
In `run_selected_experiments`, after `divide_by_recipe = bool(cfg.filter_recipes)` (around line 79), add the accumulators:
```python
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
```

### Function Calls
Pass `plot_dir` inside the two `if not divide_by_recipe` branches (around line 125):
```python
        plot_dir=target_dir,
```
And inside the `if divide_by_recipe` branches (around line 225):
```python
        plot_dir=recipe_target_dir,
```

### Data Accumulation
In the `if not divide_by_recipe` block (around line 160), right after `model_results[strategy][experiment][target_type] = exp_results`, add:
```python
                    # Accumulate for global comparison plot
                    for size, size_data in exp_results.items():
                        if "maxs" in size_data and len(size_data["maxs"]) > 0:
                            avg_iou = np.mean(size_data["maxs"])
                        elif "scores" in size_data and len(size_data["scores"]) > 0:
                            avg_iou = np.mean(size_data["scores"])
                        else:
                            avg_iou = 0.0
                        comparison_data[target_type][experiment][strategy][size] = avg_iou
```

In the `else:` (divided by recipe) block (around line 250), right before `# Aggregate for the global experiment-level output`, add:
```python
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
```

And around line 320, inside the `if not divide_by_recipe:` block near the end of the strategy loop:
```python
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
```

### Global Plots Generation
After the `aggregated_exp_results` processing logic and before saving the payload `payload = { ... }` (around line 265), re-add the global target plots loop:
```python
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
```

Finally, before `return model_results` at the very end of `run_selected_experiments`, re-add the comparative plot generator:
```python
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
```
