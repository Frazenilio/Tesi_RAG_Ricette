"""
Retrieval Inspection Script
============================
For each recipe and strategy, builds the vector DB, queries it, and reports
exactly which chunks FAISS returns — including the recipe name, chunk type
(ingredients / directions / combined), and the full chunk text.

All tuneable parameters are at the top of this file for easy access.
"""

import sys
import re
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

# Ensure project root is on sys.path when running directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.core.data import load_data, RECIPE_NAME_COL, CODE_COL
from src.core.retrieval import build_faiss_index

# ── Tuneable Parameters ─────────────────────────────────────────────────────
# Which strategies to inspect (must match config.yaml naming).
STRATEGIES = ["Singolo-Distinti", "Singolo-Aggregati", "Doppio"]

# Display names mapping for plots and thesis documentation
STRATEGY_DISPLAY_NAMES = {
    "Singolo-Distinti": "Single-Distinct",
    "Singolo-Aggregati": "Single-Aggregated",
    "Doppio": "Double",
}

# Extra chunks to retrieve beyond the number of variants for a recipe.
# Example: if a recipe has 6 variants and K_OFFSET = 2, then k = 8.
K_OFFSET = 5

# Which query types to test per recipe.
QUERY_TYPES = ["ingredients", "directions"]

# Where to save the markdown report (relative to the project root).
OUTPUT_PATH = PROJECT_ROOT / "results" / "retrieval_inspection.md"

# Maximum characters to show in the summary table per chunk.
CHUNK_PREVIEW_LEN = 80
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class ChunkMeta:
    """Metadata for a single chunk stored in a FAISS index."""
    recipe_name: str
    chunk_type: str        # "ingredients", "directions", or "combined"
    chunk_text: str


@dataclass
class RetrievedChunk:
    """One chunk returned by FAISS for a specific query."""
    faiss_id: int
    meta: ChunkMeta


@dataclass
class QueryResult:
    """Result of a single retrieval query."""
    recipe_name: str
    query_type: str        # "ingredients" or "directions"
    query_text: str
    k: int
    variant_count: int
    retrieved: list[RetrievedChunk]


def build_db_and_metadata(df, strategy, encoding_model, device, nlist, nprobe):
    """
    Build the chunk list(s), metadata registry, and FAISS index(es) for a
    given strategy.  Returns structures needed for querying.

    Returns
    -------
    For Singolo-Distinti / Singolo-Aggregati (single index):
        (index, chunk_meta_list, None, None)

    For Doppio (two indices):
        (index_ing, meta_ing, index_dir, meta_dir)
    """
    if strategy == "Doppio":
        chunks_ing: list[str] = []
        chunks_dir: list[str] = []
        meta_ing: list[ChunkMeta] = []
        meta_dir: list[ChunkMeta] = []

        for recipe_name, recipe_df in df.groupby(RECIPE_NAME_COL):
            recipe_df = recipe_df.sort_values(CODE_COL)
            for row in recipe_df.itertuples(index=False):
                chunks_ing.append(row.Ingredients)
                meta_ing.append(ChunkMeta(recipe_name, "ingredients", row.Ingredients))

                chunks_dir.append(row.Directions)
                meta_dir.append(ChunkMeta(recipe_name, "directions", row.Directions))

        # Encode
        print(f"  [{strategy}] Encoding {len(chunks_ing)} ingredients chunks...")
        emb_ing = encoding_model.encode(chunks_ing, device=device, show_progress_bar=True, batch_size=64)
        emb_ing = np.array(emb_ing).astype(np.float32)

        print(f"  [{strategy}] Encoding {len(chunks_dir)} directions chunks...")
        emb_dir = encoding_model.encode(chunks_dir, device=device, show_progress_bar=True, batch_size=64)
        emb_dir = np.array(emb_dir).astype(np.float32)

        nl_ing = max(1, min(nlist, len(chunks_ing)))
        index_ing = build_faiss_index(emb_ing, nlist=nl_ing, nprobe=nprobe)

        nl_dir = max(1, min(nlist, len(chunks_dir)))
        index_dir = build_faiss_index(emb_dir, nlist=nl_dir, nprobe=nprobe)

        return index_ing, meta_ing, index_dir, meta_dir

    # ── Singolo-Distinti / Singolo-Aggregati ──
    chunks: list[str] = []
    meta: list[ChunkMeta] = []

    for recipe_name, recipe_df in df.groupby(RECIPE_NAME_COL):
        recipe_df = recipe_df.sort_values(CODE_COL)
        for row in recipe_df.itertuples(index=False):
            if strategy == "Singolo-Distinti":
                chunks.append(row.Ingredients)
                meta.append(ChunkMeta(recipe_name, "ingredients", row.Ingredients))
                chunks.append(row.Directions)
                meta.append(ChunkMeta(recipe_name, "directions", row.Directions))
            elif strategy == "Singolo-Aggregati":
                combined = f"{row.Ingredients}\n{row.Directions}"
                chunks.append(combined)
                meta.append(ChunkMeta(recipe_name, "combined", combined))

    print(f"  [{strategy}] Encoding {len(chunks)} chunks...")
    emb = encoding_model.encode(chunks, device=device, show_progress_bar=True, batch_size=64)
    emb = np.array(emb).astype(np.float32)

    nl = max(1, min(nlist, len(chunks)))
    index = build_faiss_index(emb, nlist=nl, nprobe=nprobe)

    return index, meta, None, None


def run_queries(df, strategy, encoding_model, device, index, meta,
                index_dir, meta_dir):
    """
    For each recipe in `df`, run the configured query types and collect
    retrieved chunks with their metadata.
    """
    variant_counts = df.groupby(RECIPE_NAME_COL).size()
    results: list[QueryResult] = []

    for recipe_name in sorted(variant_counts.index):
        n_variants = int(variant_counts[recipe_name])
        k = n_variants + K_OFFSET

        for qtype in QUERY_TYPES:
            query_text = (
                f"What are the ingredients of {recipe_name}?"
                if qtype == "ingredients"
                else f"What are the directions of {recipe_name}?"
            )

            # Choose which index + meta to search
            if strategy == "Doppio":
                cur_index = index if qtype == "ingredients" else index_dir
                cur_meta = meta if qtype == "ingredients" else meta_dir
            else:
                cur_index = index
                cur_meta = meta

            # Encode query and search
            q_emb = encoding_model.encode([query_text], device=device).astype(np.float32)
            if q_emb.ndim == 1:
                q_emb = q_emb.reshape(1, -1)

            actual_k = min(k, cur_index.ntotal)
            _, raw_ids = cur_index.search(q_emb, actual_k)
            ids = raw_ids[0]

            retrieved = []
            for fid in ids:
                fid = int(fid)
                if 0 <= fid < len(cur_meta):
                    retrieved.append(RetrievedChunk(faiss_id=fid, meta=cur_meta[fid]))

            results.append(QueryResult(
                recipe_name=recipe_name,
                query_type=qtype,
                query_text=query_text,
                k=actual_k,
                variant_count=n_variants,
                retrieved=retrieved,
            ))

    return results


def _truncate(text: str, max_len: int) -> str:
    """Truncate text for table display."""
    clean = text.replace("\n", " ").strip()
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3] + "..."


def build_report(all_results: dict[str, list[QueryResult]]) -> str:
    """Build the full markdown report from all strategy results."""
    lines = [
        "# Retrieval Inspection Report",
        "",
        f"**K_OFFSET**: {K_OFFSET}  ",
        f"**Query Types**: {', '.join(QUERY_TYPES)}  ",
        f"**Strategies**: {', '.join(all_results.keys())}",
        "",
    ]

    for strategy, results in all_results.items():
        lines.append(f"## Strategy: {strategy}")
        lines.append("")

        for qr in results:
            lines.append(f"### {qr.recipe_name} — {qr.query_type}")
            lines.append(f"**Query**: `{qr.query_text}`  ")
            lines.append(f"**k** = {qr.k} ({qr.variant_count} variants + {K_OFFSET} offset)")
            lines.append("")

            # Summary table
            lines.append("| # | FAISS ID | Recipe | Type | Chunk (preview) |")
            lines.append("|---|----------|--------|------|-----------------|")
            for i, rc in enumerate(qr.retrieved, 1):
                preview = _truncate(rc.meta.chunk_text, CHUNK_PREVIEW_LEN)
                # Escape pipes in preview text
                preview = preview.replace("|", "\\|")
                lines.append(
                    f"| {i} | {rc.faiss_id} | {rc.meta.recipe_name} | "
                    f"{rc.meta.chunk_type} | {preview} |"
                )
            lines.append("")

            # Full text (collapsible)
            lines.append("<details>")
            lines.append("<summary>Full chunk texts (click to expand)</summary>")
            lines.append("")
            for i, rc in enumerate(qr.retrieved, 1):
                lines.append(f"**Chunk {i}** (FAISS ID {rc.faiss_id}, "
                             f"{rc.meta.recipe_name}, {rc.meta.chunk_type}):")
                lines.append(f"> {rc.meta.chunk_text}")
                lines.append("")
            lines.append("</details>")
            lines.append("")
            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def main():
    print("Loading config...")
    cfg = load_config()

    print("Loading data...")
    df = load_data(
        cfg.base_path,
        cfg.recipes_csv,
        cfg.ingredients_csv,
        cfg.directions_csv,
        cfg.filter_recipes,
    )
    if df.empty:
        print("Dataset is empty. Check your config and data files.")
        return

    recipe_names = sorted(df[RECIPE_NAME_COL].unique())
    variant_counts = df.groupby(RECIPE_NAME_COL).size()
    print(f"\nRecipes to inspect ({len(recipe_names)}):")
    for rn in recipe_names:
        print(f"  - {rn} ({variant_counts[rn]} variants)")

    print(f"\nLoading encoding model '{cfg.encoding_model}' on {cfg.device}...")
    encoding_model = SentenceTransformer(cfg.encoding_model, device=cfg.device)

    all_results: dict[str, list[QueryResult]] = {}

    for strategy in STRATEGIES:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy}")
        print(f"{'='*60}")

        index, meta, index_dir, meta_dir = build_db_and_metadata(
            df, strategy, encoding_model, cfg.device, cfg.nlist, cfg.nprobe
        )

        results = run_queries(
            df, strategy, encoding_model, cfg.device,
            index, meta, index_dir, meta_dir,
        )
        all_results[strategy] = results

        # Quick console summary
        for qr in results:
            own_count = sum(
                1 for rc in qr.retrieved if rc.meta.recipe_name == qr.recipe_name
            )
            print(
                f"  {qr.recipe_name} [{qr.query_type}] k={qr.k}: "
                f"{own_count}/{len(qr.retrieved)} chunks from correct recipe"
            )

    # Write report
    report = build_report(all_results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {OUTPUT_PATH.absolute()}")
    
    # ── Plotting for Documentation ──
    plot_data = []
    for strategy, res_list in all_results.items():
        for qr in res_list:
            own_count = sum(1 for rc in qr.retrieved if rc.meta.recipe_name == qr.recipe_name)
            # Recall: fraction of correct variants successfully retrieved
            recall = min(own_count / qr.variant_count, 1.0)
            plot_data.append({
                "Strategy": strategy,
                "Query Type": qr.query_type.capitalize(),
                "Recall": recall * 100  # Convert to percentage
            })
            
    plot_retrieval_metrics(plot_data=plot_data)


def parse_retrieval_inspection(report_path: Path):
    """Parses a pre-generated retrieval_inspection.md to extract plot data."""
    if not report_path.exists():
        return []

    text = report_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    current_strategy = None
    current_recipe = None
    current_qtype = None
    current_variants = 0
    own_count = 0
    in_table = False

    plot_data = []

    for line in lines:
        strat_m = re.match(r"^## Strategy:\s*(.+)$", line)
        if strat_m:
            current_strategy = strat_m.group(1).strip()
            continue

        rec_m = re.match(r"^### (.+?)\s*—\s*(ingredients|directions)$", line)
        if rec_m:
            current_recipe = rec_m.group(1).strip()
            current_qtype = rec_m.group(2).strip()
            continue

        k_m = re.search(r"\*\*k\*\*\s*=\s*\d+\s*\((\d+)\s*variants", line)
        if k_m:
            current_variants = int(k_m.group(1))
            continue

        if line.startswith("| # |"):
            in_table = True
            own_count = 0
            continue

        if in_table:
            if line.startswith("|---|"):
                continue
            row_m = re.match(r"^\|\s*\d+\s*\|\s*\d+\s*\|\s*(.+?)\s*\|\s*(\w+)\s*\|", line)
            if row_m:
                ret_rec = row_m.group(1).strip()
                if ret_rec == current_recipe:
                    own_count += 1
            else:
                in_table = False
                recall = min(own_count / current_variants, 1.0) if current_variants > 0 else 0.0
                plot_data.append({
                    "Strategy": current_strategy,
                    "Query Type": current_qtype.capitalize(),
                    "Recall": recall * 100,
                })

    return plot_data


def plot_retrieval_metrics(plot_data=None, output_path=None):
    """Generates the FAISS retrieval accuracy plot using the mapped display names."""
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        if output_path is None:
            output_path = PROJECT_ROOT / "results" / "retrieval_metrics.png"

        if plot_data is None:
            plot_data = parse_retrieval_inspection(OUTPUT_PATH)

        if not plot_data:
            print("No plot data available to generate retrieval metrics plot.")
            return

        df_plot = pd.DataFrame(plot_data)
        # Apply display name mapping to match thesis terminology
        df_plot["Strategy"] = df_plot["Strategy"].map(STRATEGY_DISPLAY_NAMES).fillna(df_plot["Strategy"])
        strategy_order = [STRATEGY_DISPLAY_NAMES.get(s, s) for s in STRATEGIES]

        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")

        # Barplot showing average recall per strategy and query type
        sns.barplot(
            x="Strategy",
            y="Recall",
            hue="Query Type",
            data=df_plot,
            order=strategy_order,
            palette="muted",
            errorbar=None,
        )

        plt.title("FAISS Retrieval Accuracy (Average Recall)", fontsize=14, pad=15)
        plt.ylabel("Average Recall (%)", fontsize=12)
        plt.xlabel("Indexing Strategy", fontsize=12)
        plt.ylim(0, 105)
        plt.legend(title="Query Type")
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Plot saved to: {output_path.absolute()}")

    except ImportError:
        print("Pandas, Matplotlib, or Seaborn not installed. Skipping plot generation.")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot_retrieval_metrics()
    else:
        main()
