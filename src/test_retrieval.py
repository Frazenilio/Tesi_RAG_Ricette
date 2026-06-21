import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from tqdm import tqdm

# Add project root to path if running directly
sys.path.append(str(Path(__file__).parent.parent))

from src.config import load_config
from src.data import load_data, RECIPE_NAME_COL, DIRECTION_COL, CODE_COL
from src.retrieval import build_faiss_index, retrieve

def calculate_metrics(retrieved_indices, correct_indices):
    retrieved_set = set(retrieved_indices)
    correct_set = set(correct_indices)
    
    intersection = retrieved_set.intersection(correct_set)
    union = retrieved_set.union(correct_set)
    
    precision = len(intersection) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(intersection) / len(correct_set) if correct_set else 0.0
    iou = len(intersection) / len(union) if union else 0.0
    
    return precision, recall, iou

def main():
    print("Loading config...")
    cfg = load_config()
    
    print("Loading data...")
    df = load_data(
        cfg.base_path,
        cfg.recipes_csv,
        cfg.ingredients_csv,
        cfg.directions_csv,
        cfg.filter_recipes
    )
    
    if df.empty:
        print("Dataset is empty. Check your config and data files.")
        return

    print(f"Loading encoding model '{cfg.encoding_model}' on {cfg.device}...")
    encoding_model = SentenceTransformer(cfg.encoding_model, device=cfg.device)

    k_values = [3, 5, 10, 15]

    for strategy in cfg.strategies:
        print(f"\n========================================================")
        print(f"EVALUATING STRATEGY: {strategy.upper()}")
        print(f"========================================================")

        # Build chunks and queries based on the strategy
        if strategy == "mixed":
            global_chunks = []
            queries = []  # list of (target_type, recipe_name, query, correct_chunk_indices)
            for recipe_name, recipe_df in df.groupby(RECIPE_NAME_COL):
                recipe_df = recipe_df.sort_values(CODE_COL)
                correct_ing_indices = []
                correct_dir_indices = []
                for row in recipe_df.itertuples(index=False):
                    start = len(global_chunks)
                    global_chunks.append(row.Ingredients)
                    correct_ing_indices.append(start)
                    global_chunks.append(row.Directions)
                    correct_dir_indices.append(start + 1)
                
                queries.append(("ingredients", recipe_name, f"What are the ingredients of {recipe_name}?", correct_ing_indices))
                queries.append(("directions", recipe_name, f"What are the directions of {recipe_name}?", correct_dir_indices))

            print(f"Total chunks in database: {len(global_chunks)}")
            print("Encoding database chunks...")
            embedded_chunks = encoding_model.encode(global_chunks, device=cfg.device, show_progress_bar=True, batch_size=64)
            embedded_chunks = np.array(embedded_chunks).astype(np.float32)

            nlist = min(cfg.nlist, len(global_chunks))
            index = build_faiss_index(embedded_chunks, nlist=max(1, nlist), nprobe=cfg.nprobe)

            for target_type in ["ingredients", "directions"]:
                print(f"\n--- Strategy: MIXED | Target: {target_type.upper()} ---")
                target_queries = [q for q in queries if q[0] == target_type]
                for k in k_values:
                    precisions, recalls, ious = [], [], []
                    for _, recipe_name, query, correct_indices in target_queries:
                        retrieved_ids, _, _ = retrieve(
                            query=query, k=k, index=index, global_chunks=global_chunks,
                            correct_indices=correct_indices, encoding_model=encoding_model, device=cfg.device
                        )
                        p, r, iou = calculate_metrics(retrieved_ids, correct_indices)
                        precisions.append(p)
                        recalls.append(r)
                        ious.append(iou)
                    print(f"k={k} | Prec: {np.mean(precisions):.2%} | Rec: {np.mean(recalls):.2%} | IoU: {np.mean(ious):.2%}")

        elif strategy == "combined":
            global_chunks = []
            queries = []
            for recipe_name, recipe_df in df.groupby(RECIPE_NAME_COL):
                recipe_df = recipe_df.sort_values(CODE_COL)
                correct_ing_indices = []
                correct_dir_indices = []
                for row in recipe_df.itertuples(index=False):
                    start = len(global_chunks)
                    combined_chunk = f"{row.Ingredients}\n{row.Directions}"
                    global_chunks.append(combined_chunk)
                    correct_ing_indices.append(start)
                    correct_dir_indices.append(start)
                
                queries.append(("ingredients", recipe_name, f"What are the ingredients of {recipe_name}?", correct_ing_indices))
                queries.append(("directions", recipe_name, f"What are the directions of {recipe_name}?", correct_dir_indices))

            print(f"Total chunks in database: {len(global_chunks)}")
            print("Encoding database chunks...")
            embedded_chunks = encoding_model.encode(global_chunks, device=cfg.device, show_progress_bar=True, batch_size=64)
            embedded_chunks = np.array(embedded_chunks).astype(np.float32)

            nlist = min(cfg.nlist, len(global_chunks))
            index = build_faiss_index(embedded_chunks, nlist=max(1, nlist), nprobe=cfg.nprobe)

            for target_type in ["ingredients", "directions"]:
                print(f"\n--- Strategy: COMBINED | Target: {target_type.upper()} ---")
                target_queries = [q for q in queries if q[0] == target_type]
                for k in k_values:
                    precisions, recalls, ious = [], [], []
                    for _, recipe_name, query, correct_indices in target_queries:
                        retrieved_ids, _, _ = retrieve(
                            query=query, k=k, index=index, global_chunks=global_chunks,
                            correct_indices=correct_indices, encoding_model=encoding_model, device=cfg.device
                        )
                        p, r, iou = calculate_metrics(retrieved_ids, correct_indices)
                        precisions.append(p)
                        recalls.append(r)
                        ious.append(iou)
                    print(f"k={k} | Prec: {np.mean(precisions):.2%} | Rec: {np.mean(recalls):.2%} | IoU: {np.mean(ious):.2%}")

        elif strategy == "separated":
            global_chunks_ingredients = []
            global_chunks_directions = []
            queries_ing = []
            queries_dir = []
            for recipe_name, recipe_df in df.groupby(RECIPE_NAME_COL):
                recipe_df = recipe_df.sort_values(CODE_COL)
                correct_ing_indices = []
                correct_dir_indices = []
                for row in recipe_df.itertuples(index=False):
                    start_ing = len(global_chunks_ingredients)
                    global_chunks_ingredients.append(row.Ingredients)
                    correct_ing_indices.append(start_ing)

                    start_dir = len(global_chunks_directions)
                    global_chunks_directions.append(row.Directions)
                    correct_dir_indices.append(start_dir)
                
                queries_ing.append(("ingredients", recipe_name, f"What are the ingredients of {recipe_name}?", correct_ing_indices))
                queries_dir.append(("directions", recipe_name, f"What are the directions of {recipe_name}?", correct_dir_indices))

            print(f"Total ingredients chunks: {len(global_chunks_ingredients)}")
            print(f"Total directions chunks: {len(global_chunks_directions)}")

            print("Encoding ingredients chunks...")
            embedded_ing = encoding_model.encode(global_chunks_ingredients, device=cfg.device, show_progress_bar=True, batch_size=64)
            embedded_ing = np.array(embedded_ing).astype(np.float32)

            print("Encoding directions chunks...")
            embedded_dir = encoding_model.encode(global_chunks_directions, device=cfg.device, show_progress_bar=True, batch_size=64)
            embedded_dir = np.array(embedded_dir).astype(np.float32)

            nlist_ing = min(cfg.nlist, len(global_chunks_ingredients))
            index_ing = build_faiss_index(embedded_ing, nlist=max(1, nlist_ing), nprobe=cfg.nprobe)

            nlist_dir = min(cfg.nlist, len(global_chunks_directions))
            index_dir = build_faiss_index(embedded_dir, nlist=max(1, nlist_dir), nprobe=cfg.nprobe)

            # Evaluate Ingredients
            print(f"\n--- Strategy: SEPARATED | Target: INGREDIENTS ---")
            for k in k_values:
                precisions, recalls, ious = [], [], []
                for _, recipe_name, query, correct_indices in queries_ing:
                    retrieved_ids, _, _ = retrieve(
                        query=query, k=k, index=index_ing, global_chunks=global_chunks_ingredients,
                        correct_indices=correct_indices, encoding_model=encoding_model, device=cfg.device
                    )
                    p, r, iou = calculate_metrics(retrieved_ids, correct_indices)
                    precisions.append(p)
                    recalls.append(r)
                    ious.append(iou)
                print(f"k={k} | Prec: {np.mean(precisions):.2%} | Rec: {np.mean(recalls):.2%} | IoU: {np.mean(ious):.2%}")

            # Evaluate Directions
            print(f"\n--- Strategy: SEPARATED | Target: DIRECTIONS ---")
            for k in k_values:
                precisions, recalls, ious = [], [], []
                for _, recipe_name, query, correct_indices in queries_dir:
                    retrieved_ids, _, _ = retrieve(
                        query=query, k=k, index=index_dir, global_chunks=global_chunks_directions,
                        correct_indices=correct_indices, encoding_model=encoding_model, device=cfg.device
                    )
                    p, r, iou = calculate_metrics(retrieved_ids, correct_indices)
                    precisions.append(p)
                    recalls.append(r)
                    ious.append(iou)
                print(f"k={k} | Prec: {np.mean(precisions):.2%} | Rec: {np.mean(recalls):.2%} | IoU: {np.mean(ious):.2%}")

if __name__ == "__main__":
    main()
