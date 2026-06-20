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
from src.data import load_data, RECIPE_NAME_COL, DIRECTION_COL
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
    df = load_data(cfg.base_path, cfg.recipes_csv, cfg.ingredients_csv, cfg.filter_recipes)
    
    if df.empty:
        print("Dataset is empty. Check your config and data files.")
        return

    # Extract all chunks from the entire dataset
    print("Extracting text chunks...")
    global_chunks = []
    queries = [] # list of (recipe_name, query, correct_chunk_indices)

    for recipe_name, recipe_df in df.groupby(RECIPE_NAME_COL):
        correct_chunk_indices = []
        for row in recipe_df.itertuples(index=False):
            start = len(global_chunks)
            correct_chunk_indices.append(start)
            chunks = [row.Ingredients] + nltk.sent_tokenize(getattr(row, DIRECTION_COL))
            global_chunks.extend(chunks)
        
        queries.append((
            recipe_name,
            f"What are the ingredients of {recipe_name}?",
            correct_chunk_indices
        ))

    print(f"Total unique recipes: {len(queries)}")
    print(f"Total chunks in database: {len(global_chunks)}")

    print(f"Loading encoding model '{cfg.encoding_model}' on {cfg.device}...")
    encoding_model = SentenceTransformer(cfg.encoding_model, device=cfg.device)

    print("Encoding database chunks (generating vector embeddings)...")
    embedded_chunks = encoding_model.encode(
        global_chunks,
        device=cfg.device,
        show_progress_bar=True,
        batch_size=64
    )
    
    # Ensure it's float32
    embedded_chunks = np.array(embedded_chunks).astype(np.float32)

    print("\nBuilding FAISS index...")
    # Adjust nlist if corpus is too small (FAISS requires nlist <= number of vectors)
    nlist = min(cfg.nlist, len(global_chunks))
    if nlist < 1:
        nlist = 1
    index = build_faiss_index(embedded_chunks, nlist=nlist, nprobe=cfg.nprobe)

    # Test retrieval for different k values
    k_values = [3, 5, 10, 15]
    print("\nEvaluating retrieval quality...")
    
    for k in k_values:
        precisions = []
        recalls = []
        ious = []
        
        for recipe_name, query, correct_indices in queries:
            # Perform FAISS retrieval
            retrieved_ids, _, _ = retrieve(
                query=query,
                k=k,
                index=index,
                global_chunks=global_chunks,
                correct_indices=correct_indices,
                encoding_model=encoding_model,
                device=cfg.device
            )
            
            p, r, iou = calculate_metrics(retrieved_ids, correct_indices)
            precisions.append(p)
            recalls.append(r)
            ious.append(iou)
            
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_iou = np.mean(ious)
        
        print(f"\n--- Metrics for k={k} ---")
        print(f"  Average Precision@{k}: {avg_precision:.2%}")
        print(f"  Average Recall@{k}:    {avg_recall:.2%}")
        print(f"  Average IoU@{k}:       {avg_iou:.2%}")

if __name__ == "__main__":
    main()
