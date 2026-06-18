import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def build_faiss_index(
    embedded_chunks: np.ndarray, nlist: int, nprobe: int
) -> faiss.Index:
    dim = embedded_chunks.shape[-1]
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    corpus_embeddings = embedded_chunks.astype(np.float32)
    index.train(corpus_embeddings)
    index.add(corpus_embeddings)
    index.nprobe = nprobe
    print(f"Total vectors in index: {index.ntotal}")
    return index


def retrieve(
    query: str,
    k: int,
    index: faiss.Index,
    global_chunks: list,
    correct_indices: list,
    encoding_model: SentenceTransformer,
    device: str,
) -> tuple:
    query_embedding = encoding_model.encode([query], device=device).astype(np.float32)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    k = int(k)
    _, raw_indices = index.search(query_embedding, k)
    indices = raw_indices[0]
    correct_chunks = [global_chunks[i] for i in indices if i in correct_indices]
    context = " ".join(global_chunks[i] for i in indices)
    return indices, correct_chunks, context


def retrieve_oracle(global_chunks: list, correct_indices: list) -> tuple:
    correct_chunks = [global_chunks[i] for i in correct_indices]
    context = " ".join(correct_chunks)
    return correct_chunks, context
