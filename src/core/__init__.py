"""
Core RAG Pipeline Engine
========================
Provides core data structures, retrieval algorithms, local LLM generation runtimes,
and experiment evaluation loops for the Recipe RAG pipeline.
"""

from src.core.config import Config, ModelSpec, load_config
from src.core.data import (
    ExperimentGroup,
    load_data,
    build_grouped_by_size,
    build_grouped_by_size_controlled,
    RECIPE_NAME_COL,
    DIRECTION_COL,
    CODE_COL,
)
from src.core.retrieval import build_faiss_index, retrieve, retrieve_oracle
from src.core.generation import (
    ModelBackendError,
    ModelRuntime,
    create_model_runtime,
    query_llm,
)
from src.core.ollama_utils import (
    check_ollama_server,
    ensure_model_available,
    delete_model,
)
from src.core.metrics import sentence_iou, compute_iou_stats

__all__ = [
    "Config",
    "ModelSpec",
    "load_config",
    "ExperimentGroup",
    "load_data",
    "build_grouped_by_size",
    "build_grouped_by_size_controlled",
    "RECIPE_NAME_COL",
    "DIRECTION_COL",
    "CODE_COL",
    "build_faiss_index",
    "retrieve",
    "retrieve_oracle",
    "ModelBackendError",
    "ModelRuntime",
    "create_model_runtime",
    "query_llm",
    "check_ollama_server",
    "ensure_model_available",
    "delete_model",
    "sentence_iou",
    "compute_iou_stats",
]
