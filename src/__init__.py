"""
Tesi RAG Ricette: Retrieval-Augmented Generation & LLM-as-a-Judge Research Platform.

Package Structure:
------------------
- `src.core`: Core RAG runtime, FAISS indexing, LLM execution (Ollama), prompt templates, and evaluation metrics.
- `src.dataset`: Dataset construction, recipe formatting, human labeling sheets, and judgment extraction.
- `src.judges`: LLM-as-a-Judge evaluation framework, local judges, multi-scale API evaluation (Groq, OpenRouter), and iterative refinement.
- `src.bias`: Bias evaluation suite for LLM judges (position, length, self-enhancement, compassion fade).
- `src.analysis`: Statistical comparison (Krippendorff alpha, Fleiss kappa, Kendall tau), retrieval diagnostics, and variance analysis.
- `src.visualization`: Publication-ready visualization tools for RAG metrics, judge comparisons, re-evaluations, and bias plots.
"""

from src import core
from src import dataset
from src import judges
from src import bias
from src import analysis
from src import visualization

__all__ = [
    "core",
    "dataset",
    "judges",
    "bias",
    "analysis",
    "visualization",
]
