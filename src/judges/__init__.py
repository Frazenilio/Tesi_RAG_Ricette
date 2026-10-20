"""
LLM-as-a-Judge Evaluation Subpackage
====================================
Local judge testing, external API evaluation suites, multi-round refinement loops,
and offline batch scoring.
"""

from src.judges.local_judges import (
    normalize_model_name,
    parse_judge_output,
)
from src.judges.iterative_refinement import parse_corrector_output
from src.judges.providers.base import APICall
from src.judges.providers.models import Model
from src.judges.providers.groq import GroqAPI
from src.judges.providers.openrouter import OpenRouterAPIRequest

__all__ = [
    "normalize_model_name",
    "parse_judge_output",
    "parse_corrector_output",
    "APICall",
    "Model",
    "GroqAPI",
    "OpenRouterAPIRequest",
]
