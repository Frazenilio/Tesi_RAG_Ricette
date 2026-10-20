"""
External API Judge Providers
============================
Integrates cloud API providers (Groq, OpenRouter) under a polymorphic APICall contract.
"""

from src.judges.providers.base import APICall
from src.judges.providers.models import Model, GROQ_NAME_API, OR_NAME_API
from src.judges.providers.groq import GroqAPI
from src.judges.providers.openrouter import OpenRouterAPIRequest

__all__ = [
    "APICall",
    "Model",
    "GROQ_NAME_API",
    "OR_NAME_API",
    "GroqAPI",
    "OpenRouterAPIRequest",
]
