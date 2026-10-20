"""
LLM-as-a-Judge Bias Testing Subpackage
======================================
Evaluation suites for position bias, verbosity/length bias, self-enhancement bias,
and compassion fade (AI vs Human attribution) bias.
"""

from src.bias.position_bias import run_position_bias
from src.bias.length_bias import run_length_bias
from src.bias.self_enchantment_bias import run_self_enchantment_bias
from src.bias.compassion_fade_bias import run_compassion_fade_bias
from src.bias.extract_retrievals import extract_base_retrievals

__all__ = [
    "run_position_bias",
    "run_length_bias",
    "run_self_enchantment_bias",
    "run_compassion_fade_bias",
    "extract_base_retrievals",
]
