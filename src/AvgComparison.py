"""Backward compatibility shim for src.AvgComparison -> src.analysis.statistical_comparison."""
from src.analysis.statistical_comparison import *
from src.analysis.statistical_comparison import (
    human_difference,
    score_variation,
    scale_metrics_comparison,
    coherence_accuracy,
    human_reliability,
    aggregate_comparison,
    combined_reliability,
    singular_judge_comparison,
)

if __name__ == "__main__":
    human_difference()
    score_variation()
    scale_metrics_comparison()
    coherence_accuracy()
    human_reliability()
    aggregate_comparison()
    combined_reliability()
    singular_judge_comparison()
