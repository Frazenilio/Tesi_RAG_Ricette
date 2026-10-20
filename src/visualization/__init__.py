"""
Visualization suite: publication-ready plots for RAG evaluations, LLM judges, re-evaluations, human comparisons, and biases.
"""

from src.visualization.judge_plots import (
    plot_overall_task_comparison_split,
    plot_cross_scale_human_judges_comparison,
    plot_per_judge_model_breakdown_split,
    plot_generator_models_breakdown_split,
    plot_correlation_heatmaps_split,
    main as run_judge_plots,
)
from src.visualization.rag_plots import (
    plot_rag_correctness_split,
    plot_rag_distributions_split,
    plot_rag_success_rate_split,
    plot_rag_scatter_split,
    plot_rag_iou_violin_split,
    plot_rag_human_violin_split,
    main as run_rag_plots,
)
from src.visualization.reevaluation_plots import (
    plot_reevaluation_task_comparison,
    plot_reevaluation_model_comparison,
    plot_per_judge_reevaluation,
    print_summary_table as print_reevaluation_summary,
    main as run_reevaluation_plots,
)
from src.visualization.human_comparison_plots import (
    load_reevaluation_human_data,
    plot_correctness_split,
    plot_distributions_split,
    plot_human_correlation_split,
    plot_mae_gap,
    plot_high_human_degradation,
    print_summary_table as print_human_summary,
    main as run_human_comparison_plots,
)
from src.visualization.bias_plots import (
    plot_self_enchantment,
    plot_compassion_fade,
    plot_length_bias,
    plot_length_bias_by_task,
    plot_position_bias,
    plot_biases_summary_dashboard,
    main as run_bias_plots,
)
from src.visualization.score_distribution_plots import (
    plot_score_variation,
    plot_coherence_accuracy,
    plot_histograms,
    main as run_score_distribution_plots,
)

__all__ = [
    "plot_overall_task_comparison_split",
    "plot_cross_scale_human_judges_comparison",
    "plot_per_judge_model_breakdown_split",
    "plot_generator_models_breakdown_split",
    "plot_correlation_heatmaps_split",
    "run_judge_plots",
    "plot_rag_correctness_split",
    "plot_rag_distributions_split",
    "plot_rag_success_rate_split",
    "plot_rag_scatter_split",
    "plot_rag_iou_violin_split",
    "plot_rag_human_violin_split",
    "run_rag_plots",
    "plot_reevaluation_task_comparison",
    "plot_reevaluation_model_comparison",
    "plot_per_judge_reevaluation",
    "print_reevaluation_summary",
    "run_reevaluation_plots",
    "load_reevaluation_human_data",
    "plot_correctness_split",
    "plot_distributions_split",
    "plot_human_correlation_split",
    "plot_mae_gap",
    "plot_high_human_degradation",
    "print_human_summary",
    "run_human_comparison_plots",
    "plot_self_enchantment",
    "plot_compassion_fade",
    "plot_length_bias",
    "plot_length_bias_by_task",
    "plot_position_bias",
    "plot_biases_summary_dashboard",
    "run_bias_plots",
    "plot_score_variation",
    "plot_coherence_accuracy",
    "plot_histograms",
    "run_score_distribution_plots",
]
