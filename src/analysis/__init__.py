"""
Analysis, statistical comparison, and diagnostic tools for RAG and LLM judges.
"""

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
from src.analysis.test_retrieval import calculate_metrics, main as run_test_retrieval
from src.analysis.inspect_retrieval import (
    build_db_and_metadata,
    run_queries,
    build_report,
    parse_retrieval_inspection,
    plot_retrieval_metrics,
    main as run_inspect_retrieval,
)
from src.analysis.analyze_score_changes import (
    find_json_files,
    parse_qa_entries,
    filter_top_recipes,
    print_detailed_report,
    export_markdown_report,
    main as run_analyze_score_changes,
)
from src.analysis.analyze_judge_variance import (
    analyze_unchanged_answers,
    print_report as print_variance_report,
    export_csv as export_variance_csv,
    main as run_analyze_judge_variance,
)
from src.analysis.read_responses import format_split_list, main as run_read_responses

__all__ = [
    "human_difference",
    "score_variation",
    "scale_metrics_comparison",
    "coherence_accuracy",
    "human_reliability",
    "aggregate_comparison",
    "combined_reliability",
    "singular_judge_comparison",
    "calculate_metrics",
    "run_test_retrieval",
    "build_db_and_metadata",
    "run_queries",
    "build_report",
    "parse_retrieval_inspection",
    "plot_retrieval_metrics",
    "run_inspect_retrieval",
    "find_json_files",
    "parse_qa_entries",
    "filter_top_recipes",
    "print_detailed_report",
    "export_markdown_report",
    "run_analyze_score_changes",
    "analyze_unchanged_answers",
    "print_variance_report",
    "export_variance_csv",
    "run_analyze_judge_variance",
    "format_split_list",
    "run_read_responses",
]
