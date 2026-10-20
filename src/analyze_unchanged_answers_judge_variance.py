"""Backward compatibility shim for src.analyze_unchanged_answers_judge_variance -> src.analysis.analyze_judge_variance."""
from src.analysis.analyze_judge_variance import *
from src.analysis.analyze_judge_variance import main

if __name__ == "__main__":
    main()
