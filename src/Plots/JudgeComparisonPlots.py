"""Backward compatibility shim for src.Plots.JudgeComparisonPlots -> src.visualization.judge_plots."""
from src.visualization.judge_plots import *
from src.visualization.judge_plots import main

if __name__ == "__main__":
    main()
