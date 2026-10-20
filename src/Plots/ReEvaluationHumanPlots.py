"""Backward compatibility shim for src.Plots.ReEvaluationHumanPlots -> src.visualization.human_comparison_plots."""
from src.visualization.human_comparison_plots import *
from src.visualization.human_comparison_plots import main

if __name__ == "__main__":
    main()
