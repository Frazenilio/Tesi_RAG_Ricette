"""Backward compatibility shim for src.Plots.ReEvaluationPlots -> src.visualization.reevaluation_plots."""
from src.visualization.reevaluation_plots import *
from src.visualization.reevaluation_plots import main

if __name__ == "__main__":
    main()
