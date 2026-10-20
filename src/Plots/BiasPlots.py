"""Backward compatibility shim for src.Plots.BiasPlots -> src.visualization.bias_plots."""
from src.visualization.bias_plots import *
from src.visualization.bias_plots import main

if __name__ == "__main__":
    main()
