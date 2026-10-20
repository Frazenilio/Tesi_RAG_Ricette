"""Backward compatibility shim for src.PlotScores -> src.visualization.score_distribution_plots."""
from src.visualization.score_distribution_plots import *
from src.visualization.score_distribution_plots import (
    main,
    plot_score_variation,
    plot_coherence_accuracy,
    plot_histograms,
)

if __name__ == "__main__":
    main()
    plot_score_variation()
    plot_coherence_accuracy()
    plot_histograms()
