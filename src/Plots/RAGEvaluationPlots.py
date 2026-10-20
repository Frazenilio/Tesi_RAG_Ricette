"""Backward compatibility shim for src.Plots.RAGEvaluationPlots -> src.visualization.rag_plots."""
from src.visualization.rag_plots import *
from src.visualization.rag_plots import main

if __name__ == "__main__":
    main()
