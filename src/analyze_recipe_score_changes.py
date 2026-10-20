"""Backward compatibility shim for src.analyze_recipe_score_changes -> src.analysis.analyze_score_changes."""
from src.analysis.analyze_score_changes import *
from src.analysis.analyze_score_changes import main

if __name__ == "__main__":
    main()
