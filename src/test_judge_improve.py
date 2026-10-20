"""Backward compatibility shim. Canonical location: src.judges.iterative_refinement"""
import runpy
from src.judges.iterative_refinement import *  # noqa: F401, F403

if __name__ == "__main__":
    runpy.run_module("src.judges.iterative_refinement", run_name="__main__")
