"""Backward compatibility shim. Canonical location: src.judges.api_evaluation"""
import runpy
from src.judges.api_evaluation import *  # noqa: F401, F403

if __name__ == "__main__":
    runpy.run_module("src.judges.api_evaluation", run_name="__main__")
