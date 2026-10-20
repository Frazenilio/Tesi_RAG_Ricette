"""Backward compatibility shim. Canonical location: src.judges.reevaluate_results"""
import runpy
from src.judges.reevaluate_results import *  # noqa: F401, F403

if __name__ == "__main__":
    runpy.run_module("src.judges.reevaluate_results", run_name="__main__")
