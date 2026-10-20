"""Backward compatibility shim. Canonical location: src.dataset.create_labeling_sheet"""
import runpy
from src.dataset.create_labeling_sheet import *  # noqa: F401, F403

if __name__ == "__main__":
    runpy.run_module("src.dataset.create_labeling_sheet", run_name="__main__")
