"""Backward compatibility shim. Canonical location: src.dataset.build_datasets"""
import runpy
from src.dataset.build_datasets import *  # noqa: F401, F403

if __name__ == "__main__":
    runpy.run_module("src.dataset.build_datasets", run_name="__main__")
