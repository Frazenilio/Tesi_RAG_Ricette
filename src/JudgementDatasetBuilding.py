"""Backward compatibility shim. Canonical location: src.dataset.build_judgment_dataset"""
import runpy
from src.dataset.build_judgment_dataset import *  # noqa: F401, F403

if __name__ == "__main__":
    runpy.run_module("src.dataset.build_judgment_dataset", run_name="__main__")
