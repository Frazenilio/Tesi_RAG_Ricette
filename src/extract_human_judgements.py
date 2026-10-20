"""Backward compatibility shim. Canonical location: src.dataset.extract_human_judgements"""
import runpy
from src.dataset.extract_human_judgements import *  # noqa: F401, F403

if __name__ == "__main__":
    runpy.run_module("src.dataset.extract_human_judgements", run_name="__main__")
