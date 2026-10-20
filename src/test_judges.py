"""Backward compatibility shim. Canonical location: src.judges.local_judges"""
import runpy
from src.judges.local_judges import *  # noqa: F401, F403

if __name__ == "__main__":
    runpy.run_module("src.judges.local_judges", run_name="__main__")
