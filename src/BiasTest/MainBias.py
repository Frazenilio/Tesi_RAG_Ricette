"""Backward compatibility shim. Canonical location: src.bias.runner"""
import runpy
from src.bias.runner import *  # noqa: F401, F403

if __name__ == "__main__":
    runpy.run_module("src.bias.runner", run_name="__main__")
