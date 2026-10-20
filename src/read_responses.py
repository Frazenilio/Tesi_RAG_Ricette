"""Backward compatibility shim for src.read_responses -> src.analysis.read_responses."""
from src.analysis.read_responses import *
from src.analysis.read_responses import main

if __name__ == "__main__":
    main()
