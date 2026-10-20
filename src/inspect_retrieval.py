"""Backward compatibility shim for src.inspect_retrieval -> src.analysis.inspect_retrieval."""
from src.analysis.inspect_retrieval import *
from src.analysis.inspect_retrieval import main

if __name__ == "__main__":
    main()
