"""Backward compatibility shim for src.test_retrieval -> src.analysis.test_retrieval."""
from src.analysis.test_retrieval import *
from src.analysis.test_retrieval import main

if __name__ == "__main__":
    main()
