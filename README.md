# RAG Ricette

A RAG (Retrieval-Augmented Generation) pipeline that evaluates how well a local LLM answers *"What are the ingredients of \<recipe\>?"* with vs. without retrieved context.

## Requirements

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) (package manager)
- [Ollama](https://ollama.com) running locally with the target model pulled
- CUDA-capable GPU recommended (set `device: cpu` in config for CPU-only)

## Setup

### 1. Install uv

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your shell (or run `source $HOME/.local/bin/env`) so that `uv` is on your `PATH`.

### 2. Install project dependencies

```bash
# Creates a virtualenv in .venv/ and installs all dependencies (including PyTorch CUDA 12.1)
uv sync
```

To run any command inside the project environment without activating the venv manually:

```bash
uv run python main.py
```

Or activate the venv once for the session:

```bash
source .venv/bin/activate
python main.py
```

### 3. Install Ollama

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS — via Homebrew
brew install ollama

# Windows
# Download the installer from https://ollama.com/download/windows
```

Start the Ollama server (runs in the background on port 11434):

```bash
ollama serve
```

On macOS and Windows the desktop app starts the server automatically.

## Data

Place two CSV files in `data/`:

| File | Required columns |
|---|---|
| `dataset_ricette.csv` | `RecipeName`, `Text` |
| `cleaned_ingredients.csv` | `Ingredients` |

The rows in both files must be aligned (same order). See `data/` for the included sample files.

## Configuration

Edit `config.yaml` to change models, paths, or dataset settings:

```yaml
data:
  base_path: "data/"
  recipes_csv: "dataset_ricette.csv"
  ingredients_csv: "cleaned_ingredients.csv"

models:
  encoding: "BAAI/bge-large-en-v1.5"   # sentence-transformers model
  language: "models.yml"
  device: "cuda"   # or "cpu"

retrieval:
  nlist: 30   # kept for separate retrieval analyses
  nprobe: 6   # kept for separate retrieval analyses
```

## Usage

```bash
# Run all three experiments with default config
uv run python main.py

# Use a custom config
uv run python main.py -c path/to/my_config.yaml

# Also persist grouped data to results/data.pkl (speeds up re-runs if you reload it manually)
uv run python main.py --save-data

```

Results are organized in the `results/` directory with a subfolder for each model/backend pair, without `base` / `instruct` suffixes in the visible result name:
- `results/<model_name>/results_<timestamp>.json`: Metrics for that specific model run.
- `results/<model_name>/plots_<timestamp>/`: Generated plots for the run.
- `results/summary_<timestamp>.json`: A global summary of all models tested in a single run.

## Experiments

| Test | Description |
|---|---|
| `rag` | LLM answers with oracle context built directly from the gold chunks in the dataset |
| `llm_only` | LLM answers from parametric knowledge alone (no context), then gets compared against the same gold chunks |
| `rag_vs_llm` | Token IoU between the very same RAG and LLM-only responses generated for the other two tests |

The generation experiments follow the thesis methodology: retrieval is bypassed, the RAG context is built from **all and only** the correct dataset chunks, and the same per-query responses are reused across the three tests.

Evaluation metric: **Sentence IoU** — token overlap (lemmatized, stopwords removed) between model response and the gold ingredient chunks, or between the paired RAG/LLM responses in `rag_vs_llm`.

## Project Structure

```
main.py              # Experiment runner entry point
config.yaml          # Runtime configuration
code/
  config.py          # Config dataclass + YAML loader
  data.py            # CSV loading, chunking, embedding, ExperimentGroup
  experiments.py     # Three experiment runners
  generation.py      # Ollama LLM wrapper
  metrics.py         # Sentence IoU metric
  plot.py            # Matplotlib result plots
  prompts.py         # System prompt templates
  retrieval.py       # FAISS index builder + retriever
data/                # CSV datasets
results/             # Output JSON results (git-ignored)
_base/               # Original Colab notebooks and raw datasets
```
