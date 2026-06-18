import ollama
from ollama import Client


class OllamaBackendError(RuntimeError):
    """Raised when the Ollama backend cannot be reached or used."""


def check_ollama_server(*, raise_on_failure: bool = True) -> bool:
    """Check if the Ollama server is running and reachable."""
    try:
        ollama.list()
        return True
    except Exception as e:
        if not raise_on_failure:
            return False
        raise OllamaBackendError(
            f"Ollama server is not running or unreachable. {e}"
        ) from e


def ensure_model_available(model_name: str) -> bool:
    """Check if the model is present locally, otherwise pull it.

    Returns True if the model was pulled (not previously present), False if it
    was already available locally. The caller can use this to decide whether to
    delete the model after use.
    """
    try:
        response = ollama.list()

        # Handle different response formats (object-oriented in newer versions, dict in older)
        if hasattr(response, "models"):
            local_models = response.models
        elif isinstance(response, dict):
            local_models = response.get("models", [])
        else:
            local_models = []

        model_names = []
        for m in local_models:
            if hasattr(m, "model"):
                model_names.append(m.model)
            elif isinstance(m, dict):
                # Check for 'model' first, then 'name' as fallback
                name = m.get("model") or m.get("name")
                if name:
                    model_names.append(name)

        # Check for exact match or model:latest match
        if model_name not in model_names and f"{model_name}:latest" not in model_names:
            print(f"Model '{model_name}' not found locally. Pulling from Ollama...")
            ollama.pull(model_name)
            print(f"Model '{model_name}' pulled successfully.")
            return True
        else:
            print(f"Model '{model_name}' found locally.")
            return False
    except Exception as e:
        raise OllamaBackendError(
            f"Error while checking/pulling model '{model_name}': {e}"
        ) from e


def unload_model(model_name: str):
    """Unload a model from Ollama memory by setting keep_alive=0."""
    try:
        Client().generate(model=model_name, keep_alive=0)
        print(f"Model '{model_name}' unloaded from memory.")
    except Exception as e:
        print(f"Warning: could not unload model '{model_name}': {e}")


def delete_model(model_name: str):
    """Delete a model from Ollama local storage."""
    try:
        ollama.delete(model_name)
        print(f"Model '{model_name}' deleted from local storage.")
    except Exception as e:
        print(f"Warning: could not delete model '{model_name}': {e}")
