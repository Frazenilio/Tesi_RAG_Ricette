import time

from ollama import Client

from .config import ModelSpec
from .ollama_utils import delete_model, ensure_model_available, unload_model


class ModelBackendError(RuntimeError):
    """Raised when no configured inference backend can run a model."""


class ModelRuntime:
    def __init__(self, spec: ModelSpec, backend: str):
        self.spec = spec
        self.backend = backend

    @property
    def results_label(self) -> str:
        return self.spec.visible_label

    @property
    def execution_label(self) -> str:
        return self.spec.execution_label

    @property
    def visible_label(self) -> str:
        return self.spec.visible_label

    @property
    def slug(self) -> str:
        return (
            self.results_label.replace("/", "_")
            .replace(":", "_")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("[", "")
            .replace("]", "")
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def query(
        self,
        system_prompt: str,
        user_message: str,
        *,
        num_ctx: int,
        num_predict: int,
        think: bool,
        temperature: float,
        keep_alive: str,
    ) -> str:
        raise NotImplementedError

    def close(self) -> None:
        pass


class OllamaModelRuntime(ModelRuntime):
    def __init__(self, spec: ModelSpec, *, timeout: int, delete_after_run: bool = False):
        super().__init__(spec, backend="ollama")
        self.model_name = spec.ollama_model
        self.client = Client(timeout=timeout)
        self.delete_after_run = delete_after_run
        self.was_pulled = ensure_model_available(self.model_name)

    def query(
        self,
        system_prompt: str,
        user_message: str,
        *,
        num_ctx: int,
        num_predict: int,
        think: bool,
        temperature: float,
        keep_alive: str,
    ) -> str:
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            stream=False,
            think=think,
            options={
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                "temperature": temperature,
            },
            keep_alive=keep_alive,
        )
        if isinstance(response, dict):
            return response["message"]["content"]
        message = response.message
        if isinstance(message, dict):
            return message["content"]
        return message.content

    def close(self) -> None:
        unload_model(self.model_name)
        if self.delete_after_run and self.was_pulled:
            delete_model(self.model_name)


def create_model_runtime(
    spec: ModelSpec,
    *,
    device: str,
    timeout: int,
    ollama_available: bool,
    delete_after_run: bool = False,
) -> ModelRuntime:
    del device
    if not ollama_available:
        raise ModelBackendError(
            f"Unable to initialize '{spec.visible_label}'. Ollama server unavailable."
        )
    return OllamaModelRuntime(spec, timeout=timeout, delete_after_run=delete_after_run)


def query_llm(
    system_prompt: str,
    user_message: str,
    model_runtime: ModelRuntime,
    *,
    call_index: int | None = None,
    num_ctx: int = 4096,
    num_predict: int = 256,
    think: bool = False,
    temperature: float = 0.2,
    timeout: int = 120,
    retries: int = 3,
    keep_alive: str = "5m",
) -> str:
    """
    Query the selected backend with timeout and retry logic.
    """
    last_error = None
    for attempt in range(retries):
        try:
            return model_runtime.query(
                system_prompt,
                user_message,
                num_ctx=num_ctx,
                num_predict=num_predict,
                think=think,
                temperature=temperature,
                keep_alive=keep_alive,
            )
        except Exception as e:
            last_error = e
            call_label = (
                f"call {call_index}" if call_index is not None else "unknown call"
            )
            print(
                f"\n[Attempt {attempt + 1}/{retries}] Error querying backend "
                f"for model '{model_runtime.visible_label}' ({call_label}): {e}"
            )
            if attempt < retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
            continue

    print(
        f"Failed to get response for model '{model_runtime.visible_label}' "
        f"after {retries} attempts. Last error: {last_error}"
    )
    return f"ERROR: backend unresponsive. {last_error}"
