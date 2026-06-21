from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
ALLOWED_EXPERIMENTS = ("rag", "llm_only", "rag_vs_llm")


@dataclass(frozen=True)
class ModelSpec:
    variant: str
    backend_order: tuple[str, ...]
    ollama_model: str

    @property
    def display_name(self) -> str:
        return self.ollama_model

    @property
    def visible_label(self) -> str:
        return self.display_name

    @property
    def execution_label(self) -> str:
        return f"{self.display_name} [{self.variant}]"


@dataclass
class Config:
    base_path: str
    recipes_csv: str
    ingredients_csv: str
    directions_csv: str
    encoding_model: str
    language_models: list[ModelSpec]
    device: str
    llm_num_ctx: int
    llm_num_predict: int
    llm_think: bool
    llm_temperature: float
    llm_timeout: int
    llm_retries: int
    llm_keep_alive: str
    nlist: int
    nprobe: int
    controlled_dataset: bool
    n_per_size: int
    experiments: list[str]
    strategies: list[str]
    delete_after_run: bool = False
    filter_recipes: list[str] = field(default_factory=list)
    language_model_types: dict[str, str] = field(default_factory=dict)


def _normalize_hf_model_id(model_id: str) -> str:
    return model_id.removeprefix("hf.co/")


def _build_model_spec(
    *,
    variant: str,
    ollama_model: str,
) -> ModelSpec:
    return ModelSpec(
        variant=variant,
        ollama_model=ollama_model,
        backend_order=("ollama",),
    )


def _build_inline_model_spec(model_name: str) -> ModelSpec:
    return _build_model_spec(
        variant="instruct",
        ollama_model=model_name,
    )


def _parse_model_entry(item: str | dict[str, object], *, variant: str) -> ModelSpec:
    if isinstance(item, str):
        return _build_model_spec(
            variant=variant,
            ollama_model=item,
        )

    if not isinstance(item, dict):
        raise TypeError(f"Unsupported model entry: {item!r}")

    raw_model = (
        item.get("ollama")
        or item.get("huggingface")
        or item.get("hf")
        or item.get("id")
    )
    if raw_model is None:
        raise ValueError(f"Missing model id in {variant} entry: {item!r}")

    return _build_model_spec(
        variant=str(item.get("variant", variant)),
        ollama_model=str(raw_model),
    )


def _parse_models_file(
    models_data: dict[str, Any],
) -> tuple[list[ModelSpec], dict[str, str]]:
    base_entries = models_data.get("base", [])
    instruct_entries = models_data.get("instruct", [])

    base_models = [_parse_model_entry(item, variant="base") for item in base_entries]
    instruct_models = [
        _parse_model_entry(item, variant="instruct") for item in instruct_entries
    ]
    all_models = base_models + instruct_models
    language_model_types = {
        _normalize_hf_model_id(model.display_name): model.variant
        for model in all_models
    }
    return all_models, language_model_types


def load_config(path: Path = _DEFAULT_CONFIG_PATH) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

    lang = raw["models"]["language"]
    if isinstance(lang, str) and lang.endswith((".yml", ".yaml")):
        models_path = Path(path).parent / lang
        with open(models_path) as mf:
            models_data = yaml.safe_load(mf)
        language_models, language_model_types = _parse_models_file(models_data)
    else:
        raw_models = [lang] if isinstance(lang, str) else lang
        language_models = [_build_inline_model_spec(str(model)) for model in raw_models]
        language_model_types = {}

    raw_experiments = raw.get("experiments", list(ALLOWED_EXPERIMENTS))
    experiments = (
        [raw_experiments] if isinstance(raw_experiments, str) else raw_experiments
    )
    invalid = [e for e in experiments if e not in ALLOWED_EXPERIMENTS]
    if invalid:
        allowed = ", ".join(ALLOWED_EXPERIMENTS)
        raise ValueError(
            f"Invalid experiment name(s): {', '.join(invalid)}. Allowed: {allowed}"
        )

    return Config(
        base_path=raw["data"]["base_path"],
        recipes_csv=raw["data"]["recipes_csv"],
        ingredients_csv=raw["data"]["ingredients_csv"],
        directions_csv=raw["data"]["directions_csv"],
        filter_recipes=raw["data"].get("filter_recipes", []),
        encoding_model=raw["models"]["encoding"],
        language_models=language_models,
        language_model_types=language_model_types,
        device=raw["models"]["device"],
        llm_num_ctx=raw["models"].get("llm_num_ctx", 4096),
        llm_num_predict=raw["models"].get("llm_num_predict", 256),
        llm_think=raw["models"].get("llm_think", False),
        llm_temperature=raw["models"].get("llm_temperature", 0.2),
        llm_timeout=raw["models"].get("llm_timeout", 120),
        llm_retries=raw["models"].get("llm_retries", 3),
        llm_keep_alive=raw["models"].get("llm_keep_alive", "5m"),
        delete_after_run=raw["models"].get("delete_after_run", False),
        nlist=raw["retrieval"]["nlist"],
        nprobe=raw["retrieval"]["nprobe"],
        controlled_dataset=raw["retrieval"].get("controlled_dataset", False),
        n_per_size=raw["retrieval"].get("n_per_size", 100),
        experiments=experiments,
        strategies=raw["retrieval"].get("strategies", ["mixed"]),
    )
