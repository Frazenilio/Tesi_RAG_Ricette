from pathlib import Path
from typing import NamedTuple

import nltk
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# RECIPE_NAME_COL = "RecipeName"
# DIRECTION_COL = "Text"

## NEW: Replaced with the new dataset
RECIPE_NAME_COL = "title"
DIRECTION_COL = "directions"
CODE_COL = "original_row"


class ExperimentGroup(NamedTuple):
    """All data for one group of recipes sharing the same number of variants."""

    size: int
    chunks: list[str]
    embeddings: np.ndarray
    questions: list[tuple[str, str, str, list[int]]]


def _normalize_ingredients_text(ingredients: object) -> str | None:
    if not isinstance(ingredients, str):
        return None
    normalized = ingredients.strip().strip('"').strip("'").strip()
    return normalized or None


def _has_substantive_ingredients_payload(recipe_name: str, ingredients: str) -> bool:
    prefix = f"The ingredients of {recipe_name} are:"
    if not ingredients.startswith(prefix):
        return False
    payload = ingredients[len(prefix) :].strip()
    return any(char.isalnum() for char in payload)


def clean_recipe_variants(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.copy()
    cleaned_df["Ingredients"] = [
        normalized
        if normalized is not None
        and _has_substantive_ingredients_payload(recipe_name, normalized)
        else None
        for recipe_name, ingredients in zip(
            cleaned_df[RECIPE_NAME_COL], cleaned_df["Ingredients"], strict=True
        )
        for normalized in [_normalize_ingredients_text(ingredients)]
    ]
    return cleaned_df[cleaned_df["Ingredients"].notna()].reset_index(drop=True)


def load_data(
    base_path: str,
    recipes_csv: str,
    ingredients_csv: str,
    filter_recipes: list[str] = None,
) -> pd.DataFrame:
    root = Path(base_path)
    df = pd.read_csv(root / recipes_csv)
    df_ingred = pd.read_csv(root / ingredients_csv)
    df["Ingredients"] = df_ingred["Ingredients"]

    # Filter by specific recipe names if requested
    if filter_recipes:
        df = df[df[RECIPE_NAME_COL].isin(filter_recipes)]

    return clean_recipe_variants(df)


def build_grouped_by_size_controlled(
    df: pd.DataFrame,
    encoding_model: SentenceTransformer,
    device: str,
    n_per_size: int = 100,
) -> list[ExperimentGroup]:
    """Build experiment groups with exactly n_per_size questions per cardinalità.

    Takes the first n_per_size size-6 recipes (sorted alphabetically) as a
    base. For each cardinalità k in [6, 5, 4, 3], a group is built by keeping
    only the first k variants (sorted by Code) of each base recipe. This
    mirrors the controlled dataset used in the original thesis experiments
    (Table 4.4: 1800 = 100 × (6+5+4+3)).
    """
    group_counts = df.groupby(RECIPE_NAME_COL).size()
    size_6_names = sorted(group_counts[group_counts == 6].index.tolist())
    if len(size_6_names) < n_per_size:
        raise ValueError(
            f"Requested n_per_size={n_per_size} but only "
            f"{len(size_6_names)} size-6 recipes available."
        )
    base_recipes = size_6_names[:n_per_size]

    grouped_by_size: list[ExperimentGroup] = []
    print(f"\nBuilding controlled experiment groups (n_per_size={n_per_size})...")
    for size in tqdm([6, 5, 4, 3], desc="Sizes"):
        questions: list[tuple[str, str, str, list[int]]] = []
        global_chunks: list[str] = []

        for recipe_name in base_recipes:
            recipe_df = (
                df[df[RECIPE_NAME_COL] == recipe_name].sort_values(CODE_COL).head(size)
            )
            code_id = "-".join(str(c) for c in recipe_df[CODE_COL].tolist())
            correct_chunk_indices: list[int] = []
            for row in recipe_df.itertuples(index=False):
                start = len(global_chunks)
                correct_chunk_indices.append(start)
                ## NOTE Here to change the chunks, what they contain
                # chunks = [row.Ingredients] + nltk.sent_tokenize(getattr(row, DIRECTION_COL))
                chunk_ing = row.Ingredients
                chunk_dir = getattr(row, DIRECTION_COL)
                global_chunks.append(chunk_ing)
                global_chunks.append(chunk_dir)

            questions.append(
                (
                    recipe_name,
                    code_id,
                    f"What are the ingredients of {recipe_name}?",
                    correct_chunk_indices,
                )
            )

        embeddings = encoding_model.encode(
            global_chunks,
            device=device,
            show_progress_bar=True,
            batch_size=64,
        )
        grouped_by_size.append(
            ExperimentGroup(size, global_chunks, embeddings, questions)
        )

    return grouped_by_size


def build_grouped_by_size(
    df: pd.DataFrame, encoding_model: SentenceTransformer, device: str
) -> list[ExperimentGroup]:
    group_counts = df.groupby(RECIPE_NAME_COL).size()
    unique_sizes = sorted([int(s) for s in group_counts.unique()], reverse=True)

    grouped_by_size: list[ExperimentGroup] = []
    print("\nBuilding experiment groups by variant size...")
    for size in tqdm(unique_sizes, desc="Sizes"):
        questions: list[tuple[str, str, str, list[int]]] = []
        global_chunks: list[str] = []

        matching_recipes = group_counts[group_counts == size].index
        group_df = df[df[RECIPE_NAME_COL].isin(matching_recipes)]

        for recipe_name, t_df in group_df.groupby(RECIPE_NAME_COL):
            t_df = t_df.sort_values(CODE_COL)
            code_id = "-".join(str(c) for c in t_df[CODE_COL].tolist())
            correct_chunk_indices: list[int] = []
            for row in t_df.itertuples(index=False):
                start = len(global_chunks)
                correct_chunk_indices.append(start)
                chunks = [row.Ingredients] + nltk.sent_tokenize(getattr(row, DIRECTION_COL))
                global_chunks.extend(chunks)

            questions.append(
                (
                    recipe_name,
                    code_id,
                    f"What are the ingredients of {recipe_name}?",
                    correct_chunk_indices,
                )
            )

        embeddings = encoding_model.encode(
            global_chunks,
            device=device,
            show_progress_bar=True,
            batch_size=64,
        )
        grouped_by_size.append(
            ExperimentGroup(size, global_chunks, embeddings, questions)
        )

    return grouped_by_size
