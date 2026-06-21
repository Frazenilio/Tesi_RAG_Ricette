from dataclasses import dataclass
from pathlib import Path

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


@dataclass
class ExperimentGroup:
    """All data for one group of recipes sharing the same number of variants."""

    size: int
    strategy: str
    global_chunks: list[str]
    embeddings: np.ndarray
    questions_ingredients: list[tuple[str, str, str, list[int]]]
    questions_directions: list[tuple[str, str, str, list[int]]]
    global_chunks_ingredients: list[str] = None
    embeddings_ingredients: np.ndarray = None
    global_chunks_directions: list[str] = None
    embeddings_directions: np.ndarray = None


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
    directions_csv: str,
    filter_recipes: list[str] = None,
) -> pd.DataFrame:
    root = Path(base_path)
    df = pd.read_csv(root / recipes_csv)
    df_ingred = pd.read_csv(root / ingredients_csv)
    df["Ingredients"] = df_ingred["Ingredients"]
    df_dirs = pd.read_csv(root / directions_csv)
    df["Directions"] = df_dirs["Directions"]

    # Filter by specific recipe names if requested
    if filter_recipes:
        df = df[df[RECIPE_NAME_COL].isin(filter_recipes)]

    return clean_recipe_variants(df)


def build_grouped_by_size_controlled(
    df: pd.DataFrame,
    encoding_model: SentenceTransformer,
    device: str,
    strategy: str,
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
    print(f"\nBuilding controlled experiment groups ({strategy}) (n_per_size={n_per_size})...")
    for size in tqdm([6, 5, 4, 3], desc="Sizes"):
        questions_ingredients: list[tuple[str, str, str, list[int]]] = []
        questions_directions: list[tuple[str, str, str, list[int]]] = []
        global_chunks: list[str] = []
        global_chunks_ingredients: list[str] = []
        global_chunks_directions: list[str] = []

        for recipe_name in base_recipes:
            recipe_df = (
                df[df[RECIPE_NAME_COL] == recipe_name].sort_values(CODE_COL).head(size)
            )
            code_id = "-".join(str(c) for c in recipe_df[CODE_COL].tolist())
            correct_ing_indices: list[int] = []
            correct_dir_indices: list[int] = []
            for row in recipe_df.itertuples(index=False):
                if strategy == "mixed":
                    start = len(global_chunks)
                    global_chunks.append(row.Ingredients)
                    correct_ing_indices.append(start)
                    global_chunks.append(row.Directions)
                    correct_dir_indices.append(start + 1)
                elif strategy == "combined":
                    start = len(global_chunks)
                    combined_chunk = f"{row.Ingredients}\n{row.Directions}"
                    global_chunks.append(combined_chunk)
                    correct_ing_indices.append(start)
                    correct_dir_indices.append(start)
                elif strategy == "separated":
                    start_ing = len(global_chunks_ingredients)
                    global_chunks_ingredients.append(row.Ingredients)
                    correct_ing_indices.append(start_ing)

                    start_dir = len(global_chunks_directions)
                    global_chunks_directions.append(row.Directions)
                    correct_dir_indices.append(start_dir)

            correct_ing_chunks = recipe_df.Ingredients.tolist()
            correct_dir_chunks = recipe_df.Directions.tolist()
            questions_ingredients.append(
                (
                    recipe_name,
                    code_id,
                    f"What are the ingredients of {recipe_name}?",
                    correct_ing_indices,
                    correct_ing_chunks,
                )
            )
            questions_directions.append(
                (
                    recipe_name,
                    code_id,
                    f"What are the directions of {recipe_name}?",
                    correct_dir_indices,
                    correct_dir_chunks,
                )
            )

        # Generate embeddings based on the selected strategy
        embeddings = None
        embeddings_ingredients = None
        embeddings_directions = None

        if strategy in ("mixed", "combined"):
            embeddings = encoding_model.encode(
                global_chunks,
                device=device,
                show_progress_bar=True,
                batch_size=64,
            )
        elif strategy == "separated":
            print(f"Encoding ingredients chunks...")
            embeddings_ingredients = encoding_model.encode(
                global_chunks_ingredients,
                device=device,
                show_progress_bar=True,
                batch_size=64,
            )
            print(f"Encoding directions chunks...")
            embeddings_directions = encoding_model.encode(
                global_chunks_directions,
                device=device,
                show_progress_bar=True,
                batch_size=64,
            )

        grouped_by_size.append(
            ExperimentGroup(
                size=size,
                strategy=strategy,
                global_chunks=global_chunks,
                embeddings=embeddings,
                questions_ingredients=questions_ingredients,
                questions_directions=questions_directions,
                global_chunks_ingredients=global_chunks_ingredients,
                embeddings_ingredients=embeddings_ingredients,
                global_chunks_directions=global_chunks_directions,
                embeddings_directions=embeddings_directions,
            )
        )

    return grouped_by_size


def build_grouped_by_size(
    df: pd.DataFrame,
    encoding_model: SentenceTransformer,
    device: str,
    strategy: str,
) -> list[ExperimentGroup]:
    group_counts = df.groupby(RECIPE_NAME_COL).size()
    unique_sizes = sorted([int(s) for s in group_counts.unique()], reverse=True)

    grouped_by_size: list[ExperimentGroup] = []
    print(f"\nBuilding experiment groups by variant size ({strategy})...")
    for size in tqdm(unique_sizes, desc="Sizes"):
        questions_ingredients: list[tuple[str, str, str, list[int]]] = []
        questions_directions: list[tuple[str, str, str, list[int]]] = []
        global_chunks: list[str] = []
        global_chunks_ingredients: list[str] = []
        global_chunks_directions: list[str] = []

        matching_recipes = group_counts[group_counts == size].index
        group_df = df[df[RECIPE_NAME_COL].isin(matching_recipes)]

        for recipe_name, t_df in group_df.groupby(RECIPE_NAME_COL):
            t_df = t_df.sort_values(CODE_COL)
            code_id = "-".join(str(c) for c in t_df[CODE_COL].tolist())
            correct_ing_indices: list[int] = []
            correct_dir_indices: list[int] = []
            for row in t_df.itertuples(index=False):
                if strategy == "mixed":
                    start = len(global_chunks)
                    global_chunks.append(row.Ingredients)
                    correct_ing_indices.append(start)
                    global_chunks.append(row.Directions)
                    correct_dir_indices.append(start + 1)
                elif strategy == "combined":
                    start = len(global_chunks)
                    combined_chunk = f"{row.Ingredients}\n{row.Directions}"
                    global_chunks.append(combined_chunk)
                    correct_ing_indices.append(start)
                    correct_dir_indices.append(start)
                elif strategy == "separated":
                    start_ing = len(global_chunks_ingredients)
                    global_chunks_ingredients.append(row.Ingredients)
                    correct_ing_indices.append(start_ing)

                    start_dir = len(global_chunks_directions)
                    global_chunks_directions.append(row.Directions)
                    correct_dir_indices.append(start_dir)

            correct_ing_chunks = t_df.Ingredients.tolist()
            correct_dir_chunks = t_df.Directions.tolist()
            questions_ingredients.append(
                (
                    recipe_name,
                    code_id,
                    f"What are the ingredients of {recipe_name}?",
                    correct_ing_indices,
                    correct_ing_chunks,
                )
            )
            questions_directions.append(
                (
                    recipe_name,
                    code_id,
                    f"What are the directions of {recipe_name}?",
                    correct_dir_indices,
                    correct_dir_chunks,
                )
            )

        # Generate embeddings based on the selected strategy
        embeddings = None
        embeddings_ingredients = None
        embeddings_directions = None

        if strategy in ("mixed", "combined"):
            embeddings = encoding_model.encode(
                global_chunks,
                device=device,
                show_progress_bar=True,
                batch_size=64,
            )
        elif strategy == "separated":
            print(f"Encoding ingredients chunks...")
            embeddings_ingredients = encoding_model.encode(
                global_chunks_ingredients,
                device=device,
                show_progress_bar=True,
                batch_size=64,
            )
            print(f"Encoding directions chunks...")
            embeddings_directions = encoding_model.encode(
                global_chunks_directions,
                device=device,
                show_progress_bar=True,
                batch_size=64,
            )

        grouped_by_size.append(
            ExperimentGroup(
                size=size,
                strategy=strategy,
                global_chunks=global_chunks,
                embeddings=embeddings,
                questions_ingredients=questions_ingredients,
                questions_directions=questions_directions,
                global_chunks_ingredients=global_chunks_ingredients,
                embeddings_ingredients=embeddings_ingredients,
                global_chunks_directions=global_chunks_directions,
                embeddings_directions=embeddings_directions,
            )
        )

    return grouped_by_size
