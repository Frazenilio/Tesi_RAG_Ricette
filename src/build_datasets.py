import ast
import re
from pathlib import Path
import pandas as pd

# ==========================================
# CONSTANTS & PATHS
# ==========================================
# Input dataset path (e.g. filtered_2M_dataset_V14.csv)
INPUT_DATASET_PATH = "data/filtered_2M_dataset_V14.csv"

# Output cleaned ingredients path
OUTPUT_CLEANED_PATH = "data/cleaned_filtered_ingredients.csv"

# Input CSV column names
INPUT_TITLE_COL = "title"
INPUT_INGREDIENTS_COL = "ingredients"


# ==========================================
# INGREDIENT CLEANING LOGIC
# ==========================================
def clean_single_ingredient(ing: str) -> str:
    """
    Cleans a single ingredient string by removing numbers, units, 
    preparation descriptors, and parentheticals.
    
    Expected input formats:
    - "n X (Y g)" -> e.g. "2 eggs, beaten (100.0 g)"
    - "Y g X"     -> e.g. "108.0 g sugar"
    """
    ing = ing.strip()
    if not ing:
        return ""
        
    # 1. Handle "n X (Y g)" format (e.g., "2 eggs, beaten (100.0 g)")
    # Group 1: quantity (optional), Group 2: Name & prep, Group 3: Gram quantity
    # Note: Fraction pattern (\d+/\d+) must precede float/int (\d+(?:\.\d+)?) in alternation
    # to prevent matching only the numerator digit and leaving the slash in Group 2.
    m_unit = re.match(
        r'^(\d+/\d+|\d+(?:\.\d+)?)?\s*(.*?)\s*\(\s*(\d+(?:\.\d+)?\s*g)\s*\)\s*$', 
        ing, 
        re.IGNORECASE
    )
    if m_unit:
        name = m_unit.group(2)
    else:
        # 2. Handle "Y g X" format (e.g., "108.0 g sugar" or "247.2 g milk, lukewarm")
        m_gram = re.match(
            r'^(\d+(?:\.\d+)?\s*g)\s*(?:of\s+)?(.*)$', 
            ing, 
            re.IGNORECASE
        )
        if m_gram:
            name = m_gram.group(2)
        else:
            # Fallback to the original string
            name = ing

    # 3. Strip details after a comma (e.g., "flour, sifted" -> "flour")
    if ',' in name:
        name = name.split(',')[0]

    # 4. Remove any remaining parentheses (e.g., comments like "(last)" or "(no substitutions)")
    name = re.sub(r'\(.*?\)', '', name)

    # 5. Clean common preparation words and adjectives
    prep_words = [
        'lukewarm', 'melted', 'softened', 'beaten', 'sifted', 'chopped', 'minced',
        'diced', 'sliced', 'drained', 'warm', 'cold', 'chilled', 'peeled', 'halved',
        'quartered', 'slivered', 'grated', 'shredded', 'crushed', 'uncooked', 'cooked',
        'fresh', 'dried', 'ground', 'powdered', 'packed', 'firmly packed', 'blended', 
        'firm', 'large', 'medium', 'small', 'thinly', 'finely', 'optional', 'to taste',
        'for garnish', 'for serving', 'as needed', 'for the pan', 'plus more'
    ]
    pattern_prep = r'\b(?:' + '|'.join(prep_words) + r')\b'
    name = re.sub(pattern_prep, '', name, flags=re.IGNORECASE)

    # 6. Clean common units (Note: butter, margarine, oleo are ingredients and should not be stripped here!)
    units = [
        'g', 'gram', 'grams', 'kg', 'kilograms', 'ml', 'liters', 'tbsp', 'tablespoon',
        'tablespoons', 'tsp', 'teaspoon', 'teaspoons', 'cup', 'cups', 'c', 'ounce',
        'ounces', 'oz', 'pound', 'pounds', 'lb', 'lbs', 'head', 'heads', 'clove',
        'cloves', 'pod', 'pods', 'can', 'cans', 'pkg', 'pkgs', 'package', 'packages',
        'bag', 'bags', 'bottle', 'bottles', 'slice', 'slices', 'piece', 'pieces', 
        'box', 'stick', 'sticks'
    ]
    pattern_units = r'\b(?:' + '|'.join(units) + r')\b'
    name = re.sub(pattern_units, '', name, flags=re.IGNORECASE)

    # 7. Strip isolated numbers/fractions that might still be left
    name = re.sub(r'\b\d+(?:\.\d+)?\b', '', name)
    name = re.sub(r'\b\d+/\d+\b', '', name)

    # 8. Clean up trailing/isolated periods or punctuation (e.g. from stripped abbreviation like "Tbsp.")
    name = re.sub(r'\s*\.\s*', ' ', name)

    # 9. Clean up leading/trailing prepositions and conjunctions
    name = re.sub(r'^\s*(?:and|or|with|of|plus|about|at|in|for|from)\s+', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+(?:and|or|with|of|plus|about|at|in|for|from)\s*$', '', name, flags=re.IGNORECASE)

    # 10. Final formatting and cleanup
    name = re.sub(r'\s*/\s*', ' ', name) # Replace slashes with spaces to clean up any leftover fraction slashes
    name = re.sub(r'\s+', ' ', name)  # Replace multiple spaces with a single space
    name = re.sub(r'-\s*$', '', name) # Remove trailing hyphens
    name = re.sub(r'^\s*-', '', name) # Remove leading hyphens
    name = name.strip()

    return name


# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================
def main():
    print(f"Reading input dataset from: {INPUT_DATASET_PATH} ...")
    if not Path(INPUT_DATASET_PATH).exists():
        raise FileNotFoundError(
            f"Input file not found. Make sure {INPUT_DATASET_PATH} exists."
        )

    df_in = pd.read_csv(INPUT_DATASET_PATH)
    
    cleaned_rows = []
    
    for idx, row in df_in.iterrows():
        recipe_name = row[INPUT_TITLE_COL]

        # Parse ingredients list
        raw_ing_str = row[INPUT_INGREDIENTS_COL]
        try:
            ingreds_list = ast.literal_eval(raw_ing_str)
        except (ValueError, SyntaxError):
            # Fallback if literal_eval fails
            ingreds_list = [ing.strip().strip("'").strip('"') for ing in raw_ing_str.strip("[]").split(",")]

        # Clean ingredients list
        cleaned_list = []
        for ing in ingreds_list:
            cleaned = clean_single_ingredient(ing)
            if cleaned:
                cleaned_list.append(cleaned)

        # Format cleaned ingredients string
        # Format: "The ingredients of {RecipeName} are: {cleaned_ing_1}; {cleaned_ing_2}; ..."
        cleaned_ingredients_formatted = f"The ingredients of {recipe_name} are: {'; '.join(cleaned_list)}."

        cleaned_rows.append({
            "Ingredients": cleaned_ingredients_formatted
        })

    # Save cleaned_ingredients structure
    df_cleaned = pd.DataFrame(cleaned_rows)
    df_cleaned.to_csv(OUTPUT_CLEANED_PATH, index=False)
    print(f"Saved cleaned ingredients to: {OUTPUT_CLEANED_PATH} (Shape: {df_cleaned.shape})")
    
    print("\nDataset creation complete! You can run this script to generate your CSV files.")


if __name__ == "__main__":
    main()
