"""Clean the raw Food.com file into data/recipes_subset.csv, then fill SQLite and ChromaDB.

Run from the repo root:
    python -m data.prepare_data               uses data/raw/recipes.parquet (or recipes.csv)
    python -m data.prepare_data --sample      uses the small sample in data/sample (no Kaggle file needed)
    python -m data.prepare_data --skip-clean  reuses recipes_subset.csv, only rebuilds SQLite + Chroma

Steps:
 1. Load the raw file (parquet preferred; the CSV stores lists as R-style c("...") strings).
 2. Parse list columns, ISO-8601 durations (PT1H30M) and numbers.
 3. Drop unusable rows (no ingredients/instructions, unrealistic time or calories).
 4. Keep well-rated recipes and sample SUBSET_SIZE of them with a fixed seed.
 5. Tag diet + allergens from ingredient names (rag/filters.py).
 6. Save the CSV, load SQLite (db.py) and embed every recipe into ChromaDB (rag/retriever.py).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import pandas as pd

import config

RAW_COLUMNS = {
    "RecipeId": "recipe_id", "Name": "name", "RecipeCategory": "category",
    "Keywords": "keywords", "RecipeIngredientQuantities": "quantities",
    "RecipeIngredientParts": "ingredients", "RecipeInstructions": "instructions",
    "TotalTime": "total_time_raw", "AggregatedRating": "rating", "ReviewCount": "review_count",
    "Calories": "calories", "FatContent": "fat", "SaturatedFatContent": "saturated_fat",
    "SodiumContent": "sodium", "CarbohydrateContent": "carbs", "FiberContent": "fiber",
    "SugarContent": "sugar", "ProteinContent": "protein", "RecipeServings": "servings",
}
LIST_COLUMNS = ["keywords", "quantities", "ingredients", "instructions"]
NUMBER_COLUMNS = ["rating", "review_count", "calories", "fat", "saturated_fat", "sodium",
                  "carbs", "fiber", "sugar", "protein", "servings"]

_R_ITEM = re.compile(r'"((?:[^"\\]|\\.)*)"|\bNA\b')
_DURATION = re.compile(r"^P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$")


def parse_list(value) -> list[str]:
    """Parse a list cell. Parquet gives arrays; the CSV gives c("a", "b") / "a" / character(0).
    NA items become "" so quantities stay aligned with ingredient names."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if not isinstance(value, str):
        return ["" if v is None else str(v).strip() for v in list(value)]
    s = value.strip()
    if s in ("", "NA", "character(0)"):
        return []
    if s.startswith("c(") and s.endswith(")"):
        return [(m.group(1) or "").replace('\\"', '"').strip() for m in _R_ITEM.finditer(s[2:-1])]
    return [s.strip('"').strip()]


def parse_duration(value) -> int | None:
    """'PT1H30M' -> 90 minutes. Returns None when missing or unparseable."""
    if not isinstance(value, str):
        return None
    m = _DURATION.match(value.strip())
    if not m or not any(m.groups()):
        return None
    d, h, mins, secs = (int(g) if g else 0 for g in m.groups())
    return d * 1440 + h * 60 + mins + round(secs / 60)


def find_raw_file(sample: bool) -> Path:
    if sample:
        return Path(config.SAMPLE_RAW_PATH)
    for name in ("recipes.parquet", "recipes.csv"):
        path = Path(config.RAW_DIR) / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No raw file in {config.RAW_DIR}/. Download recipes.parquet (or recipes.csv) from "
        "https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews and put it there, "
        "or run with --sample.")


def load_raw(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=list(RAW_COLUMNS))
    else:
        df = pd.read_csv(path, usecols=list(RAW_COLUMNS))
    return df.rename(columns=RAW_COLUMNS)


def clean(df: pd.DataFrame, size: int, seed: int = 42) -> pd.DataFrame:
    from rag.filters import tag_recipe

    df = df.copy()
    for col in LIST_COLUMNS:
        df[col] = df[col].map(parse_list)
    for col in NUMBER_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["total_minutes"] = df["total_time_raw"].map(parse_duration)
    df["name"] = df["name"].astype(str).str.strip()
    df["category"] = df["category"].fillna("").astype(str)

    usable = (
        df["name"].str.len().gt(0)
        & df["ingredients"].map(len).gt(1)
        & df["instructions"].map(len).gt(0)
        & df["total_minutes"].between(5, 240)
        & df["calories"].between(50, 2000)
        & df["protein"].notna() & df["fat"].notna()
    )
    df = df[usable]
    print(f"  usable recipes: {len(df):,}")

    rated = df[(df["rating"] >= 4.0) & (df["review_count"] >= 3)]
    if len(rated) >= size:
        df = rated
    df = df.sample(n=min(size, len(df)), random_state=seed).sort_values("recipe_id")

    tags = df["ingredients"].map(tag_recipe).apply(pd.Series)
    df = pd.concat([df, tags], axis=1)
    df["recipe_id"] = df["recipe_id"].astype(int)
    df["total_minutes"] = df["total_minutes"].astype(int)
    for col in LIST_COLUMNS:
        df[col] = df[col].map(json.dumps)
    return df.drop(columns=["total_time_raw"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true", help="use data/sample/sample_recipes.csv")
    ap.add_argument("--skip-clean", action="store_true", help="reuse existing recipes_subset.csv")
    ap.add_argument("--skip-index", action="store_true", help="do not (re)build the ChromaDB index")
    ap.add_argument("--size", type=int, default=config.SUBSET_SIZE)
    args = ap.parse_args()

    import db
    from rag import retriever

    if args.skip_clean:
        df = pd.read_csv(config.RECIPES_PATH)
        print(f"Loaded {len(df):,} recipes from {config.RECIPES_PATH}")
    else:
        raw_path = find_raw_file(args.sample)
        print(f"1/3 Cleaning {raw_path} ...")
        df = clean(load_raw(raw_path), args.size)
        df.to_csv(config.RECIPES_PATH, index=False)
        print(f"  saved {len(df):,} recipes -> {config.RECIPES_PATH}")

    print("2/3 Loading SQLite ...")
    db.save_recipes(df)

    if not args.skip_index:
        print(f"3/3 Embedding {len(df):,} recipes with {config.EMBED_MODEL} into ChromaDB ...")
        t0 = time.time()
        retriever.build_index(db.load_recipes())
        print(f"  done in {time.time() - t0:.0f}s -> {config.CHROMA_PATH}")


if __name__ == "__main__":
    main()
