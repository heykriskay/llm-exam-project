"""Hard filters: allergies, diet and max time.

These are rules, so they are enforced in code BEFORE retrieval and checked again
AFTER the LLM answers (llm/validator.py). The LLM is never trusted with safety rules.
Diet/allergen tags come from keyword matching on ingredient names (data/allergens.json).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

import config

DIETS = ["any", "vegetarian", "vegan", "pescetarian"]


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(config.ALLERGENS_PATH, encoding="utf-8") as f:
        return json.load(f)


def allergen_names() -> list[str]:
    return list(load_config()["allergens"])


def _pattern(keywords: list[str]) -> re.Pattern:
    words = sorted({k.lower() for k in keywords}, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b")


@lru_cache(maxsize=None)
def _compiled(group: str, name: str) -> re.Pattern:
    return _pattern(load_config()[group][name])


def matches(ingredients: list[str], group: str, name: str) -> list[str]:
    """Ingredient names that hit a keyword list, e.g. matches(ings, 'allergens', 'milk')."""
    pat = _compiled(group, name)
    false_friends = re.compile(r"\b(peanut butter|almond butter|cashew butter|coconut milk|"
                               r"coconut cream|soy milk|almond milk|oat milk|cream of tartar|"
                               r"butternut|vegan \w+)\b")
    hits = []
    for ing in ingredients:
        text = ing.lower()
        if group != "allergens" or name in ("milk", "egg"):
            text = false_friends.sub(" ", text)
        if pat.search(text):
            hits.append(ing)
    return hits


def tag_recipe(ingredients: list[str]) -> dict:
    """Diet and allergen flags for one recipe (stored as columns in SQLite)."""
    has_meat = bool(matches(ingredients, "diet", "meat"))
    has_seafood = bool(matches(ingredients, "diet", "seafood"))
    has_animal = bool(matches(ingredients, "diet", "animal_product"))
    tags = {
        "is_vegetarian": not has_meat and not has_seafood,
        "is_vegan": not has_meat and not has_seafood and not has_animal,
        "is_pescetarian": not has_meat,
    }
    for allergen in allergen_names():
        tags[f"contains_{allergen}"] = bool(matches(ingredients, "allergens", allergen))
    return tags


@dataclass
class HardFilters:
    diet: str = "any"
    allergies: list[str] = field(default_factory=list)
    max_minutes: int | None = None

    def describe(self) -> str:
        parts = [f"diet: {self.diet}",
                 f"avoid: {', '.join(self.allergies) if self.allergies else 'nothing'}",
                 f"max time: {self.max_minutes} min" if self.max_minutes else "max time: none"]
        return "; ".join(parts)


def apply_filters(df: pd.DataFrame, f: HardFilters) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if f.diet != "any":
        mask &= df[f"is_{f.diet}"].astype(bool)
    for allergen in f.allergies:
        mask &= ~df[f"contains_{allergen}"].astype(bool)
    if f.max_minutes:
        mask &= df["total_minutes"] <= f.max_minutes
    return df[mask]


def violations(recipe: dict, f: HardFilters) -> list[str]:
    """Human-readable reasons why a recipe breaks the hard filters (empty = OK)."""
    out = []
    if f.diet != "any" and not recipe.get(f"is_{f.diet}"):
        out.append(f"not {f.diet}")
    for allergen in f.allergies:
        if recipe.get(f"contains_{allergen}"):
            out.append(f"contains {allergen}")
    if f.max_minutes and recipe.get("total_minutes", 0) > f.max_minutes:
        out.append(f"takes {recipe['total_minutes']} min > {f.max_minutes}")
    return out
