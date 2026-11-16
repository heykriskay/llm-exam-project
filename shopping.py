"""Merged shopping list for the picked recipes.

Quantities are listed per recipe, not summed: the dataset's quantities are free text
("1 1/2", "2-3") without reliable units, so scaling/summing is out of MVP scope.
"""
from __future__ import annotations

from collections import defaultdict

PANTRY = {"salt", "pepper", "black pepper", "water", "oil", "olive oil", "vegetable oil",
          "sugar", "flour", "butter", "garlic powder", "onion powder", "salt and pepper"}


def build_list(recipes: list[dict]) -> list[dict]:
    """-> rows {ingredient, amounts, recipes, pantry}, sorted with pantry items last."""
    merged: dict[str, dict] = defaultdict(lambda: {"amounts": [], "recipes": []})
    for r in recipes:
        qty = r["quantities"] if len(r["quantities"]) == len(r["ingredients"]) else [""] * len(r["ingredients"])
        for q, ing in zip(qty, r["ingredients"]):
            key = ing.strip().lower()
            if not key:
                continue
            row = merged[key]
            if q:
                row["amounts"].append(q)
            if r["name"] not in row["recipes"]:
                row["recipes"].append(r["name"])
    rows = [{"ingredient": k, "amounts": " + ".join(v["amounts"]) or "-",
             "recipes": ", ".join(v["recipes"]), "pantry": k in PANTRY}
            for k, v in merged.items()]
    return sorted(rows, key=lambda x: (x["pantry"], x["ingredient"]))


def as_text(rows: list[dict]) -> str:
    lines = ["Shopping list", ""]
    lines += [f"[ ] {r['ingredient']} ({r['amounts']})" for r in rows if not r["pantry"]]
    pantry = [r["ingredient"] for r in rows if r["pantry"]]
    if pantry:
        lines += ["", "Check you have: " + ", ".join(pantry)]
    return "\n".join(lines)
