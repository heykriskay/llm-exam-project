"""Fridge photo -> list of ingredients, using a local multimodal model (gemma3:4b can see images).

The detected ingredients are shown to the user for correction, then used as the
retrieval query, so the vision step feeds the same RAG pipeline as typed messages.
"""
from __future__ import annotations

import json

import ollama

import config

_client = ollama.Client(host=config.OLLAMA_HOST)

VISION_PROMPT = """## Traits
You are a careful kitchen assistant who only reports food you can actually see.

## Task
List the food ingredients visible in this photo of a fridge, pantry or kitchen counter.
- Use simple, generic English ingredient names as found in recipes (e.g. "eggs", "milk", "bell pepper", "cheddar cheese").
- Do not guess hidden or unreadable items. Skip non-food items, drinks other than milk, and condiments unless clearly visible.
- Answer as JSON: {"ingredients": [...]}.

## Tone
Plain, short names only. No descriptions.

## Target
The list is used by a recipe search for a busy student or family, so recipe-style names matter more than brands."""

SCHEMA = {
    "type": "object",
    "properties": {"ingredients": {"type": "array", "items": {"type": "string"}}},
    "required": ["ingredients"],
}


def detect_ingredients(image: bytes) -> list[str]:
    resp = _client.chat(
        model=config.VISION_MODEL,
        messages=[{"role": "user", "content": VISION_PROMPT, "images": [image]}],
        format=SCHEMA,
        options={"temperature": 0, "num_ctx": config.NUM_CTX},
    )
    try:
        items = json.loads(resp["message"]["content"]).get("ingredients", [])
    except json.JSONDecodeError:
        return []
    clean = [str(i).strip().lower() for i in items if str(i).strip()]
    return list(dict.fromkeys(clean))


def fridge_matches(recipe: dict, fridge: list[str]) -> list[str]:
    """Which fridge items a recipe uses (simple word match on ingredient names)."""
    ings = " | ".join(i.lower() for i in recipe["ingredients"])
    found = []
    for item in fridge:
        words = [w for w in item.lower().split() if len(w) > 2]
        if words and any(w.rstrip("s") in ings for w in words[-1:]):
            found.append(item)
    return found
