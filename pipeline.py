"""The two flows of the app. Used by app.py and experiments/run_cases.py, so the app
and the RAG on/off tests run exactly the same code.

suggest(message, filters, use_rag) -> recipe cards with reasons
    filter (rag/filters.py) -> retrieve (rag/retriever.py) -> LLM (llm/client.py) -> validate (llm/validator.py)
answer(message, selected_ids, filters, use_rag) -> chat answer grounded in the picked + retrieved recipes
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import config
import db
from llm import client
from llm.validator import Validation, validate
from rag import retriever
from rag.filters import HardFilters, apply_filters

N_CHAT_CONTEXT = 3


@dataclass
class Card:
    recipe: dict
    reason: str


@dataclass
class SuggestResult:
    intro: str
    cards: list[Card]
    unverified: list[tuple[str, str]]
    hits: list[retriever.Hit]
    validation: Validation
    n_allowed: int
    system_prompt: str
    user_prompt: str
    raw: str


@dataclass
class AnswerResult:
    text: str
    hits: list[retriever.Hit] = field(default_factory=list)
    context_ids: list[int] = field(default_factory=list)
    system_prompt: str = ""
    user_prompt: str = ""


def recipe_line(r: dict) -> str:
    """Compact one-line summary used in the suggestion prompt (keeps a 4B model's context small)."""
    ings = ", ".join(i for i in r["ingredients"][:10] if i)
    return (f"[id {r['recipe_id']}] {r['name']} | {r['category'] or '-'} | {r['total_minutes']} min | "
            f"{r['calories']:.0f} kcal | {r['protein']:.0f} g protein | ingredients: {ings}")


def recipe_full(r: dict) -> str:
    """Full recipe text used when answering questions."""
    qty = r["quantities"] if len(r["quantities"]) == len(r["ingredients"]) else [""] * len(r["ingredients"])
    ings = "\n".join(f"- {q} {i}".replace("-  ", "- ") for q, i in zip(qty, r["ingredients"]))
    steps = "\n".join(f"{n}. {s}" for n, s in enumerate(r["instructions"], 1))
    return (f"### {r['name']} (id {r['recipe_id']})\n{r['total_minutes']} min, serves "
            f"{r['servings'] if r['servings'] == r['servings'] else '?'}, per serving: "
            f"{r['calories']:.0f} kcal, {r['protein']:.0f} g protein, {r['fat']:.0f} g fat, "
            f"{r['carbs']:.0f} g carbs\nIngredients:\n{ings}\nSteps:\n{steps}")


def suggest(message: str, filters: HardFilters, use_rag: bool = config.USE_RAG,
            n: int = config.N_SUGGESTIONS, k: int = config.TOP_K) -> SuggestResult:
    df = db.load_recipes()
    recipes = db.recipes_by_id()
    allowed = apply_filters(df, filters)
    allowed_ids = [int(i) for i in allowed["recipe_id"]]

    hits: list[retriever.Hit] = []
    if use_rag:
        system = client.load_prompt("menu")
        hits = retriever.retrieve(message, allowed_ids, k=k)
        listing = "\n".join(recipe_line(recipes[h.recipe_id]) for h in hits)
        user = (f"USER MESSAGE: {message}\nHARD FILTERS (already applied): {filters.describe()}\n"
                f"NUMBER OF MEALS: {n}\n\nRECIPES (choose only from these):\n{listing or '(none)'}")
        if not hits:
            v = Validation(intro="No recipes in the database match these filters. Try relaxing them.")
            return SuggestResult(v.intro, [], [], [], v, len(allowed_ids), system, user, "")
    else:
        system = client.load_prompt("menu_no_rag")
        user = (f"USER MESSAGE: {message}\nHARD FILTERS: {filters.describe()}\n"
                f"NUMBER OF MEALS: {n}")

    raw = client.chat(system, [{"role": "user", "content": user}], schema=client.SUGGESTION_SCHEMA)
    v = validate(raw, {h.recipe_id for h in hits}, recipes, filters, n, use_rag)
    cards = [Card(recipes[rid], reason) for rid, reason in v.valid]
    return SuggestResult(v.intro, cards, v.unverified, hits, v, len(allowed_ids), system, user, raw)


def answer(message: str, selected_ids: list[int], filters: HardFilters,
           use_rag: bool = config.USE_RAG, history: list[dict] | None = None) -> AnswerResult:
    system = client.load_prompt("chat")
    history = (history or [])[-6:]
    if not use_rag:
        user = f"MY DIET AND ALLERGIES: {filters.describe()}\n\nQUESTION: {message}"
        text = client.chat(system, [*history, {"role": "user", "content": user}])
        return AnswerResult(text, system_prompt=system, user_prompt=user)

    allowed_ids = [int(i) for i in apply_filters(db.load_recipes(), filters)["recipe_id"]]
    hits = retriever.retrieve(message, allowed_ids, k=N_CHAT_CONTEXT)
    context_ids = list(dict.fromkeys([*selected_ids, *(h.recipe_id for h in hits)]))
    recipes = db.get_recipes(context_ids)
    context = "\n\n".join(recipe_full(r) for r in recipes)
    user = (f"MY DIET AND ALLERGIES: {filters.describe()}\n"
            f"MY PICKED MEALS: {', '.join(r['name'] for r in db.get_recipes(selected_ids)) or 'none yet'}\n\n"
            f"RECIPES:\n{context or '(none)'}\n\nQUESTION: {message}")
    text = client.chat(system, [*history, {"role": "user", "content": user}])
    return AnswerResult(text, hits, context_ids, system, user)


DAY_PATTERN = re.compile(r"\b(" + "|".join(d.lower() for d in db.DAYS) + r")\b")
_SUGGEST_WORDS = re.compile(
    r"\b(suggest|ideas?|meals?|dinners?|lunch(es)?|recipes?|plan|week|something|give me|"
    r"show me|i want|i'd like|replace|swap|another|options?|cheap|quick)\b")
_QUESTION_START = re.compile(r"^(what|how|can|could|should|is|are|do|does|why|which|when|where)\b")


def route(message: str) -> tuple[str, int | None]:
    """-> ('suggest' | 'answer', target_day). Simple keyword rules, overridable in the UI.

    'replace tuesday with something vegetarian' -> ('suggest', 1)
    'what can I use instead of cream?'          -> ('answer', None)
    """
    text = message.lower().strip()
    day = None
    if re.search(r"\b(replace|swap|change)\b", text):
        m = DAY_PATTERN.search(text)
        day = db.DAYS.index(m.group(1).capitalize()) if m else None
    if day is not None:
        return "suggest", day
    if _QUESTION_START.match(text) and not re.search(r"\b(suggest|ideas?|recipes?)\b", text):
        return "answer", None
    return ("suggest", None) if _SUGGEST_WORDS.search(text) else ("answer", None)
