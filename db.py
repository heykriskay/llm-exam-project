"""SQLite, the only file that touches it.

Tables:
  recipes - loaded from recipes_subset.csv by data/prepare_data.py (list columns stored as JSON text)
  picks   - the user's week: one recipe per day (0 = Monday ... 6 = Sunday)
Chat history stays in st.session_state.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from functools import lru_cache

import pandas as pd

import config

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
LIST_COLUMNS = ["keywords", "quantities", "ingredients", "instructions"]


def _connect(path: str | None = None) -> sqlite3.Connection:
    con = sqlite3.connect(path or config.DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS picks (day INTEGER PRIMARY KEY, recipe_id INTEGER NOT NULL)")
    return con


def save_recipes(df: pd.DataFrame, path: str | None = None) -> None:
    with closing(_connect(path)) as con:
        df.to_sql("recipes", con, if_exists="replace", index=False)
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_recipe_id ON recipes(recipe_id)")
        con.execute("DELETE FROM picks WHERE recipe_id NOT IN (SELECT recipe_id FROM recipes)")
        con.commit()
    load_recipes.cache_clear()


@lru_cache(maxsize=1)
def load_recipes(path: str | None = None) -> pd.DataFrame:
    """All recipes with list columns decoded. Cached: the table only changes in prepare_data."""
    with closing(_connect(path)) as con:
        try:
            df = pd.read_sql("SELECT * FROM recipes", con)
        except Exception:
            return pd.DataFrame()
    for col in LIST_COLUMNS:
        df[col] = df[col].map(lambda s: json.loads(s) if isinstance(s, str) else [])
    return df


def recipes_by_id(path: str | None = None) -> dict[int, dict]:
    df = load_recipes(path)
    return {int(r["recipe_id"]): r for r in df.to_dict("records")}


def get_recipes(ids: list[int], path: str | None = None) -> list[dict]:
    """Recipes in the order of `ids`; unknown ids are skipped."""
    lookup = recipes_by_id(path)
    return [lookup[i] for i in ids if i in lookup]


def list_picks(path: str | None = None) -> dict[int, int]:
    with closing(_connect(path)) as con:
        return dict(con.execute("SELECT day, recipe_id FROM picks ORDER BY day").fetchall())


def add_pick(recipe_id: int, day: int | None = None, path: str | None = None) -> int | None:
    """Put a recipe on `day`, or on the first free day. Returns the day, or None if the week is full."""
    picks = list_picks(path)
    if day is None:
        free = [d for d in range(7) if d not in picks]
        if not free:
            return None
        day = free[0]
    with closing(_connect(path)) as con:
        con.execute("INSERT OR REPLACE INTO picks (day, recipe_id) VALUES (?, ?)", (day, int(recipe_id)))
        con.commit()
    return day


def remove_pick(day: int, path: str | None = None) -> None:
    with closing(_connect(path)) as con:
        con.execute("DELETE FROM picks WHERE day = ?", (day,))
        con.commit()


def clear_picks(path: str | None = None) -> None:
    with closing(_connect(path)) as con:
        con.execute("DELETE FROM picks")
        con.commit()
