"""Exam comparison: linear regression vs. local LLM + RAG on the same numerical task.

Task: predict CALORIES per serving from FAT (g) per serving, using the cleaned recipes.
  - Split with a fixed seed: N_TEST held-out test recipes, the rest is training/reference data.
  - Linear regression (scikit-learn) is fit on the training rows only.
  - The LLM gets a RAG collection built ONLY from training rows ("fat X g -> Y kcal" examples)
    plus a short background note. It never sees test targets and never runs regression code.
  - We also ask the LLM without RAG, to see whether retrieval helped.
  - Output: table of actual vs. predictions, absolute errors and MAE, plus what was retrieved.

Run from the repo root (after python -m data.prepare_data):
    python -m experiments.regression_vs_rag
Results: experiments/results/regression_vs_rag.md and .json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import ollama
import pandas as pd
from sklearn.linear_model import LinearRegression

import config

N_TEST = 5
K_EXAMPLES = 8
SEED = 42
OUT_DIR = Path(__file__).parent / "results"
_client = ollama.Client(host=config.OLLAMA_HOST)

BACKGROUND = ("Background: the reference data comes from a recipe website. Each example gives the fat "
              "content in grams per serving and the energy in kilocalories (kcal) per serving of one recipe. "
              "Energy comes from fat, carbohydrates and protein, so recipes with the same fat can differ in calories.")

SYSTEM = """## Traits
You are a careful nutrition data analyst who bases estimates on the reference examples given.

## Task
Estimate the calories (kcal per serving) of a recipe from its fat content (g per serving).
Use the REFERENCE EXAMPLES if provided. Do not write code. Answer as JSON: {"calories": <number>, "reasoning": "<one sentence>"}.

## Tone
Brief and precise.

## Target
A data science student comparing your estimate to a linear regression model."""

SCHEMA = {"type": "object",
          "properties": {"calories": {"type": "number"}, "reasoning": {"type": "string"}},
          "required": ["calories", "reasoning"]}


def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(config.RECIPES_PATH)[["recipe_id", "name", "fat", "calories"]].dropna()
    df = df[(df["fat"] >= 0) & (df["calories"] > 0)]
    test = df.sample(n=N_TEST, random_state=SEED)
    train = df.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def regression(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, LinearRegression]:
    model = LinearRegression().fit(train[["fat"]], train["calories"])
    return model.predict(test[["fat"]]), model


def example_text(row) -> str:
    return f"Reference recipe: fat {row.fat:.1f} g per serving -> {row.calories:.0f} kcal per serving."


def embed(texts: list[str]) -> np.ndarray:
    vecs = np.array(_client.embed(model=config.EMBED_MODEL, input=texts)["embeddings"], dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


class ReferenceRAG:
    """In-memory RAG over TRAINING rows only (same embedding model as the app, cosine similarity)."""

    def __init__(self, train: pd.DataFrame):
        self.docs = [example_text(r) for r in train.itertuples()]
        self.fat = train["fat"].to_numpy()
        self.vectors = np.vstack([embed(self.docs[i:i + 64]) for i in range(0, len(self.docs), 64)])

    def search(self, query: str, k: int = K_EXAMPLES) -> list[tuple[str, float, float]]:
        scores = self.vectors @ embed([query])[0]
        best = np.argsort(scores)[::-1][:k]
        return [(self.docs[i], float(scores[i]), float(self.fat[i])) for i in best]


def ask_llm(fat: float, examples: list[str] | None) -> tuple[float, str]:
    question = f"Recipe with fat {fat:.1f} g per serving. How many kcal per serving?"
    if examples:
        question = "REFERENCE EXAMPLES:\n" + BACKGROUND + "\n" + "\n".join(examples) + "\n\n" + question
    resp = _client.chat(model=config.LLM_MODEL,
                        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
                        format=SCHEMA, options={"temperature": 0, "num_ctx": config.NUM_CTX})
    try:
        data = json.loads(resp["message"]["content"])
        return float(data["calories"]), str(data.get("reasoning", ""))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return float("nan"), resp["message"]["content"][:200]


def main():
    train, test = load_split()
    print(f"Train/reference rows: {len(train)}, test rows: {len(test)}")

    reg_pred, model = regression(train, test)
    print(f"Regression: calories = {model.coef_[0]:.2f} * fat + {model.intercept_:.1f}")

    rag = ReferenceRAG(train)
    rows, retrieved_log = [], []
    for i, r in test.iterrows():
        query = f"Reference recipe: fat {r.fat:.1f} g per serving"
        hits = rag.search(query)
        llm_rag, why_rag = ask_llm(r.fat, [h[0] for h in hits])
        llm_plain, why_plain = ask_llm(r.fat, None)
        rows.append({"recipe": r["name"], "fat_g": r.fat, "actual_kcal": r.calories,
                     "regression": reg_pred[i], "llm_rag": llm_rag, "llm_no_rag": llm_plain,
                     "reason_llm_rag": why_rag, "reason_llm_no_rag": why_plain})
        retrieved_log.append({"recipe": r["name"], "query": query,
                              "retrieved": [{"text": t, "similarity": round(s, 3), "fat": f} for t, s, f in hits],
                              "mean_abs_fat_gap": round(float(np.mean([abs(f - r.fat) for _, _, f in hits])), 2)})
        print(f"  {r['name'][:30]:30} actual {r.calories:6.0f} | reg {reg_pred[i]:6.0f} | "
              f"llm+rag {llm_rag:6.0f} | llm {llm_plain:6.0f}")

    res = pd.DataFrame(rows)
    for col in ["regression", "llm_rag", "llm_no_rag"]:
        res[f"abs_err_{col}"] = (res[col] - res["actual_kcal"]).abs()
    mae = {col: float(res[f"abs_err_{col}"].mean()) for col in ["regression", "llm_rag", "llm_no_rag"]}

    OUT_DIR.mkdir(exist_ok=True)
    table = res[["recipe", "fat_g", "actual_kcal", "regression", "abs_err_regression", "llm_rag",
                 "abs_err_llm_rag", "llm_no_rag", "abs_err_llm_no_rag"]].round(1)
    md = [
        "# Linear regression vs. local LLM + RAG", "",
        f"Task: predict kcal per serving from fat (g) per serving. Data: `{config.RECIPES_PATH}` "
        f"({len(train)} training/reference rows, {len(test)} held-out test rows, seed {SEED}).", "",
        f"- Regression: `calories = {model.coef_[0]:.2f} * fat + {model.intercept_:.1f}` (scikit-learn LinearRegression)",
        f"- LLM: `{config.LLM_MODEL}`, temperature 0; RAG: top-{K_EXAMPLES} training examples by cosine "
        f"similarity with `{config.EMBED_MODEL}`", "",
        table.to_markdown(index=False), "",
        "| Approach | MAE (kcal) |", "|---|---|",
        *[f"| {k} | {v:.1f} |" for k, v in mae.items()], "",
        "## What RAG retrieved", "",
    ]
    for log in retrieved_log:
        md.append(f"**{log['recipe']}** – query `{log['query']}` – mean |fat gap| of retrieved examples: "
                  f"{log['mean_abs_fat_gap']} g")
        md += [f"- `{h['similarity']}` {h['text']}" for h in log["retrieved"]]
        md.append("")
    (OUT_DIR / "regression_vs_rag.md").write_text("\n".join(md), encoding="utf-8")
    (OUT_DIR / "regression_vs_rag.json").write_text(json.dumps(
        {"coef": model.coef_[0], "intercept": model.intercept_, "mae": mae,
         "rows": rows, "retrieved": retrieved_log}, indent=2, default=float), encoding="utf-8")
    print("\nMAE:", {k: round(v, 1) for k, v in mae.items()})
    print(f"Saved {OUT_DIR / 'regression_vs_rag.md'}")


if __name__ == "__main__":
    main()
