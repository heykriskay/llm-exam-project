# Linear regression vs. local LLM + RAG

Task: predict kcal per serving from fat (g) per serving. Data: `data/recipes_subset.csv` (1995 training/reference rows, 5 held-out test rows, seed 42).

- Regression: `calories = 11.33 * fat + 165.5` (scikit-learn LinearRegression)
- LLM: `gemma3:4b`, temperature 0; RAG: top-8 training examples by cosine similarity with `embeddinggemma`

| recipe                                                 |   fat_g |   actual_kcal |   regression |   abs_err_regression |   llm_rag |   abs_err_llm_rag |   llm_no_rag |   abs_err_llm_no_rag |
|:-------------------------------------------------------|--------:|--------------:|-------------:|---------------------:|----------:|------------------:|-------------:|---------------------:|
| Backwoods Home Baked Macaroni &amp; Cheese to Kill For |    41.4 |         679.8 |        634.7 |                 45.1 |       442 |             237.8 |         1704 |               1024.2 |
| Mongolian Beef                                         |    47.6 |         699.4 |        704.9 |                  5.5 |       748 |              48.6 |         1945 |               1245.6 |
| Marinated Antipasto Platter                            |    14.2 |         183.7 |        326.4 |                142.7 |       238 |              54.3 |          118 |                 65.7 |
| Vanilla Banana Bread                                   |    18.4 |         479.1 |        374   |                105.1 |       554 |              74.9 |          208 |                271.1 |
| Frosty Gin &amp; Tonic                                 |     0   |         270.5 |        165.5 |                105   |        86 |             184.5 |            0 |                270.5 |

| Approach | MAE (kcal) |
|---|---|
| regression | 80.7 |
| llm_rag | 120.0 |
| llm_no_rag | 575.4 |

## What RAG retrieved

**Backwoods Home Baked Macaroni &amp; Cheese to Kill For** – query `Reference recipe: fat 41.4 g per serving` – mean |fat gap| of retrieved examples: 17.19 g
- `0.877` Reference recipe: fat 21.2 g per serving -> 414 kcal per serving.
- `0.865` Reference recipe: fat 12.1 g per serving -> 414 kcal per serving.
- `0.861` Reference recipe: fat 33.7 g per serving -> 414 kcal per serving.
- `0.858` Reference recipe: fat 14.9 g per serving -> 412 kcal per serving.
- `0.857` Reference recipe: fat 20.8 g per serving -> 414 kcal per serving.
- `0.856` Reference recipe: fat 41.6 g per serving -> 442 kcal per serving.
- `0.853` Reference recipe: fat 44.0 g per serving -> 411 kcal per serving.
- `0.849` Reference recipe: fat 11.0 g per serving -> 414 kcal per serving.

**Mongolian Beef** – query `Reference recipe: fat 47.6 g per serving` – mean |fat gap| of retrieved examples: 17.73 g
- `0.875` Reference recipe: fat 47.5 g per serving -> 746 kcal per serving.
- `0.872` Reference recipe: fat 36.9 g per serving -> 472 kcal per serving.
- `0.87` Reference recipe: fat 26.4 g per serving -> 470 kcal per serving.
- `0.867` Reference recipe: fat 27.5 g per serving -> 478 kcal per serving.
- `0.867` Reference recipe: fat 22.3 g per serving -> 476 kcal per serving.
- `0.866` Reference recipe: fat 26.8 g per serving -> 472 kcal per serving.
- `0.864` Reference recipe: fat 9.7 g per serving -> 475 kcal per serving.
- `0.861` Reference recipe: fat 41.9 g per serving -> 477 kcal per serving.

**Marinated Antipasto Platter** – query `Reference recipe: fat 14.2 g per serving` – mean |fat gap| of retrieved examples: 3.3 g
- `0.887` Reference recipe: fat 14.2 g per serving -> 144 kcal per serving.
- `0.869` Reference recipe: fat 4.2 g per serving -> 142 kcal per serving.
- `0.865` Reference recipe: fat 8.2 g per serving -> 142 kcal per serving.
- `0.864` Reference recipe: fat 9.6 g per serving -> 142 kcal per serving.
- `0.859` Reference recipe: fat 14.2 g per serving -> 332 kcal per serving.
- `0.858` Reference recipe: fat 9.8 g per serving -> 142 kcal per serving.
- `0.847` Reference recipe: fat 14.2 g per serving -> 238 kcal per serving.
- `0.847` Reference recipe: fat 12.8 g per serving -> 141 kcal per serving.

**Vanilla Banana Bread** – query `Reference recipe: fat 18.4 g per serving` – mean |fat gap| of retrieved examples: 24.35 g
- `0.874` Reference recipe: fat 8.4 g per serving -> 184 kcal per serving.
- `0.868` Reference recipe: fat 18.4 g per serving -> 334 kcal per serving.
- `0.867` Reference recipe: fat 188.2 g per serving -> 1884 kcal per serving.
- `0.866` Reference recipe: fat 7.4 g per serving -> 184 kcal per serving.
- `0.861` Reference recipe: fat 18.4 g per serving -> 554 kcal per serving.
- `0.857` Reference recipe: fat 18.4 g per serving -> 288 kcal per serving.
- `0.856` Reference recipe: fat 18.4 g per serving -> 366 kcal per serving.
- `0.856` Reference recipe: fat 14.4 g per serving -> 181 kcal per serving.

**Frosty Gin &amp; Tonic** – query `Reference recipe: fat 0.0 g per serving` – mean |fat gap| of retrieved examples: 0.0 g
- `0.816` Reference recipe: fat 0.0 g per serving -> 59 kcal per serving.
- `0.815` Reference recipe: fat 0.0 g per serving -> 88 kcal per serving.
- `0.814` Reference recipe: fat 0.0 g per serving -> 76 kcal per serving.
- `0.808` Reference recipe: fat 0.0 g per serving -> 128 kcal per serving.
- `0.808` Reference recipe: fat 0.0 g per serving -> 86 kcal per serving.
- `0.804` Reference recipe: fat 0.0 g per serving -> 65 kcal per serving.
- `0.803` Reference recipe: fat 0.0 g per serving -> 168 kcal per serving.
- `0.803` Reference recipe: fat 0.0 g per serving -> 126 kcal per serving.
