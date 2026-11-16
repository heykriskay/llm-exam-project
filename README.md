# Meal Planner

A local-LLM meal planner using RAG over ~2,000 Food.com recipes (Ollama, ChromaDB, SQLite, Streamlit).

The user chats ("cheap, high-protein dinners"), gets recipe cards, picks the ones they want for the week and keeps chatting ("replace Tuesday with something vegetarian", "what can I use instead of cream?"). A photo of the fridge can be turned into a list of ingredients to cook from.

Hard filters (allergies, diet, max time) are set in the sidebar. Code filters the recipes, embeddings retrieve candidates, the local LLM picks and explains, and code validates the result.

```
message --> hard filters (code) --> retrieval (Chroma, cosine) --> LLM picks + explains (gemma3:4b, JSON)
                                                                         |
recipe cards <-- validator (code: ids must be retrieved, filters must hold) <--+
```

## Project structure

| Path | Purpose |
|---|---|
| `app.py` | Streamlit UI: chat, recipe cards with Pick, fridge photo, "My week" + shopping list, retrieved-recipes panel, RAG on/off toggle, prompt viewer |
| `pipeline.py` | `suggest()` and `answer()`: filter, retrieve, LLM, validate. `route()` decides between them |
| `db.py` | SQLite: recipes and picked meals |
| `shopping.py` | Merged shopping list |
| `config.py` | Model names, `USE_RAG`, `TOP_K`, paths (override with `.env`) |
| `rag/filters.py` | Diet/allergen tagging from `data/allergens.json` and the hard filters |
| `rag/retriever.py` | Builds and queries the ChromaDB index |
| `llm/client.py` | Ollama chat call and the JSON schema for suggestions |
| `llm/validator.py` | Checks the LLM output: invented ids, rule violations, duplicates |
| `llm/vision.py` | Fridge photo to ingredient list (multimodal `gemma3:4b`) |
| `llm/prompts/` | System prompts with the 4T's: `menu.md`, `menu_no_rag.md`, `chat.md` |
| `data/prepare_data.py` | Cleans the raw Kaggle file into `recipes_subset.csv`, fills SQLite and ChromaDB |
| `data/sample/` | 24 hand-made recipes in the Kaggle format for a quick demo without downloading data |
| `experiments/regression_vs_rag.py` | Separate exam comparison: linear regression vs. LLM + RAG |
| `experiments/run_cases.py` | Fixed requests from `test_cases.json` run with RAG on and off |

## Prerequisites

- Python 3.11+ (tested with 3.13 on Windows 11)
- [Ollama](https://ollama.com/download) running locally
- About 4 GB of free RAM/VRAM for `gemma3:4b`
- A free Kaggle account to download the recipe data

## Installation

```bash
python -m venv .venv
```

Activate it (Windows: `.venv\Scripts\activate`, macOS/Linux: `source .venv/bin/activate`), then:

```bash
pip install -r requirements.txt
```

## Model setup

Both models are local and under the 4B parameter limit:

```bash
ollama pull gemma3:4b
```

```bash
ollama pull embeddinggemma
```

`gemma3:4b` (4B) generates the answers and reads fridge photos. `embeddinggemma` (~300M) makes the embeddings. To use other models, copy `.env.example` to `.env` and change `LLM_MODEL`, `EMBED_MODEL` or `VISION_MODEL`. If you change the embedding model, rebuild the index (next section).

## Loading the RAG material

1. Download `recipes.parquet` (or `recipes.csv`) from [Food.com Recipes and Reviews](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews) (CC0) and put it in `data/raw/`. The raw file is ignored by Git.
2. Build everything (cleaning, SQLite, embeddings; takes a few minutes the first time):

```bash
python -m data.prepare_data
```

What it does:

- Parses the list columns (R-style `c("...")` strings in the CSV), the `PT1H30M` durations and the numbers.
- Drops unusable rows: no ingredients or steps, time outside 5–240 min, or calories outside 50–2000 per serving.
- Keeps well-rated recipes (rating ≥ 4, at least 3 reviews) and samples 2,000 of them with seed 42.
- Tags diet and allergens from the ingredient names.
- Embeds one document per recipe into `data/chroma`, using cosine distance.

Other options:

- `python -m data.prepare_data --sample` uses the 24 sample recipes instead (quick test, no Kaggle download).
- `python -m data.prepare_data --skip-clean` reuses the existing `recipes_subset.csv` and only rebuilds SQLite and Chroma. Use this after changing the embedding model.

## Starting the application

Make sure Ollama is running, then from the repo root:

```bash
streamlit run app.py
```

Open http://localhost:8501.

## Testing it

Manual checks in the app:

1. **Suggestions:** type `cheap high-protein dinners`. You should get recipe cards, and the "Retrieved recipes" panel shows what RAG gave the LLM.
2. **Hard filters:** choose `vegan` and avoid `peanut`, then ask for `filling dinners`. Every card must respect the filters. Anything the LLM gets wrong is removed and listed by the Validator.
3. **RAG on vs. off:** switch off "Use RAG" and repeat the message. You get dishes from the model's own knowledge, marked as not from the database, with no cards, filters or nutrition data.
4. **Prompts:** tick "Show prompts (4T's)" to see the exact system and user prompt.
5. **Week:** press Pick on a few cards and open "My week" for the plan and the shopping list. Type `replace Tuesday with something vegetarian` and pick a card to swap that day.
6. **Questions:** ask `what can I use instead of cream?`. The answer is grounded in your picked recipes and names its source.
7. **Fridge:** open "What's in my fridge?", upload a photo, press "Detect ingredients", correct the list and press "Suggest meals with these".

Scripted checks (the results are written to `experiments/results/`):

```bash
python -m experiments.run_cases
```

```bash
python -m experiments.regression_vs_rag
```

`run_cases` runs 8 fixed requests with RAG on and off and counts valid suggestions, invented ids, rule violations and unverified dishes. `regression_vs_rag` is the separate exam comparison: it predicts kcal per serving from fat (g) with scikit-learn linear regression and with `gemma3:4b`. The LLM is run once with RAG over training examples only and once without RAG. It reports a table and the MAE, and logs exactly what was retrieved.

## The 4T's

Each prompt in `llm/prompts/` has explicit **Traits**, **Task**, **Tone** and **Target** sections:

- **Traits:** an honest planner that only uses the given recipes and never invents ids or values.
- **Task:** choose N meals, cite the id, give one reason each, and return JSON.
- **Tone:** short and practical.
- **Target:** busy students and families.

The vision prompt in `llm/vision.py` and the experiment prompt in `experiments/regression_vs_rag.py` follow the same structure.

## Out of MVP scope

- Accounts and multiple users
- Scaling and summing quantities in the shopping list (the dataset's quantities have no reliable units)
- Prices and budget calculations
- Breakfast/lunch planning
- A second model comparison

Allergen detection is keyword-based and is a prototype, not a safety guarantee.
