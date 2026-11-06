# Meal Planner

Local-LLM meal planner using RAG over ~2,000 Food.com recipes (Ollama, ChromaDB, SQLite, Streamlit).
HelloFresh-style: the user chats ("cheap, high-protein dinners"), gets recipe cards, picks the ones they want for the week, and keeps chatting ("replace Tuesday with vegetarian", "what can I use instead of cream?").
Hard filters (allergies, diet, max time) are set in the sidebar. Code filters recipes, embeddings retrieve candidates, the local LLM picks and explains, code validates the ids.

| Path | Purpose | Owner |
|---|---|---|
| `app.py` | Streamlit UI: chat, recipe cards with Pick/Remove, "My week" + shopping list, retrieved-recipes panel, RAG on/off toggle | App |
| `db.py` | SQLite: recipes and picked meals | RAG |
| `pipeline.py` | `suggest()` and `answer()`: filter, retrieve, LLM, validate | RAG + LLM |
| `shopping.py` | Merged ingredient list | App |
| `config.py` | Model names, `USE_RAG`, `TOP_K`, paths | All |
| `rag/` | `filters.py` (allergy/diet/time), `retriever.py` (ChromaDB search) | RAG |
| `llm/` | `client.py` (Ollama), `validator.py` (check ids/rules), `prompts/menu.md` + `prompts/chat.md` (4T's) | LLM |
| `data/` | `prepare_data.py` (clean raw Kaggle file into `recipes_subset.csv`, fill SQLite + ChromaDB), `allergens.json` | Data |
| `experiments/` | `regression_vs_rag.py` (separate exam comparison), `run_cases.py`, `test_cases.json` | Exam |
| `tests/` | Basic tests | Exam |

## Data
- Source: [Food.com Recipes and Reviews](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews) (CC0). Put the raw file in `data/raw/` (ignored by Git).
- List columns are R-style `c("...")` strings and need parsing. Prefer the parquet file.
- Check nutrition columns and units before relying on calories.

## Build order
1. Filters, retrieval, LLM suggestions, validation, RAG on/off (exam core)
2. Cards, Pick, My week, shopping list
3. Chat questions, replace-via-chat

## Out of MVP
Accounts, quantity scaling in the shopping list, second-model comparison.

## Run (from the repo root)
1. `pip install -r requirements.txt` and `python -m data.prepare_data`
2. `streamlit run app.py`

Copy `.env.example` to `.env` to change models.
