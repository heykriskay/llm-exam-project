# Synopsis – Meal Planner with a local LLM and RAG

Draft structure (max 5 pages). Text in [brackets] must be filled in with your own measured results.

## 1. Problem, users, purpose and MVP scope

- **Problem:** deciding what to cook every week takes time. Generic chatbots suggest dishes that don't exist, ignore allergies and can't produce a shopping list.
- **Users:** busy students and small families with limited time, budget and cooking experience.
- **Purpose:** suggest real, checked recipes that match the user's wishes and hard constraints, plan the week, produce a shopping list and answer cooking questions.
- **LLM's role:** understand free-text wishes, choose among retrieved recipes, explain the choices and answer questions. Rules (allergies, diet, time) are enforced by code, not the LLM.
- **MVP:** chat-based suggestions, hard filters, recipe cards, week plan, shopping list, Q&A, fridge photo, RAG on/off toggle.
- **Out of scope:** accounts, quantity scaling, prices, breakfast/lunch.

## 2. Architecture and implementation

- **Flow:** message → hard filters (pandas) → retrieval (ChromaDB, cosine, top-15) → `gemma3:4b` picks N recipes as JSON → validator → cards.
- **Local models via Ollama:**
  - `gemma3:4b` (4B parameters) for text and images
  - `embeddinggemma` for embeddings
- **Data:** 2,000 Food.com recipes, cleaned in `data/prepare_data.py`. They are stored in SQLite for the app state and in Chroma for vector search.
- **Validator:** rejects ids that were not retrieved (invented), recipes that break the filters, and duplicates.
- **Vision:** a fridge photo goes to `gemma3:4b`, which returns an ingredient list as JSON. The user corrects the list, and it becomes the retrieval query. Cards show which fridge items each recipe uses.
- [Insert a figure of the architecture and a screenshot of the app.]

## 3. The 4T's and design decisions

| T | Implementation (`llm/prompts/menu.md`) | Why |
|---|---|---|
| Traits | Honest planner, only uses given recipes, never invents ids | Reduce hallucination; ids are checkable |
| Task | Choose N, copy id + name, one reason each, JSON | Structured output a validator can check |
| Tone | Short, friendly, practical | Cards have little space |
| Target | Busy students and families | Simple language, quick decisions |

Key design decisions:

- **Hard rules live in code**, before and after the LLM, because a 4B model does not follow rules reliably.
- **JSON schema output** (Ollama `format`) makes the answer machine-checkable.
- **Compact one-line recipes in the prompt** keep a 4B model's context small.
- **Temperature 0.2** keeps answers stable and close to the sources.

## 4. Machine learning concepts, data preparation and retrieval

- **Data quality:**
  - Parsing R-style lists and ISO durations.
  - Removing unrealistic times and calories.
  - Quantities without units, which is why the shopping list is not scaled.
  - Nutrition values that are per serving.
- **Chunking:** one recipe = one document, so a hit always contains the whole recipe and its numbers.
- **Embeddings:** a text becomes a vector, and similar meaning gives nearby vectors. The same model must be used for documents and queries.
- **Similarity:** cosine similarity = 1 − cosine distance in Chroma. Top-k is taken among recipes that passed the filters.
- **Model behaviour:** the LLM is a next-token predictor trained on general text. Without RAG it invents plausible dishes it cannot verify. [Give an example from `run_cases`.]
- **Limitation:** allergen tagging is keyword-based (e.g. "butter" vs. "peanut butter"). Embeddings also handle numbers like "under 300 kcal" poorly.

## 5. Linear regression vs. local LLM + RAG

- **Setup:** predict kcal per serving from fat (g) per serving. Training/reference data = all recipes except 5 held-out test recipes (seed 42).
  - Regression: scikit-learn `LinearRegression` on the training rows.
  - LLM: `gemma3:4b` with the top-8 training examples retrieved by embedding similarity, plus once without RAG. No test targets are in the RAG data.
- [Paste the table and the MAE from `experiments/results/regression_vs_rag.md`.]
- **Discussion points:**
  - Regression learns one global relationship from all training rows. The LLM only sees 8 examples and estimates "by feel".
  - Text embeddings are poor at numeric closeness, so the retrieved examples may not have similar fat values. Check "mean |fat gap|" in the results.
  - Did RAG help compared with no RAG? [measured]
  - A specialised model is preferable when the task is numeric, data is available and you need cheap, fast, deterministic and explainable predictions.

## 6. Limitations and possible improvements

- **Limitations:**
  - Keyword allergen detection.
  - A 4B model sometimes returns fewer meals than asked or generic reasons.
  - The keyword router between "suggest" and "answer" can guess wrong (the UI has an override).
  - The vision model can miss or mis-name items.
- **Possible improvements:**
  - Hybrid search (BM25 + embeddings).
  - Metadata ranking on calories and protein.
  - An ingredient ontology for allergens.
  - Quantity normalisation in the shopping list.
  - Evaluation with real users.
