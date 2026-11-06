# suggest(message, hard_filters, use_rag) -> 6 recipe cards with reasons
#   filter (rag/filters.py) -> retrieve (rag/retriever.py) -> LLM (llm/client.py) -> validate (llm/validator.py)
# answer(message, selected_ids, use_rag) -> chat answer grounded in the selected + retrieved recipes
# Used by app.py and experiments/run_cases.py so the app and the RAG on/off tests run the same code.
