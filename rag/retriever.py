"""ChromaDB search over recipe embeddings.

Indexing (data/prepare_data.py -> build_index):
  every recipe becomes ONE document (name, category, keywords, ingredients, time,
  nutrition). One recipe = one chunk, so a hit always brings the whole recipe.
  The text is embedded by a local Ollama embedding model and stored in a
  persistent Chroma collection that uses cosine distance.

Querying (retrieve):
  the user's message is embedded with the same model; Chroma returns the nearest
  recipes by cosine distance, restricted to the ids that passed the hard filters.
  similarity = 1 - cosine distance.
"""
from __future__ import annotations

from dataclasses import dataclass

import ollama

import config

_ollama = ollama.Client(host=config.OLLAMA_HOST)


@dataclass
class Hit:
    recipe_id: int
    similarity: float
    document: str


def recipe_document(r: dict) -> str:
    """The text that represents a recipe in vector space."""
    keywords = ", ".join(r["keywords"][:8])
    ingredients = ", ".join(i for i in r["ingredients"] if i)
    return (f"{r['name']}. Category: {r['category'] or 'unknown'}. Keywords: {keywords}. "
            f"Ingredients: {ingredients}. Ready in {int(r['total_minutes'])} minutes. "
            f"{r['calories']:.0f} kcal and {r['protein']:.0f} g protein per serving.")


def embed(texts: list[str]) -> list[list[float]]:
    return _ollama.embed(model=config.EMBED_MODEL, input=texts)["embeddings"]


def _collection(reset: bool = False):
    import chromadb
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    if reset:
        try:
            client.delete_collection(config.CHROMA_COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(
        config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine", "embed_model": config.EMBED_MODEL})


def build_index(df, batch_size: int = 64) -> int:
    col = _collection(reset=True)
    records = df.to_dict("records")
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        docs = [recipe_document(r) for r in batch]
        col.add(ids=[str(r["recipe_id"]) for r in batch],
                documents=docs,
                embeddings=embed(docs),
                metadatas=[{"recipe_id": int(r["recipe_id"])} for r in batch])
        print(f"  embedded {min(start + batch_size, len(records))}/{len(records)}", end="\r")
    print()
    return col.count()


def retrieve(query: str, allowed_ids: list[int] | None = None, k: int = config.TOP_K) -> list[Hit]:
    """Top-k recipes for `query`, only among `allowed_ids` (None = no restriction)."""
    col = _collection()
    if col.metadata.get("embed_model") != config.EMBED_MODEL:
        raise RuntimeError("The Chroma index was built with another embedding model. "
                           "Run: python -m data.prepare_data --skip-clean")
    if allowed_ids is not None and not allowed_ids:
        return []
    n = min(k, len(allowed_ids) if allowed_ids is not None else col.count())
    if n == 0:
        return []
    where = None
    if allowed_ids is not None and len(allowed_ids) < col.count():
        where = {"recipe_id": {"$in": [int(i) for i in allowed_ids]}}
    res = col.query(query_embeddings=embed([query]), n_results=n, where=where,
                    include=["documents", "distances", "metadatas"])
    return [Hit(int(m["recipe_id"]), 1 - d, doc)
            for m, d, doc in zip(res["metadatas"][0], res["distances"][0], res["documents"][0])]
