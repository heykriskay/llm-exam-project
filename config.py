import os

from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
USE_RAG = os.getenv("USE_RAG", "true").lower() == "true"
TOP_K = int(os.getenv("TOP_K", "20"))

RECIPES_PATH = "data/recipes_subset.csv"  # cleaned ~2,000 recipes, committed to the repo
ALLERGENS_PATH = "data/allergens.json"
DB_PATH = "data/app.db"  # SQLite: recipes and picked meals
CHROMA_PATH = "data/chroma"
