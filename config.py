import os

from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:4b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma")
VISION_MODEL = os.getenv("VISION_MODEL", LLM_MODEL)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
USE_RAG = os.getenv("USE_RAG", "true").lower() == "true"
TOP_K = int(os.getenv("TOP_K", "15"))
N_SUGGESTIONS = int(os.getenv("N_SUGGESTIONS", "6"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))

RAW_DIR = "data/raw"
SAMPLE_RAW_PATH = "data/sample/sample_recipes.csv"
RECIPES_PATH = "data/recipes_subset.csv"
SUBSET_SIZE = int(os.getenv("SUBSET_SIZE", "2000"))
ALLERGENS_PATH = "data/allergens.json"
DB_PATH = "data/app.db"
CHROMA_PATH = "data/chroma"
CHROMA_COLLECTION = "recipes"
