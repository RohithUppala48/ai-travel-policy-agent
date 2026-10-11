from pathlib import Path
import os
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

# Assignment structure contains .env.example
load_dotenv(BASE_DIR / ".env.example")


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2"
)

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate"
)

TOP_K = int(os.getenv("TOP_K", "3"))

MIN_RELEVANCE_SCORE = float(
    os.getenv("MIN_RELEVANCE_SCORE", "0.35")
)