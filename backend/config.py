import os
from dotenv import load_dotenv

load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma")
COLLECTION = os.getenv("COLLECTION", "cranio_lit")

DISCLAIMER = ("This assistant summarizes published research literature for educational and "
              "research purposes only. It is not medical advice, not a diagnostic tool, and "
              "does not replace a qualified clinician.")
