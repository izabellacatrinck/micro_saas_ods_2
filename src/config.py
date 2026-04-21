"""Centralized configuration: model names, paths, hyperparameters.

Loaded once at import time. Values can be overridden by environment variables
(prefix GROQ_ or RAG_) for experimentation without code changes.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
MEDIUM_DIR = DATA_DIR / "medium"
EVAL_DIR = DATA_DIR / "eval"
TRANSLATIONS_CACHE = DATA_DIR / "translations_cache.jsonl"

# --- Chroma collections ---
COLLECTION_BASELINE_EN = "rag_chunks_baseline_en"
COLLECTION_NEW_PT = "rag_chunks_pt"

# --- Models ---
EMBEDDER_MODEL = os.environ.get("RAG_EMBEDDER", "intfloat/multilingual-e5-small")
RERANKER_MODEL = os.environ.get("RAG_RERANKER", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
BASELINE_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
BASELINE_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Groq ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_LLM_MODEL = os.environ.get("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
GROQ_LLM_FAST = "llama-3.1-8b-instant"

# --- Retrieval ---
TOP_K_RETRIEVE = 15
TOP_K_RERANK = 5

# --- Chunking ---
CHUNK_MAX_TOKENS = 450
CHUNK_OVERLAP_WORDS = 80
NOISE_THRESHOLD = 0.5

# --- Libraries tracked ---
LIBRARIES = ("pandas", "numpy", "matplotlib", "seaborn")
