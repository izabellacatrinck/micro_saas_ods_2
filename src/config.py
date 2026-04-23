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
# Translation uses the fast/cheap model — Groq free tier has ~5x more TPD on
# 8b-instant vs 70b, and translation on a glossary-constrained task is already
# very high with 8b. The 70b stays as the default for RAG answer generation.
GROQ_TRANSLATION_MODEL = os.environ.get("GROQ_TRANSLATION_MODEL", GROQ_LLM_FAST)

# --- Ingestion: libraries that skip the EN→PT path ---
# pandas has 47 Google-translated PT docs (full topical coverage); translating
# its 10 EN docs would duplicate content and burn the daily token budget.
SKIP_EN_FOR_LIBRARIES = tuple(
    x.strip()
    for x in os.environ.get("RAG_SKIP_EN_LIBS", "pandas").split(",")
    if x.strip()
)

# --- Retrieval ---
TOP_K_RETRIEVE = 15
TOP_K_RERANK = 5

# --- Chunking ---
CHUNK_MAX_TOKENS = 450
CHUNK_OVERLAP_WORDS = 80
NOISE_THRESHOLD = 0.5

# --- Libraries tracked ---
LIBRARIES = ("pandas", "numpy", "matplotlib", "seaborn")
