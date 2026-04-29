"""FastAPI backend for RAG PT-BR assistant.

Endpoints:
  GET  /health  — liveness + readiness probe
  POST /ask     — full RAG pipeline (retrieve → rerank → Groq/Cerebras fallback)

Models are loaded once at startup via FastAPI lifespan — never per-request.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src import config
from src.rag_query import (
    _ensure_loaded,  # private, but no public warm-up API exists; coupling is intentional
    build_pt_prompt,
    format_citations,
    generate_answer_with_fallback,
    rerank,
    retrieve,
)

# Module-level state — set once at startup, read on every request
_state: dict = {"ready": False, "chroma_count": 0}


def _load_models() -> None:
    """Load embedder, reranker, and open Chroma. Blocks until done."""
    import chromadb
    _ensure_loaded("new")
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    _state["chroma_count"] = client.get_collection(config.COLLECTION_NEW_PT).count()
    _state["ready"] = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield


app = FastAPI(title="RAG PT-BR Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Wave 3 will restrict to Vercel URL
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    top_n: int = 5


@app.get("/health")
def health():
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="models not loaded")
    return {
        "status": "ok",
        "chroma_count": _state["chroma_count"],
        "models_loaded": True,
    }


@app.post("/ask")
def ask(req: AskRequest):
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="models not loaded")
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    chunks = retrieve(req.question, top_k=config.TOP_K_RETRIEVE)
    reranked = rerank(req.question, chunks, top_n=req.top_n)

    if not reranked:
        return {
            "answer": "Não encontrei informações relevantes sobre sua pergunta.",
            "citations": [],
        }

    prompt = build_pt_prompt(req.question, reranked)
    try:
        answer_text = generate_answer_with_fallback(prompt)
    except Exception:
        raise HTTPException(status_code=502, detail="LLM unavailable")

    return {
        "answer": answer_text,
        "citations": format_citations(reranked),
    }
