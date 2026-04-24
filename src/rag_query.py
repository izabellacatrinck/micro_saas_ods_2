"""Query orchestrator: retrieve → rerank → Groq generation in PT-BR."""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from src import config
from src.device import get_device


_embedder: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None
_collection = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBEDDER_MODEL, device=get_device())
    return _embedder


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(config.RERANKER_MODEL, device=get_device())
    return _reranker


def _get_collection(collection_name: str = config.COLLECTION_NEW_PT):
    global _collection
    if _collection is None or _collection.name != collection_name:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        _collection = client.get_collection(collection_name)
    return _collection


def retrieve(question: str, top_k: int = config.TOP_K_RETRIEVE,
             collection_name: str = config.COLLECTION_NEW_PT) -> list[dict]:
    embedder = _get_embedder()
    collection = _get_collection(collection_name)
    q_emb = embedder.encode(
        [f"query: {question}"], normalize_embeddings=True
    ).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=top_k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return [{"content": d, "metadata": m} for d, m in zip(docs, metas)]


def rerank(question: str, chunks: list[dict],
           top_n: int = config.TOP_K_RERANK) -> list[dict]:
    if not chunks:
        return []
    reranker = _get_reranker()
    pairs = [[question, c["content"]] for c in chunks]
    scores = reranker.predict(pairs).tolist()
    ranked = sorted(zip(scores, chunks), key=lambda t: -t[0])
    return [c for _, c in ranked[:top_n]]


_PT_SYSTEM = (
    "Voce e um assistente em PT-BR especializado em analise de dados com Python "
    "(pandas, numpy, matplotlib, seaborn). Responda apenas com base no CONTEXTO "
    "fornecido. Se a resposta nao estiver no contexto, diga 'Nao encontrei essa "
    "informacao na documentacao indexada.' Nao invente APIs, parametros nem erros. "
    "Seja conciso e direto - tom didatico para iniciantes. Quando citar metodo/classe, "
    "mantenha o nome em ingles (ex.: DataFrame.merge, np.array)."
)

_PT_SYSTEM_REAL = (
    "Você é um assistente em PT-BR especializado em análise de dados com Python "
    "(pandas, numpy, matplotlib, seaborn). Responda apenas com base no CONTEXTO "
    "fornecido. Se a resposta não estiver no contexto, diga 'Não encontrei essa "
    "informação na documentação indexada.' Não invente APIs, parâmetros nem erros. "
    "Seja conciso e direto — tom didático para iniciantes. Quando citar método/classe, "
    "mantenha o nome em inglês (ex.: DataFrame.merge, np.array)."
)


def build_pt_prompt(question: str, chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        header = f"[{i}] ({meta.get('library','?')} · {meta.get('section','?')})"
        blocks.append(f"{header}\n{c['content']}")
    contexto = "\n\n".join(blocks)
    return (
        f"{_PT_SYSTEM_REAL}\n\n"
        f"=== CONTEXTO ===\n{contexto}\n\n"
        f"=== PERGUNTA ===\n{question}\n\n"
        f"=== RESPOSTA (em português) ==="
    )


def format_citations(chunks: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for c in chunks:
        meta = c.get("metadata", {})
        key = (meta.get("source"), meta.get("section"))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "source": meta.get("source"),
            "section": meta.get("section"),
            "library": meta.get("library"),
        })
    return out


def generate_answer(prompt: str, model: str = config.GROQ_LLM_MODEL) -> str:
    from groq import Groq
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")
    client = Groq(api_key=config.GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=800,
    )
    return resp.choices[0].message.content.strip()


def answer(question: str,
           top_k: int = config.TOP_K_RETRIEVE,
           top_n: int = config.TOP_K_RERANK,
           collection_name: str = config.COLLECTION_NEW_PT,
           model: str = config.GROQ_LLM_MODEL) -> dict[str, Any]:
    retrieved = retrieve(question, top_k=top_k, collection_name=collection_name)
    reranked = rerank(question, retrieved, top_n=top_n)
    prompt = build_pt_prompt(question, reranked)
    llm_text = generate_answer(prompt, model=model)
    return {
        "answer": llm_text,
        "citations": format_citations(reranked),
        "retrieved_chunks": reranked,
    }


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Pergunte ao RAG PT-BR.")
    parser.add_argument("question", help="Pergunta em PT-BR")
    parser.add_argument("--top-k", type=int, default=config.TOP_K_RETRIEVE)
    parser.add_argument("--top-n", type=int, default=config.TOP_K_RERANK)
    parser.add_argument("--model", default=config.GROQ_LLM_MODEL)
    args = parser.parse_args(argv)

    result = answer(
        args.question, top_k=args.top_k, top_n=args.top_n, model=args.model,
    )
    print("\n=== RESPOSTA ===")
    print(result["answer"])
    print("\n=== CITACOES ===")
    for c in result["citations"]:
        print(f"  - {c['library']} . {c['section']} ({c['source']})")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
