# Wave 1 — Backend RAG completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deixar o RAG backend 100% funcional via CLI em PT-BR, com retrieval rerankado e geração via Groq, pronto pra ser embrulhado num FastAPI na Onda 2.

**Architecture:** Ingestão offline popula ChromaDB com chunks PT (glossary_repair já aplicado). Módulo novo `src/rag_query.py` orquestra: embed query → retrieve top-15 → rerank top-5 → prompt PT → Groq `llama-3.3-70b-versatile` → resposta + citations. Módulo dá suporte a modo `--baseline` que troca pra coleção EN + MiniLM + ms-marco + prompt EN (pra avaliação comparativa na Onda 3).

**Tech Stack:** Python 3.12, uv, ChromaDB, sentence-transformers (e5-small + cross-encoder), Groq API, pytest.

---

## Escopo desta onda

| # | Task | Arquivos alvo |
|---|---|---|
| 1 | Lock-in de section-awareness do chunker (tests) | `tests/test_chunker.py` |
| 2 | Executar ingestão end-to-end | (executa `data/main.py`) |
| 3 | Novo módulo de query (`rag_query`) com Groq + e5 + mmarco | `src/rag_query.py`, `tests/test_rag_query.py` |
| 4 | Modo baseline (EN) no mesmo módulo | `src/rag_query.py` (extensão), `tests/test_rag_query.py` |

### Fora desta onda (Ondas 2 e 3)

- FastAPI backend, HF Space deploy
- Frontend Next.js, Vercel
- Golden set, tradução PT→EN de perguntas, RAGAS, relatório, README final

### Ajuste de escopo vs spec 2026-04-23

O spec seção 5.1 assumia que o chunker atual ignorava estrutura de headings. **Falso** — `ThematicSegmenter.segment()` (`data/main.py:121-161`) já quebra por `#/##` markdown, e `SmartChunker.chunk()` roda por segmento. A ablation "fixed vs section-aware" perde sentido (seria comparar contra um chunker propositalmente pior). **Dropamos essa rodada de ablation** — voltamos às 3 rodadas originais do spec 2026-04-21 (baseline EN + novo 70B + novo 8B). Task 1 troca "implementar" por "travar comportamento atual com testes explícitos".

---

## Pré-requisitos

Variáveis de ambiente em `.env` (na raiz do worktree):

```
GROQ_API_KEY=gsk_...
# GOOGLE_API_KEY e CEREBRAS_API_KEY só serão usados em Ondas 2-3
```

Python env: `uv` gerencia `.venv/`. Rodar comandos via `uv run python ...` ou `.venv/Scripts/python.exe ...`.

---

## Task 1: Lock-in de section-awareness do chunker

**Objetivo:** Adicionar 3 testes que confirmem o contrato atual (chunk respeita fronteira de seção, metadata `section` é propagada, section longa é quebrada com overlap dentro dela mesma). Nenhuma mudança de código-fonte.

**Files:**
- Modify: `tests/test_chunker.py` (adicionar 3 testes)

- [ ] **Step 1: Escrever os 3 testes novos**

Abrir `tests/test_chunker.py` e adicionar **no final do arquivo**:

```python
def test_segmenter_splits_on_markdown_headings():
    """ThematicSegmenter should split on '# Title' markers, producing
    one segment per heading with clean title text."""
    md = (
        "# First\n"
        "Intro line.\n"
        "\n"
        "## Second\n"
        "Body of second.\n"
        "\n"
        "### Third\n"
        "Body of third.\n"
    )

    segments = ThematicSegmenter.segment(md)

    titles = [s["title"] for s in segments]
    assert "# First" in titles
    assert "## Second" in titles
    assert "### Third" in titles
    # Bodies stay attached to their own heading
    second = next(s for s in segments if s["title"] == "## Second")
    assert "Body of second" in second["content"]
    assert "Body of third" not in second["content"]


def test_chunker_respects_section_boundaries_via_pipeline():
    """Chunks should not bleed content across sections — the public pipeline
    (segment → chunk_with_metadata) must keep each section's chunks isolated."""
    md = (
        "# Intro\n"
        + ("alpha " * 200)
        + "\n\n"
        + "# Details\n"
        + ("beta " * 200)
    )

    segments = ThematicSegmenter.segment(md)
    chunker = SmartChunker(max_tokens=100, overlap=10)

    intro_chunks = chunker.chunk_with_metadata(
        text=next(s for s in segments if s["title"] == "# Intro")["content"],
        source="test.html", section="# Intro",
    )
    details_chunks = chunker.chunk_with_metadata(
        text=next(s for s in segments if s["title"] == "# Details")["content"],
        source="test.html", section="# Details",
    )

    for c in intro_chunks:
        assert "beta" not in c["content"], "Intro leaked into beta content"
        assert c["section"] == "# Intro"
    for c in details_chunks:
        assert "alpha" not in c["content"], "Details leaked into alpha content"
        assert c["section"] == "# Details"


def test_chunker_applies_overlap_within_long_section():
    """When a single section exceeds max_tokens, the sliding-window overlap
    must kick in BUT only inside that section (tested in isolation)."""
    long_section = " ".join(f"tok{i}" for i in range(500))  # 500 tokens
    chunker = SmartChunker(max_tokens=100, overlap=20)

    chunks = chunker.chunk(long_section)

    assert len(chunks) >= 5
    for i in range(1, len(chunks)):
        prev_tail = " ".join(chunks[i - 1].split()[-20:])
        assert chunks[i].startswith(prev_tail), (
            f"Chunk {i} missing overlap prefix from chunk {i-1}"
        )
```

- [ ] **Step 2: Rodar os novos testes pra confirmar que passam com o código atual**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_chunker.py -v
```
Expected: **all pass** (inclusive os 3 novos). Se falhar, o contrato do chunker atual não é o que o spec assume — parar e investigar antes de seguir.

- [ ] **Step 3: Commit**

```bash
git add tests/test_chunker.py
git commit -m "test(chunker): lock section-awareness contract (3 tests)"
```

---

## Task 2: Executar ingestão end-to-end

**Objetivo:** Rodar `data/main.py` até o fim, produzir `data/chunks.jsonl` e popular Chroma `rag_chunks_pt`. Validar ordens de grandeza (chunks, libraries, noise filter).

**Files:** (nenhum arquivo alterado — é execução de pipeline + verificação)

- [ ] **Step 1: Limpar estado anterior (caso exista resíduo)**

Run:
```bash
rm -rf data/chroma_db data/chunks.jsonl data/main_run.log
```
Expected: dirs/arquivos removidos sem erro. (Se `data/chroma_db` não existir, `rm -rf` não falha.)

- [ ] **Step 2: Rodar ingestão**

Run:
```bash
.venv/Scripts/python.exe data/main.py 2>&1 | tee data/main_run.log
```
Expected:
- Linhas `[skip] pandas/en: skipped per config.SKIP_EN_FOR_LIBRARIES` pra todas as 4 libs.
- Linhas `[PT native-MT] <arquivo>.html` pra cada HTML em `data/<lib>/pt/` (130 no total).
- Linhas `[PT medium] <arquivo>.html` se existirem HTMLs em `data/medium/raw/`.
- Linha `[quality] dropped N noisy chunks`.
- Linha final `✔ INGESTÃO FINALIZADA — X chunks em PT` com X entre 3000 e 8000.
- Linha `[index] X chunks upserted into 'rag_chunks_pt'`.

Se quebrar por rate limit ou erro de modelo → parar, reportar erro. **Não** ir pro próximo step.

- [ ] **Step 3: Verificar integridade do `chunks.jsonl`**

Run:
```bash
.venv/Scripts/python.exe -c "
import json
from collections import Counter
rows = [json.loads(l) for l in open('data/chunks.jsonl', encoding='utf-8')]
libs = Counter(r.get('library','unknown') for r in rows)
srcs = Counter(r.get('source_type','?') for r in rows)
print('total:', len(rows))
print('by library:', dict(libs))
print('by source_type:', dict(srcs))
print('sample:', rows[0]['source'], '|', rows[0]['section'][:60])
"
```
Expected:
- `total`: entre 3000-8000
- `by library`: **as 4 libs aparecem** (`pandas`, `numpy`, `matplotlib`, `seaborn`); nenhuma com menos de 100 chunks
- `by source_type`: `official_docs` dominante, `medium_article` pode aparecer ou não

Se alguma lib tiver 0 chunks → investigar (provavelmente um bug no segmenter com o HTML específico).

- [ ] **Step 4: Verificar Chroma populado**

Run:
```bash
.venv/Scripts/python.exe -c "
import chromadb
from src import config
c = chromadb.PersistentClient(path=str(config.CHROMA_DIR)).get_collection(config.COLLECTION_NEW_PT)
print('chunks no Chroma:', c.count())
sample = c.peek(3)
print('sample ids:', sample['ids'])
print('sample lib:', [m.get('library') for m in sample['metadatas']])
"
```
Expected: `chunks no Chroma:` com número = `total` do step 3. `sample ids` mostra 3 ids do formato `<arquivo>_<section>_<N>`.

- [ ] **Step 5: Commit do chunks.jsonl + log**

Nota: `data/chroma_db/` é grande — deixar pro Onda 2 decidir se committa (Git LFS) ou gera localmente. Neste commit, só versionamos o jsonl e o log.

```bash
git add data/chunks.jsonl data/main_run.log
git commit -m "feat(ingestion): complete PT-only ingestion of 4 libs"
```

---

## Task 3: Novo módulo de query (`src/rag_query.py`)

**Objetivo:** Módulo callable via CLI e via import que responde perguntas em PT usando a coleção `rag_chunks_pt`. Orquestra retrieve (top 15) → rerank (top 5) → prompt PT → Groq → resposta + citations.

**Files:**
- Create: `src/rag_query.py`
- Create: `tests/test_rag_query.py`

- [ ] **Step 1: Escrever teste `test_build_pt_prompt_format`**

Criar `tests/test_rag_query.py` com:

```python
"""Tests for rag_query orchestrator.

Integration-style tests (that touch Chroma/Groq) are skipped unless the
env var RAG_E2E=1 is set. Pure-logic tests (prompt shape, rerank math,
citation formatting) always run.
"""
from __future__ import annotations

import os
import pytest


def test_build_pt_prompt_format():
    from src.rag_query import build_pt_prompt

    chunks = [
        {"content": "DataFrame.merge combina tabelas.", "metadata": {
            "source": "pandas_merge.html", "section": "## Merge", "library": "pandas"}},
        {"content": "Use how='inner' para interseção.", "metadata": {
            "source": "pandas_merge.html", "section": "## Merge", "library": "pandas"}},
    ]
    prompt = build_pt_prompt("como fazer merge?", chunks)

    # Must be in PT with clear instruction to only use context
    assert "português" in prompt.lower() or "pt-br" in prompt.lower()
    assert "contexto" in prompt.lower()
    # Must include both chunks verbatim
    assert "DataFrame.merge combina tabelas." in prompt
    assert "Use how='inner' para interseção." in prompt
    # Must include the user question
    assert "como fazer merge?" in prompt
    # Must not leak the test scaffolding noise
    assert "<|system|>" not in prompt  # no Qwen-style markers


def test_format_citations_from_chunks():
    from src.rag_query import format_citations

    chunks = [
        {"metadata": {"source": "a.html", "section": "# X", "library": "pandas"}},
        {"metadata": {"source": "a.html", "section": "# X", "library": "pandas"}},  # dup
        {"metadata": {"source": "b.html", "section": "## Y", "library": "numpy"}},
    ]
    cites = format_citations(chunks)

    # Deduplicates by (source, section)
    assert len(cites) == 2
    assert {"source": "a.html", "section": "# X", "library": "pandas"} in cites
    assert {"source": "b.html", "section": "## Y", "library": "numpy"} in cites


@pytest.mark.skipif(os.environ.get("RAG_E2E") != "1",
                    reason="requires Chroma + Groq; set RAG_E2E=1")
def test_answer_end_to_end_returns_dict_with_expected_keys():
    from src.rag_query import answer

    result = answer("Como criar um DataFrame no pandas?")
    # Task 4 will add "variant" — use subset check so this test stays stable.
    assert {"answer", "citations", "retrieved_chunks"}.issubset(result.keys())
    assert isinstance(result["answer"], str) and result["answer"]
    assert isinstance(result["citations"], list)
    assert isinstance(result["retrieved_chunks"], list)
    assert len(result["retrieved_chunks"]) <= 5
```

- [ ] **Step 2: Rodar os testes de unidade pra confirmar que falham**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_rag_query.py -v -k "not e2e"
```
Expected: FAIL com `ModuleNotFoundError: No module named 'src.rag_query'`.

- [ ] **Step 3: Implementar `src/rag_query.py`**

Criar arquivo com:

```python
"""Query orchestrator: retrieve → rerank → Groq generation in PT-BR.

Exposes:
  - ``answer(question) -> {"answer", "citations", "retrieved_chunks"}``
    for programmatic use.
  - CLI: ``python -m src.rag_query "question"``

Baseline EN variant is added by Task 4 in the Wave 1 plan.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from src import config
from src.device import get_device


# ---------------------------------------------------------------------------
# Model loading (cached as module-level singletons — reused across queries)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Retrieval + reranking
# ---------------------------------------------------------------------------
def retrieve(question: str, top_k: int = config.TOP_K_RETRIEVE,
             collection_name: str = config.COLLECTION_NEW_PT) -> list[dict]:
    """Embed the query (with e5 'query:' prefix), retrieve top_k chunks."""
    embedder = _get_embedder()
    collection = _get_collection(collection_name)
    q_emb = embedder.encode(
        [f"query: {question}"], normalize_embeddings=True
    ).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=top_k)
    # Chroma returns parallel arrays inside 1-item outer lists
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return [{"content": d, "metadata": m} for d, m in zip(docs, metas)]


def rerank(question: str, chunks: list[dict],
           top_n: int = config.TOP_K_RERANK) -> list[dict]:
    """Cross-encode (question, chunk) pairs, keep top_n by score."""
    if not chunks:
        return []
    reranker = _get_reranker()
    pairs = [[question, c["content"]] for c in chunks]
    scores = reranker.predict(pairs).tolist()
    ranked = sorted(zip(scores, chunks), key=lambda t: -t[0])
    return [c for _, c in ranked[:top_n]]


# ---------------------------------------------------------------------------
# Prompt + citations
# ---------------------------------------------------------------------------
_PT_SYSTEM = (
    "Você é um assistente em PT-BR especializado em análise de dados com Python "
    "(pandas, numpy, matplotlib, seaborn). Responda apenas com base no CONTEXTO "
    "fornecido. Se a resposta não estiver no contexto, diga 'Não encontrei essa "
    "informação na documentação indexada.' Não invente APIs, parâmetros nem erros. "
    "Seja conciso e direto — tom didático para iniciantes. Quando citar método/classe, "
    "mantenha o nome em inglês (ex.: DataFrame.merge, np.array)."
)


def build_pt_prompt(question: str, chunks: list[dict]) -> str:
    """Assemble the PT-BR prompt with numbered context blocks."""
    blocks = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        header = f"[{i}] ({meta.get('library','?')} · {meta.get('section','?')})"
        blocks.append(f"{header}\n{c['content']}")
    contexto = "\n\n".join(blocks)
    return (
        f"{_PT_SYSTEM}\n\n"
        f"=== CONTEXTO ===\n{contexto}\n\n"
        f"=== PERGUNTA ===\n{question}\n\n"
        f"=== RESPOSTA (em português) ==="
    )


def format_citations(chunks: list[dict]) -> list[dict]:
    """Dedupe chunks by (source, section) and return citation records."""
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


# ---------------------------------------------------------------------------
# Generation via Groq
# ---------------------------------------------------------------------------
def generate_answer(prompt: str, model: str = config.GROQ_LLM_MODEL) -> str:
    """Call Groq chat completion (non-streaming) and return assistant text."""
    from groq import Groq  # lazy import — keeps module importable w/o the dep
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set — add it to .env")
    client = Groq(api_key=config.GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=800,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------
def answer(question: str,
           top_k: int = config.TOP_K_RETRIEVE,
           top_n: int = config.TOP_K_RERANK,
           collection_name: str = config.COLLECTION_NEW_PT,
           model: str = config.GROQ_LLM_MODEL) -> dict[str, Any]:
    """Full pipeline: retrieve → rerank → prompt → Groq → package."""
    retrieved = retrieve(question, top_k=top_k, collection_name=collection_name)
    reranked = rerank(question, retrieved, top_n=top_n)
    prompt = build_pt_prompt(question, reranked)
    llm_text = generate_answer(prompt, model=model)
    return {
        "answer": llm_text,
        "citations": format_citations(reranked),
        "retrieved_chunks": reranked,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
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
    print("\n=== CITAÇÕES ===")
    for c in result["citations"]:
        print(f"  - {c['library']} · {c['section']} ({c['source']})")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
```

- [ ] **Step 4: Rodar testes unitários — devem passar**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_rag_query.py -v -k "not e2e"
```
Expected: 2 passed (`test_build_pt_prompt_format`, `test_format_citations_from_chunks`).

- [ ] **Step 5: Smoke test manual via CLI**

Run:
```bash
.venv/Scripts/python.exe -m src.rag_query "Como fazer merge entre dois DataFrames no pandas?"
```
Expected:
- Carrega modelos (pode demorar 10-30s no primeiro load).
- Imprime `=== RESPOSTA ===` seguido de parágrafo em PT mencionando `DataFrame.merge`, `pd.merge` ou `how='inner'`.
- Imprime `=== CITAÇÕES ===` com 1-5 linhas, pelo menos uma mencionando `pandas`.

Se a resposta for "Não encontrei essa informação na documentação indexada" pra uma pergunta básica como essa → retrieval tá ruim; pausar e investigar (provavelmente problema de embedding ou coleção vazia).

- [ ] **Step 6: Rodar teste E2E opcional**

Run:
```bash
RAG_E2E=1 .venv/Scripts/python.exe -m pytest tests/test_rag_query.py::test_answer_end_to_end_returns_dict_with_expected_keys -v
```
Expected: PASS. (No Windows PowerShell: `$env:RAG_E2E=1; .venv/Scripts/python.exe -m pytest ...`.)

- [ ] **Step 7: Commit**

```bash
git add src/rag_query.py tests/test_rag_query.py
git commit -m "feat(rag_query): Groq + e5 + mmarco PT pipeline with citations"
```

---

## Task 4: Modo baseline (EN) no `rag_query`

**Objetivo:** Mesmo módulo suporta rodada EN (baseline) pra comparação RAGAS. `answer(..., variant="baseline")` troca embedder pra `all-MiniLM-L6-v2`, reranker pra `ms-marco-MiniLM-L-6-v2`, coleção pra `rag_chunks_baseline_en`, e prompt pra inglês. LLM continua Groq 70B (o spec 2026-04-21 troca só stack, não LLM — a variação de LLM 70B vs 8B é 3ª rodada via `model=`).

**Files:**
- Modify: `src/rag_query.py` (adicionar `variant` param + helpers EN)
- Modify: `tests/test_rag_query.py` (+2 testes)

- [ ] **Step 1: Escrever testes primeiro**

Adicionar ao fim de `tests/test_rag_query.py`:

```python
def test_build_en_prompt_format():
    from src.rag_query import build_en_prompt

    chunks = [
        {"content": "DataFrame.merge joins tables.", "metadata": {
            "source": "pandas_merge.pdf", "section": "Merge", "library": "pandas"}},
    ]
    prompt = build_en_prompt("how to merge?", chunks)

    assert "english" in prompt.lower() or "in english" in prompt.lower()
    assert "context" in prompt.lower()
    assert "DataFrame.merge joins tables." in prompt
    assert "how to merge?" in prompt
    # Must NOT be in PT (sanity: no PT-specific stopwords in system instruction)
    assert "português" not in prompt.lower()


def test_variant_baseline_uses_en_stack(monkeypatch):
    """Calling answer(variant='baseline') must route through EN models and
    EN collection. We patch the internals to capture which names get used."""
    from src import rag_query as rq

    captured = {}

    def fake_embedder_loader(model_name):
        captured["embedder"] = model_name
        class E:
            def encode(self, texts, **kw):
                # Return unit vectors of correct dimension-shape-ish
                import numpy as np
                return np.zeros((len(texts), 4))
        return E()

    def fake_reranker_loader(model_name):
        captured["reranker"] = model_name
        class R:
            def predict(self, pairs):
                return [0.5 for _ in pairs]
        return R()

    def fake_collection_loader(name):
        captured["collection"] = name
        class C:
            def query(self, **kw):
                return {
                    "documents": [["doc1"]],
                    "metadatas": [[{"source": "x", "section": "y",
                                    "library": "pandas"}]],
                }
            name = "stub"
        return C()

    def fake_generate(prompt, model):
        captured["prompt"] = prompt
        captured["model"] = model
        return "stubbed answer"

    monkeypatch.setattr(rq, "_load_embedder", fake_embedder_loader)
    monkeypatch.setattr(rq, "_load_reranker", fake_reranker_loader)
    monkeypatch.setattr(rq, "_load_collection", fake_collection_loader)
    monkeypatch.setattr(rq, "generate_answer", fake_generate)
    # reset module-level caches so the loaders run with the baseline names
    monkeypatch.setattr(rq, "_embedder", None)
    monkeypatch.setattr(rq, "_reranker", None)
    monkeypatch.setattr(rq, "_collection", None)

    result = rq.answer("how to merge?", variant="baseline")

    from src import config
    assert captured["embedder"] == config.BASELINE_EMBEDDER
    assert captured["reranker"] == config.BASELINE_RERANKER
    assert captured["collection"] == config.COLLECTION_BASELINE_EN
    assert "english" in captured["prompt"].lower()
    assert result["answer"] == "stubbed answer"
```

- [ ] **Step 2: Rodar — devem falhar (variant não existe ainda, helpers não existem)**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_rag_query.py::test_build_en_prompt_format tests/test_rag_query.py::test_variant_baseline_uses_en_stack -v
```
Expected: FAIL — `build_en_prompt` não existe, `_load_embedder` não existe, `answer` não aceita `variant`.

- [ ] **Step 3: Refatorar `src/rag_query.py` pra suportar variant**

Abrir `src/rag_query.py` e:

**(a)** Substituir o bloco "Model loading" (do topo) por:

```python
# ---------------------------------------------------------------------------
# Model loading — factored into pure loaders so tests can monkeypatch
# ---------------------------------------------------------------------------
_embedder: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None
_collection = None
_loaded_for_variant: str | None = None


def _load_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=get_device())


def _load_reranker(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name, device=get_device())


def _load_collection(collection_name: str):
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_collection(collection_name)


def _stack_for_variant(variant: str) -> tuple[str, str, str]:
    """Return (embedder_model, reranker_model, collection_name) for a variant."""
    if variant == "new":
        return (config.EMBEDDER_MODEL, config.RERANKER_MODEL,
                config.COLLECTION_NEW_PT)
    if variant == "baseline":
        return (config.BASELINE_EMBEDDER, config.BASELINE_RERANKER,
                config.COLLECTION_BASELINE_EN)
    raise ValueError(f"unknown variant {variant!r}; expected 'new' or 'baseline'")


def _ensure_loaded(variant: str) -> None:
    """Lazy-load models/collection once per variant; swap them when variant changes."""
    global _embedder, _reranker, _collection, _loaded_for_variant
    if _loaded_for_variant == variant and _embedder and _reranker and _collection:
        return
    emb_name, rr_name, coll_name = _stack_for_variant(variant)
    _embedder = _load_embedder(emb_name)
    _reranker = _load_reranker(rr_name)
    _collection = _load_collection(coll_name)
    _loaded_for_variant = variant
```

**(b)** Remover as funções `_get_embedder`, `_get_reranker`, `_get_collection` (substituídas por `_ensure_loaded`).

**(c)** Modificar `retrieve()` e `rerank()` pra usar os módulos-globais direto (em vez de `_get_*()`):

```python
def retrieve(question: str, top_k: int = config.TOP_K_RETRIEVE,
             variant: str = "new",
             query_prefix: str = "query: ") -> list[dict]:
    _ensure_loaded(variant)
    q_emb = _embedder.encode(
        [f"{query_prefix}{question}"], normalize_embeddings=True
    ).tolist()
    res = _collection.query(query_embeddings=q_emb, n_results=top_k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return [{"content": d, "metadata": m} for d, m in zip(docs, metas)]


def rerank(question: str, chunks: list[dict],
           top_n: int = config.TOP_K_RERANK,
           variant: str = "new") -> list[dict]:
    if not chunks:
        return []
    _ensure_loaded(variant)
    pairs = [[question, c["content"]] for c in chunks]
    scores = _reranker.predict(pairs).tolist()
    ranked = sorted(zip(scores, chunks), key=lambda t: -t[0])
    return [c for _, c in ranked[:top_n]]
```

**(d)** Adicionar prompt EN ao lado do PT:

```python
_EN_SYSTEM = (
    "You are an assistant specialized in Python data analysis (pandas, numpy, "
    "matplotlib, seaborn). Answer ONLY from the CONTEXT provided. If the "
    "answer is not in the context, say 'I could not find that in the indexed "
    "documentation.' Do not invent APIs, parameters, or errors. Be concise "
    "and direct — beginner-friendly tone. Answer in English."
)


def build_en_prompt(question: str, chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        header = f"[{i}] ({meta.get('library','?')} · {meta.get('section','?')})"
        blocks.append(f"{header}\n{c['content']}")
    contexto = "\n\n".join(blocks)
    return (
        f"{_EN_SYSTEM}\n\n"
        f"=== CONTEXT ===\n{contexto}\n\n"
        f"=== QUESTION ===\n{question}\n\n"
        f"=== ANSWER (in English) ==="
    )
```

**(e)** Atualizar `answer()`:

```python
def answer(question: str,
           variant: str = "new",
           top_k: int = config.TOP_K_RETRIEVE,
           top_n: int = config.TOP_K_RERANK,
           model: str = config.GROQ_LLM_MODEL) -> dict[str, Any]:
    """Full pipeline: retrieve → rerank → prompt → Groq → package.

    ``variant``:
      - ``"new"`` (default): PT pipeline (e5-small + mmarco + PT prompt, rag_chunks_pt).
      - ``"baseline"``: EN pipeline (MiniLM + ms-marco + EN prompt, rag_chunks_baseline_en).
    """
    retrieved = retrieve(question, top_k=top_k, variant=variant)
    reranked = rerank(question, retrieved, top_n=top_n, variant=variant)
    if variant == "new":
        prompt = build_pt_prompt(question, reranked)
    else:
        prompt = build_en_prompt(question, reranked)
    llm_text = generate_answer(prompt, model=model)
    return {
        "answer": llm_text,
        "citations": format_citations(reranked),
        "retrieved_chunks": reranked,
        "variant": variant,
    }
```

**(f)** Atualizar CLI:

```python
def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Pergunte ao RAG.")
    parser.add_argument("question")
    parser.add_argument("--variant", choices=["new", "baseline"], default="new")
    parser.add_argument("--top-k", type=int, default=config.TOP_K_RETRIEVE)
    parser.add_argument("--top-n", type=int, default=config.TOP_K_RERANK)
    parser.add_argument("--model", default=config.GROQ_LLM_MODEL)
    args = parser.parse_args(argv)

    result = answer(
        args.question, variant=args.variant,
        top_k=args.top_k, top_n=args.top_n, model=args.model,
    )
    header = "RESPOSTA" if args.variant == "new" else "ANSWER"
    cites_header = "CITAÇÕES" if args.variant == "new" else "CITATIONS"
    print(f"\n=== {header} ({args.variant}) ===")
    print(result["answer"])
    print(f"\n=== {cites_header} ===")
    for c in result["citations"]:
        print(f"  - {c['library']} · {c['section']} ({c['source']})")
    return 0
```

- [ ] **Step 4: Rodar todos os testes unitários**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_rag_query.py -v -k "not e2e"
```
Expected: 4 passed (os 2 antigos + 2 novos).

- [ ] **Step 5: Smoke test do baseline via CLI**

Pré-requisito: coleção `rag_chunks_baseline_en` existe (foi populada pelo commit `b4305f4`). Verificar:

```bash
.venv/Scripts/python.exe -c "
import chromadb
from src import config
c = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
colls = [x.name for x in c.list_collections()]
print('collections:', colls)
assert config.COLLECTION_BASELINE_EN in colls, f'faltou {config.COLLECTION_BASELINE_EN}'
"
```
Expected: imprime `collections: [...]` contendo `rag_chunks_baseline_en`. Se não estiver lá → rodar `.venv/Scripts/python.exe -m src.baseline_snapshot` antes do próximo step.

Smoke test:
```bash
.venv/Scripts/python.exe -m src.rag_query --variant baseline "How do I merge two DataFrames in pandas?"
```
Expected: imprime `=== ANSWER (baseline) ===` com texto em inglês + `=== CITATIONS ===` com 1-5 entradas.

- [ ] **Step 6: Smoke test comparativo PT vs baseline**

Pra confirmar que os dois modos realmente se comportam diferente:

```bash
.venv/Scripts/python.exe -m src.rag_query "Como fazer merge em pandas?"
.venv/Scripts/python.exe -m src.rag_query --variant baseline "How do I merge DataFrames in pandas?"
```
Expected: a primeira chamada responde em PT, a segunda em EN. Idealmente citam documentos diferentes (um da `data/<lib>/pt/`, outro dos PDFs EN originais).

- [ ] **Step 7: Commit**

```bash
git add src/rag_query.py tests/test_rag_query.py
git commit -m "feat(rag_query): add baseline EN variant for RAGAS comparison"
```

---

## Fim da Onda 1 — Checklist de saída

- [ ] `tests/test_chunker.py` tem 3 novos testes passando (section-awareness contract)
- [ ] `data/chunks.jsonl` existe com 3000+ chunks das 4 libs
- [ ] Chroma `rag_chunks_pt` populado com mesmo N de chunks
- [ ] Chroma `rag_chunks_baseline_en` preservado (do commit anterior)
- [ ] `src/rag_query.py` funciona via CLI em ambos os variants
- [ ] `tests/test_rag_query.py`: 4/4 unit tests passam
- [ ] `python -m src.rag_query "pergunta"` responde em PT com citações
- [ ] `python -m src.rag_query --variant baseline "question"` responde em EN
- [ ] Branch `claude/competent-mcclintock-7dc548` tem ≥ 3 commits novos (Task 1, Task 2, Tasks 3+4)

Quando tudo verde, abrir brainstorming pro **Plano 2 — Deploy backend** (FastAPI + HF Space + Git LFS + testes de `/retrieve` e `/health`).
