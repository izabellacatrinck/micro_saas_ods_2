# Design: Wave 2 — Deploy Backend (FastAPI + HF Space)

**Data:** 2026-04-26
**Status:** Aprovado
**Onda:** 2 de 3
**Pré-requisito:** Wave 1 concluída (`python -m src.rag_query` funcional, 62 testes passando)

Este spec cobre exclusivamente a Wave 2: subir o backend FastAPI no Hugging Face
Space usando `huggingface_hub` para deploy. Wave 3 (Next.js + Vercel + RAGAS) é
spec separado.

---

## 1. Objetivo

Expor o pipeline RAG (retrieve → rerank → Groq/Cerebras) como uma API HTTP
pública no HF Space free tier, acessível pelo frontend em Wave 3.

Entregável: `POST https://<user>-rag-pt-backend.hf.space/ask` retorna resposta
PT-BR + citações em < 3s (segunda chamada, pós-warmup).

---

## 2. Arquitetura

```
┌─────────────────────────────────────────────────┐
│ Hugging Face Space (Docker, CPU Basic, free)    │
│                                                 │
│  backend/app.py  (FastAPI)                      │
│    GET  /health  → status + chroma_count        │
│    POST /ask     → answer + citations           │
│         │                                       │
│         ▼                                       │
│  src/indexer.py  ── e5-small (embed query)      │
│  ChromaDB        ── top-20 candidates           │
│  mmarco reranker ── top-5 reranked              │
│  Groq SDK        ── llama-3.3-70b-versatile     │
│    └─ fallback: Cerebras llama-3.3-70b          │
│                                                 │
│  data/chroma_db/ (uploaded via deploy script)   │
└─────────────────────────────────────────────────┘
         ▲
         │  scripts/deploy_space.py
         │  (huggingface_hub.upload_folder)
         │
   [dev machine]
```

**Decisões já tomadas (não re-litigar):**
- Backend faz o pipeline completo: retrieve + rerank + LLM call.
- Chaves Groq e Cerebras ficam nos secrets do HF Space (não na Vercel).
- Fallback Groq → Cerebras acontece no backend, transparente pro cliente.
- Deploy via `huggingface_hub.upload_folder` (não git do Space).
- `torch` CPU-only (HF Space free tier não tem GPU).

---

## 3. Arquivos novos

```
backend/
  app.py              # FastAPI app
  requirements.txt    # dependências Python do Space
  Dockerfile          # build do container
  README.md           # header de configuração do HF Space

scripts/
  deploy_space.py     # upload para o HF Space via huggingface_hub
  smoke_test.py       # valida /health e /ask contra a URL pública

docs/superpowers/
  SETUP_HF_SPACE.md   # checklist manual: criar conta, Space, token, secrets
```

Nenhum arquivo existente é modificado nesta wave.

---

## 4. Contrato da API

### `GET /health`

```json
200 OK
{
  "status": "ok",
  "chroma_count": 2298,
  "models_loaded": true
}
```

### `POST /ask`

**Request:**
```json
{"question": "Como fazer merge em pandas?", "top_k": 5}
```

**Response:**
```json
200 OK
{
  "answer": "Para fazer merge entre dois DataFrames...",
  "citations": [
    {"library": "pandas", "section": "Merge, join, concatenate", "source_type": "docs"},
    ...
  ]
}
```

**Erros:**
| Condição | Status | Body |
|---|---|---|
| `question` vazio ou ausente | 422 | FastAPI default validation |
| Modelos não carregados ainda | 503 | `{"detail": "models not loaded"}` |
| Groq e Cerebras ambos falham | 502 | `{"detail": "LLM unavailable"}` |
| 0 chunks recuperados | 200 | `{"answer": "Não encontrei informações relevantes sobre sua pergunta.", "citations": []}` |

---

## 5. Implementação

### 5.1 `backend/app.py`

```python
# Estrutura de alto nível — detalhes implementados na task
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
import os

# Estado global carregado no startup
_state = {"chroma": None, "embedder": None, "reranker": None, "ready": False}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # carrega modelos e abre Chroma uma única vez
    _state["embedder"] = ...   # SentenceTransformer("intfloat/multilingual-e5-small")
    _state["reranker"] = ...   # CrossEncoder("cross-encoder/mmarco-...")
    _state["chroma"]   = ...   # chromadb.PersistentClient(path="data/chroma_db")
    _state["ready"]    = True
    yield

app = FastAPI(lifespan=lifespan)

# CORS: aceita todas as origens por agora (Wave 3 restringe para URL Vercel)
# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)

class AskRequest(BaseModel):
    question: str
    top_k: int = 5

@app.get("/health")
def health(): ...

@app.post("/ask")
def ask(req: AskRequest): ...
    # 1. embed query (prefixo "query: ")
    # 2. chroma query → top-20
    # 3. rerank → top req.top_k
    # 4. montar prompt PT-BR (template do spec 2026-04-21 seção 4.3)
    # 5. try Groq llama-3.3-70b-versatile
    #    except (429, 5xx): Cerebras llama-3.3-70b
    #    except ambos: raise HTTPException(502)
    # 6. return {answer, citations}
```

### 5.2 `backend/requirements.txt`

```
fastapi
uvicorn[standard]
chromadb
sentence-transformers
torch --index-url https://download.pytorch.org/whl/cpu
huggingface_hub
groq
cerebras-cloud-sdk
```

### 5.3 `backend/Dockerfile`

```dockerfile
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY backend/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY backend/ ./backend/
COPY src/ ./src/
COPY data/chroma_db/ ./data/chroma_db/

EXPOSE 7860
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

### 5.4 `backend/README.md`

```yaml
---
title: RAG PT-BR Backend
sdk: docker
app_port: 7860
---
```

### 5.5 `scripts/deploy_space.py`

```python
from huggingface_hub import HfApi
import os

SPACE_ID = os.environ["HF_SPACE_ID"]  # ex: "username/rag-pt-backend"
TOKEN    = os.environ["HF_TOKEN"]     # token de escrita da conta HF

api = HfApi()
api.upload_folder(
    folder_path=".",
    repo_id=SPACE_ID,
    repo_type="space",
    token=TOKEN,
    allow_patterns=["backend/**", "src/**", "data/chroma_db/**"],
    ignore_patterns=["**/__pycache__/**", "**/*.pyc", ".venv/**"],
)
print("Deploy concluído. O Space vai fazer rebuild automaticamente.")
```

**Como executar:**
```bash
HF_SPACE_ID=username/rag-pt-backend HF_TOKEN=hf_... \
  .venv/Scripts/python.exe scripts/deploy_space.py
```

### 5.6 `scripts/smoke_test.py`

```python
import httpx, os, sys

BASE = os.environ["HF_SPACE_URL"]  # ex: https://username-rag-pt-backend.hf.space

r = httpx.get(f"{BASE}/health", timeout=60)
assert r.status_code == 200 and r.json()["models_loaded"], f"health falhou: {r.text}"

r = httpx.post(f"{BASE}/ask", json={"question": "Como fazer merge em pandas?"}, timeout=60)
assert r.status_code == 200
data = r.json()
assert data["answer"] and data["citations"], f"ask falhou: {data}"

print("Smoke test OK:", data["answer"][:80], "...")
```

---

## 6. Checklist manual de setup (`docs/superpowers/SETUP_HF_SPACE.md`)

Passos únicos, feitos uma vez pelo desenvolvedor:

```
1. Criar conta em huggingface.co
2. Criar novo Space:
   - Owner: seu usuário
   - Space name: rag-pt-backend
   - SDK: Docker
   - Visibility: Public (obrigatório no free tier)
3. Gerar token de escrita:
   - HF Settings → Access Tokens → New token (role: write)
   - Guardar como HF_TOKEN na sessão do shell
4. Adicionar secrets do Space (Space → Settings → Variables and secrets):
   - GROQ_API_KEY = gsk_...
   - CEREBRAS_API_KEY = csk-...
5. Rodar deploy script (raiz do projeto):
   HF_SPACE_ID=username/rag-pt-backend HF_TOKEN=hf_... \
     .venv/Scripts/python.exe scripts/deploy_space.py
6. Acompanhar logs de build na aba "Logs" do Space (~3-5 min no primeiro build)
7. Verificar:
   curl https://username-rag-pt-backend.hf.space/health
   → {"status":"ok","models_loaded":true}
8. Rodar smoke test:
   HF_SPACE_URL=https://username-rag-pt-backend.hf.space \
     .venv/Scripts/python.exe scripts/smoke_test.py
```

---

## 7. Fora de escopo nesta wave (YAGNI)

- Streaming de tokens (Wave 3 decide se frontend precisa disso)
- Autenticação do endpoint (Space público, aceitável pra demo acadêmica)
- CORS restrito (Wave 3 configura a origem Vercel)
- Rate limiting próprio (Groq já limita por chave)
- CI/CD automático (deploy manual via script é suficiente)
- Histórico de conversas / sessões
- Frontend (Wave 3)

---

## 8. Critérios de aceite da Wave 2

Wave 2 está concluída quando:

1. `GET /health` retorna `200 {"status":"ok","models_loaded":true}` na URL pública do HF Space.
2. `POST /ask {"question":"Como fazer merge em pandas?"}` retorna resposta PT-BR + ≥ 1 citação em < 3s (segunda chamada).
3. `scripts/smoke_test.py` passa sem erros contra a URL de produção.
4. `scripts/deploy_space.py` documentado no SETUP_HF_SPACE.md e testado uma vez.

---

## 9. Riscos e mitigações

| Risco | Mitigação |
|---|---|
| Cold start do Space (~30s após 48h parado) | Chamar `/health` 5 min antes da demo; documentar no README |
| Groq 429 em demo | Fallback Cerebras automático no backend |
| `data/chroma_db/` muito grande para upload | Esperado ~30-60MB — bem abaixo dos limites do `upload_folder` |
| Primeiro build do Docker lento (~10 min) | Executar deploy antes da demo; logs visíveis na aba do Space |
| Token HF exposto acidentalmente | `HF_TOKEN` só via variável de ambiente no shell; nunca commitado |
