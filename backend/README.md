---
title: RAG PT-BR Backend
sdk: docker
app_port: 7860
---

Backend FastAPI para o assistente RAG PT-BR.

## Endpoints

- `GET /health` — liveness probe
- `POST /ask` — `{"question": str, "top_n": int = 5}` → `{"answer": str, "citations": [...]}`

## Secrets necessários (configurar no painel do HF Space)

- `GROQ_API_KEY`
- `CEREBRAS_API_KEY`

## Como fazer deploy

O `data/chroma_db/` não está no repositório git principal (é gerado localmente).
Para fazer deploy:

1. Rode a ingestão localmente: `python -m data.main`
2. Execute o script de deploy: `scripts/deploy_space.py`
   (este script faz upload de `backend/`, `src/` e `data/chroma_db/` para o HF Space)
3. O HF Space faz rebuild automático com o `data/chroma_db/` incluído

Veja `docs/superpowers/SETUP_HF_SPACE.md` para o checklist completo.
