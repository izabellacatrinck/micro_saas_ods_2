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
