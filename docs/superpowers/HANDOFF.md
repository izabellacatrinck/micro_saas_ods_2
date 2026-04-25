# Handoff — RAG PT-BR para análise de dados em Python

**Última atualização:** 2026-04-25
**Branch ativa:** `claude/competent-mcclintock-7dc548`
**Worktree path:** `.claude/worktrees/competent-mcclintock-7dc548`
**Onda 1 fechada:** ✅ (deliverable `python -m src.rag_query "pergunta"` funcional)

Este documento é o ponto de entrada pra qualquer agente (humano ou LLM) que
pegar o projeto daqui pra frente. Leia **nesta ordem**.

---

## 1. O que é o projeto

Micro-SaaS acadêmico: assistente RAG em PT-BR para bibliotecas de análise de
dados em Python (pandas, numpy, matplotlib, seaborn). Objetivo da entrega é
demonstrar o sistema funcionando ao professor, com:

- Frontend Next.js 14 (App Router) deployado na Vercel
- Backend FastAPI num Hugging Face Space (free tier CPU)
- LLM: Groq `llama-3.3-70b-versatile` (open weights) + fallback Cerebras
- Vector store: ChromaDB local (persistido no container do HF Space)
- Embedder: `intfloat/multilingual-e5-small`
- Reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- Avaliação: RAGAS + golden set PT sintético + relatório comparativo

Custo-alvo: **R$ 0** dentro dos free tiers.

---

## 2. Documentos canônicos (leia antes de mexer em código)

### Specs (decisões arquiteturais)

| Arquivo | Status | Conteúdo |
|---|---|---|
| `docs/superpowers/specs/2026-04-21-rag-pt-surgical-design.md` | Aprovado, parcialmente implementado | Decisões originais do pipeline RAG (embedder, reranker, chunker, tradução, avaliação). Tasks 1-14 executadas; 15-22 pendentes. |
| `docs/superpowers/specs/2026-04-23-deploy-and-frontend-design.md` | **Aprovado, vigente** | **Extensão do anterior.** Adiciona deploy Vercel + HF Space, frontend Next.js, LLM fallback Cerebras. Elimina tradução LLM da ingestão (agora 100% PT via `glossary_repair`). |

### Planos (roteiro de execução)

O projeto está decomposto em **3 ondas**, cada uma com seu próprio plano.
Executar na ordem: 1 → 2 → 3.

| Plano | Status | Entregável |
|---|---|---|
| `docs/superpowers/plans/2026-04-21-rag-pt-surgical.md` | Tasks 1-14 ✅ / 15-22 ⏳ | Plano original do spec 2026-04-21. Tasks 15-22 foram **redistribuídas** entre as ondas novas abaixo. |
| `docs/superpowers/plans/2026-04-23-wave1-backend-rag.md` | ✅ **CONCLUÍDA 2026-04-25** | CLI `python -m src.rag_query "pergunta"` funcional. 4 tasks executadas via `subagent-driven-development`. Commits: `4dcef42` (Task 1), `9180e08` (Task 2 — pré-existia), `c767e44`+`87b020b` (Task 3), `0c40eeb` (Task 4). 62 tests passing. |
| *(a criar)* `docs/superpowers/plans/2026-04-XX-wave2-deploy-backend.md` | **⏳ PRÓXIMA** | FastAPI + HF Space + Git LFS. Criar via `superpowers:brainstorming` → `writing-plans` → `subagent-driven-development`. |
| *(a criar)* `docs/superpowers/plans/2026-04-XX-wave3-frontend-eval.md` | Não criado ainda | Next.js + Vercel + RAGAS + report + README. Criar DEPOIS de Onda 2 concluída. |

---

## 3. Como executar os planos

Cada plano usa **superpowers:subagent-driven-development**: o agente
controlador dispara um subagent fresh por task, com revisão spec-compliance +
code-quality entre cada uma. Isso mantém o contexto limpo.

Comandos para começar (dentro do worktree):

```bash
# Garantir que .venv existe e dependências estão instaladas
uv sync

# Ler o plano e executar
# (dentro de uma sessão Claude, use a skill superpowers:subagent-driven-development
#  apontando para docs/superpowers/plans/2026-04-23-wave1-backend-rag.md)
```

---

## 4. Estado atual do código (2026-04-25)

### O que já está pronto e testado

- `src/config.py` — paths, coleções Chroma, modelos, Groq/Gemini config
- `src/device.py` — helper `get_device()` (cuda → cpu fallback)
- `src/html_extractor.py` — trafilatura em modo markdown
- `src/medium_extractor.py` — extração específica pra Medium
- `src/indexer.py` — embed com prefixo `passage: ` e upsert no Chroma
- `src/translator.py` — tradução Groq/Gemini com cache em disco *(não mais usado
  na ingestão, preservado pra eventual uso futuro)*
- `src/glossary_repair.py` — regex determinístico PT→EN pra termos de API
  (matriz→array, transmissão→broadcasting, Série→Series, etc). 19 testes.
- `src/baseline_snapshot.py` — copia coleção EN antiga pra `rag_chunks_baseline_en`.
- `data/main.py` — pipeline de ingestão end-to-end. Section-aware. **Rodado e
  validado na Onda 1 (Task 2)** — produz 2333 chunks (`data/chunks.jsonl`
  committado em `9180e08`); após dedup, 2298 vão pro Chroma `rag_chunks_pt`.
- `src/rag_query.py` (**novo, Onda 1**) — orquestrador retrieve→rerank→Groq.
  Suporta `variant="new"` (PT, default) e `variant="baseline"` (EN, p/ RAGAS).
  CLI: `python -m src.rag_query "pergunta"` ou `--variant baseline "question"`.
- `tests/test_chunker.py` — +3 testes de section-awareness (Task 1, `4dcef42`).
- `tests/test_rag_query.py` (**novo**) — 4 unit tests + 1 E2E gated por `RAG_E2E=1`.

### Corpus PT-BR

```
data/pandas/pt/     47 HTMLs (Google-translated do pandas.pydata.org)
data/numpy/pt/      31 HTMLs (Google-translated do numpy.org)
data/matplotlib/pt/ 38 HTMLs (Google-translated do matplotlib.org)
data/seaborn/pt/    14 HTMLs (Google-translated do seaborn.pydata.org)
data/medium/raw/    13 HTMLs (artigos nativos em PT curados manualmente)
```

### O que NÃO existe ainda

- `backend/` — FastAPI (Onda 2)
- `app/` (Next.js) — frontend (Onda 3)
- `data/eval/golden.jsonl` — golden set (Onda 3)
- `data/eval/report.md` — relatório RAGAS (Onda 3)

### Pendências conhecidas (follow-ups já agendados como tasks separadas via spawn_task)

1. **Bug de `library='unknown'` em `data/main.py`** — 323 dos 2333 chunks
   (~14%) caem em `library='unknown'`, incluindo HTMLs claramente do pandas
   (`dsintro.html`, `io.html`, `basics.html`). Não quebra retrieval, mas
   atrapalha filtragem por library. Fix: derivar library do parent dir
   (`data/<lib>/pt/` → `<lib>`).

2. **Coleção `rag_chunks_baseline_en` ausente neste worktree.** `data/chroma_db/`
   é gitignored — cada worktree precisa rebuild local. A Onda 1 só populou
   `rag_chunks_pt`; o baseline EN precisa ser regenerado antes do
   `--variant baseline` funcionar end-to-end e antes da Onda 3 (RAGAS comparativo).
   Os 4 unit tests da Task 4 já validam a lógica de routing via monkeypatch —
   só o smoke real depende da coleção.

---

## 5. Variáveis de ambiente necessárias

Em `.env` (gitignored) na raiz do worktree:

```
GROQ_API_KEY=gsk_...             # obrigatório, free tier em console.groq.com/keys
GOOGLE_API_KEY=AIza...           # opcional (era usado pra tradução — hoje não é chamado)
# Para Onda 2+:
# CEREBRAS_API_KEY=csk-...       # fallback LLM, free tier em cloud.cerebras.ai
```

---

## 6. Comandos de verificação rápida

```bash
# Rodar todos os testes atuais
.venv/Scripts/python.exe -m pytest tests/ -v

# Inspecionar corpus PT
ls data/*/pt/ | wc -l

# Verificar Chroma (depois da Onda 1 Task 2 executada)
.venv/Scripts/python.exe -c "import chromadb; from src import config; print([c.name for c in chromadb.PersistentClient(path=str(config.CHROMA_DIR)).list_collections()])"

# Rodar o RAG (depois da Onda 1 Task 3 concluída)
.venv/Scripts/python.exe -m src.rag_query "Como fazer merge em pandas?"
```

---

## 7. Decisões importantes já tomadas (não re-litigar)

Estas decisões foram brainstormed+aprovadas e mudar uma delas requer passar
pelo `superpowers:brainstorming` de novo:

1. **Não há tradução LLM na ingestão.** O corpus PT é 100% Google-translated +
   glossary_repair. `SKIP_EN_FOR_LIBRARIES` cobre as 4 libs.
2. **`llama-3.3-70b-versatile` via Groq** é o LLM default. `llama-3.1-8b-instant`
   é a 3ª rodada comparativa da avaliação.
3. **Embedder e reranker ficam como estão:** e5-small + mmarco. BGE-M3 foi
   considerado e rejeitado (ganho marginal, 4× mais pesado).
4. **ChromaDB fica como vector store** (não migramos pra Qdrant/Pinecone).
5. **Split Vercel + HF Space.** Vercel serverless não aguenta torch.
6. **Ablation chunker "section-aware vs fixed" foi dropada** — o chunker atual
   já é section-aware; comparar contra straw-man não traz insight.

---

## 8. Pontos de atenção (landmines conhecidas)

- **Windows line endings.** Git reclama `LF will be replaced by CRLF` em
  alguns arquivos — ignorar, não é bug.
- **UnicodeEncodeError cp1252.** Scripts que printam `→`, acentos etc.
  precisam de `sys.stdout.reconfigure(encoding="utf-8")` (ver
  `scripts/audit_pt_html.py`).
- **Venv path.** Usar `.venv/Scripts/python.exe` (Windows) ou
  `.venv/bin/python` (Unix). `uv run python ...` também funciona.
- **HF Space cold start ~30s** depois de 48h parado. No dia da demo, chamar
  `/health` ~5 min antes.
- **Commit do `data/chroma_db/`** só deve acontecer quando for subir pro repo
  do HF Space (via Git LFS). No repo principal, manter gitignored.

---

## 9. Próximo passo concreto

**Onda 1 fechada em 2026-04-25.** Deliverable verificado:

```bash
.venv/Scripts/python.exe -m src.rag_query "Como fazer merge entre dois DataFrames no pandas?"
# → resposta em PT mencionando pd.merge / how='inner' + 5 citações pandas
```

**Próximo:** abrir `superpowers:brainstorming` pra **Onda 2 — Deploy backend**
(FastAPI + HF Space + Git LFS). Rota: spec → plano → `subagent-driven-development`.

Decisões já tomadas pra Onda 2 (do spec `2026-04-23-deploy-and-frontend-design.md`,
não re-litigar):
- Backend: FastAPI no HF Space (free CPU tier).
- Vector store: ChromaDB committada via Git LFS no repo do HF Space.
- LLM call: Groq via SDK; fallback Cerebras quando Groq rate-limita.
- Endpoints mínimos: `/health`, `/ask` (POST `{question}` → `{answer, citations}`).

Antes de começar Onda 2, considerar resolver as 2 pendências conhecidas (seção 4)
ou empurrá-las pra depois — ambas têm chip de spawned-task na UI da sessão
anterior, e nenhuma é blocker pra Onda 2 (a coleção baseline só é exigida na
Onda 3).

### Desvios de spec aplicados na Onda 1 (não re-litigar)

- `src/rag_query.py:86-88` `rerank()` — usa guarda `raw.tolist() if hasattr(raw,
  "tolist") else list(raw)` pra que o monkeypatch do teste da Task 4 (que
  retorna `list` puro) e o `CrossEncoder.predict` real (ndarray) coexistam.
  Spec original só fazia `.tolist()`, o que quebrava o teste do próprio spec.
