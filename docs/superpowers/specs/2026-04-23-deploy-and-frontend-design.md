# Design: Deploy, frontend e finalização do RAG PT-BR

**Data:** 2026-04-23
**Status:** Aguardando revisão do usuário
**Tipo:** Extensão do spec `2026-04-21-rag-pt-surgical-design.md` — fecha escopo com frontend, deploy e decisões finais de LLM/corpus.

## 1. Contexto

O spec de 2026-04-21 cobriu a otimização do pipeline RAG (embedder, reranker,
LLM, chunker, tradução, avaliação RAGAS). Durante a implementação daquele plano
(tasks 1-14 concluídas, 15-22 pendentes), surgiram duas decisões que **alteram
o desenho original**:

1. **Corpus 100% PT sem tradução por LLM.** Todos os HTMLs PT foram
   obtidos via Google Translate do usuário (130 arquivos: pandas 47, numpy 31,
   matplotlib 38, seaborn 14). Um módulo `src/glossary_repair.py` (19 testes
   passando) corrige termos de API mal-traduzidos (`matriz`→`array`,
   `transmissão`→`broadcasting`, `Série`→`Series`, etc.). **A tradução via LLM
   foi eliminada da ingestão** — `SKIP_EN_FOR_LIBRARIES` agora cobre as 4
   bibliotecas.
2. **Deploy definido: Vercel + Next.js.** Trabalho acadêmico ("micro-SaaS")
   precisa de deploy funcional real, não apenas demo local, com custo zero ou
   próximo de zero.

Este spec fecha os itens em aberto, ajusta decisões em função do deploy e
define o contrato entre frontend e backend.

## 2. Objetivo

Entregar um **micro-SaaS completo** demonstrável ao professor:

- Frontend Next.js na Vercel (chat com citações).
- Backend FastAPI num Hugging Face Space (retrieval Chroma + e5-small +
  mmarco reranker).
- LLM via Groq (Llama-3.3-70B, open weights) com fallback Cerebras.
- Ingestão offline no repositório (Chroma pré-populado é committed no repo do
  HF Space).
- Avaliação RAGAS + golden set + relatório comparativo.
- Custo: R$ 0 permanente dentro das cotas dos free tiers.

## 3. Arquitetura de deploy

```
┌──────────────────────────┐                    ┌────────────────────────────┐
│ Vercel (Next.js 14)      │                    │ Hugging Face Space         │
│                          │                    │   (CPU Basic, free tier)   │
│  app/page.tsx            │                    │                            │
│   chat UI (client)       │                    │  backend/app.py            │
│         │                │   POST /retrieve   │   FastAPI                  │
│         ▼                │ ─────────────────▶ │    /retrieve   ── ChromaDB │
│  app/api/ask/route.ts    │                    │    /health    ── e5-small  │
│   orchestrator           │ ◀──────────────── │    (warmup)   ── mmarco-rr │
│         │   chunks+cites │                    │                            │
│         ▼                │                    │  data/chroma_db/ (commit)  │
│  Groq API  (stream)      │                    │  Python 3.12, uv lock      │
│   llama-3.3-70b-versatile│                    │                            │
│   fallback: Cerebras     │                    │                            │
└──────────────────────────┘                    └────────────────────────────┘
```

**Fluxo de uma pergunta:**

1. Usuário digita pergunta no chat Next.js → POST `/api/ask` (route handler
   rodando em Vercel).
2. Route handler chama `POST {HF_SPACE_URL}/retrieve` com `{question, top_k:5}`.
3. HF Space retorna `{chunks: [...], citations: [...]}` — top-5 já rerankados.
4. Route handler monta prompt PT-BR com os chunks e chama Groq (ou Cerebras em
   fallback) em modo streaming.
5. Resposta streamada de volta ao cliente via Server-Sent Events ou Vercel AI
   SDK `streamText`.

**Por que essa divisão:**

- Vercel serverless não aguenta `torch` + Chroma + e5 + reranker (250MB zip
  limit, sem GPU, sem disco persistente).
- HF Space free tier tem 2 vCPU / 16GB RAM / 50GB disco persistente —
  confortável pra e5-small (117M) + mmarco (107M) + Chroma com ~5k chunks.
- A chave do Groq fica **apenas** no ambiente Vercel (nunca exposta ao
  cliente). O HF Space não precisa de chave nenhuma.
- Backend isolado facilita trocar o frontend no futuro sem tocar no RAG.

## 4. Decisões finais de stack

| Componente | Escolha | Justificativa |
|---|---|---|
| LLM generation | **Groq `llama-3.3-70b-versatile`** (default) + **Cerebras `llama-3.3-70b`** (fallback) | Ambos servem pesos abertos (Meta Llama), free tier generoso, latência baixa. Fallback cobre rate-limit e indisponibilidade. |
| Embedder | `intfloat/multilingual-e5-small` (mantido) | PT-BR capaz, 384d (busca rápida), 117M (cabe no HF Space), já integrado com `baseline` comparativo. |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (mantido) | Treinado em mMARCO (pt-br), 107M. Reranking efetivo em queries em PT. |
| Vector store | **ChromaDB local persistido no container do HF Space** | Zero custo, sem dependência de API externa, persiste entre restarts do Space via filesystem. |
| Chunking | **Section-aware** (novo) — quebra em headings markdown emitidos por trafilatura, com limite superior de 450 tokens | Docs técnicas têm estrutura hierárquica forte (headings `##`). Quebrar por seção respeita granularidade semântica. Mantém `CHUNK_MAX_TOKENS=450` como teto. |
| Frontend framework | **Next.js 14 (App Router) + TypeScript** | Exigência do projeto; AI SDK da Vercel simplifica streaming. |
| Estilização | **Tailwind CSS + shadcn/ui** | Setup rápido, componentes acessíveis, visual profissional pra demo. |
| LLM client lib (frontend) | **Vercel AI SDK** (`ai` + `@ai-sdk/groq`) | Streaming de tokens nativo, fallback orquestrável, integra com React Server Components. |
| Ingestion | Continua em `data/main.py`, rodado manualmente local | Artefatos (Chroma) são committed no repo do HF Space. |

## 5. Componentes novos e modificações

### 5.1 Chunking section-aware (modifica `src/chunker.py`)

`trafilatura` já extrai HTML como markdown com headings (`#`, `##`, `###`).
O chunker atual ignora estrutura e aplica janela fixa. Nova lógica:

1. Primeiro passe: quebrar texto por linhas de heading (regex `^#{1,6}\s+`).
   Cada seção vira um "bloco lógico" com título + conteúdo.
2. Segundo passe: se a seção passar de 450 tokens, aplicar janela deslizante
   de 450 tokens com overlap de 80 palavras (comportamento atual) **dentro da
   seção**.
3. Metadata adicional por chunk: `section_heading` (string do título da seção
   a que pertence). Ajuda o reranker e vira parte da citação no frontend.

**Impacto nos testes:** atualizar `tests/test_chunker.py` com fixtures de
markdown com headings. Adicionar 3-5 testes novos pra confirmar boundary
behavior (seção curta = 1 chunk, seção longa = N chunks, sem headings =
fallback pro comportamento atual).

### 5.2 Backend FastAPI (`backend/app.py`)

Novo módulo, expõe dois endpoints:

```python
# POST /retrieve
# body: {"question": str, "top_k": int = 5}
# response: {
#   "chunks": [
#     {"text": str, "metadata": {...}, "score": float}, ...
#   ],
#   "citations": [
#     {"library": str, "section": str, "source_type": str}, ...
#   ]
# }

# GET /health
# response: {"status": "ok", "chroma_count": int, "models_loaded": bool}
```

Implementação usa os módulos já existentes (`src/indexer.py` pra embedding,
reranker via sentence-transformers). Carrega modelos uma única vez no startup
do FastAPI (`@app.on_event("startup")`) pra evitar reload por request.

CORS configurado pra aceitar origem Vercel:
`allow_origins=[os.environ["VERCEL_URL"]]`.

### 5.3 Orchestrator Next.js (`app/api/ask/route.ts`)

Server-side route handler que:

1. Recebe `{question: string}` do cliente.
2. Chama HF Space `/retrieve` (fetch com timeout 20s).
3. Monta prompt PT-BR com template fixo (mesmo já definido no spec
   2026-04-21 seção 4.3, portado pra TypeScript).
4. Usa Vercel AI SDK `streamText({model: groq('llama-3.3-70b-versatile'), ...})`
   com fallback: se Groq retornar 429 ou 5xx, troca pra
   `cerebras('llama-3.3-70b')` e tenta de novo.
5. Retorna stream SSE pro cliente.

Variáveis de ambiente (Vercel dashboard):

- `GROQ_API_KEY` (secret)
- `CEREBRAS_API_KEY` (secret)
- `HF_SPACE_URL` (secret, ex: `https://username-ragpt.hf.space`)

### 5.4 Frontend (`app/page.tsx`)

Página única com chat minimalista:

- Input de pergunta + botão enviar
- Lista de mensagens (usuário + assistente)
- Cada resposta do assistente mostra: texto streamado + expansível de citações
  (biblioteca, seção do doc, snippet)
- Sem histórico de sessões, sem auth — um chat simples por sessão
- Estado gerenciado com `useChat()` do Vercel AI SDK
- Layout: header com nome do projeto + área de chat + input fixo no bottom

### 5.5 Repositório HF Space

Estrutura do repo que sobe pro Space:

```
backend/
  app.py                    # FastAPI
  requirements.txt          # fastapi, sentence-transformers, chromadb, etc.
  README.md                 # HF Space config (sdk: docker, app_port: 7860)
  Dockerfile                # python:3.12-slim + uv + copy backend + chroma
src/                        # copiado do repo principal (indexer, chunker, etc.)
  indexer.py
  config.py
  glossary_repair.py
  ...
data/
  chroma_db/                # pré-populado via ingestão local; commitado
  eval/
    golden.jsonl
    report.md
```

**Fluxo de atualização de conteúdo:** rodar `python data/main.py` localmente →
`git add data/chroma_db` → push pro repo do HF Space → Space faz rebuild
automático.

**Tamanho esperado do Chroma:** ~5k chunks × 384d × 4 bytes = ~8MB só
vetores; com metadata e índice, estimar 30-60MB. Bem abaixo do limite de 50GB
do Space.

## 6. Avaliação (sem mudanças de escopo vs spec anterior)

O spec 2026-04-21 já definiu golden set sintético + RAGAS + 3 execuções
comparativas. **Mantido integralmente.** Uma adição:

- **Nova variante de ablation:** "chunker fixo (atual) vs section-aware
  (novo)" — executar 4 rodadas em vez de 3:
  1. Baseline (EN + Qwen local)
  2. Novo com fixed chunking
  3. Novo com section-aware chunking
  4. Novo com `llama-3.1-8b-instant` (trade-off qualidade vs. quota)

Isso deixa a ablation mais rica e justifica a mudança no chunker com números.

## 7. Fora de escopo (YAGNI)

- Autenticação / contas de usuário
- Histórico de conversas persistido
- Múltiplas sessões / multi-tenant
- Rate limiting próprio (Vercel + Groq já fazem)
- Dark mode / customização visual além do básico
- Caching de respostas / resultados de retrieval
- Deploy de ingestão serverless (ingestão continua offline/manual)
- Monitoramento / observability (logs básicos da Vercel já atendem)
- PWA, mobile-first específico (responsivo básico basta)
- Internacionalização do frontend (PT-BR fixo)

## 8. Critérios de sucesso

O projeto está entregue quando:

1. **Local:** `uv run python data/main.py` regenera Chroma em PT sem erros,
   com ~5k chunks e metadata completa (`language`, `source_type`, `library`,
   `section_heading`).
2. **Backend:** HF Space expõe `https://<user>-<name>.hf.space/health` retornando
   `200 {"status":"ok",...}` e `/retrieve` retornando top-5 chunks rerankados
   em < 3s em segunda chamada (primeira inclui cold start).
3. **Frontend:** URL Vercel pública responde pergunta "Como fazer merge em
   pandas?" com resposta em PT, streaming, + citações clicáveis com origem.
4. **Avaliação:** `data/eval/report.md` contém tabela RAGAS das 4 rodadas
   (baseline, novo-fixed, novo-section, novo-8b) e conclusão curta.
5. **Custo real medido:** $0.00 em Vercel + HF Space + Groq durante uma
   semana de uso moderado.
6. **README atualizado** com arquitetura, instruções de run local, link
   pra demo pública.

## 9. Riscos e mitigações

| Risco | Impacto | Mitigação |
|---|---|---|
| HF Space dorme (48h inativo) → cold start ~30s no dia da demo | Professor vê loading inicial longo | Chamar `/health` 5 min antes da demo pra esquentar; documentar isso no README como "warmup step" |
| Groq rate limit 429 em demo com várias perguntas seguidas | Fallback não ativa / lentidão | Fallback Cerebras com retry imediato em 429/5xx; se ambos falharem, mostrar mensagem clara "serviço de LLM temporariamente indisponível" |
| Chroma committed cresce além de 100MB → GitHub rejeita push | Deploy quebra | Git LFS configurado no repo do Space desde o primeiro commit (tanto `.sqlite3` quanto `.bin` do Chroma) |
| CORS mal configurado entre Vercel preview URLs e HF Space | Preview deploys não funcionam | `allow_origins` aceita regex `https://*-<team>.vercel.app` + URL de produção |
| Key Groq vazada no frontend por engano | Quota/billing risk | Chamada ao Groq apenas via `route.ts` (server-side); nenhuma env `NEXT_PUBLIC_*` para chaves |
| Chunking section-aware regride qualidade vs fixed | Tempo perdido | Ablation explícita no RAGAS valida a decisão com números antes de finalizar |
| HF Space evicted durante avaliação | Demo quebra | Plano B: rodar backend local (`uvicorn backend.app:app`) e apontar Vercel pra `ngrok` — documentar passo no README |

## 10. Plano de execução (alto nível)

Este spec vai virar um plano detalhado via `superpowers:writing-plans`. Em
linhas gerais, a sequência de tasks será:

1. **Fechar ingestão PT** (continuação das tasks 15h/15 do plano
   2026-04-21) — rodar `data/main.py` e validar Chroma.
2. **Chunker section-aware** (modificação de `src/chunker.py` + testes).
3. **Backend FastAPI** (`backend/app.py` + Dockerfile + README do HF Space).
4. **Deploy inicial no HF Space** (Chroma commitado via Git LFS, verificar
   `/health` e `/retrieve`).
5. **RAG query module refactor** (substituir `rag_qwen.py` antigo por versão
   compatível que use Groq + retrieval via backend local quando rodando de
   `scripts/`, e via HTTP quando rodando via frontend).
6. **Next.js scaffold** (App Router + Tailwind + shadcn).
7. **Orchestrator route handler** com fallback Groq→Cerebras.
8. **Chat UI** com streaming + citações.
9. **Deploy na Vercel** (env vars, teste em produção).
10. **Golden set + RAGAS + 4 rodadas + report.md** (tasks 18-21 do plano
    2026-04-21, adaptadas pra rodar contra o backend local ou via HF Space).
11. **README final** com arquitetura, links, instruções.

## 11. Próximos passos

1. Usuário revisa este spec e o diff conceitual em relação ao spec anterior.
2. Invocar `superpowers:writing-plans` pra gerar plano detalhado em
   `docs/superpowers/plans/2026-04-23-deploy-and-frontend.md`.
3. Executar plano via `superpowers:subagent-driven-development` (mesma sessão).
