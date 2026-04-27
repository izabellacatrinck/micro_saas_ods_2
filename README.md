# Data Analyst Assistant

Assistente RAG para analise de dados em Python, focado em `pandas`, `numpy`, `matplotlib` e `seaborn`. O projeto combina um backend FastAPI com retrieval sobre documentacao indexada e um frontend Next.js que envia perguntas para a API e exibe a resposta.

## Visao Geral

O sistema foi desenhado em duas camadas:

- `backend/`: API FastAPI que carrega ChromaDB, embedder, reranker e LLM fallback.
- `frontend/`: interface web em Next.js que conversa exclusivamente com o backend.

Fluxo de uma pergunta:

1. usuario abre o frontend em `localhost:3000`
2. frontend envia `POST /backend/ask`
3. Next.js faz rewrite dessa rota para o backend real
4. backend executa retrieve -> rerank -> prompt -> geracao
5. frontend recebe `answer` e `citations` e renderiza a conversa

## Arquitetura

### Backend

O backend esta em [backend/app.py] e expoe:

- `GET /health`: status da API e do indice vetorial
- `POST /ask`: endpoint principal consumido pelo frontend
- `POST /retrieve`: endpoint auxiliar de retrieval, hoje nao usado pelo frontend principal

O pipeline principal usa:

- `ChromaDB` como banco vetorial persistente
- `intfloat/multilingual-e5-small` para embeddings
- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` para rerank
- `Groq` como LLM principal
- `Cerebras` como fallback

Grande parte da logica compartilhada fica em `src/`, especialmente em:

- `src/config.py`: caminhos, modelos e configuracoes
- `src/rag_query.py`: retrieve, rerank, prompt e geracao
- `src/device.py`: escolha de device

### Frontend

O frontend esta em `app/` e usa:

- `Next.js 13`
- `React 18`
- `TypeScript`
- `Tailwind CSS`
- componentes utilitarios em `app/components/ui/`

O rewrite que liga frontend e backend esta em [app/next.config.js]. Em ambiente local, se `HF_SPACE_URL` nao estiver definida, ele usa `http://127.0.0.1:7860`.

## Estrutura do Projeto

```text
micro_saas_ods_2/
|-- app/                    # Frontend Next.js
|   |-- app/                # App Router
|   |   |-- globals.css     # Tema global
|   |   |-- layout.tsx      # Metadata e shell base
|   |   `-- page.tsx        # Tela principal do assistente
|   |-- components/ui/      # Button, badge, collapsible, etc.
|   |-- lib/                # Helpers do frontend
|   |-- package.json
|   `-- next.config.js      # Rewrite para o backend
|
|-- backend/                # API FastAPI
|   |-- app.py              # Endpoints /health, /ask e /retrieve
|   |-- requirements.txt
|   `-- README.md
|
|-- src/                    # Logica Python compartilhada
|   |-- config.py
|   |-- device.py
|   |-- glossary_repair.py
|   |-- html_extractor.py
|   |-- indexer.py
|   |-- medium_extractor.py
|   `-- rag_query.py
|
|-- data/                   # Dados, chunks e indice vetorial local
|-- tests/                  # Testes Python
|-- docs/                   # Documentacao de specs, handoff e planos
|-- pyproject.toml          # Dependencias Python
`-- .env                    # Variaveis de ambiente locais
```

## Variaveis de Ambiente

O projeto usa um `.env` na raiz.

Minimo necessario para rodar o backend:

```env
GROQ_API_KEY=...
```

Opcional:

```env
CEREBRAS_API_KEY=...
HF_SPACE_URL=https://seu-space.hf.space
```

Observacoes:

- para rodar localmente, `HF_SPACE_URL` nao e obrigatoria se o backend estiver em `localhost:7860`
- o frontend usa o rewrite do `next.config.js`, nao precisa de chamada direta ao modelo

## Como Instalar

### 1. Dependencias Python

Na raiz do projeto:

```bash
uv sync
```

### 2. Dependencias do frontend

Na pasta `app/`:

```bash
cd app
npm install
cd ..
```

### 3. Preparar o indice vetorial

Se `data/chroma_db/` ainda nao existir, rode a ingestao:

```bash
uv run python data/main.py
```

Esse passo pode demorar, porque ele processa o corpus e popula o ChromaDB.

## Como Rodar Backend e Frontend Juntos

Abra dois terminais.

### Terminal 1: backend

Na raiz do projeto:

```bash
uv run uvicorn backend.app:app --host 0.0.0.0 --port 7860 --reload
```

URLs:

- API: [http://localhost:7860](http://localhost:7860)
- docs: [http://localhost:7860/docs](http://localhost:7860/docs)
- health: [http://localhost:7860/health](http://localhost:7860/health)

### Terminal 2: frontend

Na pasta `app/`:

```bash
cd app
npm run dev
```

URL:

- frontend: [http://localhost:3000](http://localhost:3000)

### O que acontece quando voce testa

1. o frontend abre em `localhost:3000`
2. a pergunta enviada vai para `/backend/ask`
3. o rewrite do Next.js encaminha para `http://127.0.0.1:7860/ask`
4. o backend devolve a resposta e as citacoes

## Como Testar Rapidamente

### Backend

Saude da API:

```bash
curl http://localhost:7860/health
```

Pergunta manual:

```bash
curl -X POST http://localhost:7860/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"Como fazer merge em pandas?\"}"
```

### Frontend

Abra:

- [http://localhost:3000](http://localhost:3000)

Digite uma pergunta como:

- `Explique quando usar merge, join e concat no pandas.`

## Testes

### Backend / Python

Na raiz:

```bash
uv run pytest
```

### Frontend

Na pasta `app/`:

```bash
npm run build
```

`npm run lint` pode pedir inicializacao do ESLint do Next se isso ainda nao tiver sido configurado no projeto.

## Deploy

### Backend

O backend foi pensado para deploy em Hugging Face Space com Docker. Os arquivos principais para isso sao:

- `backend/requirements.txt`
- `Dockerfile`
- `scripts/deploy_space.py`
- `docs/superpowers/SETUP_HF_SPACE.md`

### Frontend

O frontend pode ser publicado na Vercel. Nesse caso, `HF_SPACE_URL` deve apontar para a URL publica do backend.

## Observacoes Importantes

- `run.bat` e `run.sh` nao sao necessarios para o projeto funcionar.
- a API principal usada pelo frontend e `/ask`.
- o endpoint `/retrieve` existe no backend, mas hoje nao e necessario para a tela principal.
- artefatos locais como `.next/`, `node_modules/` e logs de dev nao fazem parte da arquitetura do projeto.

## Estado Atual

Hoje o projeto esta organizado para:

- backend como unica camada que fala com o modelo
- frontend como cliente da API
- indice vetorial local para desenvolvimento
- deploy separado entre interface e servico de inferencia

Isso deixa a fronteira entre responsabilidade de frontend e backend mais simples e mais facil de manter.
