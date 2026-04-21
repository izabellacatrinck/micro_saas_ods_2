# Design: RAG PT-BR para análise de dados — otimização cirúrgica

**Data:** 2026-04-21
**Status:** Aprovado em brainstorming, aguardando revisão do spec
**Tipo:** Projeto acadêmico — otimização cirúrgica do protótipo existente

## 1. Contexto

O projeto é um assistente RAG para programadores iniciantes em Python, focado em
análise de dados com `pandas`, `numpy`, `matplotlib` e `seaborn`. O objetivo é
responder perguntas e explicar erros com base em documentação real, em PT-BR.

Existe um protótipo funcional no repositório:

- `rag_qwen.py` — pipeline de consulta com embedder `all-MiniLM-L6-v2`,
  reranker `ms-marco-MiniLM-L-6-v2`, LLM `Qwen2.5-1.5B-Instruct` local e
  ChromaDB persistido em `data/chroma_db/`.
- `data/main.py` — pipeline de ingestão que extrai PDFs via `pdfplumber`,
  normaliza texto, segmenta por headings, faz chunking com overlap semântico
  e salva `chunks.jsonl`.
- Corpus atual: PDFs oficiais em inglês de NumPy, Matplotlib, Pandas e Seaborn.

**Limitações do protótipo:**

- Embedder e reranker são EN-only. Corpus e queries precisam ser em PT.
- LLM local em CPU é lento e limita a qualidade a 1.5B parâmetros.
- Prompt em inglês; público-alvo é brasileiro (iniciantes).
- Sem avaliação quantitativa da qualidade do RAG.
- Bugs no chunker: `noise_score` não filtra, lógica de overlap pode duplicar,
  heurística `is_heading` gera falsos positivos em linhas curtas.

## 2. Objetivo

Entregar uma versão do RAG em PT-BR, rodando em CPU 16GB com LLM via Groq API,
com um relatório de avaliação RAGAS comparando o protótipo original ao novo
pipeline. Escopo estritamente cirúrgico: trocar peças onde há ganho claro,
manter o resto.

## 3. Decisões de projeto

| Decisão | Escolha | Motivo |
|---|---|---|
| Abordagem | Otimização cirúrgica (não reescrita, não benchmark exaustivo) | Projeto acadêmico, evitar complexidade |
| Hardware | CPU 16GB RAM + GPU NVIDIA MX570 4GB VRAM; LLM via Groq API | GPU pequena é suficiente para embedder e reranker (< 500MB cada); LLM fica no Groq para evitar lentidão de inferência local |
| Idioma | PT-BR em tudo (UI, prompt, corpus) | Público-alvo brasileiro |
| Corpus | Híbrido: PDFs oficiais traduzidos + artigos Medium curados em PT | Rigor técnico + didática em PT |
| Avaliação | Golden set sintético via LLM + métricas RAGAS, 1 rodada comparativa | Suficiente para validar mudanças |
| Vector DB | ChromaDB (mantido) | Já funciona no porte; evita mudança desnecessária |

## 4. Mudanças cirúrgicas no pipeline

### 4.1 Trocas de componentes

| Componente | Atual | Novo |
|---|---|---|
| Embedder | `sentence-transformers/all-MiniLM-L6-v2` (EN, 80MB) | `intfloat/multilingual-e5-small` (PT capaz, 117M, ~450MB) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (EN) | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilíngue, 107M) |
| LLM | `Qwen/Qwen2.5-1.5B-Instruct` local (torch) | Groq API, modelo default `llama-3.3-70b-versatile` |
| Prompt | Inglês | PT-BR |

**Consequência técnica:** `transformers` e `AutoModelForCausalLM` saem do
runtime do `rag_qwen.py` (não há mais LLM local). `torch` continua presente
como dependência transitiva de `sentence-transformers` (usado pelo embedder e
pelo `CrossEncoder` do reranker) — porém sem carregar um modelo de 1.5B na
memória. O pacote `groq` entra via `pip`.

**Uso da GPU:** o embedder (`SentenceTransformer(..., device="cuda")`) e o
reranker (`CrossEncoder(..., device="cuda")`) devem tentar CUDA primeiro e
cair em CPU se indisponível. Um helper `get_device()` centraliza essa lógica
(`"cuda" if torch.cuda.is_available() else "cpu"`). Isso também se aplica ao
pipeline de ingestão quando gera embeddings em lote.

### 4.2 Fixes no chunker (`data/main.py`)

- **`noise_score` com filtro ativo:** descartar chunks com `noise_score > 0.5`
  em `deduplicate_chunks` ou função nova `filter_by_quality`.
- **Overlap real:** substituir a concatenação das 2 últimas sentenças do chunk
  anterior por um sliding window explícito — últimas 80 palavras do chunk
  anterior prepended (alinhado ao `overlap=80` já parametrizado em
  `SmartChunker`), sem risco de duplicar conteúdo já incluído no chunk atual.
- **Heurística de heading mais restritiva:** exigir ao menos um de
  `line.isupper()` ou `line.endswith(":")` E tamanho < 80; remover o ramo
  `len(line.split()) <= 6` sozinho (gera falsos positivos em qualquer linha
  curta de texto corrido).

### 4.3 Prompt PT-BR

Substituir o `build_prompt` atual por versão em PT com as mesmas regras
(responder só com o contexto, não inventar, ser conciso), formatado para
iniciantes em análise de dados.

## 5. Corpus em PT

### 5.1 Fontes

- **PDFs oficiais** (já no repo): `numpy_docs.pdf`, `matplotlib_tutorial.pdf`,
  `data/pandas/*.pdf`, `data/seaborn/*.pdf` — traduzidos automaticamente.
- **Artigos Medium em PT-BR**: 10-15 artigos curados manualmente pelo
  responsável do projeto, salvos em `data/medium/` como HTML ou markdown + JSON
  de metadata (URL, autor, título). A curadoria (lista de URLs a incluir) é
  pré-requisito para rodar a ingestão dos Medium; a seleção fica a critério
  do autor e não é automatizada.

### 5.2 Módulo novo: `data/translator.py`

Tradução chunk-a-chunk via Groq, com três guardrails:

1. **Glossário "não-traduzir"** no system prompt: `DataFrame`, `Series`,
   `ndarray`, `dtype`, `axis`, `shape`, `index`, `columns`, `groupby`, `pivot`,
   `merge`, `plot`, `figure`, `axes`, `subplot`, `KeyError`, `ValueError`,
   `TypeError`, `AttributeError`, `IndexError`, `NaN`, `None`, `True`, `False`.
2. **Blocos de código preservados:** regex detecta blocos (linhas iniciando com
   `>>>`, `In [N]:`, `...`, ou 4+ espaços de indentação consistente) e
   substitui por placeholders `<CODE_BLOCK_N>` antes de enviar ao Groq.
   Placeholders voltam ao texto original pós-tradução.
3. **Cache em disco:** `data/translations_cache.jsonl`, chaveado por
   `sha256(texto_original)`. Evita re-tradução em reruns.

Artigos Medium em PT não passam pelo tradutor — vão direto para o segmenter.

### 5.3 Pipeline de ingestão atualizado

```
PDFs EN        → extract → normalize → TRANSLATE → segment → chunk → dedupe → jsonl → embed → Chroma
Medium PT      → extract → normalize →             segment → chunk → dedupe → jsonl → embed → Chroma
```

### 5.4 Metadados por chunk

Acrescentar aos existentes:

- `language`: `"pt"` (sempre, pós-tradução)
- `original_lang`: `"en"` (PDFs oficiais) | `"pt"` (Medium)
- `source_type`: `"official_docs"` | `"medium_article"`
- `library`: `"pandas"` | `"numpy"` | `"matplotlib"` | `"seaborn"`

`library` é inferido do caminho/arquivo de origem.

## 6. Avaliação

### 6.1 Golden set sintético

- **Tamanho:** ~100 perguntas, distribuídas proporcionalmente ao número de
  chunks por biblioteca no corpus final (ex.: se pandas tiver 40% dos chunks,
  ~40 perguntas serão geradas a partir de chunks do pandas).
- **Geração:** amostrar ~100 chunks do corpus traduzido → para cada, pedir ao
  Groq (`llama-3.3-70b-versatile`) uma pergunta de iniciante em PT que o chunk
  responde, mais resposta curta (2-3 frases) e o trecho relevante.
- **Filtro mínimo:** descartar perguntas com <5 palavras ou que fazem
  referência a metadata (ex.: "qual o título da seção?").
- **Armazenamento:** `data/eval/golden.jsonl`, versionado no git.
- **Formato:** `{question, ground_truth_answer, ground_truth_contexts, library, chunk_id}`.

### 6.2 Métricas RAGAS

Via biblioteca `ragas` (avaliador LLM = Groq):

- **Retrieval:** `context_precision`, `context_recall`
- **Generation:** `faithfulness`, `answer_relevancy`
- **Sistema (medidas fora do RAGAS):** latência p50/p95 em segundos, total de
  tokens consumidos por query.

### 6.3 Comparação

Rodar no mesmo golden set:

1. **Baseline:** pipeline antigo (MiniLM EN + ms-marco EN + Qwen 1.5B local,
   prompt EN, corpus EN).
2. **Novo:** pipeline novo (e5-small + mmarco multilíngue + Groq, prompt PT,
   corpus PT).
3. **Variação de LLM no pipeline novo:** `llama-3.3-70b-versatile` vs
   `llama-3.1-8b-instant` — mesma stack, só trocando o LLM.

**Nota sobre índices:** o baseline precisa do índice EN antigo e o pipeline
novo do índice PT novo. Para permitir as duas rodadas sem re-indexar, cada
pipeline usará um nome de coleção distinto no mesmo `PersistentClient`:
`rag_chunks_baseline_en` (construído a partir do `chunks.jsonl` atual antes
da substituição) e `rag_chunks_pt` (construído pelo pipeline novo). As
perguntas do golden set serão traduzidas para EN apenas para a rodada
baseline (via Groq), já que o baseline tem corpus em inglês. A mesma
pergunta em PT e EN alimenta respectivamente as rodadas novas e baseline.

Total: 3 execuções no golden set. Saída em `data/eval/report.md` com tabela
markdown comparativa por métrica.

### 6.4 Módulos novos

- `src/eval/synth_qa.py` — gera e salva o golden set.
- `src/eval/run_eval.py` — executa um pipeline no golden set e produz uma
  linha de métricas; orquestrador rodará para as 3 variantes e montará o
  relatório final.

## 7. Estrutura final do repositório

```
rag_qwen.py                  # atualizado: Groq + e5 + mmarco + prompt PT
data/
  main.py                    # pipeline de ingestão atualizado (+ tradução)
  translator.py              # NOVO — tradução com glossário + cache
  translations_cache.jsonl   # NOVO — gerado em runtime (gitignored)
  medium/                    # NOVO — artigos curados em PT
  chunks.jsonl               # regenerado em PT
  chunks_with_embeddings.jsonl  # regenerado com e5-small
  chroma_db/                 # regenerado com e5-small
  eval/
    golden.jsonl             # NOVO — versionado
    report.md                # NOVO — gerado
src/eval/
  synth_qa.py                # NOVO
  run_eval.py                # NOVO
.env                         # GROQ_API_KEY (gitignored)
.env.example                 # template para quem clonar
docs/superpowers/specs/2026-04-21-rag-pt-surgical-design.md  # este spec
```

## 8. Fora de escopo (YAGNI acadêmico)

Considerados durante brainstorming e rejeitados deliberadamente:

- Migração para Qdrant ou outro vector DB
- Busca híbrida BM25 + dense
- Chunking hierárquico com parent expansion
- Grid search de variantes (embedders × rerankers × LLMs × chunking)
- Embedder BGE-M3 (maior, mais pesado; e5-small basta pro porte)
- Query rewriting / HyDE
- Reescrita da doc em PT didático (só tradução fiel com glossário)
- Deploy (será discutido em conversa separada após implementação)

## 9. Riscos e mitigações

| Risco | Impacto | Mitigação |
|---|---|---|
| Tradução Groq custa muito tempo/tokens | Ingestão lenta na primeira vez | Cache em disco; só paga uma vez por chunk |
| Tradução estraga termos técnicos | Retrieval e geração ficam ruins | Glossário "não-traduzir" no system prompt + testes manuais em amostra |
| Tradução estraga blocos de código | Exemplos inúteis | Placeholders `<CODE_BLOCK_N>` antes de enviar ao Groq |
| Golden set sintético gera perguntas triviais ou ruins | RAGAS superestima qualidade | Filtros mínimos + revisão manual de amostra (10-20 perguntas) |
| Rate limit do Groq em tradução de corpus grande | Pipeline quebra no meio | Backoff exponencial + retomar do cache |
| Key do Groq exposta | Chave compromete-se | `.env` já configurado localmente (gitignored); `.env.example` com placeholder. A chave exposta durante o brainstorming deve ser rotacionada em console.groq.com/keys — o `.env` guarda a versão nova |
| Groq passa a cobrar pelo uso | Custo inesperado no trabalho | Tier gratuito atende porte acadêmico; se aparecer cobrança, plano B é rodar LLM local quantizado na MX570 (Qwen 2.5 1.5B em fp16 cabe em ~3GB VRAM) |

## 10. Critérios de sucesso

O projeto é considerado entregue quando:

1. `rag_qwen.py` responde perguntas em PT usando Groq, com retrieval em
   corpus PT, em uma única execução (CPU 16GB, <10s por query típica).
2. `data/main.py` regenera todo o corpus em PT de ponta a ponta (PDFs
   traduzidos + Medium) e popula o ChromaDB com embeddings e5-small.
3. Golden set de ~100 perguntas está em `data/eval/golden.jsonl`, versionado.
4. `data/eval/report.md` contém a tabela comparativa RAGAS das 3 execuções
   (baseline, novo com 70B, novo com 8B) e uma conclusão curta.
5. `.env.example` documenta as variáveis necessárias; README atualizado
   com instruções de execução.

## 11. Próximos passos

1. Usuário revisa este spec.
2. Invocar `superpowers:writing-plans` para gerar plano de implementação
   detalhado em `docs/superpowers/plans/`.
3. Executar o plano em sessão separada com revisão em checkpoints.
4. Conversa separada sobre deploy após implementação concluída.
