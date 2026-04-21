# RAG PT-BR Surgical Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the existing RAG prototype to PT-BR end-to-end (corpus, prompt, responses) by swapping EN-only models for multilingual equivalents, moving LLM inference to Groq API, fixing known chunker bugs, adding Medium article sources, and validating everything with a RAGAS benchmark.

**Architecture:** Three independent pipelines sharing a config module: (1) ingestion — PDFs/Medium → translate → chunk → embed → ChromaDB, (2) query — question → hybrid retrieve → rerank → Groq LLM → answer, (3) evaluation — generate synthetic golden set → run variants → RAGAS metrics → markdown report. ChromaDB holds two collections side-by-side: `rag_chunks_baseline_en` (current EN pipeline snapshot) and `rag_chunks_pt` (new PT pipeline).

**Tech Stack:** Python 3.12, uv (deps), Groq API (LLM), `sentence-transformers` (embedder + reranker), ChromaDB (vector store), pdfplumber (PDF extraction), trafilatura (HTML extraction), RAGAS (evaluation), pytest (testing).

---

## Reference: Spec

See [docs/superpowers/specs/2026-04-21-rag-pt-surgical-design.md](../specs/2026-04-21-rag-pt-surgical-design.md) for the full design. This plan implements that spec exactly.

---

## File Structure

**New files:**
- `pyproject.toml` — uv dependency manifest
- `.env.example` — template with required env vars (GROQ_API_KEY)
- `.gitignore` — ignore `.env`, `__pycache__`, `data/translations_cache.jsonl`, `data/medium/raw/`, etc.
- `src/__init__.py`
- `src/device.py` — `get_device()` CUDA/CPU helper
- `src/config.py` — centralized model names, paths, hyperparameters
- `src/translator.py` — Groq-based translation with glossary + code block preservation + disk cache
- `src/medium_extractor.py` — HTML → clean text for curated Medium articles
- `src/baseline_snapshot.py` — one-shot script: copy current `rag_chunks` collection → `rag_chunks_baseline_en`
- `src/eval/__init__.py`
- `src/eval/synth_qa.py` — generate PT golden set from corpus chunks
- `src/eval/translate_questions.py` — translate PT questions to EN for baseline run
- `src/eval/run_eval.py` — run one pipeline variant on golden set, return RAGAS metrics
- `src/eval/experiment_runner.py` — orchestrate 3 variants, build `data/eval/report.md`
- `tests/__init__.py`
- `tests/test_chunker.py`
- `tests/test_translator.py`
- `tests/test_medium_extractor.py`
- `tests/test_device.py`
- `data/medium/raw/` — directory (gitkept) for manually curated HTML
- `data/medium/metadata.jsonl` — one line per curated article: `{url, title, author, filename}`
- `data/eval/golden.jsonl` — versioned synthetic Q&A
- `data/eval/golden_en.jsonl` — questions only, translated to EN (for baseline)
- `data/eval/report.md` — final comparative table

**Modified files:**
- `rag_qwen.py` — swap embedder/reranker, remove local LLM, add Groq, PT prompt, accept collection name parameter
- `data/main.py` — wire translator, add Medium ingestion, add metadata (language, library, source_type, original_lang), filter by noise_score, fix chunker bugs
- `README.md` — update with PT usage instructions and new commands

**Unchanged (deliberate):**
- ChromaDB as vector store
- Pipeline de ingestão's overall shape (extract → normalize → segment → chunk → dedupe → save)
- `data/pandas/`, `data/seaborn/`, `data/numpy_docs.pdf`, `data/matplotlib_tutorial.pdf` (inputs)

---

## Phase 1 — Foundation

### Task 1: Project setup (deps, env, gitignore, directories)

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `src/__init__.py`
- Create: `src/eval/__init__.py`
- Create: `tests/__init__.py`
- Create: `data/medium/raw/.gitkeep`
- Create: `data/eval/.gitkeep`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "rag-assistente-pt"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "groq>=0.13.0",
    "sentence-transformers>=3.0.0",
    "chromadb>=0.5.0",
    "pdfplumber>=0.11.0",
    "trafilatura>=1.12.0",
    "python-dotenv>=1.0.0",
    "ragas>=0.2.0",
    "langchain-groq>=0.2.0",
    "datasets>=3.0.0",
    "tiktoken>=0.8.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "pytest-mock>=3.14.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 2: Write `.env.example`**

```
# Get your key at https://console.groq.com/keys
GROQ_API_KEY=gsk_your_key_here

# Default LLM for generation and evaluation judge
GROQ_LLM_MODEL=llama-3.3-70b-versatile
```

- [ ] **Step 3: Write `.gitignore`**

```
# Python
__pycache__/
*.pyc
.pytest_cache/
.venv/

# Secrets
.env

# Data artifacts (regenerable or local-only)
data/translations_cache.jsonl
data/medium/raw/
data/eval/golden.jsonl.bak
data/chunks_with_embeddings.jsonl
output_texts/

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 4: Create empty package markers and directories**

```bash
mkdir -p src/eval tests data/medium/raw data/eval
touch src/__init__.py src/eval/__init__.py tests/__init__.py
touch data/medium/raw/.gitkeep data/eval/.gitkeep
```

- [ ] **Step 5: Install dependencies with uv**

Run: `uv sync`
Expected: all packages listed in `pyproject.toml` installed, `uv.lock` regenerated.

- [ ] **Step 6: Verify install**

Run: `uv run python -c "import groq, sentence_transformers, chromadb, ragas, trafilatura; print('ok')"`
Expected: `ok`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .env.example .gitignore src tests data/medium data/eval
git commit -m "chore: set up deps, env template, and package skeleton"
```

---

### Task 2: Device helper (CUDA fallback)

**Files:**
- Create: `src/device.py`
- Create: `tests/test_device.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_device.py`:

```python
from src.device import get_device


def test_get_device_returns_string():
    device = get_device()
    assert device in ("cuda", "cpu")


def test_get_device_matches_torch_availability():
    import torch
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert get_device() == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_device.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.device'`

- [ ] **Step 3: Implement `src/device.py`**

```python
import torch


def get_device() -> str:
    """Return 'cuda' if a CUDA GPU is available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_device.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add src/device.py tests/test_device.py
git commit -m "feat(device): add CUDA/CPU auto-detection helper"
```

---

### Task 3: Central config module

**Files:**
- Create: `src/config.py`

- [ ] **Step 1: Write `src/config.py`**

```python
"""Centralized configuration: model names, paths, hyperparameters.

Loaded once at import time. Values can be overridden by environment variables
(prefix GROQ_ or RAG_) for experimentation without code changes.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
MEDIUM_DIR = DATA_DIR / "medium"
EVAL_DIR = DATA_DIR / "eval"
TRANSLATIONS_CACHE = DATA_DIR / "translations_cache.jsonl"

# --- Chroma collections ---
COLLECTION_BASELINE_EN = "rag_chunks_baseline_en"
COLLECTION_NEW_PT = "rag_chunks_pt"

# --- Models ---
EMBEDDER_MODEL = os.environ.get("RAG_EMBEDDER", "intfloat/multilingual-e5-small")
RERANKER_MODEL = os.environ.get("RAG_RERANKER", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
BASELINE_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
BASELINE_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Groq ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_LLM_MODEL = os.environ.get("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
GROQ_LLM_FAST = "llama-3.1-8b-instant"

# --- Retrieval ---
TOP_K_RETRIEVE = 15
TOP_K_RERANK = 5

# --- Chunking ---
CHUNK_MAX_TOKENS = 450
CHUNK_OVERLAP_WORDS = 80
NOISE_THRESHOLD = 0.5

# --- Libraries tracked ---
LIBRARIES = ("pandas", "numpy", "matplotlib", "seaborn")
```

- [ ] **Step 2: Smoke-test import**

Run: `uv run python -c "from src.config import GROQ_LLM_MODEL, EMBEDDER_MODEL; print(GROQ_LLM_MODEL, EMBEDDER_MODEL)"`
Expected: `llama-3.3-70b-versatile intfloat/multilingual-e5-small`

- [ ] **Step 3: Commit**

```bash
git add src/config.py
git commit -m "feat(config): add centralized model/path/hyperparam config"
```

---

## Phase 2 — Chunker fixes

### Task 4: Fix `noise_score` filtering

**Files:**
- Modify: `data/main.py` (add `filter_by_quality` function, apply in `run_pipeline`)
- Create: `tests/test_chunker.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_chunker.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

from main import SmartChunker, filter_by_quality


def test_filter_by_quality_drops_noisy_chunks():
    chunks = [
        {"content": "In [1]: df.head()", "quality_flags": {"noise_score": 0.8}},
        {"content": "Good text " * 50, "quality_flags": {"noise_score": 0.1}},
        {"content": "borderline", "quality_flags": {"noise_score": 0.5}},
    ]

    kept = filter_by_quality(chunks, threshold=0.5)

    assert len(kept) == 1
    assert kept[0]["content"].startswith("Good text")


def test_filter_by_quality_keeps_all_if_threshold_high():
    chunks = [{"content": "x", "quality_flags": {"noise_score": 0.9}}]
    assert len(filter_by_quality(chunks, threshold=1.0)) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chunker.py::test_filter_by_quality_drops_noisy_chunks -v`
Expected: FAIL with `ImportError: cannot import name 'filter_by_quality' from 'main'`

- [ ] **Step 3: Add `filter_by_quality` to `data/main.py`**

Add this function just after `deduplicate_chunks` in `data/main.py`:

```python
def filter_by_quality(chunks: List[dict], threshold: float = 0.5) -> List[dict]:
    """Drop chunks whose noise_score is strictly greater than threshold."""
    return [
        c for c in chunks
        if c.get("quality_flags", {}).get("noise_score", 0.0) <= threshold
    ]
```

- [ ] **Step 4: Wire `filter_by_quality` into `run_pipeline`**

In `data/main.py`, inside `run_pipeline`, change the final cleaning block from:

```python
    # =========================
    # FINAL CLEANING
    # =========================
    all_chunks = deduplicate_chunks(all_chunks)
    save_chunks(all_chunks)
```

to:

```python
    # =========================
    # FINAL CLEANING
    # =========================
    all_chunks = deduplicate_chunks(all_chunks)
    before = len(all_chunks)
    all_chunks = filter_by_quality(all_chunks, threshold=0.5)
    print(f"[quality] dropped {before - len(all_chunks)} noisy chunks")
    save_chunks(all_chunks)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_chunker.py -v`
Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add data/main.py tests/test_chunker.py
git commit -m "fix(chunker): filter chunks by noise_score threshold (0.5)"
```

---

### Task 5: Fix overlap logic (sliding window of 80 words)

**Files:**
- Modify: `data/main.py` — `SmartChunker.chunk`
- Modify: `tests/test_chunker.py` — add overlap tests

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chunker.py`:

```python
def test_chunk_overlap_uses_sliding_window():
    """Each chunk after the first should begin with the last N words of the prev chunk."""
    text = " ".join([f"word{i}." for i in range(1000)])  # long text, many sentences
    chunker = SmartChunker(max_tokens=100, overlap=20)

    chunks = chunker.chunk(text)

    assert len(chunks) >= 2
    for i in range(1, len(chunks)):
        prev_words = chunks[i - 1].split()
        current_words = chunks[i].split()
        overlap_words = prev_words[-20:]
        # the first 20 words of the current chunk should equal the last 20 of previous
        assert current_words[:20] == overlap_words, (
            f"Chunk {i} does not start with sliding-window overlap of prev chunk"
        )


def test_chunk_no_overlap_on_first_chunk():
    text = " ".join([f"w{i}." for i in range(200)])
    chunker = SmartChunker(max_tokens=50, overlap=10)
    chunks = chunker.chunk(text)
    # the first chunk's first word should be original text's first word
    assert chunks[0].startswith("w0.")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chunker.py::test_chunk_overlap_uses_sliding_window -v`
Expected: FAIL — current logic concatenates last 2 *sentences* of previous chunk, not a fixed 20-word window.

- [ ] **Step 3: Rewrite `SmartChunker.chunk`**

Replace the whole `chunk` method in `data/main.py` with:

```python
    def chunk(self, text: str) -> List[str]:
        sentences = SentenceSplitter.split(text)

        # First pass: pack sentences into base chunks without overlap.
        base_chunks = []
        current = []
        current_len = 0

        def flush(buffer):
            if buffer:
                base_chunks.append(" ".join(buffer).strip())

        for sent in sentences:
            sent_len = len(sent.split())

            if current_len + sent_len <= self.max_tokens:
                current.append(sent)
                current_len += sent_len
            else:
                flush(current)
                current = [sent]
                current_len = sent_len

        flush(current)

        # Second pass: apply sliding-window overlap (last `overlap` words of prev
        # chunk prepended to next). Deterministic and free of duplication risk.
        if self.overlap <= 0 or len(base_chunks) <= 1:
            return base_chunks

        final = [base_chunks[0]]
        for i in range(1, len(base_chunks)):
            prev_words = base_chunks[i - 1].split()
            overlap_prefix = " ".join(prev_words[-self.overlap:])
            final.append(f"{overlap_prefix} {base_chunks[i]}".strip())

        return final
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chunker.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add data/main.py tests/test_chunker.py
git commit -m "fix(chunker): use explicit sliding-window overlap (80 words)"
```

---

### Task 6: Tighten `is_heading` heuristic

**Files:**
- Modify: `data/main.py` — `ThematicSegmenter.is_heading`
- Modify: `tests/test_chunker.py` — add heading tests

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chunker.py`:

```python
from main import ThematicSegmenter


def test_is_heading_accepts_uppercase_short_line():
    assert ThematicSegmenter.is_heading("GROUP BY OPERATIONS")


def test_is_heading_accepts_colon_terminated_short_line():
    assert ThematicSegmenter.is_heading("Introduction:")


def test_is_heading_rejects_short_body_text():
    # A 4-word lowercase sentence is NOT a heading
    assert not ThematicSegmenter.is_heading("This is a sentence.")


def test_is_heading_rejects_long_line():
    assert not ThematicSegmenter.is_heading("A" * 120)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chunker.py::test_is_heading_rejects_short_body_text -v`
Expected: FAIL — current heuristic flags any line with ≤6 words as heading.

- [ ] **Step 3: Rewrite `ThematicSegmenter.is_heading`**

Replace the method in `data/main.py` with:

```python
    @staticmethod
    def is_heading(line: str) -> bool:
        """A heading must be short AND visually marked as one.

        Marked-as-heading means either all-caps or ending with a colon. We drop the
        bare "few words" rule — it produced too many false positives on short body
        sentences.
        """
        if not line or len(line) >= 80:
            return False
        return line.isupper() or line.endswith(":")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chunker.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add data/main.py tests/test_chunker.py
git commit -m "fix(chunker): tighten is_heading to require uppercase or trailing colon"
```

---

## Phase 3 — Translator module

### Task 7: Code block preservation

**Files:**
- Create: `src/translator.py` (skeleton with code-block helpers)
- Create: `tests/test_translator.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_translator.py`:

```python
from src.translator import extract_code_blocks, restore_code_blocks


def test_extract_code_blocks_finds_repl_lines():
    text = "Explanation.\n>>> df.head()\n>>> df.shape\nMore text."
    stripped, blocks = extract_code_blocks(text)

    assert "<CODE_BLOCK_0>" in stripped
    assert len(blocks) == 1
    assert ">>> df.head()" in blocks[0]
    assert ">>> df.shape" in blocks[0]


def test_extract_code_blocks_finds_jupyter_cells():
    text = "Intro.\nIn [1]: x = 1\nIn [2]: print(x)\nOutro."
    stripped, blocks = extract_code_blocks(text)

    assert "<CODE_BLOCK_0>" in stripped
    assert "In [1]: x = 1" in blocks[0]


def test_extract_code_blocks_finds_indented_blocks():
    text = "Before.\n    x = 1\n    y = 2\n    z = x + y\nAfter."
    stripped, blocks = extract_code_blocks(text)

    assert "<CODE_BLOCK_0>" in stripped
    assert "x = 1" in blocks[0]


def test_restore_code_blocks_reinserts_original():
    original = "Text.\n>>> a = 1\nMore text."
    stripped, blocks = extract_code_blocks(original)
    restored = restore_code_blocks(stripped, blocks)

    assert restored == original


def test_extract_returns_text_unchanged_when_no_code():
    text = "Just regular prose without code."
    stripped, blocks = extract_code_blocks(text)

    assert stripped == text
    assert blocks == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_translator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.translator'`

- [ ] **Step 3: Implement the code-block helpers**

Create `src/translator.py`:

```python
"""Translator module: EN → PT via Groq, with code-block preservation,
technical glossary, and disk cache.

Public API:
    - extract_code_blocks(text) -> (stripped_text, [blocks])
    - restore_code_blocks(stripped, blocks) -> original_text
    - translate(text) -> pt_text           (added in Task 8)
    - translate_many(texts) -> [pt_text]   (added in Task 9)
"""
from __future__ import annotations

import re


# Match a run of consecutive "code-like" lines.
#   - starts with `>>>`
#   - starts with `In [N]:` or `Out[N]:`
#   - starts with 4+ spaces (indented block)
CODE_LINE = re.compile(
    r"^(?:>>>|\.\.\.|In \[\d+\]:|Out\[\d+\]:|\s{4,}\S).*$"
)


def extract_code_blocks(text: str) -> tuple[str, list[str]]:
    """Replace consecutive code lines with <CODE_BLOCK_N> placeholders.

    Returns (stripped_text, list_of_original_blocks).
    """
    lines = text.split("\n")
    out_lines: list[str] = []
    blocks: list[str] = []

    i = 0
    while i < len(lines):
        if CODE_LINE.match(lines[i]):
            j = i
            while j < len(lines) and (CODE_LINE.match(lines[j]) or lines[j].strip() == ""):
                j += 1
            # trim trailing blank line from the block, keep it in text
            while j > i and lines[j - 1].strip() == "":
                j -= 1

            block_lines = lines[i:j]
            idx = len(blocks)
            blocks.append("\n".join(block_lines))
            out_lines.append(f"<CODE_BLOCK_{idx}>")
            i = j
        else:
            out_lines.append(lines[i])
            i += 1

    return "\n".join(out_lines), blocks


def restore_code_blocks(stripped: str, blocks: list[str]) -> str:
    """Re-insert blocks into stripped text by replacing placeholders."""
    result = stripped
    for idx, block in enumerate(blocks):
        result = result.replace(f"<CODE_BLOCK_{idx}>", block)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_translator.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/translator.py tests/test_translator.py
git commit -m "feat(translator): preserve code blocks via placeholder substitution"
```

---

### Task 8: Glossary-aware Groq translation

**Files:**
- Modify: `src/translator.py` — add `translate()` function
- Modify: `tests/test_translator.py` — add translation tests (mocked Groq)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_translator.py`:

```python
from unittest.mock import MagicMock, patch

from src.translator import translate


@patch("src.translator.groq_client")
def test_translate_sends_glossary_and_code_preserved(mock_client):
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Isto é um DataFrame.\n<CODE_BLOCK_0>"))]
    )

    english = "This is a DataFrame.\n>>> df.head()"
    result = translate(english)

    # result should have original code block restored
    assert ">>> df.head()" in result
    # DataFrame kept in English
    assert "DataFrame" in result

    # glossary should have been included in system prompt
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    system_msg = call_kwargs["messages"][0]["content"]
    assert "DataFrame" in system_msg
    assert "NÃO traduza" in system_msg


@patch("src.translator.groq_client")
def test_translate_strips_preamble(mock_client):
    """If Groq echoes preambles like 'Tradução:', they get stripped."""
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Tradução:\nTexto traduzido."))]
    )
    result = translate("Some text.")
    assert result == "Texto traduzido."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_translator.py -v`
Expected: FAIL — `translate` not defined.

- [ ] **Step 3: Implement `translate()` in `src/translator.py`**

Append to `src/translator.py`:

```python
import os

from groq import Groq

from src import config

groq_client = Groq(api_key=config.GROQ_API_KEY) if config.GROQ_API_KEY else None

GLOSSARY_TERMS = [
    "DataFrame", "Series", "ndarray", "dtype", "axis", "shape",
    "index", "columns", "groupby", "pivot", "merge", "plot",
    "figure", "axes", "subplot", "KeyError", "ValueError",
    "TypeError", "AttributeError", "IndexError", "NaN", "None",
    "True", "False", "array", "list", "dict", "tuple",
    "NumPy", "pandas", "matplotlib", "seaborn",
]


SYSTEM_PROMPT = f"""Você é um tradutor técnico de inglês para português brasileiro.

REGRAS OBRIGATÓRIAS:
1. NÃO traduza os termos técnicos a seguir — mantenha-os em inglês exatamente como estão:
   {", ".join(GLOSSARY_TERMS)}
2. NÃO altere placeholders no formato <CODE_BLOCK_N> — devolva-os exatamente como recebidos.
3. NÃO traduza nomes de métodos ou funções Python (ex.: .groupby(), .plot(), .reset_index()).
4. Preserve a estrutura de parágrafos e quebras de linha do texto original.
5. Responda SOMENTE com o texto traduzido. Não adicione preâmbulos como "Tradução:", "Aqui está:", nem comentários.
"""

_PREAMBLE_RE = re.compile(r"^\s*(?:Tradução|Traducao|Aqui está|Aqui esta|Texto traduzido)\s*:?\s*\n", re.IGNORECASE)


def _strip_preamble(text: str) -> str:
    return _PREAMBLE_RE.sub("", text, count=1).strip()


def translate(text: str) -> str:
    """Translate a single passage EN → PT, preserving code blocks and glossary terms."""
    if groq_client is None:
        raise RuntimeError("GROQ_API_KEY not set — cannot translate")

    stripped, blocks = extract_code_blocks(text)

    response = groq_client.chat.completions.create(
        model=config.GROQ_LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": stripped},
        ],
        temperature=0.0,
        max_tokens=2000,
    )
    pt = response.choices[0].message.content
    pt = _strip_preamble(pt)

    return restore_code_blocks(pt, blocks)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_translator.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/translator.py tests/test_translator.py
git commit -m "feat(translator): add Groq-based EN→PT translation with glossary"
```

---

### Task 9: Disk-based translation cache + batch helper

**Files:**
- Modify: `src/translator.py` — add `translate_many()` with cache
- Modify: `tests/test_translator.py` — add cache tests

- [ ] **Step 1: Write the failing test**

Append to `tests/test_translator.py`:

```python
import json

from src.translator import translate_many, _cache_key


def test_cache_key_is_stable():
    a = _cache_key("hello world")
    b = _cache_key("hello world")
    c = _cache_key("different")
    assert a == b
    assert a != c


@patch("src.translator.translate")
def test_translate_many_uses_cache(mock_translate, tmp_path):
    mock_translate.side_effect = lambda t: f"PT[{t}]"
    cache_path = tmp_path / "cache.jsonl"

    texts = ["one", "two", "one"]  # "one" repeats
    results = translate_many(texts, cache_path=cache_path)

    assert results == ["PT[one]", "PT[two]", "PT[one]"]
    # translate() called only twice (once per unique input)
    assert mock_translate.call_count == 2


@patch("src.translator.translate")
def test_translate_many_persists_cache(mock_translate, tmp_path):
    mock_translate.side_effect = lambda t: f"PT[{t}]"
    cache_path = tmp_path / "cache.jsonl"

    translate_many(["alpha"], cache_path=cache_path)

    # second call in a fresh process — mock must NOT be called again
    mock_translate.reset_mock()
    results = translate_many(["alpha"], cache_path=cache_path)
    assert results == ["PT[alpha]"]
    assert mock_translate.call_count == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_translator.py -v`
Expected: FAIL — `translate_many` and `_cache_key` not defined.

- [ ] **Step 3: Implement cache in `src/translator.py`**

Append to `src/translator.py`:

```python
import hashlib
import json
from pathlib import Path


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    cache: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cache[row["key"]] = row["pt"]
    return cache


def _append_cache(path: Path, key: str, pt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "pt": pt}, ensure_ascii=False) + "\n")


def translate_many(
    texts: list[str],
    cache_path: Path = config.TRANSLATIONS_CACHE,
) -> list[str]:
    """Translate a list of passages with on-disk memoization.

    Duplicates within the input list are translated once. Already-cached texts
    are never re-translated. New translations are appended to `cache_path`
    immediately so partial progress survives crashes.
    """
    cache = _load_cache(cache_path)
    results: list[str] = []

    for text in texts:
        key = _cache_key(text)
        if key in cache:
            results.append(cache[key])
            continue

        pt = translate(text)
        cache[key] = pt
        _append_cache(cache_path, key, pt)
        results.append(pt)

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_translator.py -v`
Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/translator.py tests/test_translator.py
git commit -m "feat(translator): add disk-cached batch translation"
```

---

## Phase 4 — Medium ingestion

### Task 10: Medium HTML extractor

**Files:**
- Create: `src/medium_extractor.py`
- Create: `tests/test_medium_extractor.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_medium_extractor.py`:

```python
from src.medium_extractor import extract_medium_article


def test_extract_medium_article_returns_clean_text(tmp_path):
    html = """
    <html>
      <head><title>Análise de dados com pandas</title></head>
      <body>
        <nav>Menu</nav>
        <article>
          <h1>Análise de dados com pandas</h1>
          <p>Pandas é uma biblioteca para análise de dados em Python.</p>
          <p>Ela usa DataFrames para representar tabelas.</p>
        </article>
        <footer>Rodapé</footer>
      </body>
    </html>
    """
    html_file = tmp_path / "article.html"
    html_file.write_text(html, encoding="utf-8")

    text = extract_medium_article(html_file)

    assert "Pandas é uma biblioteca" in text
    assert "DataFrames" in text
    assert "Menu" not in text  # nav stripped
    assert "Rodapé" not in text  # footer stripped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_medium_extractor.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `src/medium_extractor.py`**

```python
"""Extract clean article text from curated Medium HTML files.

Uses trafilatura, which strips boilerplate (nav, footer, ads) well across
Medium's various layouts.
"""
from pathlib import Path

import trafilatura


def extract_medium_article(html_path: Path) -> str:
    """Return cleaned article text from a saved HTML file."""
    html = html_path.read_text(encoding="utf-8")
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        favor_recall=True,
    )
    if text is None:
        raise ValueError(f"trafilatura could not extract text from {html_path}")
    return text.strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_medium_extractor.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/medium_extractor.py tests/test_medium_extractor.py
git commit -m "feat(medium): add trafilatura-based HTML extractor"
```

---

## Phase 5 — Ingestion pipeline update

### Task 11: Add `library` inference helper and update metadata schema

**Files:**
- Modify: `data/main.py`

- [ ] **Step 1: Add helper and extend `SmartChunker.chunk_with_metadata`**

In `data/main.py`, just above the `class SmartChunker:` block, add:

```python
def infer_library(source: str) -> str:
    """Return 'pandas' | 'numpy' | 'matplotlib' | 'seaborn' | 'unknown'
    based on the source filename/path."""
    s = source.lower()
    if "pandas" in s:
        return "pandas"
    if "numpy" in s:
        return "numpy"
    if "matplotlib" in s:
        return "matplotlib"
    if "seaborn" in s:
        return "seaborn"
    return "unknown"
```

Then, inside `SmartChunker.chunk_with_metadata`, change the dict returned to include the new metadata keys. Replace the return block:

```python
        return [
            {
                "content": c,
                "source": source,
                "section": section,
                "chunk_id": f"{source}_{section}_{i}",
                "position": i,
                "char_count": len(c),
                "token_estimate": len(c.split()),
                "language": language,
                "original_lang": original_lang,
                "source_type": source_type,
                "library": infer_library(source),
                "quality_flags": {
                    "has_code": "In [" in c or ">>>" in c,
                    "has_plot": any(x in c.lower() for x in ["plot", "chart", "graph", "gráfico"]),
                    "noise_score": self.noise_score(c)
                }
            }
            for i, c in enumerate(chunks)
        ]
```

Change the method signature to accept the new params:

```python
    def chunk_with_metadata(
        self,
        text: str,
        source: str,
        section: str,
        language: str = "pt",
        original_lang: str = "en",
        source_type: str = "official_docs",
    ):
```

- [ ] **Step 2: Smoke-test the new metadata**

Run: `uv run python -c "
import sys
sys.path.insert(0, 'data')
from main import SmartChunker, infer_library
chunks = SmartChunker(max_tokens=50, overlap=10).chunk_with_metadata(
    'Texto de exemplo. ' * 30, source='numpy_docs.pdf', section='intro'
)
print(chunks[0]['library'], chunks[0]['language'], chunks[0]['original_lang'], chunks[0]['source_type'])
assert chunks[0]['library'] == 'numpy'
print('ok')
"`
Expected: `numpy pt en official_docs\nok`

- [ ] **Step 3: Commit**

```bash
git add data/main.py
git commit -m "feat(chunker): attach language/library/source_type metadata"
```

---

### Task 12: Wire translator into PDF ingestion + add Medium branch

**Files:**
- Modify: `data/main.py` — replace `run_pipeline` to integrate translator and Medium

- [ ] **Step 1: Rewrite the top imports of `data/main.py`**

At the top of `data/main.py`, after the existing imports, add:

```python
import sys
from pathlib import Path as _Path

# Allow `from src.*` when running this file directly.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from src.translator import translate_many
from src.medium_extractor import extract_medium_article
from src import config
```

- [ ] **Step 2: Replace `run_pipeline` entirely**

Replace the whole `run_pipeline` function body with:

```python
def run_pipeline():
    base = Path(__file__).resolve().parent  # data/

    pandas_path = base / "pandas"
    seaborn_path = base / "seaborn"
    numpy_pdf = base / "numpy_docs.pdf"
    matplotlib_pdf = base / "matplotlib_tutorial.pdf"
    medium_dir = base / "medium" / "raw"

    extractor = PDFExtractor()
    segmenter = ThematicSegmenter()
    chunker = SmartChunker()

    all_chunks = []

    print("=" * 60)
    print("RAG PIPELINE PT-BR - INGESTION")
    print("=" * 60)

    def process_en_pdf(pdf_file: Path, source_label: str):
        print(f"\n[EN→PT] {pdf_file.name}")
        raw = extractor.extract(pdf_file, save_debug_dir=f"output_texts/{source_label}")
        if not raw.strip():
            return []

        segments = segmenter.segment(raw)
        # translate at segment level (smaller than full doc, larger than chunk → good API granularity)
        pt_segments = translate_many([s["content"] for s in segments])

        doc_chunks = []
        for seg, pt_content in zip(segments, pt_segments):
            doc_chunks.extend(
                chunker.chunk_with_metadata(
                    text=pt_content,
                    source=pdf_file.name,
                    section=seg["title"],
                    language="pt",
                    original_lang="en",
                    source_type="official_docs",
                )
            )
        return doc_chunks

    def process_pt_article(html_file: Path):
        print(f"\n[PT ] {html_file.name}")
        text = extract_medium_article(html_file)
        text = TextNormalizer.normalize(text)
        segments = segmenter.segment(text)

        doc_chunks = []
        for seg in segments:
            doc_chunks.extend(
                chunker.chunk_with_metadata(
                    text=seg["content"],
                    source=html_file.name,
                    section=seg["title"],
                    language="pt",
                    original_lang="pt",
                    source_type="medium_article",
                )
            )
        return doc_chunks

    # --- PDFs (EN, translated) ---
    if pandas_path.exists():
        for pdf_file in pandas_path.glob("*.pdf"):
            all_chunks.extend(process_en_pdf(pdf_file, "pandas"))

    if seaborn_path.exists():
        for pdf_file in seaborn_path.rglob("*.pdf"):
            all_chunks.extend(process_en_pdf(pdf_file, "seaborn"))

    if numpy_pdf.exists():
        all_chunks.extend(process_en_pdf(numpy_pdf, "numpy"))

    if matplotlib_pdf.exists():
        all_chunks.extend(process_en_pdf(matplotlib_pdf, "matplotlib"))

    # --- Medium articles (native PT, no translation) ---
    if medium_dir.exists():
        for html_file in medium_dir.glob("*.html"):
            all_chunks.extend(process_pt_article(html_file))

    # --- Final cleaning ---
    all_chunks = deduplicate_chunks(all_chunks)
    before = len(all_chunks)
    all_chunks = filter_by_quality(all_chunks, threshold=config.NOISE_THRESHOLD)
    print(f"[quality] dropped {before - len(all_chunks)} noisy chunks")

    save_chunks(all_chunks, path=str(config.CHUNKS_PATH))

    print("\n" + "=" * 60)
    print(f"✔ INGESTÃO FINALIZADA — {len(all_chunks)} chunks em PT")
    print("=" * 60)

    return all_chunks
```

- [ ] **Step 3: Smoke-test imports**

Run: `uv run python -c "
import sys; sys.path.insert(0, 'data')
from main import run_pipeline, infer_library, filter_by_quality
print('imports ok')
"`
Expected: `imports ok`

- [ ] **Step 4: Commit**

```bash
git add data/main.py
git commit -m "feat(ingestion): wire translator + Medium branch into run_pipeline"
```

---

### Task 13: Add indexing step (embed + upsert to Chroma)

**Files:**
- Create: `src/indexer.py`
- Modify: `data/main.py` — call indexer at end of pipeline

- [ ] **Step 1: Implement `src/indexer.py`**

```python
"""Embed chunks and upsert into a ChromaDB collection.

Uses the e5-small embedder. Because e5 was trained with a "passage: " prefix
for documents and "query: " prefix for queries, we apply that convention here
(documents get "passage: " prepended before encoding).
"""
from __future__ import annotations

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src import config
from src.device import get_device


def index_chunks(
    chunks_path: Path = config.CHUNKS_PATH,
    collection_name: str = config.COLLECTION_NEW_PT,
    embedder_model: str = config.EMBEDDER_MODEL,
    e5_style_prefix: bool = True,
) -> int:
    """Embed chunks from `chunks_path` and upsert into Chroma.

    Returns the number of chunks indexed.
    """
    embedder = SentenceTransformer(embedder_model, device=get_device())

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    # delete-and-recreate to ensure embeddings are consistent with current model
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name)

    with chunks_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if not rows:
        return 0

    texts = [
        (f"passage: {r['content']}" if e5_style_prefix else r["content"])
        for r in rows
    ]
    ids = [r["chunk_id"] for r in rows]
    metadatas = [
        {
            "source": r["source"],
            "section": r["section"],
            "library": r.get("library", "unknown"),
            "language": r.get("language", "pt"),
            "source_type": r.get("source_type", "official_docs"),
        }
        for r in rows
    ]

    batch = 64
    for start in range(0, len(texts), batch):
        end = start + batch
        embs = embedder.encode(
            texts[start:end],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        collection.add(
            ids=ids[start:end],
            documents=[r["content"] for r in rows[start:end]],  # store raw content (no prefix)
            embeddings=embs,
            metadatas=metadatas[start:end],
        )

    print(f"[index] {len(texts)} chunks upserted into '{collection_name}'")
    return len(texts)


if __name__ == "__main__":
    index_chunks()
```

- [ ] **Step 2: Wire into `data/main.py`**

At the end of `run_pipeline` in `data/main.py`, add before the `return all_chunks`:

```python
    from src.indexer import index_chunks
    index_chunks(
        chunks_path=config.CHUNKS_PATH,
        collection_name=config.COLLECTION_NEW_PT,
        embedder_model=config.EMBEDDER_MODEL,
    )
```

- [ ] **Step 3: Smoke-test indexer (tiny fake corpus)**

Run:
```bash
uv run python -c "
import json
from pathlib import Path
tmp = Path('/tmp/tiny_chunks.jsonl')
tmp.write_text(json.dumps({
  'chunk_id': 'x_0', 'content': 'Teste de indexação em PT.', 'source': 't', 'section': 'i',
  'library': 'pandas', 'language': 'pt', 'source_type': 'official_docs'
}) + '\n')
from src.indexer import index_chunks
n = index_chunks(chunks_path=tmp, collection_name='smoke_test')
print('indexed', n)
"
```
Expected: `indexed 1` (takes ~30s on first run while e5-small downloads).

- [ ] **Step 4: Commit**

```bash
git add src/indexer.py data/main.py
git commit -m "feat(indexer): add Chroma upsert with e5 passage prefix"
```

---

### Task 14: Baseline snapshot script

**Files:**
- Create: `src/baseline_snapshot.py`

- [ ] **Step 1: Implement the snapshot script**

```python
"""One-shot: snapshot the current `rag_chunks` collection into
`rag_chunks_baseline_en` before the pipeline is regenerated for PT.

Run this ONCE, before re-running ingestion in PT. It preserves the existing
English embeddings (from all-MiniLM-L6-v2) so the baseline variant in the
evaluation can still be executed end-to-end.
"""
import chromadb

from src import config


def snapshot_baseline(source: str = "rag_chunks", target: str = config.COLLECTION_BASELINE_EN) -> int:
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    src = client.get_collection(source)
    try:
        client.delete_collection(target)
    except Exception:
        pass
    dst = client.create_collection(target)

    # Pull everything out in one shot (include embeddings so we can re-insert without recomputing).
    data = src.get(include=["documents", "metadatas", "embeddings"])

    ids = data["ids"]
    if not ids:
        print("source collection is empty; nothing to snapshot")
        return 0

    dst.add(
        ids=ids,
        documents=data["documents"],
        metadatas=data["metadatas"],
        embeddings=data["embeddings"],
    )
    print(f"[snapshot] copied {len(ids)} items from '{source}' → '{target}'")
    return len(ids)


if __name__ == "__main__":
    snapshot_baseline()
```

- [ ] **Step 2: Run the snapshot**

Run: `uv run python -m src.baseline_snapshot`
Expected: `[snapshot] copied N items from 'rag_chunks' → 'rag_chunks_baseline_en'` (N should match the existing collection size).

- [ ] **Step 3: Verify both collections exist**

Run:
```bash
uv run python -c "
import chromadb
from src import config
c = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
for col in c.list_collections():
    print(col.name, c.get_collection(col.name).count())
"
```
Expected: lines for `rag_chunks` (original) and `rag_chunks_baseline_en` with matching counts.

- [ ] **Step 4: Commit**

```bash
git add src/baseline_snapshot.py
git commit -m "feat(baseline): add script to snapshot current EN collection"
```

---

### Task 15: Run ingestion end-to-end (manual, not a code step)

**Files:** none modified — this is a sanity-check run.

- [ ] **Step 1: User places curated Medium HTMLs**

User puts 10-15 `.html` files in `data/medium/raw/`. Also creates `data/medium/metadata.jsonl` manually (one line per file). This is out-of-band of the plan; it is a prerequisite for a representative run but not required for pipeline correctness.

If no Medium HTMLs are ready yet, the pipeline still runs — it just produces a PT corpus from official docs only.

- [ ] **Step 2: Run ingestion**

Run: `uv run python data/main.py`
Expected: console prints progress per PDF page; translator cache builds up in `data/translations_cache.jsonl`; final count of chunks printed; `data/chunks.jsonl` regenerated; `rag_chunks_pt` collection created in Chroma.

This is long-running — first pass may take 30-60 minutes depending on corpus size and Groq throughput. Reruns use the cache and are much faster.

- [ ] **Step 3: Verify PT content**

Run:
```bash
uv run python -c "
import json
with open('data/chunks.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 3: break
        c = json.loads(line)
        print(c['library'], '|', c['content'][:200])
"
```
Expected: 3 sample chunks in Portuguese with preserved technical terms (DataFrame, groupby, etc.) and intact code blocks.

- [ ] **Step 4: Commit the regenerated chunks (lightweight record)**

```bash
git add data/chunks.jsonl
git commit -m "chore(data): regenerate chunks.jsonl in PT-BR"
```

Note: `data/chroma_db/` is large and not tracked as-is. The collection is reproducible by running the pipeline; no need to commit it.

---

## Phase 6 — Query pipeline overhaul

### Task 16: Replace `rag_qwen.py` (embedder + reranker + Groq + PT prompt)

**Files:**
- Modify: `rag_qwen.py` (full rewrite — keep same filename)

- [ ] **Step 1: Fully rewrite `rag_qwen.py`**

Replace the entire file contents with:

```python
"""Query pipeline for the PT-BR RAG assistant.

Retrieves relevant chunks from Chroma, re-ranks with a cross-encoder, and
asks a Groq-hosted LLM to answer in Portuguese strictly from the retrieved
context.
"""
from __future__ import annotations

import chromadb
from groq import Groq
from sentence_transformers import CrossEncoder, SentenceTransformer

from src import config
from src.device import get_device


# --- Models loaded once at import ---
_embedder = SentenceTransformer(config.EMBEDDER_MODEL, device=get_device())
_reranker = CrossEncoder(config.RERANKER_MODEL, device=get_device())
_groq = Groq(api_key=config.GROQ_API_KEY)

_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def _get_collection(name: str):
    return _client.get_collection(name)


# --- Retrieval ---
def retrieve(
    query: str,
    collection_name: str = config.COLLECTION_NEW_PT,
    k: int = config.TOP_K_RETRIEVE,
    final_k: int = config.TOP_K_RERANK,
    e5_style_prefix: bool = True,
) -> list[dict]:
    """Return top-`final_k` chunks for `query`, dense retrieval + cross-encoder rerank.

    Each returned dict has keys: text, source, library, score.
    """
    collection = _get_collection(collection_name)
    q_text = f"query: {query}" if e5_style_prefix else query
    q_emb = _embedder.encode(q_text, normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas"],
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    if not docs:
        return []

    pairs = [(query, d) for d in docs]
    scores = _reranker.predict(pairs)

    ranked = sorted(
        (
            {
                "text": doc,
                "source": meta.get("source", ""),
                "library": meta.get("library", "unknown"),
                "score": float(score),
            }
            for doc, meta, score in zip(docs, metas, scores)
        ),
        key=lambda c: c["score"],
        reverse=True,
    )
    return ranked[:final_k]


# --- Prompt ---
SYSTEM_PROMPT_PT = """Você é um assistente técnico para programadores iniciantes em análise de dados com Python (pandas, NumPy, matplotlib, seaborn).

REGRAS:
- Responda SOMENTE com base no CONTEXTO fornecido.
- Se a resposta não estiver no contexto, diga exatamente: "Não encontrei essa informação na documentação disponível."
- Não invente funções, métodos ou parâmetros.
- Seja direto, didático e use exemplos de código quando possível.
- Preserve nomes técnicos em inglês (DataFrame, groupby, etc.).
- Responda em português do Brasil.
"""


def build_messages(chunks: list[dict], question: str) -> list[dict]:
    context_parts = []
    for i, c in enumerate(chunks, start=1):
        context_parts.append(
            f"[DOC {i} | {c['library']} | {c['source']} | score={c['score']:.2f}]\n{c['text']}"
        )
    context = "\n\n".join(context_parts) if context_parts else "(nenhum contexto relevante)"

    user_content = f"""CONTEXTO:
{context}

PERGUNTA:
{question}

RESPOSTA:"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_PT},
        {"role": "user", "content": user_content},
    ]


# --- Generation ---
def generate(messages: list[dict], model: str = config.GROQ_LLM_MODEL) -> str:
    response = _groq.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


# --- End-to-end ---
def ask(
    question: str,
    collection_name: str = config.COLLECTION_NEW_PT,
    model: str = config.GROQ_LLM_MODEL,
    e5_style_prefix: bool = True,
) -> dict:
    """Top-level entry: retrieve + rerank + generate. Returns {answer, chunks}."""
    chunks = retrieve(question, collection_name=collection_name, e5_style_prefix=e5_style_prefix)
    messages = build_messages(chunks, question)
    answer = generate(messages, model=model)
    return {"answer": answer, "chunks": chunks, "model": model}


# --- CLI ---
if __name__ == "__main__":
    print("\n🔥 RAG PT-BR iniciado\n")
    while True:
        try:
            q = input("Pergunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit", "sair"):
            break
        if not q:
            continue
        result = ask(q)
        print("\n" + "=" * 60)
        print(result["answer"])
        print("-" * 60)
        print("Fontes:", ", ".join(f"{c['source']}:{c['library']}" for c in result["chunks"]))
        print("=" * 60 + "\n")
```

- [ ] **Step 2: Smoke-test a single question**

Prerequisites: ingestion (Task 15) completed; `rag_chunks_pt` collection populated.

Run: `uv run python -c "
from rag_qwen import ask
r = ask('Como uso groupby no pandas?')
print('ANSWER:', r['answer'][:400])
print('SOURCES:', [c['source'] for c in r['chunks']])
"`
Expected: a Portuguese answer referencing pandas groupby, with 5 sources listed.

- [ ] **Step 3: Commit**

```bash
git add rag_qwen.py
git commit -m "feat(query): rewrite rag_qwen for Groq + e5-small + mmarco + PT prompt"
```

---

### Task 17: Add baseline variant to `ask()` (EN pipeline reachable)

**Files:**
- Create: `src/baseline_qa.py`

- [ ] **Step 1: Implement a thin baseline runner**

The baseline uses the original EN embedder, EN reranker, and the old Qwen-local-style prompt in English, against the `rag_chunks_baseline_en` collection. We keep it in its own module to avoid polluting `rag_qwen.py`.

```python
"""Baseline variant (pre-surgical pipeline) for comparative evaluation.

Uses:
- collection: rag_chunks_baseline_en (EN, all-MiniLM-L6-v2 embeddings)
- embedder: all-MiniLM-L6-v2
- reranker: ms-marco-MiniLM-L-6-v2
- LLM: Groq (same API client, so we're comparing retrieval quality fairly
  without dragging a 1.5B local model into the comparison — this is a
  deliberate apples-to-apples choice that isolates the retrieval stack)
- prompt: English

Note: this does NOT run the old local Qwen LLM. Running Qwen 1.5B on CPU
would add ~30s per query to the benchmark for a model we've already decided
is inferior. We test what matters: the EN retrieval stack vs the PT one,
with the same (strong) generator.
"""
from __future__ import annotations

import chromadb
from groq import Groq
from sentence_transformers import CrossEncoder, SentenceTransformer

from src import config
from src.device import get_device


_embedder = SentenceTransformer(config.BASELINE_EMBEDDER, device=get_device())
_reranker = CrossEncoder(config.BASELINE_RERANKER, device=get_device())
_groq = Groq(api_key=config.GROQ_API_KEY)
_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))


SYSTEM_PROMPT_EN = """You are a strict QA assistant for Python data analysis (pandas, NumPy, matplotlib, seaborn).

RULES:
- Answer ONLY using the provided context.
- If the answer is not in the context, say exactly: "I don't know based on the provided context."
- Do not invent information.
- Be concise and factual.
"""


def retrieve_baseline(query: str) -> list[dict]:
    col = _client.get_collection(config.COLLECTION_BASELINE_EN)
    q_emb = _embedder.encode(query, normalize_embeddings=True).tolist()

    results = col.query(
        query_embeddings=[q_emb],
        n_results=config.TOP_K_RETRIEVE,
        include=["documents", "metadatas"],
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    if not docs:
        return []

    pairs = [(query, d) for d in docs]
    scores = _reranker.predict(pairs)
    ranked = sorted(
        (
            {
                "text": doc,
                "source": meta.get("source", ""),
                "library": meta.get("library", "unknown"),
                "score": float(score),
            }
            for doc, meta, score in zip(docs, metas, scores)
        ),
        key=lambda c: c["score"],
        reverse=True,
    )
    return ranked[: config.TOP_K_RERANK]


def ask_baseline(question_en: str, model: str = config.GROQ_LLM_MODEL) -> dict:
    chunks = retrieve_baseline(question_en)
    ctx = "\n\n".join(
        f"[DOC {i+1} | score={c['score']:.2f} | {c['source']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    ) or "(no relevant context)"

    user_content = f"CONTEXT:\n{ctx}\n\nQUESTION:\n{question_en}\n\nANSWER:"
    response = _groq.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_EN},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return {
        "answer": response.choices[0].message.content.strip(),
        "chunks": chunks,
        "model": model,
    }
```

- [ ] **Step 2: Smoke-test baseline**

Run: `uv run python -c "
from src.baseline_qa import ask_baseline
r = ask_baseline('How do I use groupby in pandas?')
print('ANSWER:', r['answer'][:400])
print('N_CHUNKS:', len(r['chunks']))
"`
Expected: an English answer with 5 chunks.

- [ ] **Step 3: Commit**

```bash
git add src/baseline_qa.py
git commit -m "feat(eval): add baseline EN QA runner for comparison"
```

---

## Phase 7 — Evaluation

### Task 18: Generate synthetic PT golden set

**Files:**
- Create: `src/eval/synth_qa.py`

- [ ] **Step 1: Implement `src/eval/synth_qa.py`**

```python
"""Generate a synthetic PT-BR golden set from chunks in the PT corpus.

Sampling is proportional to library size. Each chunk produces one Q&A pair:
  {question, ground_truth_answer, ground_truth_contexts, library, chunk_id}

Questions are filtered with three rules:
  1. length >= 5 words
  2. does not reference meta ("title", "section", "capítulo", etc.)
  3. is unique (after lowercasing) in the set
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from groq import Groq

from src import config


_META_HINTS = re.compile(
    r"\b(título|titulo|section|seção|secao|capítulo|capitulo|subtítulo|subtitulo)\b",
    re.IGNORECASE,
)


GEN_SYSTEM = """Você gera perguntas didáticas para um assistente de análise de dados em Python.

Dada uma passagem, devolva JSON com os campos:
- "question": pergunta que um iniciante faria e que a passagem responde (em PT-BR)
- "answer": resposta curta e correta (2-3 frases) baseada SOMENTE na passagem

REGRAS:
- A pergunta não pode referenciar "título da seção", "capítulo", etc.
- A pergunta deve ser específica sobre o conteúdo técnico da passagem.
- Responda SOMENTE com o JSON, sem preâmbulos.
"""


def _sample_chunks(rows: list[dict], total: int) -> list[dict]:
    by_lib: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_lib[r.get("library", "unknown")].append(r)

    total_count = sum(len(v) for v in by_lib.values())
    sampled: list[dict] = []
    rng = random.Random(42)

    for lib, items in by_lib.items():
        if lib == "unknown":
            continue
        share = round(total * len(items) / total_count)
        share = max(1, min(share, len(items)))
        sampled.extend(rng.sample(items, share))

    rng.shuffle(sampled)
    return sampled[:total]


def _ask_llm(client: Groq, passage: str) -> dict | None:
    resp = client.chat.completions.create(
        model=config.GROQ_LLM_MODEL,
        messages=[
            {"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": f"PASSAGEM:\n{passage}\n\nJSON:"},
        ],
        temperature=0.2,
        max_tokens=400,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, KeyError):
        return None


def _accept(question: str, seen: set[str]) -> bool:
    q = question.strip()
    if len(q.split()) < 5:
        return False
    if _META_HINTS.search(q):
        return False
    key = q.lower()
    if key in seen:
        return False
    seen.add(key)
    return True


def generate_golden(
    chunks_path: Path = config.CHUNKS_PATH,
    target_total: int = 100,
    out_path: Path = config.EVAL_DIR / "golden.jsonl",
) -> int:
    client = Groq(api_key=config.GROQ_API_KEY)
    rows = [json.loads(l) for l in chunks_path.open("r", encoding="utf-8") if l.strip()]
    sampled = _sample_chunks(rows, target_total * 2)  # overshoot for filter rejections

    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen_questions: set[str] = set()
    written = 0

    with out_path.open("w", encoding="utf-8") as out:
        for row in sampled:
            if written >= target_total:
                break
            pair = _ask_llm(client, row["content"])
            if not pair:
                continue
            q = pair.get("question", "").strip()
            a = pair.get("answer", "").strip()
            if not q or not a or not _accept(q, seen_questions):
                continue

            entry = {
                "question": q,
                "ground_truth_answer": a,
                "ground_truth_contexts": [row["content"]],
                "library": row.get("library", "unknown"),
                "chunk_id": row["chunk_id"],
            }
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

    print(f"[synth_qa] wrote {written} Q&A pairs to {out_path}")
    return written


if __name__ == "__main__":
    generate_golden()
```

- [ ] **Step 2: Generate golden set**

Run: `uv run python -m src.eval.synth_qa`
Expected: `[synth_qa] wrote ~100 Q&A pairs to data/eval/golden.jsonl`.

- [ ] **Step 3: Manually inspect a sample**

Run:
```bash
uv run python -c "
import json
with open('data/eval/golden.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 5: break
        e = json.loads(line)
        print('Q:', e['question'])
        print('A:', e['ground_truth_answer'][:150])
        print('---')
"
```
Expected: 5 reasonable PT questions + short answers.

- [ ] **Step 4: Commit**

```bash
git add src/eval/synth_qa.py data/eval/golden.jsonl
git commit -m "feat(eval): generate PT synthetic golden set"
```

---

### Task 19: Translate questions to EN for baseline run

**Files:**
- Create: `src/eval/translate_questions.py`

- [ ] **Step 1: Implement question translation**

```python
"""Translate golden-set questions PT → EN so the EN baseline can be evaluated
on the same semantic set.

Uses the same cached translator we use for the corpus, so repeated runs are
cheap. Only the `question` field is translated; `ground_truth_answer` and
`ground_truth_contexts` stay in PT (they're only used by RAGAS against the
PT pipeline output).
"""
from __future__ import annotations

import json
from pathlib import Path

from groq import Groq

from src import config


SYS = """Traduza a pergunta a seguir do português para o inglês técnico, preservando termos como DataFrame, groupby, pandas, NumPy. Responda apenas com a tradução."""


def translate_questions(
    in_path: Path = config.EVAL_DIR / "golden.jsonl",
    out_path: Path = config.EVAL_DIR / "golden_en.jsonl",
) -> int:
    client = Groq(api_key=config.GROQ_API_KEY)
    rows = [json.loads(l) for l in in_path.open("r", encoding="utf-8") if l.strip()]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for r in rows:
            resp = client.chat.completions.create(
                model=config.GROQ_LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYS},
                    {"role": "user", "content": r["question"]},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            r_en = dict(r)
            r_en["question_en"] = resp.choices[0].message.content.strip()
            out.write(json.dumps(r_en, ensure_ascii=False) + "\n")

    print(f"[translate_questions] wrote {len(rows)} entries to {out_path}")
    return len(rows)


if __name__ == "__main__":
    translate_questions()
```

- [ ] **Step 2: Run translation**

Run: `uv run python -m src.eval.translate_questions`
Expected: `[translate_questions] wrote ~100 entries to data/eval/golden_en.jsonl`.

- [ ] **Step 3: Commit**

```bash
git add src/eval/translate_questions.py data/eval/golden_en.jsonl
git commit -m "feat(eval): translate golden questions to EN for baseline"
```

---

### Task 20: Run a single pipeline variant with RAGAS

**Files:**
- Create: `src/eval/run_eval.py`

- [ ] **Step 1: Implement `src/eval/run_eval.py`**

```python
"""Run one pipeline variant against the golden set and return RAGAS metrics.

A "variant" is a (name, ask_fn, dataset_builder) triple:
  - "new-70b":  ask_fn = rag_qwen.ask w/ llama-3.3-70b
  - "new-8b":   ask_fn = rag_qwen.ask w/ llama-3.1-8b
  - "baseline": ask_fn = src.baseline_qa.ask_baseline (uses EN questions)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from datasets import Dataset
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from langchain_huggingface import HuggingFaceEmbeddings

from src import config
from src.device import get_device


@dataclass
class VariantResult:
    name: str
    metrics: dict
    latency_p50: float
    latency_p95: float
    n_queries: int


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = int(round((p / 100) * (len(values) - 1)))
    return values[k]


def run_variant(
    name: str,
    ask_fn: Callable[[str], dict],
    golden_rows: list[dict],
    question_key: str = "question",
) -> VariantResult:
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []
    questions: list[str] = []
    latencies: list[float] = []

    for row in golden_rows:
        q = row[question_key]
        t0 = time.perf_counter()
        out = ask_fn(q)
        latencies.append(time.perf_counter() - t0)

        answers.append(out["answer"])
        contexts.append([c["text"] for c in out["chunks"]])
        ground_truths.append(row["ground_truth_answer"])
        questions.append(row["question"])  # RAGAS wants PT ref even for EN baseline

    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    judge = ChatGroq(model=config.GROQ_LLM_MODEL, temperature=0.0)
    emb = HuggingFaceEmbeddings(
        model_name=config.EMBEDDER_MODEL,
        model_kwargs={"device": get_device()},
    )

    scores = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge,
        embeddings=emb,
    )

    return VariantResult(
        name=name,
        metrics=dict(scores),
        latency_p50=_percentile(latencies, 50),
        latency_p95=_percentile(latencies, 95),
        n_queries=len(golden_rows),
    )


def load_golden(path: Path = config.EVAL_DIR / "golden.jsonl") -> list[dict]:
    return [json.loads(l) for l in path.open("r", encoding="utf-8") if l.strip()]
```

- [ ] **Step 2: Smoke-test with 3 queries**

Run:
```bash
uv run python -c "
from src.eval.run_eval import run_variant, load_golden
from rag_qwen import ask
rows = load_golden()[:3]
r = run_variant('smoke', lambda q: ask(q), rows)
print(r)
"
```
Expected: a `VariantResult` with 4 metric scores and latency numbers. First run is slow (RAGAS downloads models).

- [ ] **Step 3: Commit**

```bash
git add src/eval/run_eval.py
git commit -m "feat(eval): add per-variant RAGAS runner"
```

---

### Task 21: Experiment orchestrator + report generator

**Files:**
- Create: `src/eval/experiment_runner.py`

- [ ] **Step 1: Implement the orchestrator**

```python
"""Run the three variants (baseline, new-70b, new-8b) against the golden set
and write a markdown comparison table to data/eval/report.md.
"""
from __future__ import annotations

import json
from pathlib import Path

from rag_qwen import ask as ask_pt
from src import config
from src.baseline_qa import ask_baseline
from src.eval.run_eval import VariantResult, load_golden, run_variant


def main() -> None:
    golden_pt = load_golden(config.EVAL_DIR / "golden.jsonl")
    golden_en = load_golden(config.EVAL_DIR / "golden_en.jsonl")

    results: list[VariantResult] = []

    # 1. Baseline EN
    results.append(run_variant(
        name="baseline (EN, MiniLM + ms-marco + Llama 3.3 70B)",
        ask_fn=ask_baseline,
        golden_rows=golden_en,
        question_key="question_en",
    ))

    # 2. New pipeline, llama-3.3-70b
    results.append(run_variant(
        name="new PT (e5-small + mmarco + Llama 3.3 70B)",
        ask_fn=lambda q: ask_pt(q, model=config.GROQ_LLM_MODEL),
        golden_rows=golden_pt,
    ))

    # 3. New pipeline, llama-3.1-8b
    results.append(run_variant(
        name="new PT (e5-small + mmarco + Llama 3.1 8B)",
        ask_fn=lambda q: ask_pt(q, model=config.GROQ_LLM_FAST),
        golden_rows=golden_pt,
    ))

    write_report(results)


def write_report(results: list[VariantResult]) -> None:
    out = config.EVAL_DIR / "report.md"
    lines = [
        "# Relatório RAGAS — RAG PT-BR",
        "",
        f"Golden set: `data/eval/golden.jsonl` ({results[0].n_queries} queries).",
        "",
        "| Variante | Faithfulness | Answer Relevancy | Context Precision | Context Recall | p50 (s) | p95 (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        m = r.metrics
        lines.append(
            f"| {r.name} | "
            f"{m.get('faithfulness', 0):.3f} | "
            f"{m.get('answer_relevancy', 0):.3f} | "
            f"{m.get('context_precision', 0):.3f} | "
            f"{m.get('context_recall', 0):.3f} | "
            f"{r.latency_p50:.2f} | "
            f"{r.latency_p95:.2f} |"
        )

    lines += [
        "",
        "## Conclusão",
        "",
        "_Preencher manualmente após inspecionar a tabela._",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the experiment**

Run: `uv run python -m src.eval.experiment_runner`
Expected: long-running (~30-60 min depending on corpus size and Groq latency). Final output: `[report] wrote data/eval/report.md`.

- [ ] **Step 3: Inspect the report**

Run: `cat data/eval/report.md`
Expected: markdown table with 3 rows and realistic metric values (0.0–1.0 range).

- [ ] **Step 4: Write a brief conclusion**

Edit `data/eval/report.md` — replace the "_Preencher manualmente_" line with a 2-3 sentence summary comparing the three variants.

- [ ] **Step 5: Commit**

```bash
git add src/eval/experiment_runner.py data/eval/report.md
git commit -m "feat(eval): orchestrate 3-variant comparison and generate report"
```

---

## Phase 8 — Documentation

### Task 22: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the README content**

```markdown
# Assistente RAG PT-BR para Análise de Dados em Python

Assistente baseado em RAG (Retrieval-Augmented Generation) que responde
dúvidas e explica erros sobre `pandas`, `numpy`, `matplotlib` e `seaborn`
em português, com base em documentação real.

## Problema

Programadores iniciantes têm dificuldade em entender mensagens de erro e
documentação em inglês. Este projeto oferece respostas contextualizadas em
PT-BR, apoiadas por trechos da documentação oficial e artigos curados.

## Arquitetura

- **Embedder**: `intfloat/multilingual-e5-small`
- **Reranker**: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- **LLM**: Groq API (padrão: `llama-3.3-70b-versatile`)
- **Vector DB**: ChromaDB (coleção `rag_chunks_pt`)
- **Corpus**: PDFs oficiais traduzidos para PT-BR + artigos Medium curados

## Pré-requisitos

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- Chave do Groq (https://console.groq.com/keys)

## Setup

```bash
cp .env.example .env
# edite .env e cole sua GROQ_API_KEY

uv sync
```

## Uso

### Fazer uma pergunta

```bash
uv run python rag_qwen.py
```

Exemplo de interação:

```
Pergunta: como uso groupby no pandas?
============================================================
Para agrupar dados em um DataFrame, use o método .groupby()...
```

### (Re)construir o corpus

Requer artigos Medium curados em `data/medium/raw/*.html` (opcional).

```bash
# snapshot do índice EN atual antes de regenerar (uma única vez)
uv run python -m src.baseline_snapshot

# reingestão completa — traduz, faz chunk, indexa
uv run python data/main.py
```

Primeira rodada leva 30-60min (tradução). Reruns usam cache e são rápidos.

### Rodar a avaliação

```bash
uv run python -m src.eval.synth_qa             # gera golden set PT
uv run python -m src.eval.translate_questions  # versão EN p/ baseline
uv run python -m src.eval.experiment_runner    # roda 3 variantes + relatório
cat data/eval/report.md
```

## Estrutura

- `rag_qwen.py` — pipeline de consulta (PT)
- `data/main.py` — pipeline de ingestão
- `src/translator.py` — tradução EN→PT com glossário e cache
- `src/indexer.py` — embedding + upsert Chroma
- `src/baseline_qa.py` — pipeline baseline (EN) para comparação
- `src/eval/` — geração de Q&A sintético e orquestração RAGAS
- `docs/superpowers/specs/` — spec de design
- `docs/superpowers/plans/` — plano de implementação

## Testes

```bash
uv run pytest -v
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for PT-BR RAG setup and usage"
```

---

## Self-Review

Checking this plan against the spec requirements:

- **4.1 Trocas de componentes** ✅ Task 16 (rag_qwen rewrite)
- **4.2 Chunker fixes (noise, overlap, heading)** ✅ Tasks 4, 5, 6
- **4.3 Prompt PT-BR** ✅ Task 16 (`SYSTEM_PROMPT_PT`)
- **5.1 Fontes (PDFs + Medium)** ✅ Task 12 (process_pt_article branch)
- **5.2 Translator module (glossary + code blocks + cache)** ✅ Tasks 7, 8, 9
- **5.3 Pipeline atualizado** ✅ Task 12
- **5.4 Metadados** ✅ Task 11
- **6.1 Golden set sintético** ✅ Task 18
- **6.2 Métricas RAGAS** ✅ Task 20
- **6.3 Comparação (3 variantes com coleções distintas)** ✅ Tasks 14, 17, 21
- **6.4 Módulos novos (synth_qa, run_eval)** ✅ Tasks 18, 20, 21
- **7. Estrutura do repo** ✅ Task 1 (skeleton) + subsequent tasks populate it
- **9. Riscos — .env gitignored** ✅ Task 1
- **10. Critérios de sucesso**:
  - rag_qwen.py responde em PT via Groq ✅ Task 16 smoke test
  - Regenera corpus end-to-end ✅ Task 15
  - Golden set ~100 perguntas ✅ Task 18
  - report.md comparativo ✅ Task 21
  - README atualizado ✅ Task 22
  - .env.example ✅ Task 1

**No placeholders found.** Types and function signatures are consistent:
- `ask(question, collection_name, model)` used consistently across Tasks 16, 20, 21
- `ask_baseline(question_en, model)` consistent in Tasks 17, 21
- `extract_code_blocks` and `restore_code_blocks` from Task 7 used in Task 8
- `translate_many(texts, cache_path)` signature consistent between Tasks 9 and 12

**Scope:** Single subsystem, ~22 tasks. Not too large for one plan.
