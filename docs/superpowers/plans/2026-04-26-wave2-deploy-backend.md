# Wave 2 — Deploy Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the RAG pipeline as a public HTTP API (`/health`, `/ask`) on a Hugging Face Space Docker container, deployed via `huggingface_hub.upload_folder`.

**Architecture:** `backend/app.py` (FastAPI) wraps the existing `src/rag_query` functions — retrieve → rerank → `build_pt_prompt` → Groq with Cerebras fallback — returning `{answer, citations}`. Models load once at startup via the FastAPI lifespan handler. Deploy is triggered manually with `scripts/deploy_space.py`.

**Tech Stack:** FastAPI, uvicorn, sentence-transformers, chromadb, groq, cerebras-cloud-sdk, huggingface_hub, pytest + TestClient (httpx)

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `pyproject.toml` | Add fastapi, uvicorn, cerebras-cloud-sdk, httpx to deps |
| Modify | `src/config.py` | Add `CEREBRAS_API_KEY`, `CEREBRAS_LLM_MODEL` |
| Modify | `src/rag_query.py` | Add `_generate_cerebras()`, `generate_answer_with_fallback()` |
| Modify | `tests/test_rag_query.py` | Add 2 tests for fallback behavior |
| Create | `backend/__init__.py` | Empty — makes `backend` a Python package |
| Create | `backend/app.py` | FastAPI app: `/health`, `/ask`, lifespan model loader |
| Create | `backend/requirements.txt` | Deps for HF Space Docker image |
| Create | `backend/Dockerfile` | Container definition for HF Space |
| Create | `backend/README.md` | HF Space config header (`sdk: docker`) |
| Create | `tests/test_backend.py` | FastAPI TestClient tests for both endpoints |
| Create | `scripts/deploy_space.py` | `upload_folder` to HF Space |
| Create | `scripts/smoke_test.py` | Hits live Space URL to verify deploy |
| Create | `docs/superpowers/SETUP_HF_SPACE.md` | One-time manual checklist |

---

## Task 1: Add Cerebras fallback to `src/`

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/config.py`
- Modify: `src/rag_query.py`
- Modify: `tests/test_rag_query.py`

- [ ] **Step 1: Add `cerebras-cloud-sdk` to `pyproject.toml`**

Open `pyproject.toml`. Add these lines to the `dependencies` list and `[dependency-groups].dev`:

```toml
[project]
dependencies = [
    "groq>=0.13.0",
    "sentence-transformers>=3.0.0",
    "chromadb>=0.5.0",
    "trafilatura>=1.12.0",
    "python-dotenv>=1.0.0",
    "ragas>=0.2.0",
    "langchain-groq>=0.2.0",
    "datasets>=3.0.0",
    "tiktoken>=0.8.0",
    "google-generativeai>=0.8.6",
    "cerebras-cloud-sdk>=1.0.0",
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "pytest-mock>=3.14.0",
    "httpx>=0.27.0",
]
```

Run: `uv sync`
Expected: resolves and installs cerebras-cloud-sdk, fastapi, uvicorn, httpx.

- [ ] **Step 2: Add Cerebras config to `src/config.py`**

After the `GROQ_LLM_FAST` line (line 35), add:

```python
# --- Cerebras (fallback LLM when Groq rate-limits) ---
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")
CEREBRAS_LLM_MODEL = os.environ.get("CEREBRAS_LLM_MODEL", "llama-3.3-70b")
```

- [ ] **Step 3: Write the failing tests for Cerebras fallback**

Append to `tests/test_rag_query.py`:

```python
def test_generate_answer_with_fallback_uses_cerebras_on_rate_limit(monkeypatch):
    """When Groq raises RateLimitError, fallback calls _generate_cerebras."""
    from src import rag_query as rq

    def raise_rate_limit(prompt, model=config.GROQ_LLM_MODEL):
        from groq import RateLimitError
        import httpx
        mock_req = httpx.Request("POST", "https://api.groq.com")
        mock_resp = httpx.Response(429, request=mock_req)
        raise RateLimitError("rate limited", response=mock_resp, body=None)

    monkeypatch.setattr(rq, "generate_answer", raise_rate_limit)
    monkeypatch.setattr(rq, "_generate_cerebras", lambda prompt, model=None: "cerebras fallback answer")

    result = rq.generate_answer_with_fallback("test prompt")
    assert result == "cerebras fallback answer"


def test_generate_answer_with_fallback_returns_groq_when_ok(monkeypatch):
    """When Groq succeeds, returns its answer without calling Cerebras."""
    from src import rag_query as rq

    monkeypatch.setattr(rq, "generate_answer", lambda prompt, model=None: "groq answer")
    cerebras_called = {"called": False}
    monkeypatch.setattr(rq, "_generate_cerebras", lambda *a, **kw: cerebras_called.update({"called": True}) or "cerebras")

    result = rq.generate_answer_with_fallback("test prompt")
    assert result == "groq answer"
    assert cerebras_called["called"] is False
```

- [ ] **Step 4: Run the tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_rag_query.py::test_generate_answer_with_fallback_uses_cerebras_on_rate_limit tests/test_rag_query.py::test_generate_answer_with_fallback_returns_groq_when_ok -v
```

Expected: `AttributeError: module 'src.rag_query' has no attribute 'generate_answer_with_fallback'`

- [ ] **Step 5: Add `_generate_cerebras` and `generate_answer_with_fallback` to `src/rag_query.py`**

After the `generate_answer` function (after line 170), add:

```python
def _generate_cerebras(prompt: str, model: str = config.CEREBRAS_LLM_MODEL) -> str:
    from cerebras.cloud.sdk import Cerebras
    if not config.CEREBRAS_API_KEY:
        raise RuntimeError("CEREBRAS_API_KEY not set")
    client = Cerebras(api_key=config.CEREBRAS_API_KEY)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=800,
    )
    return resp.choices[0].message.content.strip()


def generate_answer_with_fallback(prompt: str) -> str:
    """Try Groq first; on RateLimitError or InternalServerError, retry with Cerebras."""
    from groq import RateLimitError, InternalServerError
    try:
        return generate_answer(prompt)
    except (RateLimitError, InternalServerError):
        return _generate_cerebras(prompt)
```

- [ ] **Step 6: Run the new tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_rag_query.py::test_generate_answer_with_fallback_uses_cerebras_on_rate_limit tests/test_rag_query.py::test_generate_answer_with_fallback_returns_groq_when_ok -v
```

Expected: both PASS.

- [ ] **Step 7: Run the full test suite to verify no regressions**

```bash
.venv/Scripts/python.exe -m pytest tests/ -v
```

Expected: all previously passing tests still PASS (62 + 2 new = 64 passing).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/config.py src/rag_query.py tests/test_rag_query.py
git commit -m "feat(rag_query): add Cerebras fallback for Groq rate-limit"
```

---

## Task 2: Create `backend/app.py` and tests

**Files:**
- Create: `backend/__init__.py`
- Create: `backend/app.py`
- Create: `tests/test_backend.py`

- [ ] **Step 1: Create `backend/__init__.py`**

Create an empty file at `backend/__init__.py`. It makes `backend` a Python package importable in tests.

Content: (empty file)

- [ ] **Step 2: Write the failing tests first**

Create `tests/test_backend.py`:

```python
"""Tests for backend FastAPI endpoints.

Model loading is monkeypatched — no real Chroma or sentence-transformers needed.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _make_client(monkeypatch):
    """Return a TestClient with model loading and rag functions stubbed out."""
    import backend.app as m

    # Stub out the startup loader so no real models are touched
    def fake_load():
        m._state["ready"] = True
        m._state["chroma_count"] = 42

    monkeypatch.setattr(m, "_load_models", fake_load)

    # Stub out rag functions imported into backend.app namespace
    monkeypatch.setattr(m, "retrieve", lambda question, top_k=15: [
        {"content": "DataFrame.merge combina.", "metadata": {
            "source": "pandas_merge.html", "section": "## Merge", "library": "pandas"}}
    ])
    monkeypatch.setattr(m, "rerank", lambda question, chunks, top_n=5: chunks)
    monkeypatch.setattr(m, "build_pt_prompt", lambda question, chunks: "prompt")
    monkeypatch.setattr(m, "generate_answer_with_fallback", lambda prompt: "resposta gerada")
    monkeypatch.setattr(m, "format_citations", lambda chunks: [
        {"source": "pandas_merge.html", "section": "## Merge", "library": "pandas"}
    ])

    return TestClient(m.app)


def test_health_returns_ok(monkeypatch):
    client = _make_client(monkeypatch)
    with client:
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True
    assert body["chroma_count"] == 42


def test_health_returns_503_when_not_ready(monkeypatch):
    import backend.app as m

    monkeypatch.setattr(m, "_load_models", lambda: None)  # leaves ready=False

    with TestClient(m.app) as client:
        r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["detail"] == "models not loaded"


def test_ask_returns_answer_and_citations(monkeypatch):
    client = _make_client(monkeypatch)
    with client:
        r = client.post("/ask", json={"question": "Como fazer merge em pandas?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "resposta gerada"
    assert len(body["citations"]) == 1
    assert body["citations"][0]["library"] == "pandas"


def test_ask_empty_question_returns_422(monkeypatch):
    client = _make_client(monkeypatch)
    with client:
        r = client.post("/ask", json={"question": "   "})
    assert r.status_code == 422


def test_ask_missing_question_returns_422(monkeypatch):
    client = _make_client(monkeypatch)
    with client:
        r = client.post("/ask", json={})
    assert r.status_code == 422


def test_ask_returns_empty_answer_when_no_chunks(monkeypatch):
    import backend.app as m

    def fake_load():
        m._state["ready"] = True
        m._state["chroma_count"] = 42

    monkeypatch.setattr(m, "_load_models", fake_load)
    monkeypatch.setattr(m, "retrieve", lambda question, top_k=15: [])
    monkeypatch.setattr(m, "rerank", lambda question, chunks, top_n=5: [])

    with TestClient(m.app) as client:
        r = client.post("/ask", json={"question": "pergunta sem resultado"})
    assert r.status_code == 200
    body = r.json()
    assert "Não encontrei" in body["answer"]
    assert body["citations"] == []


def test_ask_returns_502_when_llm_unavailable(monkeypatch):
    import backend.app as m

    def fake_load():
        m._state["ready"] = True
        m._state["chroma_count"] = 42

    monkeypatch.setattr(m, "_load_models", fake_load)
    monkeypatch.setattr(m, "retrieve", lambda question, top_k=15: [
        {"content": "doc", "metadata": {"source": "x", "section": "s", "library": "pandas"}}
    ])
    monkeypatch.setattr(m, "rerank", lambda question, chunks, top_n=5: chunks)
    monkeypatch.setattr(m, "build_pt_prompt", lambda question, chunks: "prompt")
    def always_raise(prompt):
        raise RuntimeError("both LLMs failed")

    monkeypatch.setattr(m, "generate_answer_with_fallback", always_raise)

    with TestClient(m.app) as client:
        r = client.post("/ask", json={"question": "Como fazer merge?"})
    assert r.status_code == 502
    assert r.json()["detail"] == "LLM unavailable"
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_backend.py -v
```

Expected: `ModuleNotFoundError: No module named 'backend'`

- [ ] **Step 4: Create `backend/app.py`**

```python
"""FastAPI backend for RAG PT-BR assistant.

Endpoints:
  GET  /health  — liveness + readiness probe
  POST /ask     — full RAG pipeline (retrieve → rerank → Groq/Cerebras fallback)

Models are loaded once at startup via FastAPI lifespan — never per-request.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src import config
from src.rag_query import (
    _ensure_loaded,
    build_pt_prompt,
    format_citations,
    generate_answer_with_fallback,
    rerank,
    retrieve,
)

# Module-level state — set once at startup, read on every request
_state: dict = {"ready": False, "chroma_count": 0}


def _load_models() -> None:
    """Load embedder, reranker, and open Chroma. Blocks until done."""
    _ensure_loaded("new")
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    _state["chroma_count"] = client.get_collection(config.COLLECTION_NEW_PT).count()
    _state["ready"] = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield


app = FastAPI(title="RAG PT-BR Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Wave 3 will restrict to Vercel URL
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


@app.get("/health")
def health():
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="models not loaded")
    return {
        "status": "ok",
        "chroma_count": _state["chroma_count"],
        "models_loaded": True,
    }


@app.post("/ask")
def ask(req: AskRequest):
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="models not loaded")
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    chunks = retrieve(req.question, top_k=config.TOP_K_RETRIEVE)
    reranked = rerank(req.question, chunks, top_n=req.top_k)

    if not reranked:
        return {
            "answer": "Não encontrei informações relevantes sobre sua pergunta.",
            "citations": [],
        }

    prompt = build_pt_prompt(req.question, reranked)
    try:
        answer_text = generate_answer_with_fallback(prompt)
    except Exception:
        raise HTTPException(status_code=502, detail="LLM unavailable")

    return {
        "answer": answer_text,
        "citations": format_citations(reranked),
    }
```

- [ ] **Step 5: Run the backend tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_backend.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 6: Run full test suite to verify no regressions**

```bash
.venv/Scripts/python.exe -m pytest tests/ -v
```

Expected: all 71 tests PASS (64 from Task 1 + 7 new).

- [ ] **Step 7: Commit**

```bash
git add backend/__init__.py backend/app.py tests/test_backend.py
git commit -m "feat(backend): add FastAPI /health and /ask endpoints with Cerebras fallback"
```

---

## Task 3: Create `backend/Dockerfile`, `requirements.txt`, `README.md`

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/Dockerfile`
- Create: `backend/README.md`

No automated tests — these are build artifacts. Verification is `docker build` (optional local check).

- [ ] **Step 1: Create `backend/requirements.txt`**

```
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
chromadb>=0.5.0
sentence-transformers>=3.0.0
torch --index-url https://download.pytorch.org/whl/cpu
huggingface_hub>=0.24.0
groq>=0.13.0
cerebras-cloud-sdk>=1.0.0
python-dotenv>=1.0.0
```

Note: `torch` CPU-only keeps the Docker image ~500 MB instead of ~2 GB. e5-small and mmarco reranker do not require GPU.

- [ ] **Step 2: Create `backend/Dockerfile`**

```dockerfile
FROM python:3.12-slim

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install Python deps before copying code (layer cache)
COPY backend/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY src/ ./src/
COPY data/chroma_db/ ./data/chroma_db/

EXPOSE 7860
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

- [ ] **Step 3: Create `backend/README.md`**

```markdown
---
title: RAG PT-BR Backend
sdk: docker
app_port: 7860
---

Backend FastAPI para o assistente RAG PT-BR.

## Endpoints

- `GET /health` — liveness probe
- `POST /ask` — `{"question": str, "top_k": int = 5}` → `{"answer": str, "citations": [...]}`

## Secrets necessários (configurar no painel do HF Space)

- `GROQ_API_KEY`
- `CEREBRAS_API_KEY`
```

- [ ] **Step 4: Commit**

```bash
git add backend/requirements.txt backend/Dockerfile backend/README.md
git commit -m "feat(backend): add Dockerfile and requirements for HF Space"
```

---

## Task 4: Create deploy and smoke test scripts

**Files:**
- Create: `scripts/deploy_space.py`
- Create: `scripts/smoke_test.py`

- [ ] **Step 1: Create `scripts/deploy_space.py`**

Create the file `scripts/deploy_space.py`:

```python
"""Upload backend to Hugging Face Space via huggingface_hub.

Usage:
    HF_SPACE_ID=username/rag-pt-backend HF_TOKEN=hf_... \
        .venv/Scripts/python.exe scripts/deploy_space.py

Environment variables:
    HF_SPACE_ID   — Space repo ID, e.g. "myuser/rag-pt-backend"
    HF_TOKEN      — HF write-access token (never commit this)
"""
import os
import sys

from huggingface_hub import HfApi


def main() -> None:
    space_id = os.environ.get("HF_SPACE_ID")
    token = os.environ.get("HF_TOKEN")

    if not space_id or not token:
        print("ERROR: HF_SPACE_ID and HF_TOKEN must be set.", file=sys.stderr)
        sys.exit(1)

    print(f"Deploying to Space: {space_id}")
    api = HfApi()
    api.upload_folder(
        folder_path=".",
        repo_id=space_id,
        repo_type="space",
        token=token,
        allow_patterns=["backend/**", "src/**", "data/chroma_db/**"],
        ignore_patterns=[
            "**/__pycache__/**",
            "**/*.pyc",
            ".venv/**",
            "**/.git/**",
        ],
    )
    print("Deploy complete. HF Space will rebuild automatically.")
    print(f"Monitor build logs at: https://huggingface.co/spaces/{space_id}/logs")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script is importable (dry-run check)**

```bash
.venv/Scripts/python.exe -c "import scripts.deploy_space; print('OK')"
```

Expected: `OK` (no import errors).

- [ ] **Step 3: Create `scripts/smoke_test.py`**

Create the file `scripts/smoke_test.py`:

```python
"""Smoke test against the live HF Space URL.

Usage:
    HF_SPACE_URL=https://username-rag-pt-backend.hf.space \
        .venv/Scripts/python.exe scripts/smoke_test.py

The first call may take ~30s if the Space is cold. This script uses a 90s timeout.
"""
import os
import sys

import httpx


def main() -> None:
    base_url = os.environ.get("HF_SPACE_URL", "").rstrip("/")
    if not base_url:
        print("ERROR: HF_SPACE_URL must be set.", file=sys.stderr)
        sys.exit(1)

    print(f"Smoke testing: {base_url}")

    # 1. Health check
    print("  GET /health ...", end=" ", flush=True)
    r = httpx.get(f"{base_url}/health", timeout=90)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    assert body.get("models_loaded") is True, f"/health: models_loaded not True: {body}"
    print(f"OK (chroma_count={body.get('chroma_count')})")

    # 2. Ask endpoint
    print("  POST /ask ...", end=" ", flush=True)
    r = httpx.post(
        f"{base_url}/ask",
        json={"question": "Como fazer merge entre dois DataFrames no pandas?"},
        timeout=90,
    )
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    assert body.get("answer"), f"/ask: empty answer: {body}"
    assert isinstance(body.get("citations"), list), f"/ask: citations not a list: {body}"
    print(f"OK")
    print(f"    Answer preview: {body['answer'][:120]}...")
    print(f"    Citations: {len(body['citations'])} returned")

    print("\nSmoke test PASSED.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify the script is importable**

```bash
.venv/Scripts/python.exe -c "import scripts.smoke_test; print('OK')"
```

Expected: `OK`.

- [ ] **Step 5: Ensure `scripts/__init__.py` exists** (needed for the import checks above)

Check if `scripts/__init__.py` exists:

```bash
ls scripts/
```

If it does not exist, create an empty `scripts/__init__.py`.

- [ ] **Step 6: Commit**

```bash
git add scripts/deploy_space.py scripts/smoke_test.py scripts/__init__.py
git commit -m "feat(scripts): add HF Space deploy and smoke test scripts"
```

---

## Task 5: Create setup checklist and update HANDOFF

**Files:**
- Create: `docs/superpowers/SETUP_HF_SPACE.md`
- Modify: `docs/superpowers/HANDOFF.md`

- [ ] **Step 1: Create `docs/superpowers/SETUP_HF_SPACE.md`**

```markdown
# HF Space Setup Checklist

One-time manual steps to create and configure the Hugging Face Space.
Run these BEFORE executing `scripts/deploy_space.py`.

---

## 1. Create a HF account

Go to https://huggingface.co and sign up (free).

## 2. Create the Space

1. Click your profile → "New Space"
2. Fill in:
   - **Owner:** your HF username
   - **Space name:** `rag-pt-backend` (or any name — just keep it consistent)
   - **SDK:** Docker
   - **Visibility:** Public (required for free CPU tier)
3. Click "Create Space"

## 3. Generate a write-access token

1. Go to HF Settings → Access Tokens
2. Click "New token"
3. Name it (e.g. `deploy-token`), set role to **write**
4. Copy the token — you will use it as `HF_TOKEN` in your shell

**Never commit this token.** Set it only as a shell variable.

## 4. Add secrets to the Space

In your Space page → Settings → Variables and secrets → "New secret":

| Name | Value |
|---|---|
| `GROQ_API_KEY` | Your Groq key from console.groq.com/keys |
| `CEREBRAS_API_KEY` | Your Cerebras key from cloud.cerebras.ai |

## 5. Run the deploy script

From the project root (with `.venv` activated):

```bash
HF_SPACE_ID=yourusername/rag-pt-backend \
HF_TOKEN=hf_yourtoken \
  .venv/Scripts/python.exe scripts/deploy_space.py
```

On Windows PowerShell:
```powershell
$env:HF_SPACE_ID="yourusername/rag-pt-backend"
$env:HF_TOKEN="hf_yourtoken"
.venv\Scripts\python.exe scripts\deploy_space.py
```

## 6. Watch the build

Go to your Space → "Logs" tab.
First build takes ~5–10 minutes (downloading torch + sentence-transformers).
Subsequent deploys rebuild from the Docker layer cache — much faster.

## 7. Verify the deploy

Once the Space shows "Running", verify:

```bash
curl https://yourusername-rag-pt-backend.hf.space/health
# Expected: {"status":"ok","models_loaded":true,"chroma_count":2298}
```

Then run the full smoke test:

```bash
HF_SPACE_URL=https://yourusername-rag-pt-backend.hf.space \
  .venv/Scripts/python.exe scripts/smoke_test.py
```

Expected: `Smoke test PASSED.`

## 8. Demo warmup note

HF Space free tier sleeps after ~48h of inactivity (cold start ~30s).
Before a demo, call `/health` at least 5 minutes in advance to warm up the container.

```bash
curl https://yourusername-rag-pt-backend.hf.space/health
```
```

- [ ] **Step 2: Update `docs/superpowers/HANDOFF.md`**

Replace the Wave 2 plan row (the `*(a criar)*` line) in the plans table with:

```markdown
| `docs/superpowers/plans/2026-04-26-wave2-deploy-backend.md` | ✅ **CONCLUÍDA** | FastAPI + HF Space via `huggingface_hub`. `/health` + `/ask` com Cerebras fallback. |
```

Also update section 9 "Próximo passo concreto" — replace the Wave 2 description with Wave 3:

```markdown
**Onda 2 fechada.** Backend FastAPI deployado no HF Space. Smoke test passando contra URL pública.

**Próximo:** abrir `superpowers:brainstorming` pra **Onda 3 — Frontend + Avaliação**
(Next.js + Vercel + RAGAS + golden set + README final).
```

And update section 4 "Estado atual do código" to reflect Wave 2 completion:

```markdown
- `backend/app.py` — FastAPI `/health` + `/ask` (retrieve → rerank → Groq/Cerebras fallback)
- `backend/Dockerfile` + `backend/requirements.txt` — container para HF Space
- `scripts/deploy_space.py` — upload para HF Space via `huggingface_hub`
- `scripts/smoke_test.py` — teste de fumaça contra URL pública
- `docs/superpowers/SETUP_HF_SPACE.md` — checklist de setup manual
```

- [ ] **Step 3: Run full test suite one final time**

```bash
.venv/Scripts/python.exe -m pytest tests/ -v
```

Expected: all 71 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/SETUP_HF_SPACE.md docs/superpowers/HANDOFF.md
git commit -m "docs: add HF Space setup checklist and update HANDOFF for Wave 2 close"
```

---

## Wave 2 Acceptance Gate

Before marking Wave 2 complete, verify these manually (require a real HF Space):

1. `GET /health` → `200 {"status":"ok","models_loaded":true}` from public URL
2. `POST /ask {"question":"Como fazer merge em pandas?"}` → PT-BR answer + ≥ 1 citation
3. `scripts/smoke_test.py` exits with `Smoke test PASSED.`
4. Second call to `/ask` responds in < 3s (models are warm)
