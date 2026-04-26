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

    monkeypatch.setattr(m, "_state", {"ready": False, "chroma_count": 0})
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

    def always_raise(prompt):
        raise RuntimeError("both LLMs failed")

    monkeypatch.setattr(m, "_load_models", fake_load)
    monkeypatch.setattr(m, "retrieve", lambda question, top_k=15: [
        {"content": "doc", "metadata": {"source": "x", "section": "s", "library": "pandas"}}
    ])
    monkeypatch.setattr(m, "rerank", lambda question, chunks, top_n=5: chunks)
    monkeypatch.setattr(m, "build_pt_prompt", lambda question, chunks: "prompt")
    monkeypatch.setattr(m, "generate_answer_with_fallback", always_raise)

    with TestClient(m.app) as client:
        r = client.post("/ask", json={"question": "Como fazer merge?"})
    assert r.status_code == 502
    assert r.json()["detail"] == "LLM unavailable"
