"""Tests for rag_query orchestrator.

Integration-style tests (that touch Chroma/Groq) are skipped unless the
env var RAG_E2E=1 is set. Pure-logic tests (prompt shape, rerank math,
citation formatting) always run.
"""
from __future__ import annotations

import os
import pytest
from src import config


def test_build_pt_prompt_format():
    from src.rag_query import build_pt_prompt

    chunks = [
        {"content": "DataFrame.merge combina tabelas.", "metadata": {
            "source": "pandas_merge.html", "section": "## Merge", "library": "pandas"}},
        {"content": "Use how='inner' para intersecão.", "metadata": {
            "source": "pandas_merge.html", "section": "## Merge", "library": "pandas"}},
    ]
    prompt = build_pt_prompt("como fazer merge?", chunks)

    # Must be in PT with clear instruction to only use context
    assert "português" in prompt.lower() or "pt-br" in prompt.lower()
    assert "contexto" in prompt.lower()
    # Must include both chunks verbatim
    assert "DataFrame.merge combina tabelas." in prompt
    assert "Use how='inner' para intersecão." in prompt
    # Must include the user question
    assert "como fazer merge?" in prompt
    # Must not leak the test scaffolding noise
    assert "<|system|>" not in prompt  # no Qwen-style markers


def test_format_citations_from_chunks():
    from src.rag_query import format_citations

    chunks = [
        {"metadata": {"source": "a.html", "section": "# X", "library": "pandas"}},
        {"metadata": {"source": "a.html", "section": "# X", "library": "pandas"}},  # dup
        {"metadata": {"source": "b.html", "section": "## Y", "library": "numpy"}},
    ]
    cites = format_citations(chunks)

    # Deduplicates by (source, section)
    assert len(cites) == 2
    assert {"source": "a.html", "section": "# X", "library": "pandas"} in cites
    assert {"source": "b.html", "section": "## Y", "library": "numpy"} in cites


@pytest.mark.skipif(os.environ.get("RAG_E2E") != "1",
                    reason="requires Chroma + Groq; set RAG_E2E=1")
def test_answer_end_to_end_returns_dict_with_expected_keys():
    from src.rag_query import answer

    result = answer("Como criar um DataFrame no pandas?")
    # Task 4 will add "variant" — use subset check so this test stays stable.
    assert {"answer", "citations", "retrieved_chunks"}.issubset(result.keys())
    assert isinstance(result["answer"], str) and result["answer"]
    assert isinstance(result["citations"], list)
    assert isinstance(result["retrieved_chunks"], list)
    assert len(result["retrieved_chunks"]) <= 5


def test_build_en_prompt_format():
    from src.rag_query import build_en_prompt

    chunks = [
        {"content": "DataFrame.merge joins tables.", "metadata": {
            "source": "pandas_merge.pdf", "section": "Merge", "library": "pandas"}},
    ]
    prompt = build_en_prompt("how to merge?", chunks)

    assert "english" in prompt.lower() or "in english" in prompt.lower()
    assert "context" in prompt.lower()
    assert "DataFrame.merge joins tables." in prompt
    assert "how to merge?" in prompt
    # Must NOT be in PT (sanity: no PT-specific stopwords in system instruction)
    assert "português" not in prompt.lower()


def test_variant_baseline_uses_en_stack(monkeypatch):
    """Calling answer(variant='baseline') must route through EN models and
    EN collection. We patch the internals to capture which names get used."""
    from src import rag_query as rq

    captured = {}

    def fake_embedder_loader(model_name):
        captured["embedder"] = model_name
        class E:
            def encode(self, texts, **kw):
                # Return unit vectors of correct dimension-shape-ish
                import numpy as np
                return np.zeros((len(texts), 4))
        return E()

    def fake_reranker_loader(model_name):
        captured["reranker"] = model_name
        class R:
            def predict(self, pairs):
                return [0.5 for _ in pairs]
        return R()

    def fake_collection_loader(name):
        captured["collection"] = name
        class C:
            def query(self, **kw):
                return {
                    "documents": [["doc1"]],
                    "metadatas": [[{"source": "x", "section": "y",
                                    "library": "pandas"}]],
                }
            name = "stub"
        return C()

    def fake_generate(prompt, model):
        captured["prompt"] = prompt
        captured["model"] = model
        return "stubbed answer"

    monkeypatch.setattr(rq, "_load_embedder", fake_embedder_loader)
    monkeypatch.setattr(rq, "_load_reranker", fake_reranker_loader)
    monkeypatch.setattr(rq, "_load_collection", fake_collection_loader)
    monkeypatch.setattr(rq, "generate_answer", fake_generate)
    # reset module-level caches so the loaders run with the baseline names
    monkeypatch.setattr(rq, "_embedder", None)
    monkeypatch.setattr(rq, "_reranker", None)
    monkeypatch.setattr(rq, "_collection", None)

    result = rq.answer("how to merge?", variant="baseline")

    from src import config
    assert captured["embedder"] == config.BASELINE_EMBEDDER
    assert captured["reranker"] == config.BASELINE_RERANKER
    assert captured["collection"] == config.COLLECTION_BASELINE_EN
    assert "english" in captured["prompt"].lower()
    assert result["answer"] == "stubbed answer"


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
