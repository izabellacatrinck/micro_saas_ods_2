"""Tests for rag_query orchestrator.

Integration-style tests (that touch Chroma/Groq) are skipped unless the
env var RAG_E2E=1 is set. Pure-logic tests (prompt shape, rerank math,
citation formatting) always run.
"""
from __future__ import annotations

import os
import pytest


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
