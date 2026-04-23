"""Post-extraction repair for Google-Translated PT-BR docs.

Google Translate aggressively translates Python API terms ("array" →
"matriz", "broadcasting" → "transmissão", "Series" → "série", etc.). That
destroys RAG retrieval quality because the LLM cannot answer a question
about ``DataFrame.merge()`` when the ingested chunk says "mesclar".

This module applies a determistic regex-based repair on extracted PT text
BEFORE chunking. It is case-aware (title case vs. lowercase) and carefully
avoids false positives by using word boundaries and context guards.

Public API:
    repair_pt_text(text: str) -> str
"""
from __future__ import annotations

import re

# Each entry is (pattern, replacement). Patterns use `(?i)` inline flag for
# case-insensitivity, and replacements use a dispatcher that preserves the
# case of the match (UPPER / Title / lower).
#
# Patterns are ordered: more specific / longer patterns first so they match
# before a shorter pattern eats their prefix.

_RULES: list[tuple[str, str]] = [
    # --- NumPy API terms -----------------------------------------------
    # Title-case headings that Google likes to produce. These run BEFORE the
    # generic `matriz` rule so they match their full multi-word context first.
    (r"\bMatrizes NumPy\b", "NumPy arrays"),
    (r"\bMatriz NumPy\b", "NumPy array"),

    # "matriz"/"matrizes" → "array"/"arrays". Guarded against math/linalg
    # contexts where "matriz" is legitimately Portuguese (matriz de
    # covariância, matriz de correlação, matriz de confusão, matriz de
    # transição).
    (r"(?i)\bmatrizes\b(?!\s+de\s+(?:covar|correl|confus|transi))", "arrays"),
    (r"(?i)\bmatriz\b(?!\s+de\s+(?:covar|correl|confus|transi))", "array"),

    # "transmissão" / "difusão" → "broadcasting" (numpy concept)
    (r"(?i)\btransmissão\b", "broadcasting"),
    (r"(?i)\btransmiss(?:ões|oes)\b", "broadcasting"),
    (r"(?i)\bdifusão\b", "broadcasting"),

    # "fatiamento" / "fatiar" / "fatia" → slicing / slice
    (r"(?i)\bfatiamento\b", "slicing"),
    (r"(?i)\bfatiar\b", "slicing"),
    (r"(?i)\bfatias?\b(?!\s+de\s+)", "slice"),  # skip "fatia de pão" etc.

    # "funções universais" (when referring to ufunc) → "ufunc" only on the
    # first mention per page is overkill; just drop the parens that Google
    # leaves dangling when it translated "ufuncs" → "funções universais (ufunc)".
    (r"Funções universais\s*\(\s*\)", "ufunc"),
    (r"funções universais\s*\(\s*\)", "ufunc"),

    # --- pandas API terms ----------------------------------------------
    # "série" → "Series" ONLY when capitalized (Série) or when preceded by
    # "um "/"uma " (= pandas Series). "série temporal" stays.
    (r"\bSérie(s)?\b(?!\s+temporal)", lambda m: "Series"),
    (r"(?<=\buma )série\b(?!\s+temporal)", "Series"),
    (r"(?<=\bum )série\b(?!\s+temporal)", "Series"),

    # Spurious phrase from Google Translate
    (r"(?i)\bmoldura de dados\b", "DataFrame"),
    (r"(?i)\bquadro de dados\b", "DataFrame"),

    # "mesclar" → "merge" (pandas method name)
    (r"(?i)\bmesclar\b", "merge"),
    (r"(?i)\bmesclag(?:em|ens)\b", "merge"),

    # "empilhar" / "desempilhar" → stack / unstack when adjacent to
    # "tabela", "DataFrame", "Series", "colun", or at start of heading
    (r"(?i)\bempilhar\b(?=.{0,80}(DataFrame|Series|tabela|coluna|dados))", "stack"),
    (r"(?i)\bdesempilhar\b", "unstack"),

    # --- Generic cleanups ----------------------------------------------
    # Google often emits "Matriz NumPy" in titles — wrong casing for "array"
    (r"\bMatriz NumPy\b", "NumPy array"),
    (r"\bMatrizes NumPy\b", "NumPy arrays"),
]


def _case_preserving_sub(pattern: str, replacement: str, text: str) -> str:
    """Like re.sub but preserves the case of the original match.

    - If the matched span is ALL CAPS (e.g. "MATRIZ"), uppercase the replacement.
    - If it is Title Case (e.g. "Matriz"), title-case the replacement.
    - Otherwise lowercase.
    """
    def repl(m: re.Match) -> str:
        original = m.group(0)
        if original.isupper():
            return replacement.upper()
        if original[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement

    return re.sub(pattern, repl, text)


def repair_pt_text(text: str) -> str:
    """Apply all glossary repairs to PT-BR text extracted from Google-Translated
    HTML, returning the repaired text.

    The repair is idempotent: running it twice is equivalent to running it once,
    because each rule only matches the PT source terms, not their EN replacements.
    """
    out = text
    for pattern, replacement in _RULES:
        if callable(replacement):
            # Lambda / function replacement — re.sub handles case preservation
            # is not possible here; we trust the lambda's output directly.
            out = re.sub(pattern, replacement, out)
        else:
            out = _case_preserving_sub(pattern, replacement, out)
    return out
