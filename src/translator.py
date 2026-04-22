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

from groq import Groq

from src import config


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


# ---------------------------------------------------------------------------
# Groq client (None when GROQ_API_KEY is not set)
# ---------------------------------------------------------------------------

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

_PREAMBLE_RE = re.compile(
    r"^\s*(?:Tradução|Traducao|Aqui está|Aqui esta|Texto traduzido)\s*:?\s*\n",
    re.IGNORECASE,
)


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
