"""Translator module: EN → PT via Groq, with code-block preservation,
technical glossary, and disk cache.

Public API:
    - extract_code_blocks(text) -> (stripped_text, [blocks])
    - restore_code_blocks(stripped, blocks) -> original_text
    - translate(text) -> pt_text           (added in Task 8)
    - translate_many(texts) -> [pt_text]   (added in Task 9)
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from groq import Groq

from src import config


# Match a run of consecutive "code-like" lines.
#   - starts with `>>>`
#   - starts with `In [N]:` or `Out[N]:`
#   - starts with 4+ spaces (indented block)
CODE_LINE = re.compile(
    r"^(?:>>>|\.\.\.|In \[\d+\]:|Out\[\d+\]:|\s{4,}\S).*$"
)

# Markdown-style fenced code block delimiter. Matches ``` or ```python etc.
FENCE = re.compile(r"^```\S*\s*$")


def extract_code_blocks(text: str) -> tuple[str, list[str]]:
    """Replace code sections with <CODE_BLOCK_N> placeholders.

    Two kinds of sections are recognized:
      - Markdown fenced code blocks (``` ... ```), typical of trafilatura
        output on HTML docs.
      - Runs of "code-like" lines (REPL prompts, Jupyter cells, or 4+ space
        indentation) — the original PDF-oriented behavior.

    Returns (stripped_text, list_of_original_blocks). Blocks preserve their
    original content verbatim, including fence lines when present.
    """
    lines = text.split("\n")
    out_lines: list[str] = []
    blocks: list[str] = []

    i = 0
    while i < len(lines):
        # Markdown fenced block takes precedence.
        if FENCE.match(lines[i]):
            j = i + 1
            while j < len(lines) and not FENCE.match(lines[j]):
                j += 1
            # Include opening and closing fences in the preserved block.
            end = j + 1 if j < len(lines) else j
            block_lines = lines[i:end]
            idx = len(blocks)
            blocks.append("\n".join(block_lines))
            out_lines.append(f"<CODE_BLOCK_{idx}>")
            i = end
            continue

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


# ---------------------------------------------------------------------------
# Disk-based translation cache + batch helper
# ---------------------------------------------------------------------------

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
