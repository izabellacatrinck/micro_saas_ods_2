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
import time
from pathlib import Path

from groq import Groq, RateLimitError, APIConnectionError

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
# Provider clients (lazy) — translator supports Groq and Gemini.
# ---------------------------------------------------------------------------

groq_client = Groq(api_key=config.GROQ_API_KEY) if config.GROQ_API_KEY else None

_gemini_model = None  # populated on first use

def _get_gemini_model():
    """Return a cached google.generativeai GenerativeModel or None if no key."""
    global _gemini_model
    if _gemini_model is not None:
        return _gemini_model
    if not config.GOOGLE_API_KEY:
        return None
    import google.generativeai as genai  # lazy import: optional dependency
    genai.configure(api_key=config.GOOGLE_API_KEY)
    _gemini_model = genai.GenerativeModel(
        model_name=config.GEMINI_TRANSLATION_MODEL,
        system_instruction=SYSTEM_PROMPT,
        generation_config={"temperature": 0.0, "max_output_tokens": 4000},
    )
    return _gemini_model

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


def _extract_retry_after(err: RateLimitError) -> float:
    """Parse 'Please try again in Xs' from Groq's rate-limit message, default 10s."""
    msg = str(err)
    m = re.search(r"try again in ([\d.]+)s", msg)
    return float(m.group(1)) + 0.5 if m else 10.0


def _translate_via_groq(stripped: str, max_retries: int) -> str:
    if groq_client is None:
        raise RuntimeError("GROQ_API_KEY not set — cannot translate via Groq")

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model=config.GROQ_TRANSLATION_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": stripped},
                ],
                temperature=0.0,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except RateLimitError as e:
            last_err = e
            wait = _extract_retry_after(e)
            print(f"  [groq rate-limit] waiting {wait:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
        except APIConnectionError as e:
            last_err = e
            wait = 2 ** attempt
            print(f"  [groq net-error] {e} — retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)

    raise RuntimeError(f"Groq translate failed after {max_retries} retries: {last_err}")


def _translate_via_gemini(stripped: str, max_retries: int) -> str:
    model = _get_gemini_model()
    if model is None:
        raise RuntimeError("GOOGLE_API_KEY not set — cannot translate via Gemini")

    # Gemini SDK raises google.api_core.exceptions.ResourceExhausted on 429 and
    # google.api_core.exceptions.ServiceUnavailable / DeadlineExceeded on network.
    from google.api_core import exceptions as gexc  # lazy

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = model.generate_content(stripped)
            # When the model refuses or returns no candidates, .text raises.
            return response.text
        except gexc.ResourceExhausted as e:
            last_err = e
            wait = 2 ** attempt * 5  # 5, 10, 20, 40, 80s
            print(f"  [gemini rate-limit] waiting {wait}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
        except (gexc.ServiceUnavailable, gexc.DeadlineExceeded, gexc.InternalServerError) as e:
            last_err = e
            wait = 2 ** attempt
            print(f"  [gemini net-error] {e} — retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)

    raise RuntimeError(f"Gemini translate failed after {max_retries} retries: {last_err}")


def translate(text: str, max_retries: int = 5, provider: str | None = None) -> str:
    """Translate a single passage EN → PT, preserving code blocks and glossary terms.

    Provider is chosen by config.TRANSLATION_PROVIDER ("gemini" or "groq"), or
    overridden per-call. Retries on rate limits and transient network errors.
    """
    provider = (provider or config.TRANSLATION_PROVIDER).lower()
    stripped, blocks = extract_code_blocks(text)

    if provider == "gemini":
        pt = _translate_via_gemini(stripped, max_retries)
    elif provider == "groq":
        pt = _translate_via_groq(stripped, max_retries)
    else:
        raise ValueError(f"Unknown TRANSLATION_PROVIDER: {provider!r}")

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
