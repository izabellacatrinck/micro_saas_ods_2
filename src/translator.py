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
