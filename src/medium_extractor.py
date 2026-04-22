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
