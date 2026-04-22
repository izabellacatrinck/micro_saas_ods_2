"""Extract clean markdown text from official documentation HTML pages.

Target: Sphinx-generated docs for pandas, numpy, matplotlib and seaborn saved
locally as .html. Uses trafilatura in markdown mode so headings stay as
`##` and code blocks stay as ``` fences — both signals are used downstream by
the segmenter and the translator code-block guardrail.

Kept as a separate module from `medium_extractor` because the two corpora may
need to diverge in extraction config later (e.g. different boilerplate rules).
"""
from pathlib import Path

import trafilatura


def extract_official_html(html_path: Path) -> str:
    """Return cleaned main-content markdown from a saved official-docs HTML file.

    Output keeps markdown structure: `#`/`##` headings and ``` fenced code
    blocks, which the downstream segmenter and translator depend on.

    Raises ValueError if trafilatura cannot extract content (empty page, pure
    boilerplate, etc.).
    """
    html = html_path.read_text(encoding="utf-8")
    text = trafilatura.extract(
        html,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        favor_recall=True,
    )
    if text is None or not text.strip():
        raise ValueError(f"trafilatura could not extract text from {html_path}")
    return text.strip()
