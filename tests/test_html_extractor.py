"""Tests for the official-docs HTML extractor.

The extractor is expected to work on pages saved from pandas.pydata.org,
numpy.org, matplotlib.org and seaborn.pydata.org — all Sphinx-generated
documentation with similar structure.
"""
from src.html_extractor import extract_official_html


def test_extract_official_html_returns_main_content(tmp_path):
    """Strips nav/footer/sidebar, keeps main body text and code blocks."""
    html = """
    <!DOCTYPE html>
    <html lang="en">
      <head><title>Intro to data structures - pandas documentation</title></head>
      <body>
        <nav class="bd-navbar">Top navigation menu</nav>
        <aside class="bd-sidebar">Sidebar links</aside>
        <main>
          <section>
            <h1>Intro to data structures</h1>
            <p>We will start with a quick overview of the fundamental data structures in pandas.</p>
            <h2>Series</h2>
            <p>Series is a one-dimensional labeled array.</p>
            <pre><code class="python">
In [1]: import pandas as pd
In [2]: s = pd.Series([1, 2, 3])
            </code></pre>
            <p>The Series object has an index attribute.</p>
          </section>
        </main>
        <footer class="bd-footer">Copyright notice and footer links</footer>
      </body>
    </html>
    """
    html_file = tmp_path / "intro.html"
    html_file.write_text(html, encoding="utf-8")

    text = extract_official_html(html_file)

    # main body preserved
    assert "Intro to data structures" in text
    assert "Series is a one-dimensional labeled array" in text
    assert "index attribute" in text
    # code preserved (text form is fine, no HTML markup required)
    assert "import pandas as pd" in text
    assert "pd.Series" in text
    # boilerplate stripped
    assert "Top navigation menu" not in text
    assert "Sidebar links" not in text
    assert "Copyright notice" not in text
    # markdown structure preserved (headings + fenced code)
    assert "# Intro to data structures" in text
    assert "## Series" in text
    assert "```" in text  # fenced code block marker


def test_extract_official_html_raises_on_empty_html(tmp_path):
    """An HTML with no extractable main content raises ValueError."""
    html_file = tmp_path / "empty.html"
    html_file.write_text("<html><body></body></html>", encoding="utf-8")

    try:
        extract_official_html(html_file)
    except ValueError as e:
        assert "could not extract" in str(e).lower()
    else:
        raise AssertionError("expected ValueError for empty HTML")
