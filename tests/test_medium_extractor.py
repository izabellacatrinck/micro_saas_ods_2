from src.medium_extractor import extract_medium_article


def test_extract_medium_article_returns_clean_text(tmp_path):
    html = """
    <html>
      <head><title>Análise de dados com pandas</title></head>
      <body>
        <nav>Menu</nav>
        <article>
          <h1>Análise de dados com pandas</h1>
          <p>Pandas é uma biblioteca para análise de dados em Python.</p>
          <p>Ela usa DataFrames para representar tabelas.</p>
        </article>
        <footer>Rodapé</footer>
      </body>
    </html>
    """
    html_file = tmp_path / "article.html"
    html_file.write_text(html, encoding="utf-8")

    text = extract_medium_article(html_file)

    assert "Pandas é uma biblioteca" in text
    assert "DataFrames" in text
    assert "Menu" not in text  # nav stripped
    assert "Rodapé" not in text  # footer stripped
