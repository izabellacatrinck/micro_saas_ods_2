import html2text
from pathlib import Path
from bs4 import BeautifulSoup
import argparse

def extract_main_content(html):
    soup = BeautifulSoup(html, "html.parser")

    # 🔥 remove lixo comum
    for tag in soup(["nav", "header", "footer", "aside", "script", "style"]):
        tag.decompose()

    # tenta pegar conteúdo principal
    main = soup.find("main")
    if main:
        return str(main)

    # fallback: body
    body = soup.find("body")
    return str(body) if body else html


def convert_html_to_md(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    converter = html2text.HTML2Text()
    converter.ignore_links = False
    converter.ignore_images = True
    converter.body_width = 0

    for html_file in input_dir.rglob("*.html"):
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()

        # 🔥 limpeza antes da conversão
        clean_html = extract_main_content(html_content)

        markdown = converter.handle(clean_html)

        output_file = output_dir / (html_file.stem + ".md")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown)

        print(f"Convertido: {html_file.name}")

    print("✅ Limpo e convertido!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    convert_html_to_md(args.input_dir, args.output_dir)