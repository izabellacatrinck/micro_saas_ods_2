import pdfplumber
from pathlib import Path
import re
import unicodedata
from typing import List, Dict
import json


# =========================
# TEXT NORMALIZER (ROBUSTO)
# =========================

class TextNormalizer:

    @staticmethod
    def normalize(text: str) -> str:

        if not text:
            return ""

        text = re.sub(r'file:///.*?\s', ' ', text)
        text = re.sub(r'\d{2}/\d{2}/\d{4},?\s*\d{2}:\d{2}', ' ', text)
        text = re.sub(r'\b\d+/\d+\b', ' ', text)

        # remove headers repetidos de docs
        text = re.sub(r'Chart visualization — pandas .*?documentation', ' ', text)

        # remove jupyter artifacts
        text = re.sub(r'In \[\d+\]:|Out\[\d+\]:', ' ', text)

        # remove múltiplos espaços
        text = re.sub(r'\s+', ' ', text)

        text = unicodedata.normalize('NFKD', text)

        text = ''.join(
            ch for ch in text
            if unicodedata.category(ch)[0] != 'C'
        )

        return text.strip()


# =========================
# SENTENCE SPLITTER (SEM NLTK)
# =========================

class SentenceSplitter:

    @staticmethod
    def split(text: str) -> List[str]:

        if not text:
            return []

        # remove linhas de código interativo
        text = re.sub(r"In \[\d+\]:.*", "", text)

        # split por frases
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

        return [
            s.strip()
            for s in sentences
            if len(s.strip()) > 20
        ]


# =========================
# PDF EXTRACTOR
# =========================

class PDFExtractor:

    @staticmethod
    def extract(pdf_path, save_debug_dir=None):

        full_text = []

        if save_debug_dir:
            save_debug_dir = Path(save_debug_dir)
            save_debug_dir.mkdir(parents=True, exist_ok=True)

        with pdfplumber.open(pdf_path) as pdf:

            for i, page in enumerate(pdf.pages):

                text = page.extract_text() or ""
                text = TextNormalizer.normalize(text)

                if len(text) < 40:
                    continue

                full_text.append(text)

                if save_debug_dir:
                    with open(save_debug_dir / f"page_{i:04d}.txt", "w", encoding="utf-8") as f:
                        f.write(text)

                print(f"[OK] {pdf_path.name} - página {i+1}/{len(pdf.pages)}")

        return "\n".join(full_text)


# =========================
# THEME SEGMENTER
# =========================

class ThematicSegmenter:

    @staticmethod
    def is_heading(line: str) -> bool:

        return (
            len(line) < 80
            and (
                line.isupper()
                or line.endswith(":")
                or len(line.split()) <= 6
            )
        )

    @staticmethod
    def segment(text: str) -> List[Dict]:

        lines = text.split("\n")

        segments = []
        current_title = "Introduction"
        buffer = []

        def flush():
            if buffer:
                segments.append({
                    "title": current_title.strip(),
                    "content": " ".join(buffer).strip()
                })

        for line in lines:

            line = line.strip()

            if ThematicSegmenter.is_heading(line):
                flush()
                current_title = line
                buffer = []
            else:
                buffer.append(line)

        flush()
        return segments


# =========================
# SMART CHUNKER (PRODUCTION RAG)
# =========================

class SmartChunker:

    def __init__(self, max_tokens=450, overlap=80):
        self.max_tokens = max_tokens
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:

        sentences = SentenceSplitter.split(text)

        chunks = []
        current = []
        current_len = 0

        def flush():
            nonlocal current, current_len
            if current:
                chunks.append(" ".join(current).strip())
                current = []
                current_len = 0

        for sent in sentences:

            sent_len = len(sent.split())

            if current_len + sent_len <= self.max_tokens:
                current.append(sent)
                current_len += sent_len
            else:
                flush()
                current = [sent]
                current_len = sent_len

        flush()

        # =========================
        # OVERLAP SEMÂNTICO REAL
        # =========================
        final = []

        for i, chunk in enumerate(chunks):

            if i == 0:
                final.append(chunk)
                continue

            prev_sentences = SentenceSplitter.split(chunks[i - 1])
            context = " ".join(prev_sentences[-2:]) if prev_sentences else ""

            final.append((context + " " + chunk).strip())

        return final

    def chunk_with_metadata(self, text: str, source: str, section: str):

        chunks = self.chunk(text)

        return [
            {
                "content": c,
                "source": source,
                "section": section,
                "chunk_id": f"{source}_{section}_{i}",
                "position": i,
                "char_count": len(c),
                "token_estimate": len(c.split()),

                # =========================
                # QUALIDADE PARA RAG
                # =========================
                "quality_flags": {
                    "has_code": "In [" in c,
                    "has_plot": any(x in c.lower() for x in ["plot", "chart", "graph"]),
                    "noise_score": self.noise_score(c)
                }
            }
            for i, c in enumerate(chunks)
        ]

    def noise_score(self, text: str) -> float:

        score = 0

        if "In [" in text:
            score += 0.4

        if "documentation" in text.lower():
            score += 0.2

        if len(text.split()) < 30:
            score += 0.3

        if text.count(".") < 2:
            score += 0.1

        return round(score, 2)


# =========================
# DEDUPLICATION
# =========================

def deduplicate_chunks(chunks: List[dict]):

    seen = set()
    clean = []

    for c in chunks:

        key = c["content"][:250]

        if key not in seen:
            seen.add(key)
            clean.append(c)

    return clean


# =========================
# SAVE
# =========================

def save_chunks(chunks, path="chunks.jsonl"):

    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


# =========================
# PIPELINE
# =========================

def run_pipeline():

    base = Path(".")

    pandas_path = base / "pandas"
    seaborn_path = base / "seaborn"
    numpy_pdf = base / "numpy_docs.pdf"
    matplotlib_pdf = base / "matplotlib_tutorial.pdf"

    extractor = PDFExtractor()
    segmenter = ThematicSegmenter()
    chunker = SmartChunker()

    all_chunks = []

    print("=" * 60)
    print("RAG PIPELINE - PRODUCTION GRADE")
    print("=" * 60)

    # =========================
    # PANDAS
    # =========================
    if pandas_path.exists():

        for pdf_file in pandas_path.glob("*.pdf"):

            print(f"\n📄 {pdf_file.name}")

            text = extractor.extract(
                pdf_file,
                save_debug_dir="output_texts/pandas"
            )

            segments = segmenter.segment(text)

            for seg in segments:

                chunks = chunker.chunk_with_metadata(
                    text=seg["content"],
                    source=pdf_file.name,
                    section=seg["title"]
                )

                all_chunks.extend(chunks)

    # =========================
    # SEABORN
    # =========================
    if seaborn_path.exists():

        for pdf_file in seaborn_path.rglob("*.pdf"):

            print(f"\n📄 {pdf_file.name}")

            text = extractor.extract(
                pdf_file,
                save_debug_dir="output_texts/seaborn"
            )

            segments = segmenter.segment(text)

            for seg in segments:

                chunks = chunker.chunk_with_metadata(
                    text=seg["content"],
                    source=pdf_file.name,
                    section=seg["title"]
                )

                all_chunks.extend(chunks)

    # =========================
    # NUMPY
    # =========================
    if numpy_pdf.exists():

        print(f"\n📄 numpy_docs.pdf")

        text = extractor.extract(
            numpy_pdf,
            save_debug_dir="output_texts/numpy"
        )

        segments = segmenter.segment(text)

        for seg in segments:

            chunks = chunker.chunk_with_metadata(
                text=seg["content"],
                source="numpy_docs.pdf",
                section=seg["title"]
            )

            all_chunks.extend(chunks)

    # =========================
    # MATPLOTLIB
    # =========================
    if matplotlib_pdf.exists():

        print(f"\n📄 matplotlib_tutorial.pdf")

        text = extractor.extract(
            matplotlib_pdf,
            save_debug_dir="output_texts/matplotlib"
        )

        segments = segmenter.segment(text)

        for seg in segments:

            chunks = chunker.chunk_with_metadata(
                text=seg["content"],
                source="matplotlib_tutorial.pdf",
                section=seg["title"]
            )

            all_chunks.extend(chunks)

    # =========================
    # FINAL CLEANING
    # =========================
    all_chunks = deduplicate_chunks(all_chunks)
    save_chunks(all_chunks)

    print("\n" + "=" * 60)
    print("✔ PIPELINE FINALIZADO")
    print(f"✔ Chunks finais: {len(all_chunks)}")
    print("✔ Qualidade: alta (RAG otimizado)")
    print("=" * 60)

    return all_chunks


if __name__ == "__main__":
    run_pipeline()