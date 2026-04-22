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
        """A heading must be short AND visually marked as one.

        Marked-as-heading means either all-caps or ending with a colon. We drop the
        bare "few words" rule — it produced too many false positives on short body
        sentences.
        """
        if not line or len(line) >= 80:
            return False
        return line.isupper() or line.endswith(":")

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
# LIBRARY INFERENCE
# =========================

def infer_library(source: str) -> str:
    """Return 'pandas' | 'numpy' | 'matplotlib' | 'seaborn' | 'unknown'
    based on the source filename/path."""
    s = source.lower()
    if "pandas" in s:
        return "pandas"
    if "numpy" in s:
        return "numpy"
    if "matplotlib" in s:
        return "matplotlib"
    if "seaborn" in s:
        return "seaborn"
    return "unknown"


# =========================
# SMART CHUNKER (PRODUCTION RAG)
# =========================

class SmartChunker:

    def __init__(self, max_tokens=450, overlap=80):
        self.max_tokens = max_tokens
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        sentences = SentenceSplitter.split(text)

        # First pass: pack sentences into base chunks without overlap.
        # Sentences that exceed max_tokens on their own are split by word count.
        base_chunks = []
        current = []
        current_len = 0

        def flush(buffer):
            if buffer:
                base_chunks.append(" ".join(buffer).strip())

        for sent in sentences:
            words = sent.split()
            sent_len = len(words)

            # If a single sentence is too long, split it into word-level sub-chunks.
            if sent_len > self.max_tokens:
                flush(current)
                current = []
                current_len = 0
                for j in range(0, sent_len, self.max_tokens):
                    sub = " ".join(words[j: j + self.max_tokens])
                    base_chunks.append(sub)
            elif current_len + sent_len <= self.max_tokens:
                current.append(sent)
                current_len += sent_len
            else:
                flush(current)
                current = [sent]
                current_len = sent_len

        flush(current)

        # Second pass: apply sliding-window overlap (last `overlap` words of prev
        # chunk prepended to next). Deterministic and free of duplication risk.
        if self.overlap <= 0 or len(base_chunks) <= 1:
            return base_chunks

        final = [base_chunks[0]]
        for i in range(1, len(base_chunks)):
            prev_words = base_chunks[i - 1].split()
            overlap_prefix = " ".join(prev_words[-self.overlap:])
            final.append(f"{overlap_prefix} {base_chunks[i]}".strip())

        return final

    def chunk_with_metadata(
        self,
        text: str,
        source: str,
        section: str,
        language: str = "pt",
        original_lang: str = "en",
        source_type: str = "official_docs",
    ):

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
                "language": language,
                "original_lang": original_lang,
                "source_type": source_type,
                "library": infer_library(source),
                "quality_flags": {
                    "has_code": "In [" in c or ">>>" in c,
                    "has_plot": any(x in c.lower() for x in ["plot", "chart", "graph", "gráfico"]),
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
# QUALITY FILTER
# =========================

def filter_by_quality(chunks: List[dict], threshold: float = 0.5) -> List[dict]:
    """Drop chunks whose noise_score is >= threshold."""
    return [
        c for c in chunks
        if c.get("quality_flags", {}).get("noise_score", 0.0) < threshold
    ]


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
    before = len(all_chunks)
    all_chunks = filter_by_quality(all_chunks, threshold=0.5)
    print(f"[quality] dropped {before - len(all_chunks)} noisy chunks")
    save_chunks(all_chunks)

    print("\n" + "=" * 60)
    print("✔ PIPELINE FINALIZADO")
    print(f"✔ Chunks finais: {len(all_chunks)}")
    print("✔ Qualidade: alta (RAG otimizado)")
    print("=" * 60)

    return all_chunks


if __name__ == "__main__":
    run_pipeline()