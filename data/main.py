from pathlib import Path
import re
import unicodedata
from typing import List, Dict
import json
import sys
from pathlib import Path as _Path

# Allow `from src.*` when running this file directly.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

# Windows consoles default to cp1252 and can't encode the unicode arrows /
# checkmarks we print below. Reconfigure stdout/stderr to UTF-8 so the
# pipeline log is readable regardless of OS locale.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from src.translator import translate_many
from src.medium_extractor import extract_medium_article
from src.html_extractor import extract_official_html
from src import config


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
# THEME SEGMENTER
# =========================

class ThematicSegmenter:

    # Markdown heading: one-or-more `#` followed by a space and content.
    _MD_HEADING = re.compile(r"^#+\s+\S")
    # Code-output markers that can terminate in `:` but are never headings.
    _CODE_MARKER = re.compile(r"^(?:In \[|Out\[)\d+\]:")

    @staticmethod
    def is_heading(line: str) -> bool:
        """A heading must be visually marked as one.

        Three signals (in priority order):
        - Markdown heading (`#`/`##`/...): recognized regardless of length.
        - All-caps line (PDF heading convention). Length-capped at 80.
        - Line ending with a colon. Length-capped at 80.

        Code output markers (`In [N]:`, `Out[N]:`) are explicitly excluded.
        """
        if not line:
            return False
        if ThematicSegmenter._MD_HEADING.match(line):
            return True
        if ThematicSegmenter._CODE_MARKER.match(line):
            return False
        if len(line) >= 80:
            return False
        return line.isupper() or line.endswith(":")

    @staticmethod
    def segment(text: str) -> List[Dict]:
        """Split text into (title, content) segments on heading lines.

        When the input contains markdown headings, only markdown headings are
        treated as segment boundaries — the colon/all-caps heuristics are
        suppressed to avoid false positives in prose that ends with `:`.
        Otherwise (no `#` headings detected) the heuristics are used, which
        matches the PDF-extracted and Medium-extracted text paths.
        """
        lines = text.split("\n")
        has_markdown = any(ThematicSegmenter._MD_HEADING.match(ln) for ln in lines)

        def line_is_heading(line: str) -> bool:
            if has_markdown:
                return bool(ThematicSegmenter._MD_HEADING.match(line))
            return ThematicSegmenter.is_heading(line)

        segments = []
        current_title = "Introduction"
        buffer = []

        def flush():
            if buffer:
                segments.append({
                    "title": current_title.strip(),
                    # Preserve line structure — downstream regexes (e.g. the
                    # translator's ``` code fence detection) need line anchors.
                    "content": "\n".join(buffer).strip()
                })

        for line in lines:
            line = line.strip()
            if line_is_heading(line):
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
        translation_source: str = "groq",
    ):
        """Split text into chunks with metadata.

        ``translation_source`` records how the PT text was obtained:
          - ``"groq"``: translated from EN by our glossary-aware Groq pipeline.
          - ``"google_translate"``: page was saved from Google Translate —
            lower MT quality, may have empty inline-code artifacts.
          - ``"native"``: content was originally authored in PT (e.g. Medium
            articles). No translation step at all.
        """

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
                "translation_source": translation_source,
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
# DOCUMENT PROCESSORS
# =========================

def process_en_official_html(
    html_file: Path,
    segmenter: "ThematicSegmenter",
    chunker: "SmartChunker",
) -> List[dict]:
    """Extract markdown from an EN official-docs HTML, segment, translate, chunk.

    The trafilatura output keeps markdown structure (headings, ``` fences),
    which the segmenter and translator guardrails rely on. We deliberately
    do NOT run TextNormalizer here — it was designed for PDF noise and
    would flatten newlines and strip code markers.
    """
    print(f"\n[EN→PT via groq] {html_file.name}")
    try:
        raw = extract_official_html(html_file)
    except ValueError as e:
        print(f"  skipped: {e}")
        return []
    if not raw.strip():
        return []

    segments = segmenter.segment(raw)
    # Translate at segment level (smaller than full doc, larger than chunk → good API granularity)
    pt_segments = translate_many([s["content"] for s in segments])

    doc_chunks = []
    for seg, pt_content in zip(segments, pt_segments):
        doc_chunks.extend(
            chunker.chunk_with_metadata(
                text=pt_content,
                source=html_file.name,
                section=seg["title"],
                language="pt",
                original_lang="en",
                source_type="official_docs",
                translation_source="groq",
            )
        )
    return doc_chunks


def process_pt_official_html(
    html_file: Path,
    segmenter: "ThematicSegmenter",
    chunker: "SmartChunker",
) -> List[dict]:
    """Extract markdown from a PT official-docs HTML (Google-translated), chunk.

    These pages were saved from Google Translate in the browser — the
    structural HTML (headings, ``` pre blocks, REPL cells) is preserved,
    but the MT quality is lower than Groq's and a small fraction of inline
    ``<code>`` tags may be empty. We tag ``translation_source="google_translate"``
    so downstream evaluation can weight them accordingly.

    Like ``process_en_official_html``, we skip TextNormalizer to keep the
    markdown line structure intact for code-fence detection.
    """
    print(f"\n[PT native-MT] {html_file.name}")
    try:
        raw = extract_official_html(html_file)
    except ValueError as e:
        print(f"  skipped: {e}")
        return []
    if not raw.strip():
        return []

    segments = segmenter.segment(raw)

    doc_chunks = []
    for seg in segments:
        doc_chunks.extend(
            chunker.chunk_with_metadata(
                text=seg["content"],
                source=html_file.name,
                section=seg["title"],
                language="pt",
                original_lang="pt",
                source_type="official_docs",
                translation_source="google_translate",
            )
        )
    return doc_chunks


def process_pt_medium_article(
    html_file: Path,
    segmenter: "ThematicSegmenter",
    chunker: "SmartChunker",
) -> List[dict]:
    """Extract a PT Medium article (native PT, author-written) and chunk.

    Unlike official docs, Medium HTML has a lot of boilerplate that
    TextNormalizer was designed to strip, so we keep that step here.
    """
    print(f"\n[PT medium] {html_file.name}")
    text = extract_medium_article(html_file)
    text = TextNormalizer.normalize(text)
    segments = segmenter.segment(text)

    doc_chunks = []
    for seg in segments:
        doc_chunks.extend(
            chunker.chunk_with_metadata(
                text=seg["content"],
                source=html_file.name,
                section=seg["title"],
                language="pt",
                original_lang="pt",
                source_type="medium_article",
                translation_source="native",
            )
        )
    return doc_chunks


# =========================
# PIPELINE
# =========================

# Per-library directory layout for official docs:
#   data/<library>/en/*.html  →  process_en_official_html (Groq translation)
#   data/<library>/pt/*.html  →  process_pt_official_html (Google Translate, no retranslation)
LIBRARY_DIRS = ["pandas", "numpy", "matplotlib", "seaborn"]


def run_pipeline():
    base = Path(__file__).resolve().parent  # data/

    medium_dir = base / "medium" / "raw"

    segmenter = ThematicSegmenter()
    chunker = SmartChunker()

    all_chunks = []

    print("=" * 60)
    print("RAG PIPELINE PT-BR - INGESTION")
    print("=" * 60)

    # --- Official docs: EN (translated via Groq) ---
    for library in LIBRARY_DIRS:
        if library in config.SKIP_EN_FOR_LIBRARIES:
            print(f"[skip] {library}/en: skipped per config.SKIP_EN_FOR_LIBRARIES "
                  f"(PT coverage is the source of truth for this library)")
            continue
        en_dir = base / library / "en"
        if not en_dir.exists():
            print(f"[skip] {library}/en: not found")
            continue
        for html_file in sorted(en_dir.glob("*.html")):
            all_chunks.extend(process_en_official_html(html_file, segmenter, chunker))

    # --- Official docs: PT (Google-translated, no retranslation) ---
    for library in LIBRARY_DIRS:
        pt_dir = base / library / "pt"
        if not pt_dir.exists():
            continue
        for html_file in sorted(pt_dir.glob("*.html")):
            all_chunks.extend(process_pt_official_html(html_file, segmenter, chunker))

    # --- Medium articles (native PT, no translation) ---
    if medium_dir.exists():
        for html_file in sorted(medium_dir.glob("*.html")):
            all_chunks.extend(process_pt_medium_article(html_file, segmenter, chunker))

    # --- Final cleaning ---
    all_chunks = deduplicate_chunks(all_chunks)
    before = len(all_chunks)
    all_chunks = filter_by_quality(all_chunks, threshold=config.NOISE_THRESHOLD)
    print(f"[quality] dropped {before - len(all_chunks)} noisy chunks")

    save_chunks(all_chunks, path=str(config.CHUNKS_PATH))

    print("\n" + "=" * 60)
    print(f"✔ INGESTÃO FINALIZADA — {len(all_chunks)} chunks em PT")
    print("=" * 60)

    from src.indexer import index_chunks
    index_chunks(
        chunks_path=config.CHUNKS_PATH,
        collection_name=config.COLLECTION_NEW_PT,
        embedder_model=config.EMBEDDER_MODEL,
    )

    return all_chunks


if __name__ == "__main__":
    run_pipeline()