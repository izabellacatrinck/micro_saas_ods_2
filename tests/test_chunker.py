import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

from main import SmartChunker, filter_by_quality
from main import ThematicSegmenter


def test_filter_by_quality_drops_noisy_chunks():
    chunks = [
        {"content": "In [1]: df.head()", "quality_flags": {"noise_score": 0.8}},
        {"content": "Good text " * 50, "quality_flags": {"noise_score": 0.1}},
        {"content": "borderline", "quality_flags": {"noise_score": 0.5}},
    ]

    kept = filter_by_quality(chunks, threshold=0.5)

    assert len(kept) == 1
    assert kept[0]["content"].startswith("Good text")


def test_filter_by_quality_keeps_all_if_threshold_high():
    chunks = [{"content": "x", "quality_flags": {"noise_score": 0.9}}]
    assert len(filter_by_quality(chunks, threshold=1.0)) == 1


def test_chunk_overlap_uses_sliding_window():
    """Each chunk after the first should begin with the last N words of the prev chunk."""
    text = " ".join([f"word{i}." for i in range(1000)])  # long text, many sentences
    chunker = SmartChunker(max_tokens=100, overlap=20)

    chunks = chunker.chunk(text)

    assert len(chunks) >= 2
    for i in range(1, len(chunks)):
        prev_words = chunks[i - 1].split()
        current_words = chunks[i].split()
        overlap_words = prev_words[-20:]
        # the first 20 words of the current chunk should equal the last 20 of previous
        assert current_words[:20] == overlap_words, (
            f"Chunk {i} does not start with sliding-window overlap of prev chunk"
        )


def test_chunk_no_overlap_on_first_chunk():
    text = " ".join([f"w{i}." for i in range(200)])
    chunker = SmartChunker(max_tokens=50, overlap=10)
    chunks = chunker.chunk(text)
    # the first chunk's first word should be original text's first word
    assert chunks[0].startswith("w0.")


def test_is_heading_accepts_uppercase_short_line():
    assert ThematicSegmenter.is_heading("GROUP BY OPERATIONS")


def test_is_heading_accepts_colon_terminated_short_line():
    assert ThematicSegmenter.is_heading("Introduction:")


def test_is_heading_rejects_short_body_text():
    # A 4-word lowercase sentence is NOT a heading
    assert not ThematicSegmenter.is_heading("This is a sentence.")


def test_is_heading_rejects_long_line():
    assert not ThematicSegmenter.is_heading("A" * 120)


from main import infer_library


def test_infer_library_detects_known_libraries():
    assert infer_library("numpy_docs.pdf") == "numpy"
    assert infer_library("data/pandas/intro.pdf") == "pandas"
    assert infer_library("Matplotlib_Tutorial.pdf") == "matplotlib"
    assert infer_library("seaborn_cheatsheet.pdf") == "seaborn"
    assert infer_library("random_doc.pdf") == "unknown"


def test_chunk_with_metadata_includes_new_fields():
    chunker = SmartChunker(max_tokens=50, overlap=10)
    text = "Texto de exemplo para testar o chunker em portugues. " * 30
    chunks = chunker.chunk_with_metadata(text, source="numpy_docs.pdf", section="intro")
    assert chunks, "expected at least one chunk"
    c = chunks[0]
    assert c["library"] == "numpy"
    assert c["language"] == "pt"
    assert c["original_lang"] == "en"
    assert c["source_type"] == "official_docs"
    # quality_flags preserved
    assert "noise_score" in c["quality_flags"]
