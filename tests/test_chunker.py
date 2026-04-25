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


def test_is_heading_accepts_markdown_h1():
    assert ThematicSegmenter.is_heading("# Introduction to pandas")


def test_is_heading_accepts_markdown_h2_h3():
    assert ThematicSegmenter.is_heading("## Series")
    assert ThematicSegmenter.is_heading("### From ndarray")


def test_is_heading_rejects_hash_without_space():
    # a trailing `#` from Sphinx anchors on a text line is NOT a heading
    assert not ThematicSegmenter.is_heading("This line ends with hash#")


def test_segment_splits_on_markdown_headings():
    text = (
        "# Introduction\n"
        "Pandas is a data analysis library.\n"
        "## Series\n"
        "A Series is a one-dimensional array.\n"
        "## DataFrame\n"
        "A DataFrame is a two-dimensional table.\n"
    )
    segs = ThematicSegmenter.segment(text)
    titles = [s["title"] for s in segs]
    assert "# Introduction" in titles
    assert "## Series" in titles
    assert "## DataFrame" in titles
    # each section has its body text
    intro = next(s for s in segs if s["title"] == "# Introduction")
    assert "data analysis library" in intro["content"]
    series = next(s for s in segs if s["title"] == "## Series")
    assert "one-dimensional array" in series["content"]


def test_segment_suppresses_colon_heuristic_when_markdown_present():
    """In markdown docs, lines ending with `:` are prose, not headings."""
    text = (
        "# Real heading\n"
        "First the output is:\n"       # prose ending in `:` — NOT a heading
        "Out[4]:\n"                     # code marker — NOT a heading
        "some result\n"
        "## Another section\n"
        "more content\n"
    )
    segs = ThematicSegmenter.segment(text)
    titles = [s["title"] for s in segs]
    # Only markdown headings become segment titles
    assert titles == ["# Real heading", "## Another section"]
    # Prose with `:` stays inside the preceding segment
    assert "First the output is:" in segs[0]["content"]
    assert "Out[4]:" in segs[0]["content"]


def test_segment_uses_colon_heuristic_when_no_markdown():
    """In plain PDF/Medium text (no `#`), the `:` heuristic still applies."""
    text = (
        "Introduction:\n"
        "This is the intro body.\n"
        "OVERVIEW:\n"
        "Overview body.\n"
    )
    segs = ThematicSegmenter.segment(text)
    titles = [s["title"] for s in segs]
    assert "Introduction:" in titles
    assert "OVERVIEW:" in titles


def test_segment_preserves_newlines_in_content():
    """Segment content must keep newlines so downstream regexes (e.g. ``` code
    fence detection in the translator) can still match line-anchored patterns.
    """
    text = (
        "# Heading\n"
        "Prose line.\n"
        "```\n"
        "code line\n"
        "```\n"
        "More prose.\n"
    )
    segs = ThematicSegmenter.segment(text)
    assert len(segs) == 1
    content = segs[0]["content"]
    # ``` fence markers must survive on their own lines
    assert "\n```\n" in content or content.startswith("```\n") or content.endswith("\n```")
    # basic content preserved
    assert "Prose line." in content
    assert "code line" in content


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
    chunks = chunker.chunk_with_metadata(text, source="numpy_docs.html", section="intro")
    assert chunks, "expected at least one chunk"
    c = chunks[0]
    assert c["library"] == "numpy"
    assert c["language"] == "pt"
    assert c["original_lang"] == "en"
    assert c["source_type"] == "official_docs"
    # default translation_source is "groq" (EN→PT via glossary-aware translation)
    assert c["translation_source"] == "groq"
    # quality_flags preserved
    assert "noise_score" in c["quality_flags"]


def test_chunk_with_metadata_accepts_translation_source():
    """PT official docs saved from Google Translate must be flagged as such
    so the evaluation can distinguish high-quality Groq translations from
    MT-based ones."""
    chunker = SmartChunker(max_tokens=50, overlap=10)
    text = "Texto em portugues nativo. " * 30
    chunks = chunker.chunk_with_metadata(
        text,
        source="10 minutos para Pandas.html",
        section="Intro",
        language="pt",
        original_lang="pt",
        source_type="official_docs",
        translation_source="google_translate",
    )
    assert chunks, "expected at least one chunk"
    c = chunks[0]
    assert c["library"] == "pandas"
    assert c["language"] == "pt"
    assert c["original_lang"] == "pt"
    assert c["source_type"] == "official_docs"
    assert c["translation_source"] == "google_translate"


def test_segmenter_splits_on_markdown_headings():
    """ThematicSegmenter should split on '# Title' markers, producing
    one segment per heading with clean title text."""
    md = (
        "# First\n"
        "Intro line.\n"
        "\n"
        "## Second\n"
        "Body of second.\n"
        "\n"
        "### Third\n"
        "Body of third.\n"
    )

    segments = ThematicSegmenter.segment(md)

    titles = [s["title"] for s in segments]
    assert "# First" in titles
    assert "## Second" in titles
    assert "### Third" in titles
    # Bodies stay attached to their own heading
    second = next(s for s in segments if s["title"] == "## Second")
    assert "Body of second" in second["content"]
    assert "Body of third" not in second["content"]


def test_chunker_respects_section_boundaries_via_pipeline():
    """Chunks should not bleed content across sections — the public pipeline
    (segment → chunk_with_metadata) must keep each section's chunks isolated."""
    md = (
        "# Intro\n"
        + ("alpha " * 200)
        + "\n\n"
        + "# Details\n"
        + ("beta " * 200)
    )

    segments = ThematicSegmenter.segment(md)
    chunker = SmartChunker(max_tokens=100, overlap=10)

    intro_chunks = chunker.chunk_with_metadata(
        text=next(s for s in segments if s["title"] == "# Intro")["content"],
        source="test.html", section="# Intro",
    )
    details_chunks = chunker.chunk_with_metadata(
        text=next(s for s in segments if s["title"] == "# Details")["content"],
        source="test.html", section="# Details",
    )

    for c in intro_chunks:
        assert "beta" not in c["content"], "Intro leaked into beta content"
        assert c["section"] == "# Intro"
    for c in details_chunks:
        assert "alpha" not in c["content"], "Details leaked into alpha content"
        assert c["section"] == "# Details"


def test_chunker_applies_overlap_within_long_section():
    """When a single section exceeds max_tokens, the sliding-window overlap
    must kick in BUT only inside that section (tested in isolation)."""
    long_section = " ".join(f"tok{i}" for i in range(500))  # 500 tokens
    chunker = SmartChunker(max_tokens=100, overlap=20)

    chunks = chunker.chunk(long_section)

    assert len(chunks) >= 5
    for i in range(1, len(chunks)):
        prev_tail = " ".join(chunks[i - 1].split()[-20:])
        assert chunks[i].startswith(prev_tail), (
            f"Chunk {i} missing overlap prefix from chunk {i-1}"
        )
