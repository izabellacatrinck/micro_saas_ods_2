"""Build the EN baseline Chroma collection (`rag_chunks_baseline_en`).

Why this exists
---------------
The original `src.baseline_snapshot` assumed an in-place EN `rag_chunks`
collection produced by a previous run of the pipeline. After the PT refactor
that source collection no longer exists, so we rebuild the EN baseline from
the raw corpus instead: `data/<library>/en/*.html`.

Pipeline (mirrors `data/main.py` but EN-only, no translation):
  1. trafilatura → markdown (via `src.html_extractor`)
  2. ThematicSegmenter + SmartChunker (reused from `data/main.py`)
  3. Save to `data/baseline_chunks.jsonl`
  4. Index into `rag_chunks_baseline_en` with `BASELINE_EMBEDDER`
     (all-MiniLM-L6-v2), keeping the e5-style "passage: " / "query: "
     prefix convention so retrieval matches `src.rag_query`.

Run once with:
    uv run python -m src.baseline_ingest
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import List

from src import config
from src.html_extractor import extract_official_html


def _load_data_main():
    """Import `data/main.py` as a module without making `data` a package."""
    path = config.DATA_DIR / "main.py"
    spec = importlib.util.spec_from_file_location("_data_main", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_data_main"] = module
    spec.loader.exec_module(module)
    return module


def build_baseline_chunks() -> List[dict]:
    dm = _load_data_main()
    segmenter = dm.ThematicSegmenter()
    chunker = dm.SmartChunker()

    base = config.DATA_DIR
    all_chunks: list[dict] = []

    print("=" * 60)
    print("EN BASELINE INGESTION (no translation)")
    print("=" * 60)

    for library in config.LIBRARIES:
        en_dir = base / library / "en"
        if not en_dir.exists():
            print(f"[skip] {library}/en: not found")
            continue
        for html_file in sorted(en_dir.glob("*.html")):
            print(f"[en] {html_file.name}")
            try:
                raw = extract_official_html(html_file)
            except ValueError as e:
                print(f"  skipped: {e}")
                continue
            if not raw.strip():
                continue
            for seg in segmenter.segment(raw):
                all_chunks.extend(
                    chunker.chunk_with_metadata(
                        text=seg["content"],
                        source=html_file.name,
                        section=seg["title"],
                        language="en",
                        original_lang="en",
                        source_type="official_docs",
                        translation_source="native",
                    )
                )

    # The chunker tags everything language="pt" by default; chunk_with_metadata
    # already accepts language="en" above, so the rows are correct. Patch any
    # rows the helper might still default-tag (defensive — current code path
    # respects the override).
    for c in all_chunks:
        c["language"] = "en"
        c["original_lang"] = "en"

    all_chunks = dm.deduplicate_chunks(all_chunks)
    before = len(all_chunks)
    all_chunks = dm.filter_by_quality(all_chunks, threshold=config.NOISE_THRESHOLD)
    print(f"[quality] dropped {before - len(all_chunks)} noisy chunks")

    return all_chunks


def save_chunks(chunks: List[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


BASELINE_CHUNKS_PATH = config.DATA_DIR / "baseline_chunks.jsonl"


def main() -> int:
    # Imported lazily so that sentence_transformers / chromadb are loaded
    # AFTER trafilatura + data/main.py — on Windows the reverse order has
    # been observed to segfault during native-lib initialization.
    from src.indexer import index_chunks

    chunks = build_baseline_chunks()
    if not chunks:
        print("no chunks produced — aborting before index")
        return 1

    save_chunks(chunks, BASELINE_CHUNKS_PATH)
    print(f"\n✔ wrote {len(chunks)} chunks to {BASELINE_CHUNKS_PATH}")

    n = index_chunks(  # noqa: F821 — bound by lazy import above
        chunks_path=BASELINE_CHUNKS_PATH,
        collection_name=config.COLLECTION_BASELINE_EN,
        embedder_model=config.BASELINE_EMBEDDER,
        e5_style_prefix=True,
    )
    print(f"✔ indexed {n} chunks into '{config.COLLECTION_BASELINE_EN}'")
    return 0


if __name__ == "__main__":
    rc = main()
    # Bypass Python's normal shutdown — sentence_transformers/torch + chromadb
    # native-lib teardown intermittently segfaults on Windows after the work
    # is already done. os._exit returns the rc immediately without finalizers.
    import os
    os._exit(rc)
