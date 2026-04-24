"""Embed chunks and upsert into a ChromaDB collection.

Uses the e5-small embedder. Because e5 was trained with a "passage: " prefix
for documents and "query: " prefix for queries, we apply that convention here
(documents get "passage: " prepended before encoding).
"""
from __future__ import annotations

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src import config
from src.device import get_device


def index_chunks(
    chunks_path: Path = config.CHUNKS_PATH,
    collection_name: str = config.COLLECTION_NEW_PT,
    embedder_model: str = config.EMBEDDER_MODEL,
    e5_style_prefix: bool = True,
) -> int:
    """Embed chunks from `chunks_path` and upsert into Chroma.

    Returns the number of chunks indexed.
    """
    embedder = SentenceTransformer(embedder_model, device=get_device())

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    # delete-and-recreate to ensure embeddings are consistent with current model
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name)

    with chunks_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if not rows:
        return 0

    texts = [
        (f"passage: {r['content']}" if e5_style_prefix else r["content"])
        for r in rows
    ]
    ids = [r["chunk_id"] for r in rows]
    metadatas = [
        {
            "source": r["source"],
            "section": r["section"],
            "library": r.get("library", "unknown"),
            "language": r.get("language", "pt"),
            "source_type": r.get("source_type", "official_docs"),
        }
        for r in rows
    ]

    batch = 64
    for start in range(0, len(texts), batch):
        end = start + batch
        embs = embedder.encode(
            texts[start:end],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        collection.add(
            ids=ids[start:end],
            documents=[r["content"] for r in rows[start:end]],  # store raw content (no prefix)
            embeddings=embs,
            metadatas=metadatas[start:end],
        )

    print(f"[index] {len(texts)} chunks upserted into '{collection_name}'")
    return len(texts)


if __name__ == "__main__":
    index_chunks()
