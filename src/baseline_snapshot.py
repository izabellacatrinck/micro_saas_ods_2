"""One-shot: snapshot the current `rag_chunks` collection into
`rag_chunks_baseline_en` before the pipeline is regenerated for PT.

Run this ONCE, before re-running ingestion in PT. It preserves the existing
English embeddings (from all-MiniLM-L6-v2) so the baseline variant in the
evaluation can still be executed end-to-end.
"""
import chromadb

from src import config


def snapshot_baseline(source: str = "rag_chunks", target: str = config.COLLECTION_BASELINE_EN) -> int:
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    src = client.get_collection(source)
    try:
        client.delete_collection(target)
    except Exception:
        pass
    dst = client.create_collection(target)

    # Pull everything out in one shot (include embeddings so we can re-insert without recomputing).
    data = src.get(include=["documents", "metadatas", "embeddings"])

    ids = data["ids"]
    if not ids:
        print("source collection is empty; nothing to snapshot")
        return 0

    dst.add(
        ids=ids,
        documents=data["documents"],
        metadatas=data["metadatas"],
        embeddings=data["embeddings"],
    )
    print(f"[snapshot] copied {len(ids)} items from '{source}' -> '{target}'")
    return len(ids)


if __name__ == "__main__":
    snapshot_baseline()
