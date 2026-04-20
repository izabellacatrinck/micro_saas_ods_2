import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from tqdm import tqdm


# =========================
# MODEL
# =========================
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# =========================
# LOAD JSONL
# =========================
def load_chunks(path="./chunks.jsonl") -> List[Dict]:
    chunks = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    return chunks


# =========================
# EMBEDDING BATCH
# =========================
def embed_chunks(chunks: List[Dict]) -> List[Dict]:

    texts = [c["content"] for c in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    enriched = []

    for chunk, emb in zip(chunks, embeddings):

        chunk["embedding"] = emb.tolist()
        enriched.append(chunk)

    return enriched


# =========================
# SAVE
# =========================
def save(path, chunks):

    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


# =========================
# PIPELINE
# =========================
def run():

    print("🔵 Loading chunks...")
    chunks = load_chunks()

    print(f"✔ Loaded: {len(chunks)} chunks")

    print("🧠 Creating embeddings...")
    chunks = embed_chunks(chunks)

    print("💾 Saving enriched JSONL...")
    save("chunks_with_embeddings.jsonl", chunks)

    print("✔ Done!")


if __name__ == "__main__":
    run()