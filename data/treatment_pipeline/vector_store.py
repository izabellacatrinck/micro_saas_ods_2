import json
import chromadb


# =========================
# LOAD JSONL
# =========================
def load(path="chunks_with_embeddings.jsonl"):
    chunks = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    return chunks


# =========================
# CHROMA DB
# =========================
def create_db():

    client = chromadb.PersistentClient(path="./chroma_db")

    collection = client.get_or_create_collection(
        name="rag_chunks"
    )

    return collection


# =========================
# STORE
# =========================
def store(collection, chunks):

    ids = [c["chunk_id"] for c in chunks]
    texts = [c["content"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]

    metadatas = []

    for c in chunks:

        metadatas.append({
            "source": c["source"],
            "section": c["section"],
            "position": c["position"],
            "tokens": c["token_estimate"],
            "has_code": c["quality_flags"]["has_code"],
            "noise_score": c["quality_flags"]["noise_score"]
        })

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )


# =========================
# QUERY
# =========================
def query(collection, model, text, k=5):

    query_emb = model.encode([text]).tolist()

    return collection.query(
        query_embeddings=query_emb,
        n_results=k
    )


# =========================
# RUN
# =========================
def run():

    print("📦 Loading data...")
    chunks = load()

    print(f"✔ chunks: {len(chunks)}")

    print("🧠 Connecting ChromaDB...")
    collection = create_db()

    print("💾 Inserting vectors...")
    store(collection, chunks)

    print("✔ DONE")


if __name__ == "__main__":
    run()