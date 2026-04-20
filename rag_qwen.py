import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# =========================
# MODELOS
# =========================

# embedding (recall)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 🔥 reranker (precision)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# LLM
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)


# =========================
# CHROMA
# =========================

client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_collection("rag_chunks")


# =========================
# RETRIEVE + RE-RANK
# =========================

def retrieve(query, k=15, final_k=5):
    """
    1. pega top-k com embeddings (recall)
    2. re-ranqueia com cross-encoder (precision)
    """

    # embedding da query
    query_emb = embedder.encode(query, normalize_embeddings=True).tolist()

    # busca inicial (recall alto)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=k,
        include=["documents", "metadatas"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # =========================
    # RE-RANK
    # =========================

    pairs = [(query, d) for d in docs]

    scores = reranker.predict(pairs)

    ranked = []

    for doc, meta, score in zip(docs, metas, scores):
        ranked.append({
            "text": doc,
            "source": meta.get("source", ""),
            "score": float(score)
        })

    # ordena pelo score do cross-encoder
    ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)

    # pega os melhores
    top_chunks = ranked[:final_k]

    # =========================
    # CONTEXTO
    # =========================

    context = []

    for i, c in enumerate(top_chunks):
        context.append(
            f"[DOC {i+1} | score={c['score']:.2f} | {c['source']}]\n{c['text']}"
        )

    return "\n\n".join(context)


# =========================
# PROMPT BUILDER
# =========================

def build_prompt(context, question):

    return f"""
You are a strict QA assistant.

RULES:
- Answer ONLY using the provided context
- If the answer is not in the context, say:
  "I don't know based on the provided context"
- Do NOT invent information
- Be concise and factual

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


# =========================
# GENERATE
# =========================

def generate(prompt):

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.0,
            do_sample=False,
            use_cache=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# =========================
# PIPELINE RAG
# =========================

def ask(question):

    context = retrieve(question)

    if not context.strip():
        return "I don't know based on the provided context."

    prompt = build_prompt(context, question)

    response = generate(prompt)

    return response


# =========================
# TESTE
# =========================

if __name__ == "__main__":

    print("\n🔥 RAG com RE-RANKER iniciado\n")

    while True:

        q = input("Pergunta: ")

        if q.lower() in ["exit", "quit"]:
            break

        answer = ask(q)

        print("\n" + "="*60)
        print(answer)
        print("="*60)