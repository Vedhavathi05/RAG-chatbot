"""
Retriever
Vector retrieval + CrossEncoder reranking
Aligned with rag_index ingestion pipeline
"""

import os
import chromadb
import traceback

from models.embedding_model import get_embedder
from models.reranker import get_reranker


# ---------------------------------------------------
# Resolve PROJECT ROOT (where run.py exists)
# ---------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.getenv(
    "PROJECT_ROOT",
    os.path.abspath(os.path.join(CURRENT_DIR, ".."))
)

RAG_INDEX = os.path.join(PROJECT_ROOT, "rag_index")
CHROMA_PATH = os.path.join(RAG_INDEX, "chroma_db")

print(f"[Retriever] Project root: {PROJECT_ROOT}")
print(f"[Retriever] Chroma path: {CHROMA_PATH}")


# ---------------------------------------------------
# Initialize ChromaDB
# ---------------------------------------------------
client = None

try:
    print("[Retriever] Initializing ChromaDB client...")

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    print("[Retriever] ✓ ChromaDB client created")

except Exception as e:
    print(f"[Retriever] ✗ ChromaDB init failed: {e}")
    traceback.print_exc()


# ---------------------------------------------------
# Load Collection
# ---------------------------------------------------
collection = None

try:
    if client:
        print("[Retriever] Loading collection 'medical_guidelines'...")
        collection = client.get_collection("medical_guidelines")
        print("[Retriever] ✓ Collection loaded")

except Exception as e:
    print(f"[Retriever] ✗ Collection not found: {e}")
    print("[Retriever] Run build_index.py first.")
    traceback.print_exc()


# ---------------------------------------------------
# Load Embedder
# ---------------------------------------------------
embedder = None

try:
    print("[Retriever] Loading embedder...")
    embedder = get_embedder()
    print("[Retriever] ✓ Embedder loaded")

except Exception as e:
    print(f"[Retriever] ✗ Embedder load failed: {e}")
    traceback.print_exc()


# ---------------------------------------------------
# Load Reranker
# ---------------------------------------------------
reranker = None

try:
    print("[Retriever] Loading reranker...")
    reranker = get_reranker()
    print("[Retriever] ✓ Reranker loaded")

except Exception as e:
    print(f"[Retriever] ✗ Reranker load failed: {e}")
    traceback.print_exc()


# ---------------------------------------------------
# Deduplicate chunks
# ---------------------------------------------------
def deduplicate_chunks(results):
    seen = set()
    unique = []

    for item in results:
        key = item["text"][:120]

        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


# ---------------------------------------------------
# Retrieval
# ---------------------------------------------------
def retrieve(query, k=5):

    if not collection or not embedder:
        return [{
            "rank": 1,
            "chunk_id": "error",
            "text": "Retriever not initialized.",
            "score": 0.0,
            "source": "System",
            "position": 0,
        }]

    try:

        # -------------------------------
        # BGE query formatting
        # -------------------------------
        bge_query = (
            "Represent this sentence for searching relevant passages: "
            + query.strip()
        )

        vec = embedder.encode(
            [bge_query],
            normalize_embeddings=True
        )[0].tolist()

        # retrieve more candidates for reranking
        candidate_k = max(80, k * 10)

        # -------------------------------
        # Vector Search (Chroma)
        # -------------------------------
        result = collection.query(
            query_embeddings=[vec],   # ✅ FIXED
            n_results=candidate_k,    # ✅ FIXED
            include=["documents", "metadatas", "distances"]
        )

        docs = result["documents"][0]
        ids = result["ids"][0]
        dists = result["distances"][0]
        metas = result["metadatas"][0]

        output = []

        for i, (doc, cid, dist, meta) in enumerate(
            zip(docs, ids, dists, metas)
        ):
            meta = meta or {}

            output.append({
                "rank": i + 1,
                "chunk_id": cid,
                "text": doc,
                "vector_score": 1 - float(dist),
                "source": meta.get("source", "Unknown"),
                "position": meta.get("position", 0),
            })

        output = deduplicate_chunks(output)

        # -------------------------------
        # CrossEncoder Reranking
        # -------------------------------
        if reranker and output:

            pairs = [(query, c["text"]) for c in output]
            scores = reranker.predict(pairs)

            for c, rs in zip(output, scores):
                c["final_score"] = (
                    0.3 * c["vector_score"]
                    + 0.7 * float(rs)
                )

            output.sort(
                key=lambda x: x["final_score"],
                reverse=True
            )

        else:
            output.sort(
                key=lambda x: x["vector_score"],
                reverse=True
            )

        # Final ranking
        for i, item in enumerate(output):
            item["rank"] = i + 1
            item["score"] = item.get(
                "final_score",
                item["vector_score"]
            )

        return output[:k]

    except Exception as e:
        print(f"[Retriever] Retrieval error: {e}")
        traceback.print_exc()

        return [{
            
    "rank": int,
    "chunk_id": str,
    "text": str,
    "score": float,
    "source": str,
    "position": int,

        }]