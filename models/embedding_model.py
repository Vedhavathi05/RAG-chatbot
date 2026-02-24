from sentence_transformers import SentenceTransformer

_model = None


def get_embedder():
    global _model

    if _model is None:
        # Excellent retrieval model for RAG
        _model = SentenceTransformer(
            "BAAI/bge-base-en-v1.5"
        )

    return _model