from typing import Any

from embeddings import DEFAULT_EMBEDDING_MODEL, embed_text
from vector_store import VectorStore


def retrieve_code(
    question: str,
    store: VectorStore,
    *,
    top_k: int = 5,
    client: Any | None = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[dict[str, Any]]:
    """Retrieve code chunks relevant to a natural-language question."""
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    query_embedding = embed_text(
        question,
        client=client,
        model=model,
    )

    return store.search(query_embedding, top_k=top_k)
