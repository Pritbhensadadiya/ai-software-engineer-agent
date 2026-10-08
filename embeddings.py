import math
import os
from typing import Any

from dotenv import load_dotenv
from google import genai

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"

def embed_text(
    text: str,
    *,
    client: Any | None = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[float]:
    """Generate and validate an embedding vector for a text input.

    Pass a client explicitly to make the function easy to test
    without making real API requests.
    """
    if not text or not text.strip():
        raise ValueError("Text to embed must not be empty.")

    if not model or not model.strip():
        raise ValueError("Embedding model must not be empty.")

    if client is None:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    response = client.models.embed_content(
        model=model,
        contents=text,
    )

    embeddings = getattr(response, "embeddings", None)

    if not embeddings:
        raise ValueError("Embedding API returned no embeddings.")

    values = getattr(embeddings[0], "values", None)

    if not values:
        raise ValueError("Embedding API returned an empty vector.")

    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("Embedding vector contains invalid values.") from exc

    if not all(math.isfinite(value) for value in vector):
        raise ValueError("Embedding vector contains non-finite values.")

    return vector


def embed_chunks(
    chunks: list[dict[str, Any]],
    *,
    client: Any | None = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[dict[str, Any]]:
    """Generate embeddings while preserving each chunk's metadata."""
    embedded_chunks = []

    for chunk in chunks:
        vector = embed_text(
            chunk["content"],
            client=client,
            model=model,
        )

        embedded_chunks.append(
            {
                **chunk,
                "embedding": vector,
            }
        )

    return embedded_chunks
