import json
import math
from pathlib import Path
from typing import Any


INDEX_VERSION = 1


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """Calculate cosine similarity between two vectors."""
    if not vector_a or not vector_b:
        raise ValueError("Vectors must not be empty.")

    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimensions.")

    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in vector_a + vector_b
    ):
        raise ValueError("Vectors must contain only finite numbers.")

    norm_a = math.sqrt(sum(value * value for value in vector_a))
    norm_b = math.sqrt(sum(value * value for value in vector_b))

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Zero vectors are not supported.")

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    return dot_product / (norm_a * norm_b)


def _validate_chunk_metadata(chunk: dict[str, Any]) -> None:
    """Validate the required metadata for a code chunk."""
    required_fields = {
        "file_path": str,
        "start_line": int,
        "end_line": int,
        "language": str,
        "content": str,
    }

    for field, expected_type in required_fields.items():
        value = chunk.get(field)

        if not isinstance(value, expected_type):
            raise ValueError(f"Each chunk must have valid {field!r} metadata.")

    if not chunk["file_path"].strip():
        raise ValueError("Chunk file_path must not be empty.")

    if not chunk["language"].strip():
        raise ValueError("Chunk language must not be empty.")

    start_line = chunk["start_line"]
    end_line = chunk["end_line"]

    if isinstance(start_line, bool) or isinstance(end_line, bool):
        raise ValueError("Chunk line numbers must be integers.")

    if start_line <= 0:
        raise ValueError("Chunk start_line must be positive.")

    if end_line < start_line:
        raise ValueError("Chunk end_line must be greater than or equal to start_line.")


class VectorStore:
    """Store embedded code chunks and retrieve similar chunks."""

    def __init__(self) -> None:
        self._chunks: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}

    @property
    def chunks(self) -> list[dict[str, Any]]:
        """Return a copy of the indexed chunks."""
        return [dict(chunk) for chunk in self._chunks]

    def __len__(self) -> int:
        """Return the number of stored chunks."""
        return len(self._chunks)

    def add(self, chunks: list[dict[str, Any]]) -> None:
        """Validate an entire batch before modifying the store."""
        pending_chunks: list[dict[str, Any]] = []

        expected_dimensions = (
            len(self._chunks[0]["embedding"]) if self._chunks else None
        )

        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise ValueError("Each chunk must be a dictionary.")

            embedding = chunk.get("embedding")

            if not isinstance(embedding, list) or not embedding:
                raise ValueError("Each chunk must have a non-empty embedding.")

            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in embedding
            ):
                raise ValueError("Embeddings must contain finite numbers.")

            _validate_chunk_metadata(chunk)

            if expected_dimensions is None:
                expected_dimensions = len(embedding)
            elif len(embedding) != expected_dimensions:
                raise ValueError("All embeddings must have the same dimensions.")

            pending_chunks.append(dict(chunk))

        # Modify the store only after the entire batch is validated.
        self._chunks.extend(pending_chunks)

    def remove_file(self, file_path: str) -> int:
        """Remove all chunks belonging to a file.

        Returns:
            The number of chunks removed.
        """
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("file_path must be a non-empty string.")

        original_count = len(self._chunks)

        self._chunks = [
            chunk for chunk in self._chunks if chunk["file_path"] != file_path
        ]

        return original_count - len(self._chunks)

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the most similar chunks in descending score order."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if not isinstance(query_embedding, list) or not query_embedding:
            raise ValueError("Query embedding must not be empty.")

        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in query_embedding
        ):
            raise ValueError("Query embedding must contain finite numbers.")

        results: list[dict[str, Any]] = []

        for chunk in self._chunks:
            score = cosine_similarity(
                query_embedding,
                chunk["embedding"],
            )
            results.append({**chunk, "score": score})

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results[:top_k]

    def save(
        self,
        path: str | Path,
        *,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save the index and its metadata as JSON."""
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ValueError("model must be a non-empty string or None.")

        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary or None.")

        index_metadata = metadata if metadata is not None else self.metadata

        if not isinstance(index_metadata, dict):
            raise ValueError("Index metadata must be a dictionary.")

        index_data = {
            "version": INDEX_VERSION,
            "model": model,
            "metadata": index_metadata,
            "chunks": self._chunks,
        }

        try:
            with Path(path).open("w", encoding="utf-8") as file:
                json.dump(
                    index_data,
                    file,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Index contains data that cannot be serialized to JSON."
            ) from exc

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_model: str | None = None,
    ) -> "VectorStore":
        """Load and validate an index from a JSON file."""
        if expected_model is not None and (
            not isinstance(expected_model, str) or not expected_model.strip()
        ):
            raise ValueError("expected_model must be a non-empty string or None.")

        try:
            with Path(path).open("r", encoding="utf-8") as file:
                index_data = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError("Index file contains invalid JSON.") from exc
        except UnicodeDecodeError as exc:
            raise ValueError("Index file must contain valid UTF-8 text.") from exc

        if not isinstance(index_data, dict):
            raise ValueError("Index file must contain a JSON object.")

        version = index_data.get("version")

        if type(version) is not int or version != INDEX_VERSION:
            raise ValueError(f"Unsupported index version: {version!r}.")

        model = index_data.get("model")

        if model is not None and not isinstance(model, str):
            raise ValueError("Index model must be a string or None.")

        if expected_model is not None and model != expected_model:
            raise ValueError(
                f"Index model mismatch: expected {expected_model!r}, found {model!r}."
            )

        chunks = index_data.get("chunks")

        if not isinstance(chunks, list):
            raise ValueError("Index chunks must be a list.")

        metadata = index_data.get("metadata", {})

        if not isinstance(metadata, dict):
            raise ValueError("Index metadata must be a dictionary.")

        store = cls()
        store.add(chunks)
        store.metadata = metadata

        return store
