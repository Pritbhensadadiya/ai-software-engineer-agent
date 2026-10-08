from pathlib import Path
from typing import Any

from code_indexer import build_code_index, update_code_index
from embeddings import DEFAULT_EMBEDDING_MODEL
from tools import PROJECT_ROOT
from vector_store import VectorStore


DEFAULT_INDEX_PATH = PROJECT_ROOT / ".code_index.json"


def load_or_build_code_index(
    root: Path = PROJECT_ROOT,
    *,
    index_path: Path = DEFAULT_INDEX_PATH,
    client: Any | None = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
    max_chunks: int = 2_000,
    exclude_dirs: set[str] | None = None,
    exclude_files: set[str] | None = None,
) -> VectorStore:
    """
    Load an existing code index and incrementally update it.

    If no saved index exists, build a new index.
    """
    root = root.resolve()
    index_path = index_path.resolve()

    if index_path.exists():
        store = VectorStore.load(index_path)

        updated_store = update_code_index(
            store,
            root,
            client=client,
            model=model,
            max_chunks=max_chunks,
            exclude_dirs=exclude_dirs,
            exclude_files=exclude_files,
        )

        updated_store.save(index_path)

        return updated_store

    store = build_code_index(
        root,
        client=client,
        model=model,
        max_chunks=max_chunks,
        exclude_dirs=exclude_dirs,
        exclude_files=exclude_files,
    )

    store.save(index_path)

    return store


def refresh_code_index(
    store: VectorStore,
    root: Path = PROJECT_ROOT,
    *,
    index_path: Path = DEFAULT_INDEX_PATH,
    client: Any | None = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
    max_chunks: int = 2_000,
    exclude_dirs: set[str] | None = None,
    exclude_files: set[str] | None = None,
) -> VectorStore:
    """
    Incrementally update a code index and persist it.
    """
    updated_store = update_code_index(
        store,
        root,
        client=client,
        model=model,
        max_chunks=max_chunks,
        exclude_dirs=exclude_dirs,
        exclude_files=exclude_files,
    )

    updated_store.save(index_path)

    return updated_store
