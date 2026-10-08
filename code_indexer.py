from pathlib import Path
from typing import Any

from code_chunker import chunk_files, chunk_repository_with_status
from embeddings import DEFAULT_EMBEDDING_MODEL, embed_chunks
from index_manifest import build_source_manifest, compare_manifests
from tools import PROJECT_ROOT
from vector_store import VectorStore


def build_code_index(
    root: Path = PROJECT_ROOT,
    *,
    client: Any | None = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
    max_chunks: int = 2_000,
    exclude_dirs: set[str] | None = None,
    exclude_files: set[str] | None = None,
) -> VectorStore:
    """Build an in-memory semantic index for a source repository."""
    chunks, chunk_limit_reached = chunk_repository_with_status(
        root,
        max_chunks=max_chunks,
        exclude_dirs=exclude_dirs or set(),
        exclude_files=exclude_files or set(),
    )

    embedded_chunks = embed_chunks(
        chunks,
        client=client,
        model=model,
    )

    store = VectorStore()
    store.add(embedded_chunks)

    source_manifest = build_source_manifest(
        root,
        exclude_dirs=exclude_dirs,
        exclude_files=exclude_files,
    )

    indexed_files = sorted({chunk["file_path"] for chunk in embedded_chunks})

    store.metadata = {
        "root": str(root.resolve()),
        "model": model,
        "source_manifest": source_manifest,
        "indexed_files": indexed_files,
        "unindexed_source_files": sorted(set(source_manifest) - set(indexed_files)),
        "exclude_dirs": sorted(exclude_dirs or set()),
        "exclude_files": sorted(exclude_files or set()),
        "max_chunks": max_chunks,
        "chunk_limit_reached": chunk_limit_reached,
    }

    return store


def update_code_index(
    store: VectorStore,
    root: Path = PROJECT_ROOT,
    *,
    client: Any | None = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
    max_chunks: int | None = None,
    exclude_dirs: set[str] | None = None,
    exclude_files: set[str] | None = None,
) -> VectorStore:
    """Update an existing code index using only changed source files."""
    root = root.resolve()
    excluded_dirs = exclude_dirs or set()
    excluded_files = exclude_files or set()

    current_manifest = build_source_manifest(
        root,
        exclude_dirs=excluded_dirs,
        exclude_files=excluded_files,
    )

    metadata = store.metadata

    if max_chunks is None:
        max_chunks = metadata.get("max_chunks", 2_000)

    # Check whether the existing index can be updated incrementally.
    compatible = (
        metadata.get("root") == str(root)
        and metadata.get("model") == model
        and metadata.get("max_chunks") == max_chunks
        and metadata.get("exclude_dirs") == sorted(excluded_dirs)
        and metadata.get("exclude_files") == sorted(excluded_files)
        and isinstance(metadata.get("source_manifest"), dict)
        and not metadata.get("chunk_limit_reached", True)
    )

    # Rebuild if the index is incomplete or its configuration changed.
    if not compatible:
        return build_code_index(
            root,
            client=client,
            model=model,
            max_chunks=max_chunks,
            exclude_dirs=excluded_dirs,
            exclude_files=excluded_files,
        )

    changes = compare_manifests(
        metadata["source_manifest"],
        current_manifest,
    )

    changed_files = set(changes["added"] + changes["modified"])
    deleted_files = set(changes["deleted"])

    # Nothing changed, so preserve the existing index.
    if not changed_files and not deleted_files:
        return store

    # Remove deleted files and replace modified files.
    files_to_remove = deleted_files | set(changes["modified"])

    # Only chunk files that were added or modified.

    # Only chunk files that were added or modified.
    new_chunks = (
        chunk_files(
            changed_files,
            root=root,
            exclude_dirs=excluded_dirs,
            exclude_files=excluded_files,
        )
        if changed_files
        else []
    )

    # Calculate the expected size after removing stale chunks.
    old_chunk_count = sum(
        1 for chunk in store.chunks if chunk["file_path"] in files_to_remove
    )

    projected_count = len(store) - old_chunk_count + len(new_chunks)

    # Rebuild if the updated index would exceed its configured limit.
    if projected_count > max_chunks:
        return build_code_index(
            root,
            client=client,
            model=model,
            max_chunks=max_chunks,
            exclude_dirs=excluded_dirs,
            exclude_files=excluded_files,
        )

    # Embed only chunks from added and modified files.

    # Embed only when there are chunks to process.
    if new_chunks:
        embedded_chunks = embed_chunks(
            new_chunks,
            client=client,
            model=model,
        )
    else:
        embedded_chunks = []

    # Replace stale chunks only after preparing the new chunks.
    for file_path in files_to_remove:
        store.remove_file(file_path)

    store.add(embedded_chunks)

    # Refresh metadata for the next incremental update.
    indexed_files = sorted({chunk["file_path"] for chunk in store.chunks})

    store.metadata = {
        **metadata,
        "root": str(root),
        "model": model,
        "source_manifest": current_manifest,
        "indexed_files": indexed_files,
        "unindexed_source_files": sorted(set(current_manifest) - set(indexed_files)),
        "exclude_dirs": sorted(excluded_dirs),
        "exclude_files": sorted(excluded_files),
        "max_chunks": max_chunks,
        "chunk_limit_reached": False,
    }

    return store
