from pathlib import Path

import code_indexer
from code_indexer import build_code_index, update_code_index
from vector_store import VectorStore


class FakeModels:
    def __init__(self):
        self.calls = []

    def embed_content(self, *, model, contents):
        self.calls.append({"model": model, "contents": contents})

        # Deterministic vectors for offline tests.
        values = [1.0, 0.0] if "path" in contents.lower() else [0.0, 1.0]
        embedding = type("Embedding", (), {"values": values})()
        return type("EmbeddingResponse", (), {"embeddings": [embedding]})()


class FakeClient:
    def __init__(self):
        self.models = FakeModels()


def make_chunk(
    file_path,
    content="original code",
    embedding=None,
    start_line=1,
    end_line=1,
):
    return {
        "file_path": file_path,
        "start_line": start_line,
        "end_line": end_line,
        "language": "Python",
        "content": content,
        "embedding": embedding if embedding is not None else [1.0, 0.0],
    }


def make_indexed_store(root, chunks, manifest, max_chunks=10):
    store = VectorStore()
    store.add(chunks)
    store.metadata = {
        "root": str(root.resolve()),
        "model": code_indexer.DEFAULT_EMBEDDING_MODEL,
        "source_manifest": manifest,
        "indexed_files": sorted({chunk["file_path"] for chunk in chunks}),
        "unindexed_source_files": [],
        "exclude_dirs": [],
        "exclude_files": [],
        "max_chunks": max_chunks,
        "chunk_limit_reached": False,
    }
    return store


# ---------------------------------------------------------------------------
# Full index builds
# ---------------------------------------------------------------------------


def test_build_code_index_embeds_and_stores_chunks(monkeypatch):
    chunks = [
        {
            "file_path": "tools.py",
            "start_line": 1,
            "end_line": 3,
            "language": "Python",
            "content": "def validate_path(path): pass",
        },
        {
            "file_path": "agent.py",
            "start_line": 5,
            "end_line": 8,
            "language": "Python",
            "content": "def run_agent(): pass",
        },
    ]

    def fake_chunk_repository_with_status(
        root, *, max_chunks, exclude_dirs, exclude_files
    ):
        assert max_chunks == 10
        assert exclude_dirs == set()
        assert exclude_files == set()
        return chunks, False

    monkeypatch.setattr(
        code_indexer,
        "chunk_repository_with_status",
        fake_chunk_repository_with_status,
    )

    client = FakeClient()
    store = build_code_index(Path("."), client=client, max_chunks=10)

    assert len(store) == 2
    assert len(client.models.calls) == 2

    results = store.search([1.0, 0.0], top_k=2)
    assert results[0]["file_path"] == "tools.py"
    assert results[0]["embedding"] == [1.0, 0.0]
    assert results[1]["file_path"] == "agent.py"


def test_build_code_index_returns_empty_store_for_empty_repository(monkeypatch):
    monkeypatch.setattr(
        code_indexer,
        "chunk_repository_with_status",
        lambda root, **kwargs: ([], False),
    )

    client = FakeClient()
    store = build_code_index(Path("."), client=client)

    assert len(store) == 0
    assert client.models.calls == []


def test_build_code_index_passes_model_to_embedding_client(monkeypatch):
    chunks = [make_chunk("example.py", "print('hello')")]
    chunks[0].pop("embedding")

    monkeypatch.setattr(
        code_indexer,
        "chunk_repository_with_status",
        lambda root, **kwargs: (chunks, False),
    )

    client = FakeClient()
    build_code_index(Path("."), client=client, model="custom-model")

    assert client.models.calls[0]["model"] == "custom-model"


def test_build_code_index_passes_exclusions_to_chunker(monkeypatch):
    chunks = [make_chunk("src/app.py", "print('hello')")]
    chunks[0].pop("embedding")

    def fake_chunk_repository_with_status(
        root, *, max_chunks, exclude_dirs, exclude_files
    ):
        assert exclude_dirs == {"scratch", "scripts"}
        assert exclude_files == {"tests/test_tools.py"}
        return chunks, False

    monkeypatch.setattr(
        code_indexer,
        "chunk_repository_with_status",
        fake_chunk_repository_with_status,
    )

    client = FakeClient()
    store = build_code_index(
        Path("."),
        client=client,
        exclude_dirs={"scratch", "scripts"},
        exclude_files={"tests/test_tools.py"},
    )

    assert len(store) == 1
    assert len(client.models.calls) == 1


def test_build_code_index_records_indexed_and_unindexed_files(monkeypatch, tmp_path):
    chunks = [make_chunk("included.py", "print('hello')")]
    chunks[0].pop("embedding")

    monkeypatch.setattr(
        code_indexer,
        "chunk_repository_with_status",
        lambda root, **kwargs: (chunks, False),
    )
    monkeypatch.setattr(
        code_indexer,
        "build_source_manifest",
        lambda root, **kwargs: {
            "included.py": {"size": 14, "mtime_ns": 100},
            "not_indexed.py": {"size": 10, "mtime_ns": 200},
        },
    )

    store = build_code_index(tmp_path, client=FakeClient())

    assert store.metadata["indexed_files"] == ["included.py"]
    assert store.metadata["unindexed_source_files"] == ["not_indexed.py"]
    assert set(store.metadata["source_manifest"]) == {
        "included.py",
        "not_indexed.py",
    }


def test_build_code_index_records_chunk_limit_status(monkeypatch, tmp_path):
    chunks = [make_chunk("example.py", "print('hello')")]
    chunks[0].pop("embedding")

    monkeypatch.setattr(
        code_indexer,
        "chunk_repository_with_status",
        lambda root, **kwargs: (chunks, True),
    )
    monkeypatch.setattr(
        code_indexer,
        "build_source_manifest",
        lambda root, **kwargs: {
            "example.py": {"size": 14, "mtime_ns": 100},
        },
    )

    store = build_code_index(tmp_path, client=FakeClient(), max_chunks=1)

    assert store.metadata["max_chunks"] == 1
    assert store.metadata["chunk_limit_reached"] is True


# ---------------------------------------------------------------------------
# Incremental updates
# ---------------------------------------------------------------------------


def test_update_code_index_does_not_reembed_unchanged_files(monkeypatch, tmp_path):
    manifest = {"app.py": {"size": 10, "mtime_ns": 100}}
    store = make_indexed_store(
        tmp_path,
        [make_chunk("app.py", "original code")],
        manifest,
    )

    monkeypatch.setattr(
        code_indexer,
        "build_source_manifest",
        lambda root, **kwargs: manifest,
    )

    def unexpected_call(*args, **kwargs):
        raise AssertionError("Unchanged files should not be reprocessed")

    monkeypatch.setattr(code_indexer, "chunk_files", unexpected_call)
    monkeypatch.setattr(code_indexer, "embed_chunks", unexpected_call)

    result = update_code_index(store, tmp_path)

    assert result is store
    assert len(store) == 1
    assert store.chunks[0]["content"] == "original code"


def test_update_code_index_indexes_added_file(monkeypatch, tmp_path):
    saved_manifest = {"app.py": {"size": 10, "mtime_ns": 100}}
    current_manifest = {
        **saved_manifest,
        "new.py": {"size": 12, "mtime_ns": 200},
    }
    store = make_indexed_store(
        tmp_path,
        [make_chunk("app.py")],
        saved_manifest,
    )
    new_chunk = make_chunk("new.py", "new code")
    new_chunk.pop("embedding")
    embedded_files = []

    monkeypatch.setattr(
        code_indexer,
        "build_source_manifest",
        lambda root, **kwargs: current_manifest,
    )

    def fake_chunk_files(file_paths, root, **kwargs):
        assert file_paths == {"new.py"}
        return [new_chunk]

    def fake_embed_chunks(chunks, **kwargs):
        embedded_files.extend(chunk["file_path"] for chunk in chunks)
        return [{**chunk, "embedding": [0.0, 1.0]} for chunk in chunks]

    monkeypatch.setattr(code_indexer, "chunk_files", fake_chunk_files)
    monkeypatch.setattr(code_indexer, "embed_chunks", fake_embed_chunks)

    result = update_code_index(store, tmp_path)

    assert result is store
    assert len(store) == 2
    assert embedded_files == ["new.py"]
    assert store.metadata["indexed_files"] == ["app.py", "new.py"]


def test_update_code_index_replaces_modified_file_chunks(monkeypatch, tmp_path):
    saved_manifest = {"app.py": {"size": 10, "mtime_ns": 100}}
    current_manifest = {"app.py": {"size": 20, "mtime_ns": 200}}
    store = make_indexed_store(
        tmp_path,
        [make_chunk("app.py", "old code")],
        saved_manifest,
    )
    new_chunk = make_chunk("app.py", "updated code")
    new_chunk.pop("embedding")
    embedded_files = []

    monkeypatch.setattr(
        code_indexer,
        "build_source_manifest",
        lambda root, **kwargs: current_manifest,
    )
    monkeypatch.setattr(
        code_indexer,
        "chunk_files",
        lambda file_paths, root, **kwargs: [new_chunk],
    )

    def fake_embed_chunks(chunks, **kwargs):
        embedded_files.extend(chunk["file_path"] for chunk in chunks)
        return [{**chunk, "embedding": [0.0, 1.0]} for chunk in chunks]

    monkeypatch.setattr(code_indexer, "embed_chunks", fake_embed_chunks)

    update_code_index(store, tmp_path)

    assert len(store) == 1
    assert store.chunks[0]["content"] == "updated code"
    assert embedded_files == ["app.py"]
    assert store.metadata["source_manifest"] == current_manifest


def test_update_code_index_removes_deleted_file_without_embedding(
    monkeypatch, tmp_path
):
    saved_manifest = {
        "app.py": {"size": 10, "mtime_ns": 100},
        "old.py": {"size": 8, "mtime_ns": 100},
    }
    current_manifest = {"app.py": saved_manifest["app.py"]}
    store = make_indexed_store(
        tmp_path,
        [
            make_chunk("app.py"),
            make_chunk("old.py", "deleted code"),
        ],
        saved_manifest,
    )

    monkeypatch.setattr(
        code_indexer,
        "build_source_manifest",
        lambda root, **kwargs: current_manifest,
    )

    def unexpected_call(*args, **kwargs):
        raise AssertionError("Deleting a file should not require embedding")

    monkeypatch.setattr(code_indexer, "chunk_files", unexpected_call)
    monkeypatch.setattr(code_indexer, "embed_chunks", unexpected_call)

    update_code_index(store, tmp_path)

    assert len(store) == 1
    assert store.chunks[0]["file_path"] == "app.py"
    assert store.metadata["source_manifest"] == current_manifest
    assert store.metadata["indexed_files"] == ["app.py"]
