import json

import pytest

from vector_store import VectorStore


@pytest.fixture
def sample_chunks():
    return [
        {
            "file_path": "tools.py",
            "start_line": 1,
            "end_line": 5,
            "language": "Python",
            "content": "def read_file(path): ...",
            "embedding": [1.0, 0.0],
        },
        {
            "file_path": "agent.py",
            "start_line": 10,
            "end_line": 15,
            "language": "Python",
            "content": "def run_agent(): ...",
            "embedding": [0.0, 1.0],
        },
    ]


def test_save_and_load_preserves_chunks(tmp_path, sample_chunks):
    store = VectorStore()
    store.add(sample_chunks)

    index_path = tmp_path / "code_index.json"
    store.save(index_path, model="gemini-embedding-001")

    loaded = VectorStore.load(
        index_path,
        expected_model="gemini-embedding-001",
    )

    assert len(loaded) == 2

    results = loaded.search([1.0, 0.0], top_k=1)

    assert results[0]["file_path"] == "tools.py"
    assert results[0]["content"] == "def read_file(path): ..."
    assert results[0]["score"] == pytest.approx(1.0)


def test_load_rejects_model_mismatch(tmp_path, sample_chunks):
    store = VectorStore()
    store.add(sample_chunks)

    index_path = tmp_path / "code_index.json"
    store.save(index_path, model="gemini-embedding-001")

    with pytest.raises(ValueError, match="model mismatch"):
        VectorStore.load(
            index_path,
            expected_model="different-model",
        )


def test_load_rejects_invalid_json(tmp_path):
    index_path = tmp_path / "broken.json"
    index_path.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        VectorStore.load(index_path)


def test_load_rejects_unsupported_version(tmp_path):
    index_path = tmp_path / "old.json"
    index_path.write_text(
        json.dumps(
            {
                "version": 999,
                "model": None,
                "chunks": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported index version"):
        VectorStore.load(index_path)


def test_load_rejects_invalid_embeddings(tmp_path):
    index_path = tmp_path / "invalid_embedding.json"
    index_path.write_text(
        json.dumps(
            {
                "version": 1,
                "model": None,
                "chunks": [
                    {
                        "file_path": "example.py",
                        "embedding": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-empty embedding"):
        VectorStore.load(index_path)


def test_save_and_load_empty_store(tmp_path):
    store = VectorStore()
    index_path = tmp_path / "empty.json"

    store.save(index_path)
    loaded = VectorStore.load(index_path)

    assert len(loaded) == 0


def test_load_missing_file_raises_error(tmp_path):
    index_path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        VectorStore.load(index_path)
