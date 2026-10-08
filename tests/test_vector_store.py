import json

import pytest

from vector_store import VectorStore, cosine_similarity


def make_chunk(
    path,
    embedding,
    content=None,
    start_line=1,
    end_line=5,
):
    """Create a valid test chunk."""
    return {
        "file_path": path,
        "start_line": start_line,
        "end_line": end_line,
        "language": "Python",
        "content": content if content is not None else f"Code from {path}",
        "embedding": embedding,
    }


def test_cosine_similarity_identical_vectors():
    assert cosine_similarity(
        [1.0, 0.0],
        [1.0, 0.0],
    ) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    ) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors():
    assert cosine_similarity(
        [1.0, 0.0],
        [-1.0, 0.0],
    ) == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ([], [1.0]),
        ([1.0], []),
        ([1.0], [1.0, 2.0]),
        ([0.0, 0.0], [1.0, 0.0]),
        ([float("nan")], [1.0]),
    ],
)
def test_cosine_similarity_rejects_invalid_vectors(left, right):
    with pytest.raises(ValueError):
        cosine_similarity(left, right)


def test_vector_store_search_returns_best_matches_first():
    store = VectorStore()
    store.add(
        [
            make_chunk("unrelated.py", [0.0, 1.0]),
            make_chunk("relevant.py", [1.0, 0.0]),
            make_chunk("opposite.py", [-1.0, 0.0]),
        ]
    )

    results = store.search([1.0, 0.0], top_k=2)

    assert [item["file_path"] for item in results] == [
        "relevant.py",
        "unrelated.py",
    ]
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[1]["score"] == pytest.approx(0.0)


def test_vector_store_preserves_metadata():
    store = VectorStore()
    store.add([make_chunk("tools.py", [1.0, 0.0])])

    result = store.search([1.0, 0.0])[0]

    assert result["file_path"] == "tools.py"
    assert result["start_line"] == 1
    assert result["end_line"] == 5
    assert result["language"] == "Python"
    assert "embedding" in result


def test_vector_store_rejects_invalid_top_k():
    store = VectorStore()

    with pytest.raises(ValueError, match="top_k"):
        store.search([1.0, 0.0], top_k=0)


def test_vector_store_rejects_missing_embedding():
    store = VectorStore()

    with pytest.raises(ValueError, match="embedding"):
        store.add([{"file_path": "empty.py"}])


def test_vector_store_starts_empty():
    assert len(VectorStore()) == 0


def test_load_rejects_missing_index_file(tmp_path):
    with pytest.raises(OSError):
        VectorStore.load(tmp_path / "missing.json")


def test_load_rejects_invalid_utf8(tmp_path):
    index_path = tmp_path / "invalid.json"
    index_path.write_bytes(b"\xff\xfe\xfa")

    with pytest.raises((ValueError, UnicodeDecodeError)):
        VectorStore.load(index_path)


def test_load_rejects_non_dictionary_metadata(tmp_path):
    index_path = tmp_path / "index.json"
    index_path.write_text(
        '{"version": 1, "model": null, "metadata": [], "chunks": []}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="metadata"):
        VectorStore.load(index_path)


def test_load_rejects_model_mismatch(tmp_path):
    index_path = tmp_path / "index.json"

    store = VectorStore()
    store.add([make_chunk("app.py", [1.0, 0.0])])
    store.save(index_path, model="model-a")

    with pytest.raises(ValueError, match="model mismatch"):
        VectorStore.load(index_path, expected_model="model-b")


def test_load_rejects_chunk_without_required_metadata(tmp_path):
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "version": 1,
                "model": None,
                "chunks": [{"embedding": [1.0, 0.0]}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        VectorStore.load(index_path)


def test_add_is_atomic_when_later_chunk_is_invalid():
    store = VectorStore()

    valid_chunk = make_chunk("first.py", [1.0, 0.0])
    invalid_chunk = make_chunk("second.py", [0.0, 1.0, 0.0])

    with pytest.raises(ValueError, match="same dimensions"):
        store.add([valid_chunk, invalid_chunk])

    assert len(store) == 0


def test_add_preserves_existing_chunks_when_batch_fails():
    store = VectorStore()
    store.add([make_chunk("existing.py", [1.0, 0.0])])

    valid_chunk = make_chunk("new.py", [0.0, 1.0])
    invalid_chunk = make_chunk("invalid.py", [])

    with pytest.raises(ValueError, match="non-empty embedding"):
        store.add([valid_chunk, invalid_chunk])

    assert len(store) == 1
    assert store.search([1.0, 0.0])[0]["file_path"] == "existing.py"


def test_remove_file_removes_all_chunks_for_file():
    store = VectorStore()
    store.add(
        [
            make_chunk("tools.py", [1.0, 0.0], content="chunk one"),
            make_chunk("tools.py", [0.8, 0.2], content="chunk two"),
            make_chunk("agent.py", [0.0, 1.0], content="agent chunk"),
        ]
    )

    removed = store.remove_file("tools.py")

    assert removed == 2
    assert len(store) == 1
    assert store.search([0.0, 1.0], top_k=5)[0]["file_path"] == "agent.py"


def test_remove_file_returns_zero_when_file_is_not_indexed():
    store = VectorStore()
    store.add([make_chunk("agent.py", [0.0, 1.0])])

    assert store.remove_file("missing.py") == 0
    assert len(store) == 1


@pytest.mark.parametrize("file_path", ["", "   ", None, 123])
def test_remove_file_rejects_invalid_path(file_path):
    store = VectorStore()

    with pytest.raises(ValueError, match="file_path"):
        store.remove_file(file_path)
