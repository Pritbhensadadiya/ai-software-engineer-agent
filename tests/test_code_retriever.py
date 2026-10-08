from types import SimpleNamespace

import pytest

from code_retriever import retrieve_code
from vector_store import VectorStore


class FakeModels:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def embed_content(self, *, model, contents):
        self.calls.append({"model": model, "contents": contents})
        return SimpleNamespace(embeddings=[SimpleNamespace(values=self.values)])


class FakeClient:
    def __init__(self, values):
        self.models = FakeModels(values)


def make_chunk(path, embedding):
    return {
        "file_path": path,
        "start_line": 1,
        "end_line": 5,
        "language": "Python",
        "content": f"Code from {path}",
        "embedding": embedding,
    }


def test_retrieve_code_returns_most_relevant_chunks():
    store = VectorStore()
    store.add(
        [
            make_chunk("unrelated.py", [0.0, 1.0]),
            make_chunk("tools.py", [1.0, 0.0]),
            make_chunk("other.py", [-1.0, 0.0]),
        ]
    )
    client = FakeClient([1.0, 0.0])

    results = retrieve_code(
        "Where is path traversal prevented?",
        store,
        top_k=2,
        client=client,
    )

    assert [item["file_path"] for item in results] == [
        "tools.py",
        "unrelated.py",
    ]
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[1]["score"] == pytest.approx(0.0)
    assert client.models.calls[0]["contents"] == ("Where is path traversal prevented?")


def test_retrieve_code_preserves_source_metadata():
    store = VectorStore()
    store.add([make_chunk("tools.py", [1.0, 0.0])])

    results = retrieve_code(
        "Find path validation",
        store,
        client=FakeClient([1.0, 0.0]),
    )

    assert results[0]["file_path"] == "tools.py"
    assert results[0]["start_line"] == 1
    assert results[0]["end_line"] == 5
    assert results[0]["language"] == "Python"
    assert results[0]["content"] == "Code from tools.py"


@pytest.mark.parametrize("question", ["", " ", "\n"])
def test_retrieve_code_rejects_empty_question(question):
    with pytest.raises(ValueError, match="Question"):
        retrieve_code(
            question,
            VectorStore(),
            client=FakeClient([1.0, 0.0]),
        )


def test_retrieve_code_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="top_k"):
        retrieve_code(
            "Find path validation",
            VectorStore(),
            top_k=0,
            client=FakeClient([1.0, 0.0]),
        )


def test_retrieve_code_returns_empty_results_for_empty_store():
    results = retrieve_code(
        "Find path validation",
        VectorStore(),
        client=FakeClient([1.0, 0.0]),
    )

    assert results == []
