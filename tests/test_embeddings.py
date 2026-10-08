from types import SimpleNamespace

import pytest

from embeddings import DEFAULT_EMBEDDING_MODEL, embed_text


class FakeModels:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def embed_content(self, *, model, contents):
        self.calls.append({"model": model, "contents": contents})
        return self.response


class FakeClient:
    def __init__(self, response):
        self.models = FakeModels(response)


def make_response(values):
    return SimpleNamespace(embeddings=[SimpleNamespace(values=values)])


def test_embed_text_returns_vector():
    client = FakeClient(make_response([0.1, -0.2, 0.3]))

    result = embed_text("Find path traversal protection", client=client)

    assert result == [0.1, -0.2, 0.3]
    assert client.models.calls == [
        {
            "model": DEFAULT_EMBEDDING_MODEL,
            "contents": "Find path traversal protection",
        }
    ]


def test_embed_text_supports_custom_model():
    client = FakeClient(make_response([0.5, 0.6]))

    result = embed_text(
        "Search code",
        client=client,
        model="custom-embedding-model",
    )

    assert result == [0.5, 0.6]
    assert client.models.calls[0]["model"] == "custom-embedding-model"


@pytest.mark.parametrize("text", ["", " ", "\n\t"])
def test_embed_text_rejects_empty_input(text):
    client = FakeClient(make_response([0.1]))

    with pytest.raises(ValueError, match="must not be empty"):
        embed_text(text, client=client)

    assert client.models.calls == []


def test_embed_text_rejects_empty_model():
    client = FakeClient(make_response([0.1]))

    with pytest.raises(ValueError, match="model must not be empty"):
        embed_text("some code", client=client, model=" ")

    assert client.models.calls == []


def test_embed_text_rejects_missing_embeddings():
    client = FakeClient(SimpleNamespace(embeddings=None))

    with pytest.raises(ValueError, match="no embeddings"):
        embed_text("some code", client=client)


def test_embed_text_rejects_empty_vector():
    client = FakeClient(make_response([]))

    with pytest.raises(ValueError, match="empty vector"):
        embed_text("some code", client=client)


def test_embed_text_rejects_invalid_values():
    client = FakeClient(make_response(["not-a-number"]))

    with pytest.raises(ValueError, match="invalid values"):
        embed_text("some code", client=client)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_embed_text_rejects_non_finite_values(value):
    client = FakeClient(make_response([value]))

    with pytest.raises(ValueError, match="non-finite"):
        embed_text("some code", client=client)


def test_embed_chunks_preserves_metadata_and_adds_embeddings():
    from embeddings import embed_chunks

    client = FakeClient(make_response([0.1, 0.2, 0.3]))
    chunks = [
        {
            "file_path": "tools.py",
            "start_line": 1,
            "end_line": 5,
            "language": "Python",
            "content": "def list_files(): pass",
        },
        {
            "file_path": "agent.py",
            "start_line": 10,
            "end_line": 15,
            "language": "Python",
            "content": "def run_agent(): pass",
        },
    ]

    result = embed_chunks(chunks, client=client)

    assert len(result) == 2
    assert result[0]["file_path"] == "tools.py"
    assert result[0]["start_line"] == 1
    assert result[0]["embedding"] == [0.1, 0.2, 0.3]
    assert result[1]["file_path"] == "agent.py"
    assert result[1]["embedding"] == [0.1, 0.2, 0.3]
    assert len(client.models.calls) == 2


def test_embed_chunks_handles_empty_list():
    from embeddings import embed_chunks

    assert embed_chunks([], client=FakeClient(None)) == []
