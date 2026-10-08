import json

from agent import run_agent


class FakeCall:
    def __init__(self, name, arguments, call_id):
        self.type = "function_call"
        self.name = name
        self.arguments = arguments
        self.id = call_id


class FakeStep:
    def __init__(self, step_type, **kwargs):
        self.type = step_type

        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeInteraction:
    def __init__(self, interaction_id, steps, output_text=None):
        self.id = interaction_id
        self.steps = steps
        self.output_text = output_text


class FakeInteractions:
    def __init__(self):
        self.calls = []
        self.call_number = 0

    def create(self, **kwargs):
        self.calls.append(kwargs)
        self.call_number += 1

        if self.call_number == 1:
            return FakeInteraction(
                interaction_id="interaction-1",
                steps=[
                    FakeCall(
                        name="retrieve_code",
                        arguments={"question": ("How does incremental indexing work?")},
                        call_id="call-1",
                    )
                ],
            )

        if self.call_number == 2:
            return FakeInteraction(
                interaction_id="interaction-2",
                steps=[],
                output_text=(
                    "Incremental indexing compares the repository "
                    "manifest and re-embeds only changed files."
                ),
            )

        raise AssertionError(
            "Fake Gemini received more interaction calls than expected."
        )


class FakeGeminiClient:
    def __init__(self):
        self.interactions = FakeInteractions()


class FakeEmbeddingModels:
    def __init__(self):
        self.calls = []

    def embed_content(self, *, model, contents):
        self.calls.append(
            {
                "model": model,
                "contents": contents,
            }
        )

        class Embedding:
            values = [1.0, 0.0, 0.0]

        class Response:
            embeddings = [Embedding()]

        return Response()


class FakeEmbeddingClient:
    def __init__(self):
        self.models = FakeEmbeddingModels()


class FakeVectorStore:
    def __init__(self):
        self.calls = []

    def search(self, query_embedding, top_k=5):
        self.calls.append(
            {
                "query_embedding": query_embedding,
                "top_k": top_k,
            }
        )

        return [
            {
                "file_path": "code_indexer.py",
                "start_line": 100,
                "end_line": 140,
                "language": "python",
                "content": ("Compare source manifests and re-index changed files."),
                "score": 0.96,
            }
        ]


def test_agent_uses_retrieve_code_and_returns_final_answer(
    monkeypatch,
):
    gemini_client = FakeGeminiClient()
    embedding_client = FakeEmbeddingClient()
    store = FakeVectorStore()

    monkeypatch.setenv(
        "GEMINI_MODEL",
        "fake-model",
    )

    answer = run_agent(
        "How does incremental indexing work?",
        client=gemini_client,
        code_store=store,
        embedding_client=embedding_client,
    )

    assert answer == (
        "Incremental indexing compares the repository "
        "manifest and re-embeds only changed files."
    )

    assert len(gemini_client.interactions.calls) == 2

    first_call = gemini_client.interactions.calls[0]

    assert first_call["model"] == "fake-model"

    assert "previous_interaction_id" not in first_call

    tool_names = [tool["name"] for tool in first_call["tools"]]

    assert "retrieve_code" in tool_names

    second_call = gemini_client.interactions.calls[1]

    assert second_call["previous_interaction_id"] == "interaction-1"

    function_results = second_call["input"]

    assert len(function_results) == 1

    function_result = function_results[0]

    assert function_result["type"] == "function_result"
    assert function_result["name"] == "retrieve_code"
    assert function_result["call_id"] == "call-1"

    payload = json.loads(function_result["result"][0]["text"])

    assert payload["ok"] is True

    assert payload["result"][0]["file_path"] == ("code_indexer.py")

    assert payload["result"][0]["score"] == 0.96

    assert embedding_client.models.calls == [
        {
            "model": "gemini-embedding-001",
            "contents": "How does incremental indexing work?",
        }
    ]

    assert store.calls == [
        {
            "query_embedding": [1.0, 0.0, 0.0],
            "top_k": 5,
        }
    ]
