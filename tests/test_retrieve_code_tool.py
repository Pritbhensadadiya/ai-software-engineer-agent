from code_retriever import retrieve_code
from tool_registry import ToolRegistry, ToolSpec


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
                "start_line": 10,
                "end_line": 25,
                "language": "python",
                "content": "def update_code_index(...):",
                "score": 0.95,
            }
        ]


def test_retrieve_code_returns_relevant_chunks():
    embedding_client = FakeEmbeddingClient()
    store = FakeVectorStore()

    results = retrieve_code(
        "How does incremental indexing work?",
        store,
        client=embedding_client,
        top_k=3,
    )

    assert results == [
        {
            "file_path": "code_indexer.py",
            "start_line": 10,
            "end_line": 25,
            "language": "python",
            "content": "def update_code_index(...):",
            "score": 0.95,
        }
    ]

    assert embedding_client.models.calls == [
        {
            "model": "gemini-embedding-001",
            "contents": "How does incremental indexing work?",
        }
    ]

    assert store.calls == [
        {
            "query_embedding": [1.0, 0.0, 0.0],
            "top_k": 3,
        }
    ]


def test_retrieve_code_can_be_exposed_through_registry():
    embedding_client = FakeEmbeddingClient()
    store = FakeVectorStore()

    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name="retrieve_code",
            description="Retrieve semantically relevant code.",
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": ("Natural-language question about the code."),
                    }
                },
                "required": ["question"],
            },
            function=lambda question: retrieve_code(
                question,
                store,
                client=embedding_client,
                top_k=5,
            ),
        )
    )

    result = registry.invoke(
        "retrieve_code",
        {
            "question": "Where is repository indexing implemented?",
        },
    )

    assert len(result) == 1
    assert result[0]["file_path"] == "code_indexer.py"
    assert result[0]["score"] == 0.95

    assert embedding_client.models.calls == [
        {
            "model": "gemini-embedding-001",
            "contents": "Where is repository indexing implemented?",
        }
    ]

    assert store.calls == [
        {
            "query_embedding": [1.0, 0.0, 0.0],
            "top_k": 5,
        }
    ]
