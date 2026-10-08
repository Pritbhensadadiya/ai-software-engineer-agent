from agent import _make_tool_definitions
from tool_registry import build_default_registry


class DummyStore:
    pass


def test_default_registry_does_not_enable_code_retrieval_without_store():
    registry = build_default_registry()

    names = [tool["name"] for tool in registry.describe_tools()]

    assert names == [
        "list_files",
        "read_file",
        "write_file",
        "search_code",
        "run_tests",
        "analyze_repository",
        "apply_patch",
    ]


def test_registry_registers_retrieve_code_when_store_is_provided(monkeypatch):
    store = DummyStore()
    embedding_client = object()
    captured = {}

    def fake_retrieve_code(question, received_store, *, client=None, model=None, top_k=5):
        captured.update(
            question=question,
            store=received_store,
            client=client,
        )
        return [{"file_path": "example.py", "content": "return 42"}]

    monkeypatch.setattr("tool_registry.retrieve_code", fake_retrieve_code)

    registry = build_default_registry(
        code_store=store,
        embedding_client=embedding_client,
    )

    names = [tool["name"] for tool in registry.describe_tools()]
    assert names[-1] == "retrieve_code"

    result = registry.invoke(
        "retrieve_code",
        {"question": "Where is the answer calculated?"},
    )

    assert result == [{"file_path": "example.py", "content": "return 42"}]
    assert captured == {
        "question": "Where is the answer calculated?",
        "store": store,
        "client": embedding_client,
    }


def test_retrieve_code_definition_exposes_only_question():
    registry = build_default_registry(code_store=DummyStore())

    definitions = _make_tool_definitions(registry)
    retrieval = next(item for item in definitions if item["name"] == "retrieve_code")

    assert retrieval["parameters"] == {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "Natural-language question about the project code.",
            }
        },
        "required": ["question"],
        "additionalProperties": False,
    }
