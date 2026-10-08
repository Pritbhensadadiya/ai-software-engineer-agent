from agent import run_agent
from tool_registry import build_default_registry
from tools import apply_patch
import pytest

def bind_tool(registry, tool_name, function):
    original = registry._tools[tool_name]

    registry._tools[tool_name] = original.__class__(
        name=original.name,
        description=original.description,
        parameters=original.parameters,
        function=function,
    )


class FakeEmbeddingModels:
    def embed_content(self, *, model, contents):
        class Response:
            embeddings = [type("Embedding", (), {"values": [1.0, 0.0]})()]

        return Response()


class FakeEmbeddingClient:
    def __init__(self):
        self.models = FakeEmbeddingModels()


class FakeStore:
    def search(self, embedding, top_k):
        return [
            {
                "file_path": "example.py",
                "start_line": 1,
                "end_line": 2,
                "language": "python",
                "content": 'def hello():\n    return "hello"',
                "embedding": [1.0, 0.0],
            }
        ]


class FakeInteraction:
    def __init__(self, interaction_id, steps, output_text=None):
        self.id = interaction_id
        self.steps = steps
        self.output_text = output_text


class FakeFunctionCall:
    type = "function_call"

    def __init__(self, name, arguments, call_id):
        self.name = name
        self.arguments = arguments
        self.id = call_id


class FakeGeminiClient:
    def __init__(self):
        self.calls = []
        self.interactions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)

        call_number = len(self.calls)

        if call_number == 1:
            return FakeInteraction(
                "interaction-1",
                [
                    FakeFunctionCall(
                        "retrieve_code",
                        {
                            "question": "Where is hello implemented?",
                        },
                        "call-1",
                    )
                ],
            )

        if call_number == 2:
            return FakeInteraction(
                "interaction-2",
                [
                    FakeFunctionCall(
                        "apply_patch",
                        {
                            "file_path": "example.py",
                            "old_text": 'return "hello"',
                            "new_text": 'return "hello world"',
                        },
                        "call-2",
                    )
                ],
            )

        if call_number == 3:
            return FakeInteraction(
                "interaction-3",
                [
                    FakeFunctionCall(
                        "read_file",
                        {
                            "path": "example.py",
                        },
                        "call-3",
                    )
                ],
            )

        return FakeInteraction(
            "interaction-4",
            [],
            'Updated example.py to return "hello world".',
        )


class FakeMultiToolGeminiClient(FakeGeminiClient):
    def create(self, **kwargs):
        self.calls.append(kwargs)

        call_number = len(self.calls)

        if call_number == 1:
            return FakeInteraction(
                "interaction-1",
                [
                    FakeFunctionCall(
                        "retrieve_code",
                        {
                            "question": "Where is hello implemented?",
                        },
                        "call-1",
                    ),
                    FakeFunctionCall(
                        "read_file",
                        {
                            "path": "example.py",
                        },
                        "call-2",
                    ),
                ],
            )

        return FakeInteraction(
            "interaction-2",
            [],
            "I inspected the relevant code successfully.",
        )


class FakeVerificationGeminiClient(FakeGeminiClient):
    def create(self, **kwargs):
        self.calls.append(kwargs)

        call_number = len(self.calls)

        if call_number == 1:
            return FakeInteraction(
                "interaction-1",
                [
                    FakeFunctionCall(
                        "read_file",
                        {
                            "path": "example.py",
                        },
                        "call-1",
                    )
                ],
            )

        if call_number == 2:
            return FakeInteraction(
                "interaction-2",
                [
                    FakeFunctionCall(
                        "apply_patch",
                        {
                            "file_path": "example.py",
                            "old_text": 'return "hello"',
                            "new_text": 'return "hello world"',
                        },
                        "call-2",
                    )
                ],
            )

        if call_number == 3:
            return FakeInteraction(
                "interaction-3",
                [
                    FakeFunctionCall(
                        "read_file",
                        {
                            "path": "example.py",
                        },
                        "call-3",
                    )
                ],
            )

        return FakeInteraction(
            "interaction-4",
            [],
            'Verified example.py now returns "hello world".',
        )


class FakeFailedPatchGeminiClient(FakeGeminiClient):
    def create(self, **kwargs):
        self.calls.append(kwargs)

        call_number = len(self.calls)

        if call_number == 1:
            return FakeInteraction(
                "interaction-1",
                [
                    FakeFunctionCall(
                        "read_file",
                        {
                            "path": "example.py",
                        },
                        "call-1",
                    )
                ],
            )

        if call_number == 2:
            return FakeInteraction(
                "interaction-2",
                [
                    FakeFunctionCall(
                        "apply_patch",
                        {
                            "file_path": "example.py",
                            "old_text": 'return "missing"',
                            "new_text": 'return "hello world"',
                        },
                        "call-2",
                    )
                ],
            )

        return FakeInteraction(
            "interaction-3",
            [],
            "I could not apply the requested change because the target "
            "text was not found.",
        )


class FakeUnsafePatchGeminiClient:
    def __init__(self):
        self.interactions = self
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        if len(self.calls) == 1:
            return FakeInteraction(
                interaction_id="unsafe-patch-1",
                steps=[
                    FakeFunctionCall(
                        name="apply_patch",
                        arguments={
                            "file_path": "example.py",
                            "old_text": 'return "hello"',
                            "new_text": 'return "hello world"',
                        },
                        call_id="call-1",
                    )
                ],
            )

        return FakeInteraction(
            interaction_id="unsafe-patch-2",
            steps=[],
            output_text="I could not safely modify the file.",
        )


class FakeUnverifiedPatchGeminiClient(FakeGeminiClient):
    def create(self, **kwargs):
        self.calls.append(kwargs)

        call_number = len(self.calls)

        if call_number == 1:
            return FakeInteraction(
                "interaction-1",
                [
                    FakeFunctionCall(
                        "read_file",
                        {
                            "path": "example.py",
                        },
                        "call-1",
                    )
                ],
            )

        if call_number == 2:
            return FakeInteraction(
                "interaction-2",
                [
                    FakeFunctionCall(
                        "apply_patch",
                        {
                            "file_path": "example.py",
                            "old_text": 'return "hello"',
                            "new_text": 'return "hello world"',
                        },
                        "call-2",
                    )
                ],
            )

        return FakeInteraction(
            "interaction-3",
            [],
            "I changed the file successfully.",
        )

class FakeWrongFileVerificationGeminiClient(FakeGeminiClient):
    def create(self, **kwargs):
        self.calls.append(kwargs)

        call_number = len(self.calls)

        if call_number == 1:
            return FakeInteraction(
                "interaction-1",
                [
                    FakeFunctionCall(
                        "read_file",
                        {
                            "path": "example.py",
                        },
                        "call-1",
                    )
                ],
            )

        if call_number == 2:
            return FakeInteraction(
                "interaction-2",
                [
                    FakeFunctionCall(
                        "apply_patch",
                        {
                            "file_path": "example.py",
                            "old_text": 'return "hello"',
                            "new_text": 'return "hello world"',
                        },
                        "call-2",
                    )
                ],
            )

        if call_number == 3:
            return FakeInteraction(
                "interaction-3",
                [
                    FakeFunctionCall(
                        "read_file",
                        {
                            "path": "other.py",
                        },
                        "call-3",
                    )
                ],
            )

        return FakeInteraction(
            "interaction-4",
            [],
            "I changed the file successfully.",
        )
def test_agent_requires_verification_of_patched_file(tmp_path):
    file_path = tmp_path / "example.py"
    file_path.write_text(
        'def hello():\n    return "hello"\n',
        encoding="utf-8",
    )

    other_file = tmp_path / "other.py"
    other_file.write_text(
        'def other():\n    return "other"\n',
        encoding="utf-8",
    )

    registry = build_default_registry()
    client = FakeWrongFileVerificationGeminiClient()

    bind_tool(
        registry,
        "read_file",
        lambda path: (
            file_path.read_text(encoding="utf-8")
            if path == "example.py"
            else other_file.read_text(encoding="utf-8")
        ),
    )

    bind_tool(
        registry,
        "apply_patch",
        lambda file_path, old_text, new_text: apply_patch(
            file_path,
            old_text,
            new_text,
            root=tmp_path,
        ),
    )

    with pytest.raises(RuntimeError, match="verification"):
        run_agent(
            "Change hello to return hello world.",
            client=client,
            registry=registry,
        )

    assert file_path.read_text(encoding="utf-8") == (
        'def hello():\n    return "hello world"\n'
    )

    assert len(client.calls) == 4
def test_agent_retrieves_code_then_applies_patch(tmp_path):
    file_path = tmp_path / "example.py"

    file_path.write_text(
        'def hello():\n    return "hello"\n',
        encoding="utf-8",
    )

    embedding_client = FakeEmbeddingClient()
    store = FakeStore()
    client = FakeGeminiClient()

    registry = build_default_registry(
        code_store=store,
        embedding_client=embedding_client,
    )

    bind_tool(
        registry,
        "apply_patch",
        lambda file_path, old_text, new_text: apply_patch(
            file_path,
            old_text,
            new_text,
            root=tmp_path,
        ),
    )
    bind_tool(
        registry,
        "read_file",
        lambda path: file_path.read_text(encoding="utf-8"),
    )
    answer = run_agent(
        "Change hello to return hello world.",
        client=client,
        registry=registry,
    )

    assert answer == 'Updated example.py to return "hello world".'

    assert file_path.read_text(encoding="utf-8") == (
        'def hello():\n    return "hello world"\n'
    )

    assert len(client.calls) == 4

    assert client.calls[1]["previous_interaction_id"] == "interaction-1"
    assert client.calls[2]["previous_interaction_id"] == "interaction-2"

    second_call_input = client.calls[1]["input"]

    assert second_call_input[0]["type"] == "function_result"
    assert second_call_input[0]["name"] == "retrieve_code"
    assert '"ok": true' in second_call_input[0]["result"][0]["text"]
    assert "example.py" in second_call_input[0]["result"][0]["text"]

    third_call_input = client.calls[2]["input"]

    assert third_call_input[0]["type"] == "function_result"
    assert third_call_input[0]["name"] == "apply_patch"
    assert (
        "Successfully patched example.py" in (third_call_input[0]["result"][0]["text"])
    )
    assert '+    return \\"hello world\\"' in (third_call_input[0]["result"][0]["text"])


def test_agent_returns_patch_error_to_gemini(tmp_path):
    file_path = tmp_path / "example.py"

    original_content = 'def hello():\n    return "hello"\n'

    file_path.write_text(
        original_content,
        encoding="utf-8",
    )

    client = FakeFailedPatchGeminiClient()

    registry = build_default_registry()

    bind_tool(
        registry,
        "read_file",
        lambda path: file_path.read_text(encoding="utf-8"),
    )

    bind_tool(
        registry,
        "apply_patch",
        lambda file_path, old_text, new_text: apply_patch(
            file_path,
            old_text,
            new_text,
            root=tmp_path,
        ),
    )

    answer = run_agent(
        "Change hello to return hello world.",
        client=client,
        registry=registry,
    )

    assert answer == (
        "I could not apply the requested change because the target text was not found."
    )

    assert file_path.read_text(encoding="utf-8") == original_content

    assert len(client.calls) == 3

    failed_patch_result = client.calls[2]["input"][0]

    assert failed_patch_result["type"] == "function_result"
    assert failed_patch_result["name"] == "apply_patch"
    assert '"ok": false' in failed_patch_result["result"][0]["text"]
    assert "Patch target was not found" in (failed_patch_result["result"][0]["text"])


def test_agent_cannot_patch_outside_project_root(tmp_path):
    file_path = tmp_path / "example.py"
    outside_file = tmp_path.parent / "outside.py"

    file_path.write_text(
        'def hello():\n    return "hello"\n',
        encoding="utf-8",
    )

    outside_file.write_text(
        'secret = "unchanged"\n',
        encoding="utf-8",
    )

    registry = build_default_registry()

    bind_tool(
        registry,
        "apply_patch",
        lambda file_path, old_text, new_text: apply_patch(
            file_path,
            old_text,
            new_text,
            root=tmp_path,
        ),
    )

    try:
        registry.invoke(
            "apply_patch",
            {
                "file_path": "../outside.py",
                "old_text": 'secret = "unchanged"',
                "new_text": 'secret = "modified"',
            },
        )
    except ValueError as exc:
        assert str(exc) == "Path is outside the allowed project directory."
    else:
        raise AssertionError("Path traversal should have been rejected.")

    assert outside_file.read_text(encoding="utf-8") == ('secret = "unchanged"\n')


def test_agent_executes_multiple_tool_calls_in_one_round(tmp_path):
    file_path = tmp_path / "example.py"

    file_path.write_text(
        'def hello():\n    return "hello"\n',
        encoding="utf-8",
    )

    embedding_client = FakeEmbeddingClient()
    store = FakeStore()
    client = FakeMultiToolGeminiClient()

    registry = build_default_registry(
        code_store=store,
        embedding_client=embedding_client,
    )

    bind_tool(
        registry,
        "read_file",
        lambda path: file_path.read_text(encoding="utf-8"),
    )

    answer = run_agent(
        "Inspect example.py.",
        client=client,
        registry=registry,
    )

    assert answer == "I inspected the relevant code successfully."

    assert len(client.calls) == 2

    first_response = client.calls[0]

    assert first_response["input"].endswith("Inspect example.py.")

    second_call = client.calls[1]

    assert second_call["previous_interaction_id"] == "interaction-1"

    results = second_call["input"]

    assert len(results) == 2

    assert results[0]["type"] == "function_result"
    assert results[0]["name"] == "retrieve_code"
    assert results[0]["call_id"] == "call-1"

    assert results[1]["type"] == "function_result"
    assert results[1]["name"] == "read_file"
    assert results[1]["call_id"] == "call-2"

    assert '"ok": true' in results[0]["result"][0]["text"]
    assert '"ok": true' in results[1]["result"][0]["text"]
    assert "def hello()" in results[1]["result"][0]["text"]


def test_agent_verifies_file_after_patch(tmp_path):
    file_path = tmp_path / "example.py"

    file_path.write_text(
        'def hello():\n    return "hello"\n',
        encoding="utf-8",
    )

    client = FakeVerificationGeminiClient()

    registry = build_default_registry()

    bind_tool(
        registry,
        "read_file",
        lambda path: file_path.read_text(encoding="utf-8"),
    )

    bind_tool(
        registry,
        "apply_patch",
        lambda file_path, old_text, new_text: apply_patch(
            file_path,
            old_text,
            new_text,
            root=tmp_path,
        ),
    )

    answer = run_agent(
        "Change hello to return hello world and verify the change.",
        client=client,
        registry=registry,
    )

    assert answer == 'Verified example.py now returns "hello world".'

    assert file_path.read_text(encoding="utf-8") == (
        'def hello():\n    return "hello world"\n'
    )

    assert len(client.calls) == 4

    assert client.calls[1]["previous_interaction_id"] == "interaction-1"
    assert client.calls[2]["previous_interaction_id"] == "interaction-2"
    assert client.calls[3]["previous_interaction_id"] == "interaction-3"

    verification_input = client.calls[3]["input"]

    assert verification_input[0]["type"] == "function_result"
    assert verification_input[0]["name"] == "read_file"
    assert verification_input[0]["call_id"] == "call-3"
    assert 'return \\"hello world\\"' in (verification_input[0]["result"][0]["text"])


def test_agent_blocks_patch_before_repository_evidence(tmp_path):
    """The agent must not patch before inspecting repository evidence."""

    file_path = tmp_path / "example.py"
    original = 'def hello():\n    return "hello"\n'

    file_path.write_text(original, encoding="utf-8")

    registry = build_default_registry()

    patch_attempts = []

    def fake_apply_patch(file_path, old_text, new_text):
        patch_attempts.append(
            {
                "file_path": file_path,
                "old_text": old_text,
                "new_text": new_text,
            }
        )

        return "This patch should not have been executed."

    bind_tool(
        registry,
        "apply_patch",
        fake_apply_patch,
    )

    client = FakeUnsafePatchGeminiClient()

    result = run_agent(
        "Change hello() to return hello world.",
        client=client,
        registry=registry,
    )

    assert "safely modify" in result
    assert patch_attempts == []
    assert file_path.read_text(encoding="utf-8") == original

    second_input = client.calls[1]["input"]

    assert len(second_input) == 1

    tool_result = second_input[0]

    assert tool_result["type"] == "function_result"
    assert tool_result["name"] == "apply_patch"
    assert tool_result["call_id"] == "call-1"

    error_text = tool_result["result"][0]["text"]

    assert '"ok": false' in error_text
    assert "apply_patch is blocked" in error_text
    assert "evidence" in error_text.lower()
def test_agent_requires_verification_after_successful_patch(tmp_path):
    file_path = tmp_path / "example.py"

    file_path.write_text(
        'def hello():\n    return "hello"\n',
        encoding="utf-8",
    )

    registry = build_default_registry()
    client = FakeUnverifiedPatchGeminiClient()

    bind_tool(
        registry,
        "read_file",
        lambda path: file_path.read_text(encoding="utf-8"),
    )

    bind_tool(
        registry,
        "apply_patch",
        lambda file_path, old_text, new_text: apply_patch(
            file_path,
            old_text,
            new_text,
            root=tmp_path,
        ),
    )

    with pytest.raises(RuntimeError, match="verification"):
        run_agent(
            "Change hello to return hello world.",
            client=client,
            registry=registry,
        )

    assert file_path.read_text(encoding="utf-8") == (
        'def hello():\n    return "hello world"\n'
    )

    assert len(client.calls) == 3
    