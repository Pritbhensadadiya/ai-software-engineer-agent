import json
from types import SimpleNamespace

import pytest

from agent import run_agent
from test_agent import run_test_agent
from tool_registry import build_default_registry


def bind_tool(registry, tool_name, function):
    original = registry._tools[tool_name]

    registry._tools[tool_name] = original.__class__(
        name=original.name,
        description=original.description,
        parameters=original.parameters,
        function=function,
    )


class FakeInteractions:
    """Simulates Gemini responses without making network requests."""

    def __init__(self, return_tool_call=True):
        self.calls = []
        self.return_tool_call = return_tool_call

    def create(self, **kwargs):
        self.calls.append(kwargs)

        # First request: simulate Gemini asking to search the code.
        if len(self.calls) == 1 and self.return_tool_call:
            function_call = SimpleNamespace(
                type="function_call",
                id="fake-call-001",
                name="search_code",
                arguments={"query": "analyze_task"},
            )

            return SimpleNamespace(
                id="fake-interaction-001",
                steps=[function_call],
                output_text=None,
            )

        # Follow-up request: verify the tool result was sent back.
        if len(self.calls) == 2 and self.return_tool_call:
            submitted_results = kwargs["input"]

            assert len(submitted_results) == 1
            assert submitted_results[0]["type"] == "function_result"
            assert submitted_results[0]["name"] == "search_code"
            assert submitted_results[0]["call_id"] == "fake-call-001"

            payload_text = submitted_results[0]["result"][0]["text"]
            payload = json.loads(payload_text)

            assert isinstance(payload["ok"], bool)

            return SimpleNamespace(
                id="fake-interaction-002",
                steps=[SimpleNamespace(type="message")],
                output_text=(
                    "I searched the project for analyze_task. "
                    "The search results were returned by the tool."
                ),
            )

        # Simulate Gemini answering directly without calling tools.
        return SimpleNamespace(
            id="fake-interaction-final",
            steps=[SimpleNamespace(type="message")],
            output_text="This is a simulated final answer.",
        )


class FakeClient:
    def __init__(self, return_tool_call=True):
        self.interactions = FakeInteractions(return_tool_call)


class FakeTestAgentClient:
    """Simulates a Gemini response for the Test Agent."""

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        return SimpleNamespace(output_text="Test plan: add a test for example.py.")


class FakeToolCallingTestAgentClient:
    """Simulates a Test Agent that requests a project tool."""

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        if len(self.calls) == 1:
            return SimpleNamespace(
                id="test-agent-interaction-1",
                steps=[
                    SimpleNamespace(
                        type="function_call",
                        id="test-call-001",
                        name="read_file",
                        arguments={"path": "example.py"},
                    )
                ],
                output_text=None,
            )

        return SimpleNamespace(
            id="test-agent-interaction-2",
            steps=[SimpleNamespace(type="message")],
            output_text="Test plan created after inspecting example.py.",
        )

class FakeMultiRoundTestAgentClient:
    """Simulates a Test Agent that needs two tool-calling rounds."""

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        if len(self.calls) == 1:
            return SimpleNamespace(
                id="test-agent-interaction-1",
                steps=[
                    SimpleNamespace(
                        type="function_call",
                        id="test-call-001",
                        name="read_file",
                        arguments={"path": "example.py"},
                    )
                ],
                output_text=None,
            )

        if len(self.calls) == 2:
            submitted_results = kwargs["input"]

            assert len(submitted_results) == 1
            assert submitted_results[0]["type"] == "function_result"
            assert submitted_results[0]["name"] == "read_file"
            assert submitted_results[0]["call_id"] == "test-call-001"

            payload_text = submitted_results[0]["result"][0]["text"]
            payload = json.loads(payload_text)

            assert payload["ok"] is True
            assert "def example():" in payload["result"]

            return SimpleNamespace(
                id="test-agent-interaction-2",
                steps=[
                    SimpleNamespace(
                        type="function_call",
                        id="test-call-002",
                        name="read_file",
                        arguments={"path": "test_example.py"},
                    )
                ],
                output_text=None,
            )
        submitted_results = kwargs["input"]

        assert len(submitted_results) == 1
        assert submitted_results[0]["type"] == "function_result"
        assert submitted_results[0]["name"] == "read_file"
        assert submitted_results[0]["call_id"] == "test-call-002"

        payload_text = submitted_results[0]["result"][0]["text"]
        payload = json.loads(payload_text)

        assert payload["ok"] is True
        assert "def test_example():" in payload["result"]

        return SimpleNamespace(
            id="test-agent-interaction-3",
            steps=[SimpleNamespace(type="message")],
            output_text="Test plan created after inspecting both files.",
        )
def test_agent_executes_tool_and_returns_final_answer():
    client = FakeClient()
    registry = build_default_registry()

    answer = run_agent(
        "Find where analyze_task is defined.",
        client=client,
        registry=registry,
    )

    assert "analyze_task" in answer
    assert len(client.interactions.calls) == 2

    # First request starts the interaction.
    first_request = client.interactions.calls[0]
    assert first_request["model"]
    assert first_request["tools"]

    # Second request continues the original interaction.
    second_request = client.interactions.calls[1]
    assert second_request["previous_interaction_id"] == ("fake-interaction-001")


def test_agent_rejects_empty_task():
    client = FakeClient()

    with pytest.raises(ValueError, match="Task must not be empty"):
        run_agent("   ", client=client)

    # Invalid tasks must not trigger an API request.
    assert len(client.interactions.calls) == 0


def test_agent_returns_answer_without_tool_calls():
    client = FakeClient(return_tool_call=False)

    answer = run_agent(
        "Explain the project briefly.",
        client=client,
    )

    assert answer == "This is a simulated final answer."
    assert len(client.interactions.calls) == 1


def test_agent_sends_tool_error_back_to_model():
    from tool_registry import ToolRegistry, ToolSpec

    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name="search_code",
            description="Search project code",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
            function=lambda query: (_ for _ in ()).throw(
                RuntimeError("Simulated tool failure")
            ),
        )
    )

    client = FakeClient()

    answer = run_agent(
        "Search for analyze_task.",
        client=client,
        registry=registry,
    )

    # Verify the tool result sent back to Gemini.
    assert len(client.interactions.calls) == 2

    follow_up = client.interactions.calls[1]
    submitted_results = follow_up["input"]

    assert len(submitted_results) == 1
    assert submitted_results[0]["type"] == "function_result"
    assert submitted_results[0]["name"] == "search_code"
    assert submitted_results[0]["call_id"] == "fake-call-001"

    payload_text = submitted_results[0]["result"][0]["text"]
    payload = json.loads(payload_text)

    assert payload["ok"] is False
    assert "Simulated tool failure" in payload["error"]

    # Verify the agent still receives a final answer.
    assert "analyze_task" in answer


def test_test_agent_returns_test_plan():
    client = FakeTestAgentClient()

    result = run_test_agent(
        "Add tests for example.py",
        client=client,
    )

    assert result == "Test plan: add a test for example.py."
    assert len(client.calls) == 1
    assert "You are a software testing agent." in client.calls[0]["input"]
    assert "Task:\nAdd tests for example.py" in client.calls[0]["input"]


def test_test_agent_rejects_empty_task():
    client = FakeTestAgentClient()

    with pytest.raises(ValueError, match="Task must not be empty"):
        run_test_agent("   ", client=client)

    assert len(client.calls) == 0


def test_test_agent_accepts_registry():
    client = FakeTestAgentClient()
    registry = build_default_registry()

    result = run_test_agent(
        "Add tests for example.py",
        client=client,
        registry=registry,
    )

    assert result == "Test plan: add a test for example.py."
    assert len(client.calls) == 1


def test_test_agent_executes_tool_and_returns_final_answer(tmp_path):
    test_file = tmp_path / "example.py"
    test_file.write_text(
        "def example():\n    return 42\n",
        encoding="utf-8",
    )

    client = FakeToolCallingTestAgentClient()
    registry = build_default_registry()

    bind_tool(
        registry,
        "read_file",
        lambda path: test_file.read_text(encoding="utf-8"),
    )

    result = run_test_agent(
        "Create tests for example.py",
        client=client,
        registry=registry,
    )

    assert result == "Test plan created after inspecting example.py."
    assert len(client.calls) == 2

    second_request = client.calls[1]

    assert second_request["previous_interaction_id"] == ("test-agent-interaction-1")

    submitted_results = second_request["input"]

    assert len(submitted_results) == 1
    assert submitted_results[0]["type"] == "function_result"
    assert submitted_results[0]["name"] == "read_file"
    assert submitted_results[0]["call_id"] == "test-call-001"
def test_test_agent_sends_tool_error_back_to_model():
    from tool_registry import ToolRegistry, ToolSpec

    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name="read_file",
            description="Read a project file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
            function=lambda path: (_ for _ in ()).throw(
                RuntimeError("Simulated read failure")
            ),
        )
    )

    client = FakeToolCallingTestAgentClient()

    result = run_test_agent(
        "Create tests for example.py",
        client=client,
        registry=registry,
    )

    assert result == "Test plan created after inspecting example.py."
    assert len(client.calls) == 2

    follow_up = client.calls[1]
    submitted_results = follow_up["input"]

    assert len(submitted_results) == 1
    assert submitted_results[0]["type"] == "function_result"
    assert submitted_results[0]["name"] == "read_file"
    assert submitted_results[0]["call_id"] == "test-call-001"

    payload_text = submitted_results[0]["result"][0]["text"]
    payload = json.loads(payload_text)

    assert payload["ok"] is False
    assert "Simulated read failure" in payload["error"]
class FakeWriteFileTestAgentClient:
    """Simulates a Test Agent that requests write_file."""

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        if len(self.calls) == 1:
            return SimpleNamespace(
                id="test-agent-write-interaction-1",
                steps=[
                    SimpleNamespace(
                        type="function_call",
                        id="write-call-001",
                        name="write_file",
                        arguments={
                            "path": "test_example.py",
                            "content": ("def test_example():\n    assert 2 + 2 == 4\n"),
                        },
                    )
                ],
                output_text=None,
            )

        return SimpleNamespace(
            id="test-agent-write-interaction-2",
            steps=[SimpleNamespace(type="message")],
            output_text="Test file created successfully.",
        )

class FakeWriteAndRunTestAgentClient:
    """Simulates a Test Agent that writes a test and then runs it."""

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        if len(self.calls) == 1:
            return SimpleNamespace(
                id="test-agent-write-run-1",
                steps=[
                    SimpleNamespace(
                        type="function_call",
                        id="write-call-001",
                        name="write_file",
                        arguments={
                            "path": "test_example.py",
                            "content": ("def test_example():\n    assert 2 + 2 == 4\n"),
                        },
                    )
                ],
                output_text=None,
            )

        if len(self.calls) == 2:
            submitted_results = kwargs["input"]

            assert len(submitted_results) == 1
            assert submitted_results[0]["name"] == "write_file"
            assert submitted_results[0]["call_id"] == "write-call-001"

            return SimpleNamespace(
                id="test-agent-write-run-2",
                steps=[
                    SimpleNamespace(
                        type="function_call",
                        id="run-call-001",
                        name="run_tests",
                        arguments={"test_path": "test_example.py"},
                    )
                ],
                output_text=None,
            )

        submitted_results = kwargs["input"]

        assert len(submitted_results) == 1
        assert submitted_results[0]["name"] == "run_tests"
        assert submitted_results[0]["call_id"] == "run-call-001"

        return SimpleNamespace(
            id="test-agent-write-run-3",
            steps=[SimpleNamespace(type="message")],
            output_text="Test created and executed successfully.",
        )


def test_test_agent_creates_and_runs_a_test():
    client = FakeWriteAndRunTestAgentClient()
    registry = build_default_registry()

    written_files = {}

    bind_tool(
        registry,
        "write_file",
        lambda path, content: (
            written_files.update({path: content}) or f"Successfully wrote {path}."
        ),
    )

    bind_tool(
        registry,
        "run_tests",
        lambda test_path: {
            "return_code": 0,
            "stdout": "1 passed",
            "stderr": "",
        },
    )

    result = run_test_agent(
        "Create a test for example.py and run it.",
        client=client,
        registry=registry,
    )

    assert result == "Test created and executed successfully."

    assert len(client.calls) == 3

    assert written_files["test_example.py"] == (
        "def test_example():\n    assert 2 + 2 == 4\n"
    )

    assert client.calls[1]["previous_interaction_id"] == ("test-agent-write-run-1")

    assert client.calls[2]["previous_interaction_id"] == ("test-agent-write-run-2")
def test_test_agent_executes_write_file_and_returns_final_answer():
    client = FakeWriteFileTestAgentClient()
    registry = build_default_registry()

    written_files = {}

    bind_tool(
        registry,
        "write_file",
        lambda path, content: (
            written_files.update({path: content}) or f"Successfully wrote {path}."
        ),
    )

    result = run_test_agent(
        "Create tests for example.py",
        client=client,
        registry=registry,
    )

    assert result == "Test file created successfully."
    assert len(client.calls) == 2

    assert written_files["test_example.py"] == (
        "def test_example():\n    assert 2 + 2 == 4\n"
    )

    second_request = client.calls[1]

    assert second_request["previous_interaction_id"] == (
        "test-agent-write-interaction-1"
    )

    submitted_results = second_request["input"]

    assert len(submitted_results) == 1
    assert submitted_results[0]["type"] == "function_result"
    assert submitted_results[0]["name"] == "write_file"
    assert submitted_results[0]["call_id"] == "write-call-001"
def test_test_agent_handles_unknown_tool():
    client = FakeToolCallingTestAgentClient()
    registry = build_default_registry()

    # Make Gemini request a tool that does not exist.
    client.create = lambda **kwargs: (
        client.calls.append(kwargs)
        or (
            SimpleNamespace(
                id="test-agent-interaction-unknown",
                steps=[
                    SimpleNamespace(
                        type="function_call",
                        id="unknown-call-001",
                        name="nonexistent_tool",
                        arguments={},
                    )
                ],
                output_text=None,
            )
            if len(client.calls) == 1
            else SimpleNamespace(
                id="test-agent-final",
                steps=[SimpleNamespace(type="message")],
                output_text="Handled unknown tool correctly.",
            )
        )
    )

    result = run_test_agent(
        "Create tests.",
        client=client,
        registry=registry,
    )

    assert result == "Handled unknown tool correctly."
    assert len(client.calls) == 2

    submitted_results = client.calls[1]["input"]

    assert submitted_results[0]["type"] == "function_result"
    assert submitted_results[0]["name"] == "nonexistent_tool"
    assert submitted_results[0]["call_id"] == "unknown-call-001"

    payload_text = submitted_results[0]["result"][0]["text"]
    payload = json.loads(payload_text)

    assert payload["ok"] is False
    assert "Unknown tool" in payload["error"]
def test_test_agent_enforces_tool_call_limit():
    client = FakeToolCallingTestAgentClient()
    registry = build_default_registry()

    client.create = lambda **kwargs: (
        client.calls.append(kwargs)
        or SimpleNamespace(
            id=f"interaction-{len(client.calls)}",
            steps=[
                SimpleNamespace(
                    type="function_call",
                    id=f"call-{len(client.calls)}",
                    name="read_file",
                    arguments={"path": "example.py"},
                )
                for _ in range(21)
            ],
            output_text=None,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="maximum number of tool calls",
    ):
        run_test_agent(
            "Create tests.",
            client=client,
            registry=registry,
        )

def test_test_agent_supports_multiple_tool_rounds(tmp_path):
    example_file = tmp_path / "example.py"
    example_file.write_text(
        "def example():\n    return 42\n",
        encoding="utf-8",
    )

    test_example_file = tmp_path / "test_example.py"
    test_example_file.write_text(
        "def test_example():\n    assert example() == 42\n",
        encoding="utf-8",
    )

    client = FakeMultiRoundTestAgentClient()
    registry = build_default_registry()

    def fake_read_file(path):
        if path == "example.py":
            return example_file.read_text(encoding="utf-8")
        if path == "test_example.py":
            return test_example_file.read_text(encoding="utf-8")
        raise ValueError(f"Unexpected path: {path}")

    bind_tool(
        registry,
        "read_file",
        fake_read_file,
    )

    result = run_test_agent(
        "Create tests after inspecting both files.",
        client=client,
        registry=registry,
    )

    assert result == "Test plan created after inspecting both files."
    assert len(client.calls) == 3

    assert client.calls[1]["previous_interaction_id"] == ("test-agent-interaction-1")
    assert client.calls[2]["previous_interaction_id"] == ("test-agent-interaction-2")

    second_round_results = client.calls[2]["input"]

    assert len(second_round_results) == 1
    assert second_round_results[0]["type"] == "function_result"
    assert second_round_results[0]["name"] == "read_file"
    assert second_round_results[0]["call_id"] == "test-call-002"

def test_test_agent_uses_repository_tools_before_final_answer():
    client = FakeToolCallingTestAgentClient()
    registry = build_default_registry()

    result = run_test_agent(
        "Create tests for example.py",
        client=client,
        registry=registry,
    )

    assert result == "Test plan created after inspecting example.py."
    assert len(client.calls) == 2

    first_request = client.calls[0]

    assert first_request["tools"]

    tool_names = {tool["name"] for tool in first_request["tools"]}

    assert "read_file" in tool_names
    assert "search_code" in tool_names
    assert "apply_patch" in tool_names

def test_test_agent_creates_a_test_file(tmp_path):
    """Test Agent should create a test file when asked to add tests."""
    test_file = tmp_path / "test_example.py"

    test_file.write_text(
        "def test_example():\n    assert 2 + 2 == 4\n",
        encoding="utf-8",
    )

    assert test_file.exists()
    assert "assert 2 + 2 == 4" in test_file.read_text(encoding="utf-8")

def test_test_agent_writes_a_test_file():
    """Test Agent registry should expose the write_file tool."""
    registry = build_default_registry()

    tools = registry.describe_tools()
    tool_names = [tool["name"] for tool in tools]

    assert "write_file" in tool_names

class FakeFailingRunTestAgentClient:
    """Simulates a Test Agent that receives a failed test run."""

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        if len(self.calls) == 1:
            return SimpleNamespace(
                id="test-agent-failure-1",
                steps=[
                    SimpleNamespace(
                        type="function_call",
                        id="run-call-failure-001",
                        name="run_tests",
                        arguments={"test_path": "test_example.py"},
                    )
                ],
                output_text=None,
            )

        submitted_results = kwargs["input"]

        payload_text = submitted_results[0]["result"][0]["text"]
        payload = json.loads(payload_text)

        assert payload["ok"] is True
        assert payload["result"]["return_code"] != 0
        assert "1 failed" in payload["result"]["stdout"]

        return SimpleNamespace(
            id="test-agent-failure-2",
            steps=[SimpleNamespace(type="message")],
            output_text="The generated test failed.",
        )


def test_test_agent_receives_failed_test_result():
    client = FakeFailingRunTestAgentClient()
    registry = build_default_registry()

    bind_tool(
        registry,
        "run_tests",
        lambda test_path: {
            "return_code": 1,
            "stdout": "1 failed",
            "stderr": "",
        },
    )

    result = run_test_agent(
        "Run the generated tests.",
        client=client,
        registry=registry,
    )

    assert result == "The generated test failed."
    assert len(client.calls) == 2

class FakeFailureCorrectionTestAgentClient:
    """Simulates a Test Agent that fixes a failing test."""

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        if len(self.calls) == 1:
            return SimpleNamespace(
                id="test-agent-correction-1",
                steps=[
                    SimpleNamespace(
                        type="function_call",
                        id="run-call-001",
                        name="run_tests",
                        arguments={"test_path": "test_example.py"},
                    )
                ],
                output_text=None,
            )

        if len(self.calls) == 2:
            submitted_results = kwargs["input"]

            payload = json.loads(submitted_results[0]["result"][0]["text"])

            assert payload["ok"] is True
            assert payload["result"]["return_code"] != 0

            return SimpleNamespace(
                id="test-agent-correction-2",
                steps=[
                    SimpleNamespace(
                        type="function_call",
                        id="patch-call-001",
                        name="apply_patch",
                        arguments={
                            "file_path": "test_example.py",
                            "old_text": "assert 2 + 2 == 5",
                            "new_text": "assert 2 + 2 == 4",
                        },
                    )
                ],
                output_text=None,
            )

        submitted_results = kwargs["input"]

        assert len(submitted_results) == 1
        assert submitted_results[0]["name"] == "apply_patch"
        assert submitted_results[0]["call_id"] == "patch-call-001"

        return SimpleNamespace(
            id="test-agent-correction-3",
            steps=[SimpleNamespace(type="message")],
            output_text="The failing test was corrected.",
        )


def test_test_agent_corrects_failing_test():
    client = FakeFailureCorrectionTestAgentClient()
    registry = build_default_registry()

    patch_calls = []

    bind_tool(
        registry,
        "run_tests",
        lambda test_path: {
            "return_code": 1,
            "stdout": "1 failed",
            "stderr": "",
        },
    )

    bind_tool(
        registry,
        "apply_patch",
        lambda file_path, old_text, new_text: (
            patch_calls.append(
                {
                    "file_path": file_path,
                    "old_text": old_text,
                    "new_text": new_text,
                }
            )
            or "Successfully patched test_example.py."
        ),
    )

    result = run_test_agent(
        "Run the tests and fix any failures.",
        client=client,
        registry=registry,
    )

    assert result == "The failing test was corrected."
    assert len(client.calls) == 3

    assert patch_calls == [
        {
            "file_path": "test_example.py",
            "old_text": "assert 2 + 2 == 5",
            "new_text": "assert 2 + 2 == 4",
        }
    ]
class FakeFixAndRerunTestAgentClient:
    """Simulates a Test Agent that fixes a test and reruns it."""

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        if len(self.calls) == 1:
            return SimpleNamespace(
                id="test-agent-rerun-1",
                steps=[
                    SimpleNamespace(
                        type="function_call",
                        id="run-call-001",
                        name="run_tests",
                        arguments={"test_path": "test_example.py"},
                    )
                ],
                output_text=None,
            )

        if len(self.calls) == 2:
            return SimpleNamespace(
                id="test-agent-rerun-2",
                steps=[
                    SimpleNamespace(
                        type="function_call",
                        id="patch-call-001",
                        name="apply_patch",
                        arguments={
                            "file_path": "test_example.py",
                            "old_text": "assert 2 + 2 == 5",
                            "new_text": "assert 2 + 2 == 4",
                        },
                    )
                ],
                output_text=None,
            )

        if len(self.calls) == 3:
            submitted_results = kwargs["input"]

            assert submitted_results[0]["name"] == "apply_patch"
            assert submitted_results[0]["call_id"] == "patch-call-001"

            return SimpleNamespace(
                id="test-agent-rerun-3",
                steps=[
                    SimpleNamespace(
                        type="function_call",
                        id="run-call-002",
                        name="run_tests",
                        arguments={"test_path": "test_example.py"},
                    )
                ],
                output_text=None,
            )

        submitted_results = kwargs["input"]

        assert submitted_results[0]["name"] == "run_tests"
        assert submitted_results[0]["call_id"] == "run-call-002"

        payload = json.loads(submitted_results[0]["result"][0]["text"])

        assert payload["ok"] is True
        assert payload["result"]["return_code"] == 0
        assert "1 passed" in payload["result"]["stdout"]

        return SimpleNamespace(
            id="test-agent-rerun-4",
            steps=[SimpleNamespace(type="message")],
            output_text="The test was fixed and now passes.",
        )


def test_test_agent_fixes_and_reruns_failing_test():
    client = FakeFixAndRerunTestAgentClient()
    registry = build_default_registry()

    run_count = {"value": 0}

    bind_tool(
        registry,
        "run_tests",
        lambda test_path: (
            run_count.update(value=run_count["value"] + 1)
            or (
                {
                    "return_code": 1,
                    "stdout": "1 failed",
                    "stderr": "",
                }
                if run_count["value"] == 1
                else {
                    "return_code": 0,
                    "stdout": "1 passed",
                    "stderr": "",
                }
            )
        ),
    )

    bind_tool(
        registry,
        "apply_patch",
        lambda file_path, old_text, new_text: "Successfully patched test_example.py.",
    )

    result = run_test_agent(
        "Run the tests, fix failures, and rerun them.",
        client=client,
        registry=registry,
    )

    assert result == "The test was fixed and now passes."
    assert len(client.calls) == 4
    assert run_count["value"] == 2

def test_test_agent_sends_patch_error_back_to_model():
    class FakePatchFailureClient:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)

            if len(self.calls) == 1:
                return SimpleNamespace(
                    id="patch-failure-1",
                    steps=[
                        SimpleNamespace(
                            type="function_call",
                            id="patch-call-failure-001",
                            name="apply_patch",
                            arguments={
                                "file_path": "test_example.py",
                                "old_text": "missing text",
                                "new_text": "replacement",
                            },
                        )
                    ],
                    output_text=None,
                )

            submitted_results = kwargs["input"]

            payload = json.loads(submitted_results[0]["result"][0]["text"])

            assert payload["ok"] is False
            assert "not found" in payload["error"].lower()

            return SimpleNamespace(
                id="patch-failure-2",
                steps=[SimpleNamespace(type="message")],
                output_text="The patch could not be applied.",
            )

    client = FakePatchFailureClient()
    registry = build_default_registry()

    def failing_patch(file_path, old_text, new_text):
        raise ValueError("Patch target was not found in the file.")

    bind_tool(
        registry,
        "apply_patch",
        failing_patch,
    )

    result = run_test_agent(
        "Fix the failing test.",
        client=client,
        registry=registry,
    )

    assert result == "The patch could not be applied."
    assert len(client.calls) == 2

class FakeRepeatedFailureTestAgentClient:
    """Simulates a Test Agent whose attempted fix still fails."""

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        if len(self.calls) == 1:
            return SimpleNamespace(
                id="repeated-failure-1",
                steps=[
                    SimpleNamespace(
                        type="function_call",
                        id="run-call-001",
                        name="run_tests",
                        arguments={"test_path": "test_example.py"},
                    )
                ],
                output_text=None,
            )

        if len(self.calls) == 2:
            return SimpleNamespace(
                id="repeated-failure-2",
                steps=[
                    SimpleNamespace(
                        type="function_call",
                        id="patch-call-001",
                        name="apply_patch",
                        arguments={
                            "file_path": "test_example.py",
                            "old_text": "assert 2 + 2 == 5",
                            "new_text": "assert 2 + 2 == 4",
                        },
                    )
                ],
                output_text=None,
            )

        if len(self.calls) == 3:
            return SimpleNamespace(
                id="repeated-failure-3",
                steps=[
                    SimpleNamespace(
                        type="function_call",
                        id="run-call-002",
                        name="run_tests",
                        arguments={"test_path": "test_example.py"},
                    )
                ],
                output_text=None,
            )

        submitted_results = kwargs["input"]

        assert submitted_results[0]["name"] == "run_tests"
        assert submitted_results[0]["call_id"] == "run-call-002"

        payload = json.loads(submitted_results[0]["result"][0]["text"])

        assert payload["ok"] is True
        assert payload["result"]["return_code"] != 0
        assert "1 failed" in payload["result"]["stdout"]

        return SimpleNamespace(
            id="repeated-failure-4",
            steps=[SimpleNamespace(type="message")],
            output_text="The test still fails after the attempted fix.",
        )


def test_test_agent_handles_test_that_still_fails_after_fix():
    client = FakeRepeatedFailureTestAgentClient()
    registry = build_default_registry()

    run_count = {"value": 0}

    def fake_run_tests(test_path):
        run_count["value"] += 1

        return {
            "return_code": 1,
            "stdout": "1 failed",
            "stderr": "",
        }

    bind_tool(
        registry,
        "run_tests",
        fake_run_tests,
    )

    bind_tool(
        registry,
        "apply_patch",
        lambda file_path, old_text, new_text: "Successfully patched test_example.py.",
    )

    result = run_test_agent(
        "Run the tests, fix failures, and report if they still fail.",
        client=client,
        registry=registry,
    )

    assert result == "The test still fails after the attempted fix."
    assert len(client.calls) == 4
    assert run_count["value"] == 2