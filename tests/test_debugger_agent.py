import json
from types import SimpleNamespace

import pytest

from debugger_agent import run_debugger
from tool_registry import build_default_registry
def bind_tool(registry, name, function):
    registry._tools[name] = registry._tools[name].__class__(
        name=registry._tools[name].name,
        description=registry._tools[name].description,
        parameters=registry._tools[name].parameters,
        function=function,
    )

class FakeDebuggerClient:
    """Simulates a debugger that inspects a file and returns a diagnosis."""

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        if len(self.calls) == 1:
            return SimpleNamespace(
                id="debugger-interaction-1",
                steps=[
                    SimpleNamespace(
                        type="function_call",
                        id="debug-call-001",
                        name="read_file",
                        arguments={"path": "example.py"},
                    )
                ],
                output_text=None,
            )

        submitted_results = kwargs["input"]

        assert len(submitted_results) == 1
        assert submitted_results[0]["name"] == "read_file"
        assert submitted_results[0]["call_id"] == "debug-call-001"

        payload = json.loads(submitted_results[0]["result"][0]["text"])

        assert payload["ok"] is True
        assert "def example():" in payload["result"]

        return SimpleNamespace(
            id="debugger-interaction-2",
            steps=[SimpleNamespace(type="message")],
            output_text="The failure is caused by an incorrect expected value.",
        )


def test_debugger_rejects_empty_task():
    with pytest.raises(ValueError, match="must not be empty"):
        run_debugger("", client=FakeDebuggerClient())


def test_debugger_requires_client():
    with pytest.raises(RuntimeError, match="client is required"):
        run_debugger("Debug the failing test.")


def test_debugger_inspects_code_and_returns_diagnosis():
    client = FakeDebuggerClient()

    from tool_registry import build_default_registry

    registry = build_default_registry()

    bind_tool(
        registry,
        "read_file",
        lambda path: "def example():\n    return 42\n",
    )

    result = run_debugger(
        "Debug the failing test in example.py.",
        client=client,
        registry=registry,
    )

    assert result == ("The failure is caused by an incorrect expected value.")
    assert len(client.calls) == 2

    assert client.calls[1]["previous_interaction_id"] == ("debugger-interaction-1")
class FakeDebuggerFixClient:
    """Simulates a debugger that fixes a failure and reruns tests."""

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        if len(self.calls) == 1:
            return SimpleNamespace(
                id="debug-fix-1",
                steps=[
                    SimpleNamespace(
                        type="function_call",
                        id="debug-run-001",
                        name="run_tests",
                        arguments={"test_path": "test_example.py"},
                    )
                ],
                output_text=None,
            )

        if len(self.calls) == 2:
            return SimpleNamespace(
                id="debug-fix-2",
                steps=[
                    SimpleNamespace(
                        type="function_call",
                        id="debug-patch-001",
                        name="apply_patch",
                        arguments={
                            "file_path": "example.py",
                            "old_text": "return 41",
                            "new_text": "return 42",
                        },
                    )
                ],
                output_text=None,
            )

        if len(self.calls) == 3:
            return SimpleNamespace(
                id="debug-fix-3",
                steps=[
                    SimpleNamespace(
                        type="function_call",
                        id="debug-run-002",
                        name="run_tests",
                        arguments={"test_path": "test_example.py"},
                    )
                ],
                output_text=None,
            )

        return SimpleNamespace(
            id="debug-fix-4",
            steps=[SimpleNamespace(type="message")],
            output_text="The failure was fixed and the tests now pass.",
        )


def test_debugger_fixes_and_reruns_tests():
    client = FakeDebuggerFixClient()
    registry = build_default_registry()

    run_count = {"value": 0}
    patch_calls = []

    def fake_run_tests(test_path):
        run_count["value"] += 1

        if run_count["value"] == 1:
            return {
                "return_code": 1,
                "stdout": "AssertionError: assert example() == 42",
                "stderr": "",
            }

        return {
            "return_code": 0,
            "stdout": "1 passed",
            "stderr": "",
        }

    def fake_apply_patch(file_path, old_text, new_text):
        patch_calls.append(
            {
                "file_path": file_path,
                "old_text": old_text,
                "new_text": new_text,
            }
        )
        return "Successfully patched example.py."

    bind_tool(registry, "run_tests", fake_run_tests)
    bind_tool(registry, "apply_patch", fake_apply_patch)

    result = run_debugger(
        "Debug the failing test, fix the problem, and rerun the tests.",
        client=client,
        registry=registry,
    )

    assert result == "The failure was fixed and the tests now pass."
    assert len(client.calls) == 4
    assert run_count["value"] == 2

    assert patch_calls == [
        {
            "file_path": "example.py",
            "old_text": "return 41",
            "new_text": "return 42",
        }
    ]
def test_debugger_reports_persistent_failure():
    client = FakeDebuggerFixClient()
    registry = build_default_registry()

    run_count = {"value": 0}

    def always_failing_tests(test_path):
        run_count["value"] += 1
        return {
            "return_code": 1,
            "stdout": "1 failed",
            "stderr": "",
        }

    bind_tool(registry, "run_tests", always_failing_tests)

    bind_tool(
        registry,
        "apply_patch",
        lambda file_path, old_text, new_text: "Successfully patched example.py.",
    )

    result = run_debugger(
        "Fix the failing test and report if it still fails.",
        client=client,
        registry=registry,
    )

    assert result == "The failure was fixed and the tests now pass."
    assert run_count["value"] == 2


def test_debugger_handles_tool_errors():
    class FakeToolErrorClient:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)

            if len(self.calls) == 1:
                return SimpleNamespace(
                    id="debug-error-1",
                    steps=[
                        SimpleNamespace(
                            type="function_call",
                            id="debug-error-call",
                            name="run_tests",
                            arguments={"test_path": "missing.py"},
                        )
                    ],
                    output_text=None,
                )

            payload = json.loads(kwargs["input"][0]["result"][0]["text"])

            assert payload["ok"] is False
            assert "missing" in payload["error"].lower()

            return SimpleNamespace(
                id="debug-error-2",
                steps=[SimpleNamespace(type="message")],
                output_text="The test execution failed.",
            )

    registry = build_default_registry()

    def failing_run_tests(test_path):
        raise ValueError("Test path missing.py does not exist.")

    bind_tool(registry, "run_tests", failing_run_tests)

    result = run_debugger(
        "Run the missing test.",
        client=FakeToolErrorClient(),
        registry=registry,
    )

    assert result == "The test execution failed."