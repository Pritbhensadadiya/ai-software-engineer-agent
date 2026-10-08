import json
from types import SimpleNamespace

import pytest

from code_reviewer_agent import run_code_reviewer
from tool_registry import build_default_registry


def bind_tool(registry, name, function):
    registry._tools[name] = registry._tools[name].__class__(
        name=registry._tools[name].name,
        description=registry._tools[name].description,
        parameters=registry._tools[name].parameters,
        function=function,
    )


class FakeCodeReviewerClient:
    """Simulates a reviewer that inspects changed code."""

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        if len(self.calls) == 1:
            return SimpleNamespace(
                id="review-interaction-1",
                steps=[
                    SimpleNamespace(
                        type="function_call",
                        id="review-call-001",
                        name="read_file",
                        arguments={"path": "example.py"},
                    )
                ],
                output_text=None,
            )

        submitted_results = kwargs["input"]

        assert len(submitted_results) == 1
        assert submitted_results[0]["name"] == "read_file"
        assert submitted_results[0]["call_id"] == "review-call-001"

        payload = json.loads(submitted_results[0]["result"][0]["text"])

        assert payload["ok"] is True
        assert "def example():" in payload["result"]

        return SimpleNamespace(
            id="review-interaction-2",
            steps=[SimpleNamespace(type="message")],
            output_text=(
                "Changes requested: verify that the function should return a value."
            ),
        )


def test_code_reviewer_rejects_empty_task():
    with pytest.raises(ValueError, match="must not be empty"):
        run_code_reviewer(
            "",
            client=FakeCodeReviewerClient(),
        )


def test_code_reviewer_requires_client():
    with pytest.raises(RuntimeError, match="client is required"):
        run_code_reviewer("Review the changes.")


def test_code_reviewer_inspects_code_and_returns_findings():
    client = FakeCodeReviewerClient()
    registry = build_default_registry()

    bind_tool(
        registry,
        "read_file",
        lambda path: "def example():\n    return None\n",
    )

    result = run_code_reviewer(
        "Review example.py.",
        client=client,
        registry=registry,
    )

    assert result == (
        "Changes requested: verify that the function should return a value."
    )

    assert len(client.calls) == 2
    assert client.calls[1]["previous_interaction_id"] == ("review-interaction-1")
