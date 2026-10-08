
import pytest

from llm import LLMServiceError, analyze_task
from models import TaskResponse


class FakeInteraction:
    def __init__(self, output_text):
        self.output_text = output_text


class FakeInteractions:
    def __init__(self, output_text=None, error=None):
        self.output_text = output_text
        self.error = error

    def create(self, **kwargs):
        if self.error:
            raise self.error

        return FakeInteraction(self.output_text)


class FakeClient:
    def __init__(self, output_text=None, error=None):
        self.interactions = FakeInteractions(output_text, error)


def test_analyze_task_returns_structured_response():
    client = FakeClient(
        output_text=TaskResponse(
            summary="Add a login endpoint.",
            implementation_plan=[
                "Inspect existing routes",
                "Implement the endpoint",
                "Add tests",
            ],
            assumptions=["A user model exists"],
            risks=["Validate credentials securely"],
        ).model_dump_json()
    )

    result = analyze_task("Add a login endpoint.", client=client)

    assert isinstance(result, TaskResponse)
    assert result.summary == "Add a login endpoint."
    assert len(result.implementation_plan) == 3


def test_analyze_task_rejects_empty_input():
    with pytest.raises(ValueError, match="must not be empty"):
        analyze_task("   ", client=FakeClient())


def test_analyze_task_handles_api_failure():
    client = FakeClient(error=RuntimeError("Simulated API failure"))

    with pytest.raises(LLMServiceError, match="request or response"):
        analyze_task("Add a login endpoint.", client=client)


def test_analyze_task_rejects_invalid_response():
    client = FakeClient(output_text='{"summary": "Incomplete"}')

    with pytest.raises(LLMServiceError, match="validation failed"):
        analyze_task("Add a login endpoint.", client=client)
