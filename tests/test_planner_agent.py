from types import SimpleNamespace

import pytest

from planner_agent import run_planner


class FakePlannerClient:
    def __init__(self):
        self.interactions = self

    def create(self, **kwargs):
        return SimpleNamespace(
            id="planner-1",
            steps=[SimpleNamespace(type="message")],
            output_text=(
                "1. Inspect the relevant files.\n"
                "2. Implement the requested change.\n"
                "3. Run tests."
            ),
        )


def test_planner_rejects_empty_task():
    with pytest.raises(ValueError, match="must not be empty"):
        run_planner("", client=FakePlannerClient())


def test_planner_returns_plan():
    result = run_planner(
        "Add a logging feature.",
        client=FakePlannerClient(),
    )

    assert "Inspect the relevant files." in result
    assert "Run tests." in result
