from workflow import run_workflow


def test_run_workflow_rejects_empty_task():
    try:
        run_workflow("")
    except ValueError as exc:
        assert str(exc) == "Task must not be empty."
    else:
        raise AssertionError("Expected ValueError.")


def test_run_workflow_builds_orchestrator(monkeypatch):
    captured = {}

    fake_agents = {
        "planner": lambda task: "plan",
        "developer": lambda plan: "implementation",
        "tester": lambda implementation: "passed",
        "debugger": lambda result: "passed",
        "reviewer": lambda implementation: "approved",
    }

    monkeypatch.setattr(
        "workflow.build_agents",
        lambda **kwargs: fake_agents,
    )

    result = run_workflow("Build a feature.")

    assert result.status == "completed"
    assert result.output == "approved"
    assert result.stages == [
        "planner",
        "developer",
        "tester",
        "reviewer",
    ]
