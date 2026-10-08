import pytest

from agent_factory import build_agents
from orchestrator import SoftwareEngineerOrchestrator


def test_orchestrator_runs_all_stages():
    calls = []

    def planner(task):
        calls.append(("planner", task))
        return "plan"

    def developer(plan):
        calls.append(("developer", plan))
        return "implementation"

    def tester(implementation):
        calls.append(("tester", implementation))
        return "passed"

    def debugger(result):
        calls.append(("debugger", result))
        return "passed"

    def reviewer(implementation):
        calls.append(("reviewer", implementation))
        return "approved"

    orchestrator = SoftwareEngineerOrchestrator(
        planner,
        developer,
        tester,
        debugger,
        reviewer,
    )

    result = orchestrator.run("Add a feature.")

    assert result.status == "completed"
    assert result.stages == [
        "planner",
        "developer",
        "tester",
        "reviewer",
    ]
    assert result.output == "approved"

    assert calls == [
        ("planner", "Add a feature."),
        ("developer", "plan"),
        ("tester", "implementation"),
        ("reviewer", "implementation"),
    ]


def test_orchestrator_runs_debugger_after_test_failure():
    calls = []

    def planner(task):
        calls.append("planner")
        return "plan"

    def developer(plan):
        calls.append("developer")
        return "implementation"

    def tester(implementation):
        calls.append("tester")
        return "failed"

    def debugger(result):
        calls.append("debugger")
        return "passed"

    def reviewer(implementation):
        calls.append("reviewer")
        return "approved"

    orchestrator = SoftwareEngineerOrchestrator(
        planner,
        developer,
        tester,
        debugger,
        reviewer,
    )

    result = orchestrator.run("Fix the feature.")

    assert result.status == "completed"
    assert result.stages == [
        "planner",
        "developer",
        "tester",
        "debugger",
        "reviewer",
    ]
    assert result.output == "approved"

    assert calls == [
        "planner",
        "developer",
        "tester",
        "debugger",
        "reviewer",
    ]


def test_orchestrator_stops_after_failed_debugging():
    def planner(task):
        return "plan"

    def developer(plan):
        return "implementation"

    def tester(implementation):
        return "failed"

    def debugger(result):
        return "still failing"

    def reviewer(implementation):
        pytest.fail("Reviewer should not run after unrecoverable failure.")

    orchestrator = SoftwareEngineerOrchestrator(
        planner,
        developer,
        tester,
        debugger,
        reviewer,
    )

    result = orchestrator.run("Fix the feature.")

    assert result.status == "failed"
    assert result.stages == [
        "planner",
        "developer",
        "tester",
        "debugger",
        "debugger",
    ]
    assert result.output == "still failing"


def test_orchestrator_rejects_empty_task():
    orchestrator = SoftwareEngineerOrchestrator(
        lambda task: None,
        lambda plan: None,
        lambda implementation: "passed",
        lambda result: "passed",
        lambda implementation: "approved",
    )

    with pytest.raises(ValueError, match="must not be empty"):
        orchestrator.run("")


def test_orchestrator_from_agents_requires_all_agents():
    agents = {
        "planner": lambda task: "plan",
        "developer": lambda plan: "implementation",
        "tester": lambda implementation: "passed",
        "debugger": lambda result: "passed",
    }

    with pytest.raises(ValueError, match="reviewer"):
        SoftwareEngineerOrchestrator.from_agents(agents)


def test_agent_factory_builds_all_agents():
    agents = build_agents()

    assert set(agents) == {
        "planner",
        "developer",
        "tester",
        "debugger",
        "reviewer",
    }

    assert all(callable(agent) for agent in agents.values())


def test_full_multi_agent_workflow():
    calls = []

    def planner(task):
        calls.append(("planner", task))
        return "PLAN: implement logging"

    def developer(plan):
        calls.append(("developer", plan))
        return "IMPLEMENTATION: logging added"

    def tester(implementation):
        calls.append(("tester", implementation))
        return "passed"

    def debugger(result):
        calls.append(("debugger", result))
        return "passed"

    def reviewer(implementation):
        calls.append(("reviewer", implementation))
        return "approved"

    orchestrator = SoftwareEngineerOrchestrator(
        planner=planner,
        developer=developer,
        tester=tester,
        debugger=debugger,
        reviewer=reviewer,
    )

    result = orchestrator.run("Add logging.")

    assert result.status == "completed"
    assert result.stages == [
        "planner",
        "developer",
        "tester",
        "reviewer",
    ]
    assert result.output == "approved"

    assert calls == [
        ("planner", "Add logging."),
        ("developer", "PLAN: implement logging"),
        ("tester", "IMPLEMENTATION: logging added"),
        ("reviewer", "IMPLEMENTATION: logging added"),
    ]


def test_full_multi_agent_workflow_stops_on_unrecoverable_failure():
    calls = []

    def planner(task):
        calls.append("planner")
        return "PLAN"

    def developer(plan):
        calls.append("developer")
        return "IMPLEMENTATION"

    def tester(implementation):
        calls.append("tester")
        return "failed"

    def debugger(result):
        calls.append("debugger")
        return "still failing"

    def reviewer(implementation):
        calls.append("reviewer")
        pytest.fail("Reviewer must not run after unrecoverable failure.")

    orchestrator = SoftwareEngineerOrchestrator(
        planner=planner,
        developer=developer,
        tester=tester,
        debugger=debugger,
        reviewer=reviewer,
    )

    result = orchestrator.run("Build the feature.")

    assert result.status == "failed"
    assert result.stages == [
        "planner",
        "developer",
        "tester",
        "debugger",
        "debugger",
    ]
    assert result.output == "still failing"

    assert calls == [
        "planner",
        "developer",
        "tester",
        "debugger",
        "debugger",
    ]


def test_orchestrator_limits_debug_attempts():
    debug_calls = []

    def debugger(result):
        debug_calls.append(result)
        return "still failing"

    orchestrator = SoftwareEngineerOrchestrator(
        planner=lambda task: "plan",
        developer=lambda plan: "implementation",
        tester=lambda implementation: "failed",
        debugger=debugger,
        reviewer=lambda implementation: "approved",
    )

    result = orchestrator.run("Fix the feature.")

    assert result.status == "failed"
    assert len(debug_calls) == 2
    assert result.stages == [
        "planner",
        "developer",
        "tester",
        "debugger",
        "debugger",
    ]
    assert result.output == "still failing"


def test_orchestrator_stage_limit():
    orchestrator = SoftwareEngineerOrchestrator(
        planner=lambda task: "plan",
        developer=lambda plan: "implementation",
        tester=lambda implementation: "passed",
        debugger=lambda result: "passed",
        reviewer=lambda implementation: "approved",
    )

    orchestrator._check_stage_limit(["stage"] * 10)

    with pytest.raises(RuntimeError, match="maximum number of stages"):
        orchestrator._check_stage_limit(["stage"] * 11)
def test_orchestrator_uses_workflow_logger():
    orchestrator = SoftwareEngineerOrchestrator(
        planner=lambda task: "plan",
        developer=lambda plan: "implementation",
        tester=lambda implementation: "passed",
        debugger=lambda result: "passed",
        reviewer=lambda implementation: "approved",
    )

    result = orchestrator.run("Build the feature.")

    assert result.status == "completed"