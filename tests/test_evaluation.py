from types import SimpleNamespace

from evaluation import evaluate_workflow


def test_successful_workflow_passes_evaluation():
    result = SimpleNamespace(
        status="completed",
        stages=[
            "planner",
            "developer",
            "tester",
            "reviewer",
        ],
        output="approved",
    )

    evaluation = evaluate_workflow(result)

    assert evaluation.passed is True
    assert evaluation.score == 3
    assert "Workflow completed successfully." in evaluation.checks
    assert "All required workflow stages executed." in evaluation.checks


def test_workflow_with_debugger_gets_extra_credit():
    result = SimpleNamespace(
        status="completed",
        stages=[
            "planner",
            "developer",
            "tester",
            "debugger",
            "reviewer",
        ],
        output="approved",
    )

    evaluation = evaluate_workflow(result)

    assert evaluation.passed is True
    assert evaluation.score == 4
    assert "Debugger was invoked." in evaluation.checks


def test_failed_workflow_does_not_pass():
    result = SimpleNamespace(
        status="failed",
        stages=[
            "planner",
            "developer",
            "tester",
            "debugger",
        ],
        output="still failing",
    )

    evaluation = evaluate_workflow(result)

    assert evaluation.passed is False
    assert evaluation.score == 2
def test_empty_evaluation_summary():
    from evaluation import summarize_evaluations

    summary = summarize_evaluations([])

    assert summary == {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "pass_rate": 0.0,
        "average_score": 0.0,
    }


def test_evaluation_summary():
    from evaluation import EvaluationResult, summarize_evaluations

    evaluations = [
        EvaluationResult(True, 4, []),
        EvaluationResult(True, 3, []),
        EvaluationResult(False, 2, []),
    ]

    summary = summarize_evaluations(evaluations)

    assert summary["total"] == 3
    assert summary["passed"] == 2
    assert summary["failed"] == 1
    assert summary["pass_rate"] == 2 / 3
    assert summary["average_score"] == 3.0

from orchestrator import SoftwareEngineerOrchestrator
from evaluation import evaluate_workflow


def test_evaluate_actual_orchestrator_result():
    orchestrator = SoftwareEngineerOrchestrator(
        planner=lambda task: "plan",
        developer=lambda plan: "implementation",
        tester=lambda implementation: "passed",
        debugger=lambda result: "passed",
        reviewer=lambda implementation: "approved",
    )

    workflow_result = orchestrator.run("Build a feature.")
    evaluation = evaluate_workflow(workflow_result)

    assert workflow_result.status == "completed"
    assert evaluation.passed is True
    assert evaluation.score == 3