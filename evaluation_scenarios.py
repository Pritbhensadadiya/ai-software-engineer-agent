from dataclasses import dataclass
from types import SimpleNamespace

from evaluation import evaluate_workflow


@dataclass
class EvaluationScenario:
    name: str
    result: object
    expected_pass: bool


SCENARIOS = [
    EvaluationScenario(
        name="successful workflow",
        result=SimpleNamespace(
            status="completed",
            stages=[
                "planner",
                "developer",
                "tester",
                "reviewer",
            ],
            output="approved",
        ),
        expected_pass=True,
    ),
    EvaluationScenario(
        name="successful workflow with debugging",
        result=SimpleNamespace(
            status="completed",
            stages=[
                "planner",
                "developer",
                "tester",
                "debugger",
                "reviewer",
            ],
            output="approved",
        ),
        expected_pass=True,
    ),
    EvaluationScenario(
        name="unrecoverable failure",
        result=SimpleNamespace(
            status="failed",
            stages=[
                "planner",
                "developer",
                "tester",
                "debugger",
            ],
            output="still failing",
        ),
        expected_pass=False,
    ),
]


def run_evaluation_scenarios() -> list[dict]:
    """Evaluate all predefined workflow scenarios."""
    results = []

    for scenario in SCENARIOS:
        evaluation = evaluate_workflow(scenario.result)

        results.append(
            {
                "name": scenario.name,
                "passed": evaluation.passed,
                "expected_pass": scenario.expected_pass,
                "score": evaluation.score,
                "checks": evaluation.checks,
            }
        )

    return results
