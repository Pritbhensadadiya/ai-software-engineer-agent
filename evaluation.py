from dataclasses import dataclass


@dataclass
class EvaluationResult:
    passed: bool
    score: int
    checks: list[str]


def evaluate_workflow(result) -> EvaluationResult:
    """Evaluate whether a multi-agent workflow completed correctly."""
    checks = []
    score = 0

    if result.status == "completed":
        checks.append("Workflow completed successfully.")
        score += 1
    else:
        checks.append("Workflow did not complete successfully.")

    required_stages = {
        "planner",
        "developer",
        "tester",
        "reviewer",
    }

    actual_stages = set(result.stages)

    if required_stages.issubset(actual_stages):
        checks.append("All required workflow stages executed.")
        score += 1
    else:
        checks.append("One or more required stages were missing.")

    if "debugger" in actual_stages:
        checks.append("Debugger was invoked.")
        score += 1
    else:
        checks.append("Debugger was not required.")

    if result.output:
        checks.append("Workflow produced an output.")
        score += 1
    else:
        checks.append("Workflow produced no output.")

    return EvaluationResult(
        passed=score >= 3,
        score=score,
        checks=checks,
    )

def summarize_evaluations(evaluations: list[EvaluationResult]) -> dict:
    """Summarize a collection of evaluation results."""
    if not evaluations:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "average_score": 0.0,
        }

    passed = sum(1 for evaluation in evaluations if evaluation.passed)
    total = len(evaluations)
    average_score = sum(evaluation.score for evaluation in evaluations) / total

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total,
        "average_score": average_score,
    }
