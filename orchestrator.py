from dataclasses import dataclass

from workflow_logger import get_workflow_logger


MAX_WORKFLOW_STAGES = 10
MAX_DEBUG_ATTEMPTS = 2


@dataclass
class WorkflowResult:
    status: str
    stages: list[str]
    output: str
    error: str | None = None


class SoftwareEngineerOrchestrator:
    """Coordinate the software engineering agents."""

    def __init__(
        self,
        planner,
        developer,
        tester,
        debugger,
        reviewer,
    ):
        self.planner = planner
        self.developer = developer
        self.tester = tester
        self.debugger = debugger
        self.reviewer = reviewer
        self.logger = get_workflow_logger()

    @classmethod
    def from_agents(cls, agents):
        """Build an orchestrator from an agent mapping."""
        required = {
            "planner",
            "developer",
            "tester",
            "debugger",
            "reviewer",
        }

        missing = required - agents.keys()

        if missing:
            raise ValueError(f"Missing required agents: {', '.join(sorted(missing))}")

        return cls(
            planner=agents["planner"],
            developer=agents["developer"],
            tester=agents["tester"],
            debugger=agents["debugger"],
            reviewer=agents["reviewer"],
        )

    def _check_stage_limit(self, stages):
        """Ensure the workflow does not exceed the stage limit."""
        if len(stages) > MAX_WORKFLOW_STAGES:
            raise RuntimeError("Workflow exceeded the maximum number of stages.")

    def _failure_result(self, stages, exc):
        """Create a structured workflow failure result."""
        return WorkflowResult(
            status="failed",
            stages=stages,
            output="",
            error=f"{type(exc).__name__}: {exc}",
        )

    def run(self, task: str) -> WorkflowResult:
        if not task or not task.strip():
            raise ValueError("Task must not be empty.")

        stages = []

        try:
            self.logger.info("Starting planner stage.")
            plan = self.planner(task)
            stages.append("planner")
            self._check_stage_limit(stages)

            self.logger.info("Starting developer stage.")
            implementation = self.developer(plan)
            stages.append("developer")
            self._check_stage_limit(stages)

            self.logger.info("Starting tester stage.")
            test_result = self.tester(implementation)
            stages.append("tester")
            self._check_stage_limit(stages)

            if test_result != "passed":
                debug_result = None

                for _ in range(MAX_DEBUG_ATTEMPTS):
                    self.logger.info("Starting debugger stage.")
                    debug_result = self.debugger(test_result)
                    stages.append("debugger")
                    self._check_stage_limit(stages)

                    if debug_result == "passed":
                        break

                    test_result = debug_result

                if debug_result != "passed":
                    self.logger.error("Workflow failed after debugger attempts.")

                    return WorkflowResult(
                        status="failed",
                        stages=stages,
                        output=str(debug_result),
                    )

            self.logger.info("Starting reviewer stage.")
            review = self.reviewer(implementation)
            stages.append("reviewer")
            self._check_stage_limit(stages)

            self.logger.info("Workflow completed successfully.")

            return WorkflowResult(
                status="completed",
                stages=stages,
                output=str(review),
            )

        except Exception as exc:
            self.logger.error("Workflow failed: %s", exc)
            return self._failure_result(stages, exc)
