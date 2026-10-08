from agent import run_agent


PLANNER_INSTRUCTIONS = """
You are the planning stage of an AI software engineering system.

Your job is to analyze the user's software task and produce a clear,
ordered implementation plan.

Focus on:
- understanding the requested behavior,
- identifying relevant files,
- identifying dependencies between changes,
- defining implementation steps,
- defining how the result should be tested.

Do not modify files.
Do not implement the solution.
Return only the implementation plan.
"""


def run_planner(
    task: str,
    client=None,
    registry=None,
    code_store=None,
    embedding_client=None,
) -> str:
    """Create an implementation plan for a software engineering task."""
    if not task or not task.strip():
        raise ValueError("Task must not be empty.")

    planning_task = f"{PLANNER_INSTRUCTIONS}\n\nUser task:\n{task}"

    return run_agent(
        planning_task,
        client=client,
        registry=registry,
        code_store=code_store,
        embedding_client=embedding_client,
    )
