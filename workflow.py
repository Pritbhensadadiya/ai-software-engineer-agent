from agent_factory import build_agents
from orchestrator import SoftwareEngineerOrchestrator


def run_workflow(
    task: str,
    client=None,
    registry=None,
    code_store=None,
    embedding_client=None,
):
    """Run the complete software engineering workflow."""

    if not task or not task.strip():
        raise ValueError("Task must not be empty.")

    agents = build_agents(
        client=client,
        registry=registry,
        code_store=code_store,
        embedding_client=embedding_client,
    )

    orchestrator = SoftwareEngineerOrchestrator.from_agents(agents)

    return orchestrator.run(task)
