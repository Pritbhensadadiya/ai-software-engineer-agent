from code_reviewer_agent import run_code_reviewer
from debugger_agent import run_debugger
from developer_agent import run_developer
from planner_agent import run_planner
from test_agent import run_test_agent


def build_agents(client=None, registry=None, code_store=None, embedding_client=None):
    """Build all software engineering agents."""

    return {
        "planner": lambda task: run_planner(
            task,
            client=client,
            registry=registry,
            code_store=code_store,
            embedding_client=embedding_client,
        ),
        "developer": lambda plan: run_developer(
            plan,
            client=client,
            registry=registry,
            code_store=code_store,
            embedding_client=embedding_client,
        ),
        "tester": lambda task: run_test_agent(
            task,
            client=client,
            registry=registry,
        ),
        "debugger": lambda task: run_debugger(
            task,
            client=client,
            registry=registry,
        ),
        "reviewer": lambda task: run_code_reviewer(
            task,
            client=client,
            registry=registry,
        ),
    }
