from agent import run_agent


DEVELOPER_INSTRUCTIONS = """
You are the implementation stage of an AI software engineering system.

Your job is to implement the supplied software engineering plan.

Before modifying files:
- inspect the relevant repository code,
- understand the existing implementation,
- identify the smallest safe changes.

Use targeted patches rather than rewriting entire files.

After modifying files:
- verify every modified file,
- preserve existing functionality,
- explain what was changed.

Do not claim a change succeeded unless the corresponding tool operation
actually succeeded.
"""


def run_developer(
    plan: str,
    client=None,
    registry=None,
    code_store=None,
    embedding_client=None,
) -> str:
    """Implement a previously generated software engineering plan."""
    if not plan or not plan.strip():
        raise ValueError("Plan must not be empty.")

    implementation_task = f"{DEVELOPER_INSTRUCTIONS}\n\nImplementation plan:\n{plan}"

    return run_agent(
        implementation_task,
        client=client,
        registry=registry,
        code_store=code_store,
        embedding_client=embedding_client,
    )
