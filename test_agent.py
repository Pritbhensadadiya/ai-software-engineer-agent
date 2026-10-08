import json

from tool_registry import build_default_registry


MAX_TOOL_CALLS = 20

TEST_AGENT_INSTRUCTIONS = """
You are a software testing agent.

Your job is to inspect the existing project code and determine what tests
should be added or updated.

Before proposing tests, inspect the relevant implementation using the
available project tools.

Focus on:
- understanding the existing behavior,
- identifying important success and failure cases,
- preserving existing functionality,
- following the project's existing testing conventions,
- avoiding unnecessary or duplicate tests.
- creating or updating tests when the task requires it,
- running the relevant tests after making changes,

Do not claim that tests were created or executed unless the corresponding
tool operation actually succeeded.
"""
def run_test_agent(
    task: str,
    client=None,
    registry=None,
) -> str:
    if not task or not task.strip():
        raise ValueError("Task must not be empty.")

    if client is None:
        raise RuntimeError("A client is required for testing.")

    if registry is None:
        registry = build_default_registry()

    tool_call_count = 0

    interaction = client.create(
        model="test-agent",
        input=f"{TEST_AGENT_INSTRUCTIONS}\n\nTask:\n{task}",
        tools=registry.describe_tools(),
    )

    while True:
        function_calls = [
            step
            for step in getattr(interaction, "steps", [])
            if getattr(step, "type", None) == "function_call"
        ]

        if not function_calls:
            break

        tool_call_count += len(function_calls)

        if tool_call_count > MAX_TOOL_CALLS:
            raise RuntimeError("Test Agent exceeded the maximum number of tool calls.")

        function_results = []

        for call in function_calls:
            try:
                result = registry.invoke(
                    call.name,
                    call.arguments,
                )

                payload = {
                    "ok": True,
                    "result": result,
                }
            except Exception as exc:
                payload = {
                    "ok": False,
                    "error": str(exc),
                }

            function_results.append(
                {
                    "type": "function_result",
                    "name": call.name,
                    "call_id": call.id,
                    "result": [
                        {
                            "type": "text",
                            "text": json.dumps(payload),
                        }
                    ],
                }
            )

        interaction = client.create(
            model="test-agent",
            input=function_results,
            previous_interaction_id=interaction.id,
            tools=registry.describe_tools(),
        )

    answer = getattr(interaction, "output_text", None)

    if not answer or not answer.strip():
        raise RuntimeError("Gemini returned no final answer.")

    return answer
