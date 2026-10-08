import json

from tool_registry import build_default_registry


MAX_TOOL_CALLS = 20

DEBUGGER_INSTRUCTIONS = """
You are a software debugging agent.

Your job is to investigate failing tests and determine the cause of the
failure, then make a safe correction when appropriate.

Workflow:
- inspect the relevant source and test files,
- understand the failure,
- identify the likely root cause,
- make the smallest appropriate code or test change,
- run the relevant tests,
- if the tests still fail, continue investigating and correcting,
- never claim a fix succeeded unless the tests actually pass.

Focus on preserving existing functionality and making minimal changes.
"""


def run_debugger(
    task: str,
    client=None,
    registry=None,
) -> str:
    if not task or not task.strip():
        raise ValueError("Task must not be empty.")

    if client is None:
        raise RuntimeError("A client is required for debugging.")

    if registry is None:
        registry = build_default_registry()

    tool_call_count = 0

    interaction = client.create(
        model="debugger",
        input=f"{DEBUGGER_INSTRUCTIONS}\n\nTask:\n{task}",
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
            raise RuntimeError("Debugger exceeded the maximum number of tool calls.")

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
            model="debugger",
            input=function_results,
            previous_interaction_id=interaction.id,
            tools=registry.describe_tools(),
        )

    answer = getattr(interaction, "output_text", None)

    if not answer or not answer.strip():
        raise RuntimeError("Gemini returned no final answer.")

    return answer
