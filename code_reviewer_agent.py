import json

from tool_registry import build_default_registry


MAX_TOOL_CALLS = 20

CODE_REVIEWER_INSTRUCTIONS = """
You are a software code review agent.

Your job is to review code changes for correctness, regressions,
maintainability, and potential problems.

Before reviewing, inspect the relevant project files using the available
tools.

Focus on:
- correctness,
- unintended behavior,
- regressions,
- error handling,
- maintainability,
- unnecessary complexity.

Return clear findings with severity and explain why each issue matters.

Do not claim that a file was inspected unless the corresponding tool
operation actually succeeded.
"""


def run_code_reviewer(
    task: str,
    client=None,
    registry=None,
) -> str:
    if not task or not task.strip():
        raise ValueError("Task must not be empty.")

    if client is None:
        raise RuntimeError("A client is required for code review.")

    if registry is None:
        registry = build_default_registry()

    tool_call_count = 0

    interaction = client.create(
        model="code-reviewer",
        input=f"{CODE_REVIEWER_INSTRUCTIONS}\n\nTask:\n{task}",
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
            raise RuntimeError(
                "Code Reviewer exceeded the maximum number of tool calls."
            )

        function_results = []

        for call in function_calls:
            try:
                result = registry.invoke(call.name, call.arguments)
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
            model="code-reviewer",
            input=function_results,
            previous_interaction_id=interaction.id,
            tools=registry.describe_tools(),
        )

    answer = getattr(interaction, "output_text", None)

    if not answer or not answer.strip():
        raise RuntimeError("Gemini returned no final answer.")

    return answer
