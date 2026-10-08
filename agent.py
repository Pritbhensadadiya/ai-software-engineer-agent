import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from google import genai

from tool_registry import ToolRegistry, build_default_registry


load_dotenv()

MAX_TOOL_ROUNDS = 8
MAX_TOOL_CALLS = 20
MAX_RETRY_ATTEMPTS = 3

EVIDENCE_TOOLS = {
    "read_file",
    "search_code",
    "retrieve_code",
}

AGENT_INSTRUCTIONS = """
You are an AI Software Engineer Agent working inside an approved project directory.

You can inspect project files using the available tools.
Use tools when you need evidence from the repository.
Use retrieve_code for natural-language questions where semantically relevant code
context will help answer the question.
Never claim you inspected a file unless a tool returned its contents.
Explain findings using the actual tool results.
Do not attempt to call tools that were not provided.
You may modify project files using the apply_patch tool when appropriate.
Before modifying a file, inspect or retrieve the relevant code so the change is based
on actual repository evidence.
Prefer small, targeted patches that preserve existing functionality.
Never claim that a file was modified unless the apply_patch tool returned a successful result.
After every successful `apply_patch`, use `read_file` to verify the modified file before giving a final answer.
Do not execute shell commands; shell execution tools are not available yet.
When you have enough evidence, provide a clear final answer instead of continuing
to call tools unnecessarily.
"""


def _make_tool_definitions(
    registry: ToolRegistry,
) -> list[dict[str, Any]]:
    """Convert registered tools into Gemini function declarations."""
    definitions = []

    for spec in registry.describe_tools():
        definitions.append(
            {
                "type": "function",
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["parameters"],
            }
        )

    return definitions


def create_interaction_with_retry(client, **kwargs):
    """Retry temporary Gemini service failures up to three times."""

    retryable_markers = (
        "503",
        "service_unavailable",
        "429",
        "resource_exhausted",
        "500",
        "502",
        "504",
        "internal_server_error",
        "deadline_exceeded",
        "temporarily unavailable",
    )

    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            return client.interactions.create(**kwargs)

        except Exception as exc:
            message = str(exc).lower()
            is_retryable = any(marker in message for marker in retryable_markers)

            if not is_retryable or attempt == MAX_RETRY_ATTEMPTS - 1:
                raise

            delay = 2**attempt

            print(
                f"[Agent] Gemini temporarily unavailable. "
                f"Retrying in {delay}s "
                f"({attempt + 1}/{MAX_RETRY_ATTEMPTS - 1})..."
            )

            time.sleep(delay)

    raise RuntimeError("Gemini request failed after retries.")


def _get_function_calls(interaction) -> list:
    """Extract function-call steps from a Gemini interaction."""
    return [
        step
        for step in interaction.steps
        if getattr(step, "type", None) == "function_call"
    ]


def run_agent(
    task: str,
    client=None,
    registry: ToolRegistry | None = None,
    code_store: Any | None = None,
    embedding_client: Any | None = None,
) -> str:
    """Run the agent's bounded tool-calling loop."""

    if not task or not task.strip():
        raise ValueError("Task must not be empty.")

    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    if client is None:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file.")

        client = genai.Client(api_key=api_key)

    registry = registry or build_default_registry(
        code_store=code_store,
        embedding_client=embedding_client,
    )

    tool_definitions = _make_tool_definitions(registry)

    print(f"[Agent] Model: {model}")
    print("[Agent] Sending task to Gemini...")

    interaction = create_interaction_with_retry(
        client,
        model=model,
        input=f"{AGENT_INSTRUCTIONS}\n\nUser task:\n{task}",
        tools=tool_definitions,
    )

    total_tool_calls = 0
    evidence_obtained = False
    pending_verifications: set[str] = set()

    for round_number in range(1, MAX_TOOL_ROUNDS + 1):
        print(f"\n[Agent] Tool round {round_number}/{MAX_TOOL_ROUNDS}")

        function_calls = _get_function_calls(interaction)

        if not function_calls:
            if pending_verifications:
                pending_files = ", ".join(sorted(pending_verifications))

                raise RuntimeError(
                    "Patch verification is required before the agent can "
                    "return a final answer. "
                    f"Unverified file(s): {pending_files}"
                )

            answer = getattr(interaction, "output_text", None)

            if answer and answer.strip():
                print("[Agent] Final answer received.")
                return answer

            raise RuntimeError(
                "Gemini returned neither tool calls nor a final text answer."
            )

        print(f"[Agent] Gemini requested {len(function_calls)} tool call(s).")

        function_results = []

        for call in function_calls:
            total_tool_calls += 1

            if total_tool_calls > MAX_TOOL_CALLS:
                raise RuntimeError("Agent exceeded the tool-call limit.")

            name = call.name
            arguments = call.arguments or {}

            print(f"\n[Agent tool call] {name}({arguments})")

            try:
                if name == "apply_patch" and not evidence_obtained:
                    raise RuntimeError(
                        "apply_patch is blocked until relevant repository "
                        "evidence has been obtained with read_file, "
                        "search_code, or retrieve_code."
                    )

                result = registry.invoke(name, arguments)

                if name in EVIDENCE_TOOLS:
                    evidence_obtained = True

                if name == "apply_patch":
                    pending_verifications.add(arguments["file_path"])

                if name == "read_file":
                    pending_verifications.discard(arguments["path"])

                result_payload = {
                    "ok": True,
                    "result": result,
                }

                print(f"[Agent] Tool '{name}' completed.")

            except Exception as exc:
                result_payload = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }

                print(f"[Agent] Tool '{name}' failed: {type(exc).__name__}: {exc}")

            function_results.append(
                {
                    "type": "function_result",
                    "name": name,
                    "call_id": call.id,
                    "result": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                result_payload,
                                ensure_ascii=False,
                                default=str,
                            ),
                        }
                    ],
                }
            )

        print(
            f"\n[Agent] Sending {len(function_results)} "
            "tool result(s) back to Gemini..."
        )

        previous_interaction_id = interaction.id

        try:
            interaction = create_interaction_with_retry(
                client,
                model=model,
                previous_interaction_id=previous_interaction_id,
                input=function_results,
                tools=tool_definitions,
            )

            print("[Agent] Gemini response received.")

        except Exception as exc:
            raise RuntimeError(
                "Gemini failed after tool execution. "
                f"Details: {type(exc).__name__}: {exc}"
            ) from exc

    raise RuntimeError(
        f"Agent reached the maximum of {MAX_TOOL_ROUNDS} tool rounds "
        f"and {total_tool_calls} tool calls without a final answer."
    )


if __name__ == "__main__":
    print("AI Software Engineer Agent — Phase 2.3")

    try:
        task = input("\nWhat should I investigate in this project?\n> ")

        answer = run_agent(task)

        print("\nAgent answer")
        print("------------")
        print(answer)

    except KeyboardInterrupt:
        print("\n[Agent] Stopped by user.")

    except Exception as exc:
        print(f"\nAgent error: {type(exc).__name__}: {exc}")
