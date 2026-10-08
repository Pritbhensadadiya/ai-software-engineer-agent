import json
from dataclasses import dataclass
from typing import Any, Callable

from code_retriever import retrieve_code
from repository_analyzer import analyze_repository
from tools import (
    apply_patch,
    list_files,
    read_file,
    run_tests,
    search_code,
    write_file,
)


@dataclass(frozen=True)
class ToolSpec:
    """Description and execution function for one approved tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., Any]


class ToolRegistry:
    """Allowlisted collection of tools the agent can invoke."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")

        self._tools[spec.name] = spec

    def describe_tools(self) -> list[dict[str, Any]]:
        """Return tool definitions suitable for inspection or API adaptation."""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            }
            for spec in self._tools.values()
        ]

    def invoke(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Validate a tool request and execute an approved function."""
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")

        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object.")

        spec = self._tools[name]
        schema = spec.parameters

        allowed = set(schema.get("properties", {}))
        required = set(schema.get("required", []))
        supplied = set(arguments)

        unexpected = supplied - allowed
        missing = required - supplied

        if unexpected:
            raise ValueError(f"Unexpected arguments for {name}: {sorted(unexpected)}")

        if missing:
            raise ValueError(f"Missing arguments for {name}: {sorted(missing)}")

        # This first version accepts only string-valued arguments.
        for key, value in arguments.items():
            if not isinstance(value, str):
                raise ValueError(f"Argument '{key}' must be a string.")

        return spec.function(**arguments)


def build_default_registry(
    *,
    code_store: Any | None = None,
    embedding_client: Any | None = None,
) -> ToolRegistry:
    """Register the tools currently available to the agent."""
    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name="list_files",
            description="List files within an allowed project directory.",
            parameters={
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": (
                            "Relative directory; use '.' for the project root."
                        ),
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
            function=list_files,
        )
    )

    registry.register(
        ToolSpec(
            name="read_file",
            description="Read a UTF-8 text file inside the project.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path of the file to read.",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            function=read_file,
        )
    )
    registry.register(
        ToolSpec(
            name="write_file",
            description="Create or overwrite a UTF-8 text file inside the project.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path of the file to create or overwrite.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete UTF-8 text content to write.",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            function=write_file,
        )
    )

    registry.register(
        ToolSpec(
            name="search_code",
            description="Search project text files for a literal string.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search for.",
                    },
                    "directory": {
                        "type": "string",
                        "description": (
                            "Relative directory to search; defaults to '.'."
                        ),
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            function=search_code,
        )
    )

    registry.register(
        ToolSpec(
            name="run_tests",
            description="Run the project's pytest test suite.",
            parameters={
                "type": "object",
                "properties": {
                    "test_path": {
                        "type": "string",
                        "description": ("Test file or directory to run."),
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
            function=run_tests,
        )
    )

    registry.register(
        ToolSpec(
            name="analyze_repository",
            description=(
                "Analyze the project structure and summarize its "
                "languages, entry points, configuration files, "
                "dependency files, and tests."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            function=analyze_repository,
        )
    )

    registry.register(
        ToolSpec(
            name="apply_patch",
            description=(
                "Safely modify a project file by replacing exactly one "
                "matching text block with new text."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": ("Relative path of the file to modify."),
                    },
                    "old_text": {
                        "type": "string",
                        "description": (
                            "Exact existing text that must occur exactly "
                            "once in the file."
                        ),
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                },
                "required": [
                    "file_path",
                    "old_text",
                    "new_text",
                ],
                "additionalProperties": False,
            },
            function=apply_patch,
        )
    )

    if code_store is not None:

        def retrieve_code_for_agent(
            question: str,
        ) -> list[dict[str, Any]]:
            return retrieve_code(
                question,
                code_store,
                client=embedding_client,
            )

        registry.register(
            ToolSpec(
                name="retrieve_code",
                description=(
                    "Retrieve semantically relevant source-code chunks "
                    "from the project's code index for a natural-language "
                    "question."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": (
                                "Natural-language question about the project code."
                            ),
                        }
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
                function=retrieve_code_for_agent,
            )
        )

    return registry


def invoke_tool_json(
    registry: ToolRegistry,
    name: str,
    arguments_json: str,
) -> Any:
    """Convenience wrapper for tool arguments received as JSON text."""
    try:
        arguments = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        raise ValueError("Tool arguments are not valid JSON.") from exc

    return registry.invoke(name, arguments)
