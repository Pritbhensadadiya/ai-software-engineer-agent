from agent import _make_tool_definitions
from tool_registry import build_default_registry


def test_agent_exposes_repository_analyzer_to_gemini():
    registry = build_default_registry()

    definitions = _make_tool_definitions(registry)

    analyzer = next(
        tool for tool in definitions if tool["name"] == "analyze_repository"
    )

    assert analyzer["type"] == "function"
    assert analyzer["description"]
    assert analyzer["parameters"]["type"] == "object"
    assert analyzer["parameters"]["properties"] == {}


def test_agent_exposes_apply_patch_to_gemini():
    registry = build_default_registry()

    definitions = _make_tool_definitions(registry)

    patch_tool = next(tool for tool in definitions if tool["name"] == "apply_patch")

    assert patch_tool["type"] == "function"
    assert patch_tool["description"]
    assert patch_tool["parameters"]["type"] == "object"

    assert patch_tool["parameters"]["required"] == [
        "file_path",
        "old_text",
        "new_text",
    ]

    assert set(patch_tool["parameters"]["properties"]) == {
        "file_path",
        "old_text",
        "new_text",
    }

    assert patch_tool["parameters"]["additionalProperties"] is False
