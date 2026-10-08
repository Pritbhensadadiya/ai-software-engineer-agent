from tool_registry import build_default_registry


def test_default_registry_has_read_only_and_patch_tools():
    registry = build_default_registry()

    names = {tool["name"] for tool in registry.describe_tools()}

    assert names == {
        "list_files",
        "read_file",
        "write_file",
        "search_code",
        "run_tests",
        "analyze_repository",
        "apply_patch",
    }


def test_apply_patch_tool_has_expected_schema():
    registry = build_default_registry()

    tools = {tool["name"]: tool for tool in registry.describe_tools()}

    apply_patch = tools["apply_patch"]

    assert apply_patch["parameters"]["required"] == [
        "file_path",
        "old_text",
        "new_text",
    ]

    assert set(apply_patch["parameters"]["properties"]) == {
        "file_path",
        "old_text",
        "new_text",
    }

    assert apply_patch["parameters"]["additionalProperties"] is False
def test_registry_can_invoke_apply_patch(tmp_path):
    file_path = tmp_path / "example.py"

    file_path.write_text(
        'def hello():\n    return "hello"\n',
        encoding="utf-8",
    )

    registry = build_default_registry()

    # The default tool uses the real project root, so for this isolated
    # test we register a temporary-root version of the patch function.
    from tools import apply_patch

    registry._tools["apply_patch"] = registry._tools["apply_patch"].__class__(
        name="apply_patch",
        description=registry._tools["apply_patch"].description,
        parameters=registry._tools["apply_patch"].parameters,
        function=lambda file_path, old_text, new_text: apply_patch(
            file_path,
            old_text,
            new_text,
            root=tmp_path,
        ),
    )

    result = registry.invoke(
        "apply_patch",
        {
            "file_path": "example.py",
            "old_text": 'return "hello"',
            "new_text": 'return "hello world"',
        },
    )

    assert "Successfully patched" in result
    assert file_path.read_text(encoding="utf-8") == (
        'def hello():\n    return "hello world"\n'
    )