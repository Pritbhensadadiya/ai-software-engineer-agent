
import tool_registry


def test_analyze_repository_is_registered(monkeypatch):
    expected_inventory = {
        "root": "/sample/project",
        "total_files": 1,
        "languages": {"Python": 1},
        "source_files": ["main.py"],
        "configuration_files": [],
        "dependency_files": [],
        "test_files": [],
        "entry_points": ["main.py"],
        "files": ["main.py"],
    }

    def fake_analyze_repository():
        return expected_inventory

    monkeypatch.setattr(
        tool_registry,
        "analyze_repository",
        fake_analyze_repository,
    )

    registry = tool_registry.build_default_registry()

    result = registry.invoke("analyze_repository", {})

    assert result == expected_inventory
