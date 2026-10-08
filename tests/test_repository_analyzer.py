from pathlib import Path

from repository_analyzer import (
    analyze_repository,
    format_repository_summary,
)


def test_analyze_repository_classifies_files(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text(
        "print('hello')",
        encoding="utf-8",
    )
    (tmp_path / "requirements.txt").write_text(
        "pytest",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        "[project]",
        encoding="utf-8",
    )

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_main.py").write_text(
        "def test_example(): pass",
        encoding="utf-8",
    )

    inventory = analyze_repository(root=tmp_path)

    assert inventory["total_files"] == 4
    assert inventory["languages"] == {"Python": 2}
    assert inventory["entry_points"] == ["main.py"]
    assert "requirements.txt" in inventory["dependency_files"]
    assert "pyproject.toml" in inventory["configuration_files"]
    assert "tests/test_main.py" in inventory["test_files"]


def test_analyze_repository_rejects_missing_root(tmp_path):
    missing_root = tmp_path / "does_not_exist"

    try:
        analyze_repository(root=missing_root)
    except ValueError as exc:
        assert "existing directory" in str(exc)
    else:
        raise AssertionError("Expected ValueError for a missing root")


def test_format_repository_summary():
    inventory = {
        "root": "/sample/project",
        "total_files": 1,
        "languages": {"Python": 1},
        "entry_points": ["main.py"],
        "configuration_files": [],
        "dependency_files": ["requirements.txt"],
        "test_files": [],
    }

    summary = format_repository_summary(inventory)

    assert "Repository Analysis" in summary
    assert "Python (1 files)" in summary
    assert "main.py" in summary
    assert "requirements.txt" in summary
    assert "None detected" in summary
