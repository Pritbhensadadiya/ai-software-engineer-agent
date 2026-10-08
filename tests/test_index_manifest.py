from index_manifest import build_source_manifest, compare_manifests


def test_manifest_includes_supported_source_files(tmp_path):
    (tmp_path / "app.py").write_text("print('hello')", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not source code", encoding="utf-8")

    manifest = build_source_manifest(tmp_path)

    assert "app.py" in manifest
    assert "notes.txt" not in manifest
    assert manifest["app.py"]["size"] > 0
    assert manifest["app.py"]["mtime_ns"] > 0


def test_manifest_respects_exclusions(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()

    (tmp_path / "src" / "app.py").write_text("pass", encoding="utf-8")
    (tmp_path / "src" / "generated.py").write_text("pass", encoding="utf-8")
    (tmp_path / "scripts" / "helper.py").write_text("pass", encoding="utf-8")

    manifest = build_source_manifest(
        tmp_path,
        exclude_dirs={"scripts"},
        exclude_files={"src/generated.py"},
    )

    assert "src/app.py" in manifest
    assert "src/generated.py" not in manifest
    assert "scripts/helper.py" not in manifest


def test_compare_manifests_detects_added_modified_and_deleted_files():
    saved = {
        "same.py": {"size": 10, "mtime_ns": 100},
        "changed.py": {"size": 10, "mtime_ns": 100},
        "removed.py": {"size": 10, "mtime_ns": 100},
    }
    current = {
        "same.py": {"size": 10, "mtime_ns": 100},
        "changed.py": {"size": 15, "mtime_ns": 200},
        "new.py": {"size": 5, "mtime_ns": 300},
    }

    changes = compare_manifests(saved, current)

    assert changes == {
        "added": ["new.py"],
        "modified": ["changed.py"],
        "deleted": ["removed.py"],
    }


def test_compare_manifests_returns_empty_changes_when_identical():
    manifest = {"app.py": {"size": 10, "mtime_ns": 100}}

    assert compare_manifests(manifest, manifest) == {
        "added": [],
        "modified": [],
        "deleted": [],
    }

def test_format_manifest_changes_lists_changed_files():
    from index_manifest import format_manifest_changes

    changes = {
        "added": ["new.py"],
        "modified": ["app.py"],
        "deleted": ["old.py"],
    }

    messages = format_manifest_changes(changes)

    assert messages == [
        "Added files (1):",
        "  - new.py",
        "Modified files (1):",
        "  - app.py",
        "Deleted files (1):",
        "  - old.py",
    ]


def test_format_manifest_changes_returns_empty_list_when_fresh():
    from index_manifest import format_manifest_changes

    assert format_manifest_changes({"added": [], "modified": [], "deleted": []}) == []
