from pathlib import Path

import pytest

from tools import apply_patch


def test_apply_patch_replaces_exact_match(tmp_path: Path):
    file_path = tmp_path / "example.py"

    file_path.write_text(
        'def hello():\n    return "hello"\n',
        encoding="utf-8",
    )

    result = apply_patch(
        "example.py",
        'return "hello"',
        'return "hello world"',
        root=tmp_path,
    )

    assert "Successfully patched" in result
    assert file_path.read_text(encoding="utf-8") == (
        'def hello():\n    return "hello world"\n'
    )


def test_apply_patch_rejects_missing_target(tmp_path: Path):
    file_path = tmp_path / "example.py"

    original = 'def hello():\n    return "hello"\n'

    file_path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="Patch target was not found"):
        apply_patch(
            "example.py",
            'return "missing"',
            'return "changed"',
            root=tmp_path,
        )

    assert file_path.read_text(encoding="utf-8") == original


def test_apply_patch_rejects_ambiguous_target(tmp_path: Path):
    file_path = tmp_path / "example.py"

    original = 'def first():\n    return "same"\n\ndef second():\n    return "same"\n'

    file_path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="Patch target is ambiguous"):
        apply_patch(
            "example.py",
            'return "same"',
            'return "changed"',
            root=tmp_path,
        )

    assert file_path.read_text(encoding="utf-8") == original


def test_apply_patch_rejects_path_traversal(tmp_path: Path):
    outside_file = tmp_path.parent / "outside.py"

    outside_file.write_text(
        'print("outside")\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside the allowed project"):
        apply_patch(
            "../outside.py",
            'print("outside")',
            'print("changed")',
            root=tmp_path,
        )


def test_apply_patch_returns_unified_diff(tmp_path: Path):
    file_path = tmp_path / "example.py"

    file_path.write_text(
        'def hello():\n    return "hello"\n',
        encoding="utf-8",
    )

    result = apply_patch(
        "example.py",
        'return "hello"',
        'return "hello world"',
        root=tmp_path,
    )

    assert "--- example.py" in result
    assert "+++ example.py" in result
    assert '-    return "hello"' in result
    assert '+    return "hello world"' in result


def test_apply_patch_preserves_original_when_temporary_write_fails(
    tmp_path: Path,
    monkeypatch,
):
    file_path = tmp_path / "example.py"

    original = 'def hello():\n    return "hello"\n'

    file_path.write_text(original, encoding="utf-8")

    class FailingTemporaryFile:
        def __enter__(self):
            raise OSError("Simulated write failure")

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(
        "tools.tempfile.NamedTemporaryFile",
        lambda **kwargs: FailingTemporaryFile(),
    )

    with pytest.raises(OSError, match="Simulated write failure"):
        apply_patch(
            "example.py",
            'return "hello"',
            'return "hello world"',
            root=tmp_path,
        )

    assert file_path.read_text(encoding="utf-8") == original


def test_apply_patch_allows_noop_replacement(tmp_path: Path):
    file_path = tmp_path / "example.py"

    original = 'def hello():\n    return "hello"\n'

    file_path.write_text(original, encoding="utf-8")

    result = apply_patch(
        "example.py",
        'return "hello"',
        'return "hello"',
        root=tmp_path,
    )

    assert "Successfully patched example.py" in result
    assert file_path.read_text(encoding="utf-8") == original
