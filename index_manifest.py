from pathlib import Path
from typing import Any

from code_chunker import LANGUAGE_BY_EXTENSION
from tools import (
    IGNORED_DIRS,
    MAX_FILE_BYTES,
    PROJECT_ROOT,
    list_repository_files,
)


def build_source_manifest(
    root: Path = PROJECT_ROOT,
    *,
    exclude_dirs: set[str] | None = None,
    exclude_files: set[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Build a manifest of supported source files eligible for indexing."""
    root = root.resolve()

    if not root.is_dir():
        raise ValueError("Repository root must be an existing directory.")

    excluded_dirs = IGNORED_DIRS | (exclude_dirs or set())
    excluded_files_normalized = {
        Path(file_path).as_posix() for file_path in (exclude_files or set())
    }

    manifest: dict[str, dict[str, int]] = {}

    for relative_path in list_repository_files(root=root):
        path = Path(relative_path)
        normalized_path = path.as_posix()

        if normalized_path in excluded_files_normalized:
            continue

        if any(part in excluded_dirs for part in path.parts[:-1]):
            continue

        if path.suffix.lower() not in LANGUAGE_BY_EXTENSION:
            continue

        absolute_path = root / path

        try:
            stat = absolute_path.stat()
        except OSError:
            continue

        if stat.st_size > MAX_FILE_BYTES:
            continue

        manifest[normalized_path] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    return manifest


def compare_manifests(
    saved: dict[str, dict[str, Any]],
    current: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    """Identify added, modified, and deleted files between two manifests."""
    saved_paths = set(saved)
    current_paths = set(current)

    added = sorted(current_paths - saved_paths)
    deleted = sorted(saved_paths - current_paths)
    modified = sorted(
        path for path in saved_paths & current_paths if saved[path] != current[path]
    )

    return {
        "added": added,
        "modified": modified,
        "deleted": deleted,
    }
def format_manifest_changes(
    changes: dict[str, list[str]],
) -> list[str]:
    """Format manifest differences into readable warning lines."""
    messages = []

    for change_type in ("added", "modified", "deleted"):
        paths = changes.get(change_type, [])
        if paths:
            label = change_type.capitalize()
            messages.append(f"{label} files ({len(paths)}):")
            messages.extend(f"  - {path}" for path in paths)

    return messages